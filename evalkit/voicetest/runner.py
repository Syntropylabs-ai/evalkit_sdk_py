"""The SDK run (RUN_API section 1): ``evalkit voice-test run`` and ``evalkit.voicetest.run()``.

It creates the run with the subscription key (section 3.1), places the plan's calls from this process
(so a bot on localhost or a private network is reachable), reports each through the section 4
endpoints, completes the run and waits for the evaluators' verdict.

Interrupting it (SIGINT, Ctrl-C) stops new calls and lets the calls in progress finish; a second
SIGINT, or a SIGTERM, ends those now. Either way every call is reported (the ones never started as
``not_run``) and the run is completed, so it never stays open on the server.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import datetime
import json
import logging
import os
import shutil
import signal
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .api import DEFAULT_API_URL, VoiceTestApi, VoiceTestApiError
from .net import RetryPolicy
from .plan import PlanError, RunPlan, parse_plan, target_label
from .planned_call import CallOutcome, run_planned_call
from .script import FIRST_SPEAKERS

log = logging.getLogger("evalkit.voicetest.runner")

EXIT_PASSED = 0
EXIT_FAILED = 1
EXIT_SETUP = 2
EXIT_INTERRUPTED = 130

# Statuses of a run that is still being called or scored. RUN_API does not list them, so a run is
# taken as decided once it has a boolean `passed` and a status that is not one of these.
PENDING_STATUSES = frozenset({
    "pending", "queued", "created", "open", "started", "running", "in_progress", "calling",
    "completing", "scoring", "evaluating", "processing",
})
TERMINAL_STATUSES = frozenset({
    "completed", "complete", "done", "finished", "scored", "passed", "failed", "error", "errored",
    "cancelled", "canceled",
})
_POLL_DELAYS_S = (1.0, 1.0, 2.0, 2.0, 3.0, 5.0)
_ANALYSIS_WORKERS = 1  # one analysis at a time leaves the live calls the most CPU
_IO_WORKERS = 4


class VoiceTestError(RuntimeError):
    pass


class VoiceTestSetupError(VoiceTestError):
    """The run could not start: a bad option, a missing key or extra, or the API refused to create it."""


@dataclass
class VoiceTestRun:
    """What a run came to. `exit_code` is the CLI's: 0 passed, 1 failed, 2 setup or network failure
    (including a verdict that could not be read), 130 interrupted."""

    run_id: Optional[str]
    name: str
    console_url: Optional[str] = None
    calls: List[CallOutcome] = field(default_factory=list)
    passed: Optional[bool] = None
    status: Optional[str] = None
    summary: Optional[Dict[str, Any]] = field(default=None, repr=False)
    completed: bool = False
    interrupted: bool = False
    timing_only: bool = False
    out_dir: Optional[str] = None
    problems: List[str] = field(default_factory=list)

    @property
    def reporting_failed(self) -> bool:
        return not self.completed or any(not c.reported for c in self.calls)

    @property
    def exit_code(self) -> int:
        if self.interrupted:
            return EXIT_INTERRUPTED
        if self.reporting_failed or any(c.setup_failure for c in self.calls):
            return EXIT_SETUP
        if self.passed is True:
            return EXIT_PASSED
        if self.passed is False:
            return EXIT_FAILED
        return EXIT_SETUP

    @property
    def verdict(self) -> str:
        if self.interrupted:
            return "interrupted"
        if self.exit_code == EXIT_SETUP:
            return "error" if (self.reporting_failed or any(c.setup_failure for c in self.calls)) else "undecided"
        return "passed" if self.passed else "failed"

    def to_json(self) -> Dict[str, Any]:
        return {
            "runId": self.run_id,
            "name": self.name,
            "consoleUrl": self.console_url,
            "passed": self.passed,
            "status": self.status,
            "verdict": self.verdict,
            "exitCode": self.exit_code,
            "completed": self.completed,
            "interrupted": self.interrupted,
            "timingOnly": self.timing_only,
            "outDir": self.out_dir,
            "problems": list(self.problems),
            "calls": [c.to_json() for c in self.calls],
            "run": self.summary,
        }


def _require_extra() -> None:
    missing = []
    for name in ("numpy", "websockets", "webrtcvad"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise VoiceTestSetupError(
            f"voice tests need {', '.join(missing)}: pip install 'syntropylabs-evalkit[voice]'")


def resolve_credentials(subscription_key: Optional[str] = None, api_url: Optional[str] = None,
                        client: Any = None):
    """The key and API base a run uses: the arguments, else the client's (``evalkit.init()``'s when no
    client is given), else ``EVALKIT_API_KEY`` / ``EVALKIT_SUBSCRIPTION_KEY`` and ``EVALKIT_API_URL``."""
    if client is None:
        import evalkit

        client = evalkit._instance
    opts = getattr(client, "_opts", None) if client is not None and getattr(client, "enabled", False) else None
    key = (subscription_key or getattr(opts, "subscription_key", None) or os.getenv("EVALKIT_API_KEY")
           or os.getenv("EVALKIT_SUBSCRIPTION_KEY") or "")
    base = api_url or os.getenv("EVALKIT_API_URL") or getattr(opts, "api_url", None) or DEFAULT_API_URL
    return key.strip(), base.rstrip("/")


def _trace_client(key: str, base: str, client: Any):
    """The client the voice.call traces go through, and whether this run owns it (and destroys it).

    The process's ``evalkit.init()`` client when it sends under the run's key; otherwise one of the
    run's own, so the traces land in the run's project without configuring the process's tracing.
    """
    from evalkit.core.config import disabled_by_env

    from .trace import ingest_client

    if client is not None:
        return client, False
    import evalkit

    existing = evalkit._instance
    if existing is not None and getattr(existing, "enabled", False):
        if getattr(existing._opts, "subscription_key", None) == key:
            return existing, False
    if disabled_by_env():
        return None, False
    return ingest_client(key, base), True


def _create_body(name: str, target: Any, url: str, scenario_set: str, scenarios: Sequence[str],
                 repeats: int, agent: Optional[str], voice: Optional[Mapping[str, Any]],
                 timing_only: bool = False, first_speaker: Optional[str] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "name": name,
        # A label only: the query and the headers can carry the bot's credentials and never leave here.
        "target": {"url": target_label(url), "format": target.format, "sampleRate": target.sample_rate},
        "scenarioSetId": scenario_set,
        "repeats": int(repeats),
        # The server must know too: a timing-only run refuses recordings and reads as one in the console.
        "timingOnly": bool(timing_only),
    }
    if first_speaker:
        # Who opens the call sets line 0's wait in the plan; left out, the API's default applies.
        body["target"]["firstSpeaker"] = first_speaker
    if scenarios:
        body["scenarioIds"] = list(scenarios)
    if agent:
        body["agentId"] = agent
    if voice:
        body["voice"] = dict(voice)
    return body


def _decided(body: Mapping[str, Any]) -> bool:
    status = str(body.get("status") or "").strip().lower()
    if status in TERMINAL_STATUSES:
        return True
    return isinstance(body.get("passed"), bool) and status not in PENDING_STATUSES


class _Signals:
    """SIGINT and SIGTERM while a run is in progress, restored after it."""

    def __init__(self, loop: asyncio.AbstractEventLoop, on_signal: Callable[[int], None]) -> None:
        self.loop = loop
        self.on_signal = on_signal
        self._previous: Dict[int, Any] = {}
        self._installed: List[int] = []
        self._via_loop = True

    def install(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
            if sig is None:
                continue
            try:
                self._previous[sig] = signal.getsignal(sig)
            except (ValueError, OSError):
                continue
            try:
                self.loop.add_signal_handler(sig, self.on_signal, sig)
            except (NotImplementedError, RuntimeError, ValueError):
                # No loop signal handlers (Windows): a plain handler that hands over to the loop.
                self._via_loop = False
                try:
                    signal.signal(sig, lambda s, _f: self.loop.call_soon_threadsafe(self.on_signal, s))
                except (ValueError, OSError):
                    continue
            self._installed.append(sig)

    def restore(self) -> None:
        for sig in self._installed:
            if self._via_loop:
                try:
                    self.loop.remove_signal_handler(sig)
                except (ValueError, RuntimeError):
                    pass
            try:
                signal.signal(sig, self._previous[sig])
            except (ValueError, OSError, TypeError):
                pass
        self._installed = []


class _PlanOrder:
    """The run's limit on calls at once, granted in the plan's order.

    Each call fetches its audio on a worker thread before it asks for a slot, and those threads finish
    in any order; with a plain semaphore the call whose thread finished first would start first. Here
    a call may ask for a slot only once every call before it has one or will never ask, so the calls
    start in plan order whatever the threads do.
    """

    def __init__(self, limit: int, count: int) -> None:
        self._slot = asyncio.Semaphore(limit)
        self._asked = [asyncio.Event() for _ in range(count)]

    def turn(self, position: int) -> "_Turn":
        return _Turn(self, position)


class _Turn:
    """One call's place in a `_PlanOrder`, used as the slot `run_planned_call` acquires and releases."""

    def __init__(self, order: _PlanOrder, position: int) -> None:
        self._order = order
        self._position = position

    async def acquire(self) -> bool:
        try:
            if self._position:
                await self._order._asked[self._position - 1].wait()
            return await self._order._slot.acquire()
        finally:
            self.done()  # granted or given up: either way the next call may ask

    def release(self) -> None:
        self._order._slot.release()

    def done(self) -> None:
        """Passes the turn on; for a call that never asks (its script or audio failed)."""
        self._order._asked[self._position].set()


class _Run:
    def __init__(self, *, api: VoiceTestApi, target: Any, plan: RunPlan, out_root: str,
                 timing_only: bool, trace_client: Any, cache_dir: Optional[str],
                 on_call: Optional[Callable[[CallOutcome], None]], say: Callable[[str], None],
                 retry: Optional[RetryPolicy]) -> None:
        self.api = api
        self.target = target
        self.plan = plan
        self.out_root = out_root
        self.timing_only = timing_only
        self.trace_client = trace_client
        self.cache_dir = cache_dir
        self.on_call = on_call
        self.say = say
        self.retry = retry
        self.stopping = False
        self.abort: Optional[asyncio.Event] = None
        self._signals: Optional[_Signals] = None
        self.signals = 0
        self.live = 0

    def on_signal(self, sig: int) -> None:
        self.signals += 1
        hard = sig != signal.SIGINT or self.signals > 1
        if not self.stopping:
            self.stopping = True
        if hard:
            if self.abort is not None and not self.abort.is_set():
                self.abort.set()
                self.say("stopping now: ending the calls in progress, then reporting what was recorded")
            if self._signals is not None:
                self._signals.restore()  # a further Ctrl-C ends the process as usual
        else:
            self.say(f"stopping: no new calls start; {self.live} in progress will finish "
                     "(Ctrl-C again ends them now)")

    async def execute(self, handle_signals: bool) -> List[CallOutcome]:
        from .audio_cache import AudioCache
        from .sources import PlanAudioSource, default_sources

        loop = asyncio.get_running_loop()
        self.abort = asyncio.Event()
        self._signals = _Signals(loop, self.on_signal) if handle_signals else None
        if self._signals is not None:
            self._signals.install()
        cache = AudioCache(self.cache_dir, retry=self.retry)
        sources = default_sources()
        audio = PlanAudioSource(cache)
        sources["audio"] = audio
        io = concurrent.futures.ThreadPoolExecutor(_IO_WORKERS, thread_name_prefix="evalkit-voicetest")
        analysis = concurrent.futures.ThreadPoolExecutor(_ANALYSIS_WORKERS,
                                                         thread_name_prefix="evalkit-voicetest-analysis")
        order = _PlanOrder(self.plan.max_concurrent_calls, len(self.plan.calls))
        try:
            await self._prefetch(loop, io, audio)
            tasks = [asyncio.ensure_future(self._one(call, sources, cache, order.turn(position), io, analysis))
                     for position, call in enumerate(self.plan.calls)]
            return list(await asyncio.gather(*tasks))
        finally:
            if self._signals is not None:
                self._signals.restore()
            io.shutdown(wait=True)
            analysis.shutdown(wait=True)

    async def _prefetch(self, loop, io, audio) -> None:
        """Every distinct line audio of the plan, fetched in parallel before the first call, so no
        call waits on a download and a failed one is known before any call starts."""
        specs: Dict[str, Mapping[str, Any]] = {}
        for call in self.plan.calls:
            for line in (call.script.get("lines") or []):
                spec = line.get("audio") if isinstance(line, Mapping) else None
                if isinstance(spec, Mapping) and isinstance(spec.get("sha256"), str):
                    specs.setdefault(spec["sha256"].strip().lower(), spec)
        if not specs:
            return
        rate = self.target.sample_rate

        def fetch(spec):
            try:
                audio.prepare(spec, rate)
                return None
            except Exception as exc:
                return str(exc)

        results = await asyncio.gather(*(loop.run_in_executor(io, fetch, s) for s in specs.values()))
        failed = [r for r in results if r]
        fetched = len(specs) - len(failed)
        self.say(f"caller audio: {fetched} of {len(specs)} clips ready "
                 f"({audio.cache.downloaded} downloaded, {audio.cache.hits} from the cache)")
        for message in failed:
            self.say(f"  {message}")

    async def _one(self, call, sources, cache, turn: _Turn, io, analysis) -> CallOutcome:
        out_dir = os.path.join(self.out_root, f"call-{call.call_index:03d}")

        def may_start() -> bool:
            if self.stopping:
                return False
            self.live += 1
            return True

        try:
            outcome = await run_planned_call(
                call,
                run_id=self.plan.run_id,
                target=self.target,
                out_dir=out_dir,
                api=self.api,
                trace_client=self.trace_client,
                limits=self.plan.limits,
                timing_only=self.timing_only,
                stop=self.abort,
                sources=sources,
                audio_cache=cache,
                slots=(turn,),
                may_start=may_start,
                executor=io,
                analysis_executor=analysis,
            )
        finally:
            turn.done()
        if outcome.ended_by is not None:
            self.live -= 1
        if self.on_call is not None:
            try:
                self.on_call(outcome)
            except Exception:
                log.debug("on_call failed", exc_info=True)
        return outcome


async def run_async(
    url: str,
    *,
    format: str,
    scenario_set: str,
    sample_rate: int = 16000,
    headers: Optional[Mapping[str, str]] = None,
    scenarios: Optional[Sequence[str]] = None,
    repeats: int = 1,
    agent: Optional[str] = None,
    name: Optional[str] = None,
    voice: Optional[Mapping[str, Any]] = None,
    timing_only: bool = False,
    first_speaker: Optional[str] = None,
    out_dir: Optional[str] = None,
    subscription_key: Optional[str] = None,
    api_url: Optional[str] = None,
    client: Any = None,
    session_query_param: str = "evalkit_session",
    send_start: bool = True,
    wait: bool = True,
    wait_timeout_s: float = 600.0,
    cache_dir: Optional[str] = None,
    handle_signals: bool = True,
    on_call: Optional[Callable[[CallOutcome], None]] = None,
    on_message: Optional[Callable[[str], None]] = None,
    retry: Optional[RetryPolicy] = None,
) -> VoiceTestRun:
    """`run`, for code already inside an event loop. Signal handling applies only in the main thread."""
    from .transport import Target

    say = on_message or (lambda message: log.info("%s", message))
    _require_extra()
    if not isinstance(scenario_set, str) or not scenario_set.strip():
        raise VoiceTestSetupError("a scenario set id is required")
    if int(repeats) < 1:
        raise VoiceTestSetupError("repeats must be at least 1")
    if first_speaker is not None and first_speaker not in FIRST_SPEAKERS:
        raise VoiceTestSetupError(f"first_speaker must be one of {', '.join(FIRST_SPEAKERS)}")
    try:
        target = Target(url=url, format=format, sample_rate=sample_rate, headers=dict(headers or {}),
                        session_query_param=session_query_param, send_start=send_start)
    except ValueError as exc:
        raise VoiceTestSetupError(str(exc)) from None
    key, base = resolve_credentials(subscription_key, api_url, client)
    if not key:
        raise VoiceTestSetupError("a subscription key is required: pass --key, set EVALKIT_API_KEY, "
                                  "or call evalkit.init() first")
    api = VoiceTestApi.with_subscription_key(base, key, retry=retry)
    title = name or f"voice test {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M} UTC"
    loop = asyncio.get_running_loop()
    body = _create_body(title, target, url, scenario_set.strip(), list(scenarios or ()), repeats, agent, voice,
                        timing_only, first_speaker)
    try:
        created = await loop.run_in_executor(None, api.create_run, body)
        plan = parse_plan(created)
    except VoiceTestApiError as exc:
        raise VoiceTestSetupError(str(exc)) from None
    except PlanError as exc:
        raise VoiceTestSetupError(f"the API answered with a plan the runner cannot use: {exc}") from None

    report = VoiceTestRun(run_id=plan.run_id, name=title, console_url=plan.console_url,
                          timing_only=timing_only)
    say(f"run {plan.run_id}: {len(plan.calls)} call(s), up to {plan.max_concurrent_calls} at once"
        + (f"; {plan.console_url}" if plan.console_url else ""))
    keep = out_dir is not None
    out_root = os.path.join(out_dir, plan.run_id) if keep else tempfile.mkdtemp(prefix="evalkit-voicetest-")
    os.makedirs(out_root, exist_ok=True)
    report.out_dir = out_root if keep else None
    trace_client, own_client = _trace_client(key, base, client)
    if trace_client is None:
        report.problems.append("tracing is disabled in this process, so no voice.call traces were sent")
    runner = _Run(api=api, target=target, plan=plan, out_root=out_root, timing_only=timing_only,
                  trace_client=trace_client, cache_dir=cache_dir, on_call=on_call, say=say, retry=retry)
    try:
        report.calls = await runner.execute(handle_signals)
        report.interrupted = runner.stopping
        try:
            summary = await loop.run_in_executor(None, api.complete, plan.run_id)
            report.completed = True
            report.summary = summary or None
        except VoiceTestApiError as exc:
            report.problems.append(str(exc))
            say(str(exc))
        if report.completed and wait and not report.interrupted:
            await _wait_for_verdict(loop, api, report, wait_timeout_s, say)
        elif report.summary:
            _take_verdict(report, report.summary)
    finally:
        if own_client:
            try:
                await loop.run_in_executor(None, trace_client.destroy)
            except Exception:
                log.debug("closing the trace client failed", exc_info=True)
        if keep:
            try:
                with open(os.path.join(out_root, "run.json"), "w", encoding="utf-8") as fh:
                    json.dump(report.to_json(), fh, indent=2, default=str)
                    fh.write("\n")
            except OSError as exc:
                log.warning("could not write run.json: %s", exc)
        else:
            shutil.rmtree(out_root, ignore_errors=True)
    return report


def _take_verdict(report: VoiceTestRun, body: Mapping[str, Any]) -> None:
    report.summary = dict(body)
    status = body.get("status")
    report.status = str(status) if status is not None else None
    passed = body.get("passed")
    report.passed = passed if isinstance(passed, bool) else None
    if report.console_url is None and body.get("consoleUrl"):
        report.console_url = str(body["consoleUrl"])


async def _wait_for_verdict(loop, api: VoiceTestApi, report: VoiceTestRun, timeout_s: float,
                            say: Callable[[str], None]) -> None:
    if report.summary and _decided(report.summary):
        _take_verdict(report, report.summary)
        return
    deadline = loop.time() + max(0.0, timeout_s)
    attempt = 0
    while True:
        try:
            body = await loop.run_in_executor(None, api.get_run, report.run_id)
        except VoiceTestApiError as exc:
            report.problems.append(str(exc))
            say(str(exc))
            return
        _take_verdict(report, body)
        if _decided(body):
            return
        if loop.time() >= deadline:
            report.passed = None
            report.problems.append(f"the evaluators had not reported after {timeout_s:.0f} s "
                                   f"(status {report.status or 'unknown'})")
            return
        delay = _POLL_DELAYS_S[min(attempt, len(_POLL_DELAYS_S) - 1)]
        attempt += 1
        await asyncio.sleep(min(delay, max(0.0, deadline - loop.time())))


def run(url: str, **kwargs: Any) -> VoiceTestRun:
    """Runs a voice test of the bot at `url` (``ws://`` or ``wss://``) and returns its `VoiceTestRun`.

    Keyword arguments: `format` (``twilio`` | ``pcm16``) and `scenario_set` (an id) are required;
    `sample_rate` (pcm16: 8000, 16000 or 24000), `headers` (sent to the bot only), `scenarios`
    (a subset of the set), `repeats`, `agent` (score under a registered agent), `name`, `voice`
    (``{"provider", "model", "voice"}``), `timing_only` (upload no audio), `first_speaker` (``auto``,
    ``bot`` or ``caller``: who opens each call, which sets the wait before the first line; the API's
    default, ``auto``, when None), `out_dir` (keep the
    recordings under ``out_dir/<runId>/``), `subscription_key` and `api_url` (default: the
    ``evalkit.init()`` client's, then ``EVALKIT_API_KEY`` and ``EVALKIT_API_URL``), `client` (the
    EvalKit client for the voice.call traces), `wait` and `wait_timeout_s` (for the verdict),
    `cache_dir` (the caller audio cache), `on_call(outcome)` and `on_message(text)` for progress.

    Raises VoiceTestSetupError when the run cannot start. A call that fails does not raise: it is
    reported with its error and the run goes on. ``sys.exit(run(...).exit_code)`` gates a CI job.
    """
    return asyncio.run(run_async(url, **kwargs))
