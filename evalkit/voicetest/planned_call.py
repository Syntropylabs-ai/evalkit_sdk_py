"""One planned call, placed and reported (RUN_API section 4). The SDK run and the hosted runner both
use it, so a call is measured and reported the same way whichever started it.

The steps, in order: the line audio is fetched and decoded; the call runs; the recording is analysed;
the recording is uploaded through its signed URL (unless the run is timing-only); the ``voice.call``
trace is sent; the result is posted. A call that fails still reports, with ``error`` set and whatever
was recorded, and never raises: the run goes on to its next call.

Everything that blocks (downloads, analysis, uploads, posts) runs in an executor, so another call's
20 ms frame loop on the same event loop keeps its timing.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional

from .api import VoiceTestApi, VoiceTestApiError
from .plan import PlanError, PlannedCall, prepare_script_audio, script_from_plan_call
from .script import Script, ScriptError

log = logging.getLogger("evalkit.voicetest.call")

# Kinds of `error` a result can carry. The ones in SETUP_ERROR_KINDS are failures on the testing side
# (our audio, the caller itself), which make the run's verdict say nothing about the bot. A connector
# (`Target.connector`) names why it refused to connect with a `kind` on the exception it raises:
# the hosted runner's outbound guard reports guard_refused, dns_failed or redirect_refused.
ERROR_KINDS = (
    "invalid_script",
    "audio_unavailable",
    "connect_failed",
    "guard_refused",
    "dns_failed",
    "redirect_refused",
    "bot_closed_early",
    "max_call",
    "aborted",
    "not_run",
    "caller_error",
    "analysis_failed",
)
SETUP_ERROR_KINDS = frozenset({"invalid_script", "audio_unavailable", "caller_error", "analysis_failed"})
# A call runs to its own maxCallMs; this is only the backstop for a socket that will not close.
GUARD_MS = 60_000
_MESSAGE_CAP = 1000


def _error(kind: str, message: str) -> Dict[str, str]:
    return {"kind": kind, "message": str(message)[:_MESSAGE_CAP]}


@dataclass
class CallOutcome:
    """What one planned call came to, and what of it reached the API."""

    call_index: int
    session_token: str
    scenario_name: str = ""
    scenario_id: Optional[str] = None
    repeat: int = 0
    out_dir: Optional[str] = None
    ended_by: Optional[str] = None
    duration_ms: Optional[int] = None
    scorecard: Optional[Dict[str, Any]] = field(default=None, repr=False)
    error: Optional[Dict[str, str]] = None
    recording: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None
    reported: bool = False
    # What went wrong around the call without making it fail: an upload, the trace, the result post.
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def caller_healthy(self) -> bool:
        health = (self.scorecard or {}).get("caller_health") or {}
        return bool(health.get("healthy", True))

    @property
    def setup_failure(self) -> bool:
        return self.error is not None and self.error.get("kind") in SETUP_ERROR_KINDS

    def result_body(self) -> Dict[str, Any]:
        """The ``POST .../calls/:callIndex/result`` body (RUN_API section 4)."""
        from .caller import CALLER_VERSION, VAD_VERSION

        vad = ((self.scorecard or {}).get("vad") or {}).get("version") or VAD_VERSION
        return {
            "scorecard": self.scorecard,
            "recording": self.recording,
            "traceId": self.trace_id,
            "callerVersion": CALLER_VERSION,
            "vadVersion": vad,
            "callerHealthy": self.caller_healthy,
            "error": self.error,
        }

    def figures(self) -> Dict[str, Any]:
        call = (self.scorecard or {}).get("call") or {}
        keys = ("first_word_ms", "response_delay_p50_ms", "response_delay_p95_ms", "talk_over_ms",
                "talk_over_count", "interruption_count", "interruption_stop_p50_ms",
                "interruption_stop_p95_ms", "max_silence_ms", "turn_count", "scored_turn_count")
        return {k: call.get(k) for k in keys if k in call}

    def to_json(self) -> Dict[str, Any]:
        return {
            "callIndex": self.call_index,
            "scenarioId": self.scenario_id,
            "scenarioName": self.scenario_name,
            "repeat": self.repeat,
            "sessionToken": self.session_token,
            "endedBy": self.ended_by,
            "durationMs": self.duration_ms,
            "error": self.error,
            "callerHealthy": self.caller_healthy if self.scorecard else None,
            "figures": self.figures(),
            "recording": self.recording,
            "traceId": self.trace_id,
            "reported": self.reported,
            "problems": list(self.problems),
            "outDir": self.out_dir,
        }


async def _acquire(slot: asyncio.Semaphore, stop: Optional[asyncio.Event]) -> bool:
    """Waits for `slot`; False when `stop` was set first."""
    if stop is None:
        await slot.acquire()
        return True
    acquire = asyncio.ensure_future(slot.acquire())
    stopped = asyncio.ensure_future(stop.wait())
    try:
        await asyncio.wait({acquire, stopped}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        stopped.cancel()
    if not acquire.done():
        acquire.cancel()
    try:
        await acquire
        return True
    except asyncio.CancelledError:
        return False


def _classify(result: Any, script: Script) -> Optional[Dict[str, str]]:
    ended = result.ended_by
    if ended == "script":
        return None
    if ended == "max_call":
        return _error("max_call", f"the call reached maxCallMs ({script.max_call_ms} ms) before the "
                                  "script ended")
    if ended == "aborted":
        return _error("aborted", "the run was interrupted during this call")
    if ended == "bot_closed":
        spoken = sum(1 for e in result.events if e.get("type") == "line_end")
        if spoken >= len(script.lines):
            return None  # hanging up after the caller's last line is how a call ends
        closed = [e for e in result.events if e.get("type") == "closed" and e.get("by") == "bot"]
        code = closed[-1].get("code") if closed else None
        how = f" (close code {code})" if code is not None else ""
        return _error("bot_closed_early", f"the bot closed the connection{how} after {spoken} of "
                                          f"{len(script.lines)} caller lines")
    return _error("caller_error", result.errors[-1] if result.errors else "the call failed")


def _analyse(out_dir: str) -> Dict[str, Any]:
    from .analysis.scorecard import write_scorecard

    path = write_scorecard(out_dir)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _recorded(out_dir: str) -> bool:
    """Whether the directory holds a whole recording: the WAV and a call.json with the call's start."""
    from .recording import CALL_NAME, RECORDING_NAME

    if not os.path.exists(os.path.join(out_dir, RECORDING_NAME)):
        return False
    try:
        with open(os.path.join(out_dir, CALL_NAME), "r", encoding="utf-8") as fh:
            call = json.load(fh)
    except (OSError, ValueError):
        return False
    return isinstance(call, dict) and isinstance(call.get("t0_epoch_ms"), (int, float))


def _upload(api: VoiceTestApi, run_id: str, call_index: int, path: str) -> Dict[str, Any]:
    upload = api.recording_upload(run_id, call_index)
    try:
        size, sha256 = api.upload_recording(upload, path)
    except VoiceTestApiError as exc:
        # A signed URL that expired or was refused: ask for a fresh one, once.
        if exc.status not in (400, 401, 403):
            raise
        upload = api.recording_upload(run_id, call_index)
        size, sha256 = api.upload_recording(upload, path)
    object_path = upload.get("objectPath")
    if not isinstance(object_path, str) or not object_path:
        raise VoiceTestApiError("the recording upload answer has no objectPath")
    return {"objectPath": object_path, "bytes": size, "sha256": sha256}


def _send_trace(outcome: CallOutcome, client: Any, flush_timeout_s: float) -> None:
    from .trace import send_call_trace

    uri = (outcome.recording or {}).get("objectPath")
    status = "OK" if outcome.error is None else "ERROR"
    outcome.trace_id = send_call_trace(outcome.scorecard, client, recording_uri=uri, status=status,
                                       flush=False)
    if not client.flush(timeout=flush_timeout_s):
        outcome.problems.append("the voice.call trace was queued but not confirmed sent")


async def run_planned_call(
    call: Any,
    *,
    run_id: str,
    target: Any,
    out_dir: str,
    api: Optional[VoiceTestApi] = None,
    trace_client: Any = None,
    limits: Optional[Mapping[str, Any]] = None,
    timing_only: bool = False,
    stop: Optional[asyncio.Event] = None,
    connector: Optional[Callable[..., Awaitable[Any]]] = None,
    sources: Optional[Mapping[str, Any]] = None,
    audio_cache: Any = None,
    slots: Iterable[asyncio.Semaphore] = (),
    may_start: Optional[Callable[[], bool]] = None,
    executor: Any = None,
    analysis_executor: Any = None,
    trace_flush_timeout_s: float = 10.0,
) -> CallOutcome:
    """Runs one plan call (a RUN_API section 3.3 ``calls[]`` entry, or a `PlannedCall`) against
    `target` and reports it through `api`. Never raises for the call's own failures: they come back
    as the outcome's ``error``, and are posted as the result's ``error``.

    - `target`: the `Target`; `connector`, when given, replaces its WebSocket connector.
    - `api`: the endpoints to report to, with their credential (`VoiceTestApi.with_subscription_key`
      or `VoiceTestApi.with_run_token` on the callback base). None reports nothing.
    - `trace_client`: the EvalKit client the ``voice.call`` trace goes through (the process's
      ``evalkit.init()`` client, or `trace.ingest_client(ingest_key, ...)`). None sends no trace.
    - `stop`: when set, a call in progress ends at its next frame and is reported as ``aborted``;
      a call not yet started is reported as ``not_run``.
    - `out_dir`: where the call's recording.wav, events.jsonl, call.json and scorecard.json go.
    - `slots`: semaphores held for the live part of the call only (the run's and, say, a pod's limit
      on calls at once); `may_start` is asked once they are held.
    - `sources`/`audio_cache`: line sources shared by the run's calls, so audio is fetched once.

    Raises PlanError only when `call` itself is malformed (no callIndex, sessionToken or script),
    since there is then no call to report against.
    """
    from .audio_cache import AudioError
    from .caller import CallError, run_call

    loop = asyncio.get_running_loop()
    planned = call if isinstance(call, PlannedCall) else PlannedCall.from_json(call)
    outcome = CallOutcome(
        call_index=planned.call_index,
        session_token=planned.session_token,
        scenario_name=planned.scenario_name,
        scenario_id=planned.scenario_id,
        repeat=planned.repeat,
        out_dir=str(out_dir),
    )
    if connector is not None:
        target = dataclasses.replace(target, connector=connector)
    os.makedirs(out_dir, exist_ok=True)

    script: Optional[Script] = None
    chosen: Dict[str, Any] = {}
    try:
        script, chosen = script_from_plan_call(planned, limits=limits, audio_cache=audio_cache,
                                               sources=sources)
    except (PlanError, ScriptError) as exc:
        outcome.error = _error("invalid_script", str(exc))
    if script is not None:
        try:
            await loop.run_in_executor(executor, prepare_script_audio, script, chosen, target.sample_rate)
        except (AudioError, OSError, ValueError) as exc:
            outcome.error = _error("audio_unavailable", str(exc))

    if outcome.error is None and script is not None:
        held: List[asyncio.Semaphore] = []
        try:
            started = True
            for slot in slots:
                if not await _acquire(slot, stop):
                    started = False
                    break
                held.append(slot)
            if not started or (stop is not None and stop.is_set()) or (may_start is not None and not may_start()):
                outcome.error = _error("not_run", "the run was stopped before this call started")
            else:
                guard_s = (script.max_call_ms + GUARD_MS) / 1000.0
                try:
                    result = await asyncio.wait_for(
                        run_call(target, script, str(out_dir), session=planned.session_token,
                                 sources=chosen, stop=stop),
                        guard_s,
                    )
                except CallError as exc:
                    outcome.ended_by = "error"
                    kind = getattr(exc.__cause__, "kind", None)
                    if not (isinstance(kind, str) and kind in ERROR_KINDS):
                        kind = "connect_failed"
                    outcome.error = _error(kind, str(exc))
                except asyncio.TimeoutError:
                    outcome.ended_by = "max_call"
                    outcome.error = _error("max_call", f"the call did not end within maxCallMs "
                                                       f"({script.max_call_ms} ms)")
                except AudioError as exc:
                    outcome.error = _error("audio_unavailable", str(exc))
                except Exception as exc:  # the caller's own fault; reported, never raised
                    outcome.ended_by = "error"
                    outcome.error = _error("caller_error", f"{type(exc).__name__}: {exc}")
                else:
                    outcome.ended_by = result.ended_by
                    outcome.duration_ms = result.duration_ms
                    outcome.error = _classify(result, script)
        finally:
            for slot in reversed(held):
                slot.release()

    recorded = outcome.ended_by is not None and _recorded(str(out_dir))
    if recorded:
        try:
            outcome.scorecard = await loop.run_in_executor(analysis_executor, _analyse, str(out_dir))
        except Exception as exc:  # the call happened; a broken analysis must not lose its report
            log.warning("call %d: the analysis failed: %s", planned.call_index, exc)
            if outcome.error is None:
                outcome.error = _error("analysis_failed", f"{type(exc).__name__}: {exc}")

    # Uploaded even when the analysis failed: the recording is then what shows why.
    if api is not None and not timing_only and recorded:
        from .recording import RECORDING_NAME

        try:
            outcome.recording = await loop.run_in_executor(
                executor, _upload, api, run_id, planned.call_index, os.path.join(str(out_dir), RECORDING_NAME))
        except VoiceTestApiError as exc:
            outcome.problems.append(f"the recording was not uploaded: {exc}")
            log.warning("call %d: %s", planned.call_index, outcome.problems[-1])

    if trace_client is not None and outcome.scorecard is not None and getattr(trace_client, "enabled", True):
        try:
            await loop.run_in_executor(executor, _send_trace, outcome, trace_client, trace_flush_timeout_s)
        except Exception as exc:  # tracing must never cost the call its result
            outcome.problems.append(f"the voice.call trace was not sent: {type(exc).__name__}: {exc}")
            log.warning("call %d: %s", planned.call_index, outcome.problems[-1])

    if api is not None:
        try:
            await loop.run_in_executor(executor, api.post_result, run_id, planned.call_index,
                                       outcome.result_body())
            outcome.reported = True
        except VoiceTestApiError as exc:
            outcome.problems.append(f"the result was not sent: {exc}")
            log.warning("call %d: %s", planned.call_index, outcome.problems[-1])
    return outcome
