"""The run plan the SyntropyLabs API returns (RUN_API section 3.3), turned into what the caller runs.

The plan is written in the API's camelCase (``maxCallMs``, ``wait.rule``, ``end.afterLastLine``);
the caller's script format is snake_case (README "Script format"). `script_from_plan_call` maps one
onto the other and accepts either spelling, so a plan in the script's own format works too. Fields
the caller does not know are left out rather than refused: the server adds fields over time, and a
plan from a newer server must still run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

from .script import DEFAULT_SOURCE_KINDS, Script, ScriptError, parse_script

log = logging.getLogger("evalkit.voicetest.plan")

PLAN_SOURCE_KINDS = DEFAULT_SOURCE_KINDS + ("audio",)


class PlanError(ValueError):
    """The plan cannot be run as given."""


def _pick(obj: Mapping[str, Any], camel: str, snake: str, default: Any = None) -> Any:
    if obj.get(camel) is not None:
        return obj[camel]
    if obj.get(snake) is not None:
        return obj[snake]
    return default


def _int(value: Any, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < minimum:
        raise PlanError(f"{where} must be a number of at least {minimum}")
    return int(value)


@dataclass
class PlannedCall:
    """One entry of the plan's ``calls``."""

    call_index: int
    session_token: str
    script: Dict[str, Any]
    scenario_id: Optional[str] = None
    scenario_name: str = ""
    repeat: int = 0
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "PlannedCall":
        if not isinstance(obj, Mapping):
            raise PlanError("a planned call must be an object")
        index = _int(_pick(obj, "callIndex", "call_index"), "callIndex")
        token = _pick(obj, "sessionToken", "session_token")
        if not isinstance(token, str) or not token.strip():
            raise PlanError(f"call {index} has no sessionToken")
        script = obj.get("script")
        if not isinstance(script, Mapping):
            raise PlanError(f"call {index} has no script")
        scenario_id = _pick(obj, "scenarioId", "scenario_id")
        return cls(
            call_index=index,
            session_token=token,
            script=dict(script),
            scenario_id=str(scenario_id) if scenario_id is not None else None,
            scenario_name=str(_pick(obj, "scenarioName", "scenario_name", "") or ""),
            repeat=int(_pick(obj, "repeat", "repeat", 0) or 0),
            raw=dict(obj),
        )


    @property
    def mode(self) -> str:
        """``dynamic`` when the call asks for its lines during the call (RUN_API section 3.4), else
        ``scripted``: a plan without ``mode`` is scripted."""
        return "dynamic" if str(self.script.get("mode") or "").strip().lower() == "dynamic" else "scripted"


@dataclass
class RunPlan:
    run_id: str
    calls: List[PlannedCall]
    console_url: Optional[str] = None
    max_call_ms: Optional[int] = None
    max_concurrent_calls: int = 1
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)
    # The model that decides a dynamic call's lines, as `{provider, model, keySource}` (`keySource`
    # `request`, `organization` or `platform`); None when no call is dynamic.
    caller: Optional[Dict[str, Any]] = None
    # The organization's SyntropyLabs model minutes left this month; None when its plan has none, -1
    # when they are unlimited.
    platform_minutes_left: Optional[float] = None

    @property
    def limits(self) -> Dict[str, Any]:
        return {"maxCallMs": self.max_call_ms, "maxConcurrentCalls": self.max_concurrent_calls}


def parse_plan(body: Mapping[str, Any]) -> RunPlan:
    """The plan as the API answered it (``POST /v1/voice-tests`` or a hosted runner's job)."""
    if not isinstance(body, Mapping):
        raise PlanError("the plan is not a JSON object")
    run_id = _pick(body, "runId", "run_id")
    if not isinstance(run_id, str) or not run_id:
        raise PlanError("the plan has no runId")
    limits = body.get("limits") if isinstance(body.get("limits"), Mapping) else {}
    max_call = _pick(limits, "maxCallMs", "max_call_ms")
    concurrent = _pick(limits, "maxConcurrentCalls", "max_concurrent_calls", 1)
    raw_calls = body.get("calls")
    if not isinstance(raw_calls, list):
        raise PlanError("the plan has no calls list")
    calls = [PlannedCall.from_json(c) for c in raw_calls]
    indices = [c.call_index for c in calls]
    if len(set(indices)) != len(indices):
        raise PlanError("the plan repeats a callIndex")
    console = _pick(body, "consoleUrl", "console_url")
    caller = body.get("caller")
    minutes = _pick(body, "platformMinutesLeft", "platform_minutes_left")
    return RunPlan(
        run_id=run_id,
        calls=sorted(calls, key=lambda c: c.call_index),
        console_url=str(console) if console else None,
        max_call_ms=_int(max_call, "limits.maxCallMs", 1) if max_call is not None else None,
        max_concurrent_calls=max(1, _int(concurrent, "limits.maxConcurrentCalls", 1)),
        raw=dict(body),
        # Read, never relied on: a plan from a server without them still runs.
        caller=dict(caller) if isinstance(caller, Mapping) else None,
        platform_minutes_left=(minutes if isinstance(minutes, (int, float)) and not isinstance(minutes, bool)
                               else None),
    )


def _wait_rule(obj: Any, *, end: bool = False) -> Optional[Dict[str, Any]]:
    if obj is None:
        return None
    if isinstance(obj, str):  # "end": "bot_done"
        return {"mode": obj}
    if not isinstance(obj, Mapping):
        raise ScriptError("a wait rule must be an object")
    mode = obj.get("rule") or obj.get("mode") or (obj.get("afterLastLine") if end else None)
    out: Dict[str, Any] = {}
    if mode is not None:
        out["mode"] = mode
    for camel, snake in (("silenceMs", "silence_ms"), ("maxMs", "max_ms"), ("ms", "ms"),
                         ("startWithinMs", "start_within_ms")):
        value = _pick(obj, camel, snake)
        if value is not None:
            out[snake] = value
    return out


def _line(raw: Any, index: int) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ScriptError(f"lines[{index}]: a line must be an object")
    out: Dict[str, Any] = {"text": raw.get("text") or ""}
    for kind in PLAN_SOURCE_KINDS:
        if raw.get(kind) is not None:
            out[kind] = raw[kind]
    wait = _wait_rule(raw.get("wait"))
    if wait is not None:
        out["wait"] = wait
    interrupt = _pick(raw, "interruptAfterBotMs", "interrupt_after_bot_ms")
    if interrupt is not None:
        out["interrupt_after_bot_ms"] = interrupt
    known = set(out) | {"wait", "interruptAfterBotMs", "interrupt_after_bot_ms"}
    ignored = sorted(set(raw) - known)
    if ignored:
        log.debug("lines[%d]: ignoring fields the caller does not use: %s", index, ignored)
    return out


def _turns(raw: Any) -> Dict[str, Any]:
    """A dynamic script's ``turns`` (RUN_API section 3.3) in the script format."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ScriptError("turns: must be an object")
    out: Dict[str, Any] = {}
    if raw.get("max") is not None:
        out["max"] = raw["max"]
    wait = _wait_rule(raw.get("wait"))
    if wait is not None:
        out["wait"] = wait
    interrupt = raw.get("interrupt")
    if interrupt is not None:
        if not isinstance(interrupt, Mapping):
            raise ScriptError("turns.interrupt: must be {turn, afterBotMs}")
        out["interrupt"] = {"turn": interrupt.get("turn"),
                            "after_bot_ms": _pick(interrupt, "afterBotMs", "after_bot_ms", 0)}
    return out


def script_json_from_plan(script: Mapping[str, Any], *, name: str = "",
                          max_call_ms: Optional[int] = None) -> Dict[str, Any]:
    """A plan call's ``script`` in the caller's script format. `max_call_ms` (the run's limit) caps
    the script's own ``maxCallMs``. A dynamic script keeps its ``mode`` and ``turns``; a scripted one is
    written without ``mode``, as before there was one."""
    if not isinstance(script, Mapping):
        raise ScriptError("a script must be a JSON object")
    lines = script.get("lines")
    if not isinstance(lines, list) or not lines:
        raise ScriptError("a script needs a non-empty lines list")
    mode = str(script.get("mode") or "scripted").strip().lower()
    if mode not in ("scripted", "dynamic"):
        # A call this caller cannot play: refused as invalid_script rather than played wrongly.
        raise ScriptError(f"the script's mode {mode!r} is not one this caller plays (scripted, dynamic)")
    out: Dict[str, Any] = {"lines": [_line(raw, i) for i, raw in enumerate(lines)]}
    if mode == "dynamic":
        out["mode"] = "dynamic"
        out["turns"] = _turns(script.get("turns"))
    title = script.get("name") or name
    if title:
        out["name"] = str(title)
    end = _wait_rule(script.get("end"), end=True)
    if end is not None:
        out["end"] = end
    own = _pick(script, "maxCallMs", "max_call_ms")
    caps = [v for v in (own, max_call_ms) if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if caps:
        out["max_call_ms"] = int(min(caps))
    return out


def redact_script(raw: Dict[str, Any]) -> Dict[str, Any]:
    """The script as call.json keeps it: audio URLs without their signed query."""
    from .net import display_url

    out = dict(raw)
    lines = []
    for line in raw.get("lines") or []:
        line = dict(line)
        audio = line.get("audio")
        if isinstance(audio, Mapping) and isinstance(audio.get("url"), str):
            line["audio"] = dict(audio, url=display_url(audio["url"]))
        lines.append(line)
    out["lines"] = lines
    return out


def script_from_plan_call(
    call: Any,
    *,
    limits: Optional[Mapping[str, Any]] = None,
    audio_cache: Any = None,
    sources: Optional[Mapping[str, Any]] = None,
) -> Tuple[Script, Dict[str, Any]]:
    """One plan call (a RUN_API section 3.3 ``calls[]`` entry, or a `PlannedCall`) as the caller's
    `Script`, with the line sources to render it.

    The sources are the synthetic and WAV sources plus a `PlanAudioSource` over `audio_cache` (a
    fresh `AudioCache` when None), which downloads each line's signed URL and checks its sha256.
    Pass `sources` to share one prepared `PlanAudioSource` between the calls of a run. Raises
    PlanError or ScriptError when the call cannot be run.
    """
    from .sources import PlanAudioSource, default_sources

    planned = call if isinstance(call, PlannedCall) else PlannedCall.from_json(call)
    cap = None
    if limits:
        value = _pick(limits, "maxCallMs", "max_call_ms")
        cap = int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    obj = script_json_from_plan(planned.script, name=planned.scenario_name, max_call_ms=cap)
    script = parse_script(obj, source_kinds=PLAN_SOURCE_KINDS)
    script.raw = redact_script(script.raw)
    chosen: Dict[str, Any] = dict(sources) if sources is not None else default_sources()
    if "audio" not in chosen:
        chosen["audio"] = PlanAudioSource(audio_cache)
    return script, chosen


def prepare_script_audio(script: Script, sources: Mapping[str, Any], rate: int) -> None:
    """Download and decode every ``audio`` line now (blocking; run it off the event loop), so the call
    itself only reads memory."""
    source = sources.get("audio")
    if source is None or not hasattr(source, "prepare"):
        return
    for line in script.lines:
        if line.source == "audio":
            source.prepare(line.spec, rate)


def target_from_settings(settings: Mapping[str, Any], *, connector: Any = None,
                         send_start: Optional[bool] = None):
    """A `Target` from target settings in the API's shape (RUN_API section 2): ``url``, ``format``,
    ``sampleRate``, ``headers`` (``[{"name", "value"}]`` or a mapping) and ``sessionQueryParam``."""
    from .transport import Target
    from .transport.base import DEFAULT_SESSION_PARAM

    if not isinstance(settings, Mapping):
        raise PlanError("the target must be an object")
    raw_headers = settings.get("headers") or {}
    headers: Dict[str, str] = {}
    if isinstance(raw_headers, Mapping):
        headers = {str(k): str(v) for k, v in raw_headers.items()}
    elif isinstance(raw_headers, list):
        for item in raw_headers:
            if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
                raise PlanError("each target header needs a name and a value")
            headers[item["name"]] = str(item.get("value", ""))
    if send_start is None:
        start = _pick(settings, "sendStart", "send_start")
        send_start = True if start is None else bool(start)
    return Target(
        url=str(settings.get("url") or ""),
        format=str(settings.get("format") or ""),
        sample_rate=int(_pick(settings, "sampleRate", "sample_rate", 16000)),
        headers=headers,
        session_query_param=str(_pick(settings, "sessionQueryParam", "session_query_param",
                                      DEFAULT_SESSION_PARAM)),
        send_start=bool(send_start),
        connector=connector,
    )


def target_label(url: str) -> str:
    """The target URL as the API stores it for an SDK run: scheme, host and path only, so no query
    (which can hold credentials) and no user info ever leave the machine."""
    parts = urlsplit(url)
    host = parts.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parts.scheme, host, parts.path, "", ""))
