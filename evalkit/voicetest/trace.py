"""Our side of a test call as an EvalKit trace (CONTRACT section 6).

One ``voice.call`` root span and one ``voice.turn`` child per turn. The spans are built after the call
from the scorecard, so they sit at the times the recording measured (``t0`` plus sample-clock ms),
never at the moment this code runs. ``session.id`` is the call's session token; the bot's own spans
join ours by that session and by time, and neither trace is nested inside the other.

In the developer's process it sends through the EvalKit client they already initialised, under their
subscription key. A runner that sends for others (the hosted runner, with a run's ingest key) builds
a client of its own with `ingest_client`, which leaves the process's ``evalkit.init()`` client alone.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_log = logging.getLogger("evalkit.voicetest")

NS = "evalkit.voice.v1."
CALL_SPAN = "voice.call"
TURN_SPAN = "voice.turn"


def _iso(epoch_ms: float) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _put(attrs: Dict[str, Any], key: str, value: Any) -> None:
    """Absent rather than null: a metric the call has no value for (no bot speech) is left out."""
    if value is not None:
        attrs[NS + key] = value


def _caller_health_json(health: Dict[str, Any]) -> str:
    compact = dict(health)
    crosscheck = compact.get("vad_crosscheck")
    if isinstance(crosscheck, dict):
        # The per-line deltas stay in scorecard.json; the span carries the summary.
        compact["vad_crosscheck"] = {k: v for k, v in crosscheck.items() if k != "lines"}
    return json.dumps(compact, separators=(",", ":"), sort_keys=True)


def call_attributes(scorecard: Dict[str, Any], recording_uri: Optional[str] = None) -> Dict[str, Any]:
    """The ``voice.call`` attributes, named as in CONTRACT section 6."""
    call = scorecard["call"]
    attrs: Dict[str, Any] = {}
    _put(attrs, "transport", call.get("transport"))
    _put(attrs, "sample_rate", call.get("sample_rate"))
    _put(attrs, "duration_ms", call.get("duration_ms"))
    _put(attrs, "first_word_ms", call.get("first_word_ms"))
    _put(attrs, "response_delay_p50_ms", call.get("response_delay_p50_ms"))
    _put(attrs, "response_delay_p95_ms", call.get("response_delay_p95_ms"))
    _put(attrs, "talk_over_ms", call.get("talk_over_ms"))
    _put(attrs, "talk_over_count", call.get("talk_over_count"))
    _put(attrs, "max_silence_ms", call.get("max_silence_ms"))
    _put(attrs, "vad_version", (scorecard.get("vad") or {}).get("version"))
    _put(attrs, "analysis_version", (scorecard.get("analysis") or {}).get("version"))
    _put(attrs, "recording_uri", recording_uri)
    _put(attrs, "caller_health", _caller_health_json(scorecard.get("caller_health") or {}))
    _put(attrs, "session", call.get("session"))
    _put(attrs, "call.mode", call.get("mode"))
    return attrs


def turn_attributes(turn: Dict[str, Any], mode: Optional[str] = None) -> Dict[str, Any]:
    """One ``voice.turn``'s attributes, named as in CONTRACT section 6. `mode` is the call's: a dynamic
    call's turns also carry the caller's think time."""
    attrs: Dict[str, Any] = {}
    caller = turn.get("caller") or {}
    bot = turn.get("bot") or {}
    _put(attrs, "turn.index", turn.get("index"))
    _put(attrs, "caller.speech_start_ms", caller.get("speech_start_ms"))
    _put(attrs, "caller.speech_end_ms", caller.get("speech_end_ms"))
    _put(attrs, "bot.speech_start_ms", bot.get("speech_start_ms"))
    _put(attrs, "bot.speech_end_ms", bot.get("speech_end_ms"))
    _put(attrs, "response_delay_ms", turn.get("response_delay_ms"))
    _put(attrs, "talk_over_ms", turn.get("talk_over_ms"))
    _put(attrs, "caller_overlap_ms", turn.get("caller_overlap_ms"))
    _put(attrs, "interruption.planned", bool(turn.get("interruption_planned")))
    _put(attrs, "interruption.status", turn.get("interruption_status"))
    _put(attrs, "interruption.stop_ms", turn.get("interruption_stop_ms"))
    _put(attrs, "silence_ms", turn.get("silence_ms"))
    _put(attrs, "caller_unhealthy", bool(turn.get("caller_unhealthy")))
    _put(attrs, "transcript.caller", turn.get("caller_text") or turn.get("text") or "")
    # What the turn endpoint heard the bot say (dynamic calls); empty for a scripted call.
    _put(attrs, "transcript.bot", turn.get("heard") or turn.get("bot_transcript") or "")
    if mode == "dynamic":
        _put(attrs, "turn.think_ms", int(caller.get("think_ms") or 0))
    return attrs


def build_call_spans(
    scorecard: Dict[str, Any],
    *,
    recording_uri: Optional[str] = None,
    service_name: Optional[str] = None,
    environment: Optional[str] = None,
    app_version: Optional[str] = None,
    base_attributes: Optional[Dict[str, Any]] = None,
    status: str = "OK",
    trace_id: Optional[str] = None,
) -> List[Any]:
    """The call's spans as SDK envelopes, root first. Pure: sends nothing.

    Raises ValueError when the scorecard has no call start time (``t0``), since the spans could then
    only be placed at the wrong time.
    """
    from evalkit._version import SDK_VERSION
    from evalkit.core.types import TraceEnvelope

    call = scorecard["call"]
    t0 = call.get("t0_epoch_ms")
    if not isinstance(t0, (int, float)):
        raise ValueError("the scorecard has no t0_epoch_ms; call.json must record the call's start time")
    session = call.get("session")
    trace_id = trace_id or secrets.token_hex(16)
    root_id = secrets.token_hex(8)
    duration = int(call.get("duration_ms") or 0)
    common = dict(
        service_name=service_name,
        environment=environment,
        app_version=app_version,
        session_id=session,
        # Empty, not None: None would make the SDK stamp the ambient agent (the bot under test) on
        # spans that are the test caller's.
        agent_name="",
        sdk_version=SDK_VERSION,
    )

    def attributes(own: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base_attributes or {})
        merged.update(own)
        return merged

    root_start = _iso(t0)
    spans = [
        TraceEnvelope(
            trace_id=trace_id,
            span_id=root_id,
            span_type="log",
            status=status,  # type: ignore[arg-type]
            operation=CALL_SPAN,
            attributes=attributes(call_attributes(scorecard, recording_uri)),
            latency_ms=duration,
            start_time=root_start,
            end_time=_iso(t0 + duration),
            created_at=root_start,
            **common,  # type: ignore[arg-type]
        )
    ]
    mode = call.get("mode")
    for turn in scorecard.get("turns") or []:
        window = turn.get("window") or {}
        lo = int(window.get("start_ms", 0))
        hi = int(window.get("end_ms", lo))
        start = _iso(t0 + lo)
        spans.append(
            TraceEnvelope(
                trace_id=trace_id,
                span_id=secrets.token_hex(8),
                parent_span_id=root_id,
                # The bot's own voice.turn spans are "agent" spans; the caller's match them, and a
                # leaf "log" row would be hidden by the waterfall's default log filter.
                span_type="agent",
                status="OK",
                operation=TURN_SPAN,
                attributes=attributes(turn_attributes(turn, mode)),
                latency_ms=max(0, hi - lo),
                start_time=start,
                end_time=_iso(t0 + hi),
                created_at=start,
                **common,  # type: ignore[arg-type]
            )
        )
    return spans


def ingest_client(
    ingest_key: str,
    base_url: str = "https://api.syntropylabs.ai",
    *,
    service_name: str = "evalkit-voicetest",
    environment: Optional[str] = None,
) -> Any:
    """A client of its own that sends to ``base_url``'s ingest under ``ingest_key``: for the voice.call
    traces of a run made on someone else's behalf. It patches nothing and is not the process's
    ``evalkit.init()`` client. Call ``destroy()`` on it once the run's traces are flushed."""
    from evalkit.core.client import EvalKitClient
    from evalkit.core.types import EvalKitOptions

    if not isinstance(ingest_key, str) or not ingest_key.strip():
        raise ValueError("an ingest key is required")
    return EvalKitClient(
        EvalKitOptions(
            subscription_key=ingest_key.strip(),
            base_url=base_url.rstrip("/"),
            api_url=base_url.rstrip("/"),
            service_name=service_name,
            environment=environment,
        )
    )


def _default_client() -> Any:
    import evalkit

    return evalkit._get_instance()


def send_call_trace(
    scorecard: Dict[str, Any],
    client: Any = None,
    *,
    recording_uri: Optional[str] = None,
    status: str = "OK",
    flush: bool = True,
    flush_timeout_s: float = 10.0,
) -> str:
    """Queues the call's trace on ``client`` (default: the process's ``evalkit.init()`` client) and
    returns its trace id. The client's own redaction, content and export settings apply."""
    client = client if client is not None else _default_client()
    opts = getattr(client, "_opts", None)
    meta = client._meta_attrs() if hasattr(client, "_meta_attrs") else {}
    spans = build_call_spans(
        scorecard,
        recording_uri=recording_uri,
        service_name=getattr(opts, "service_name", None),
        environment=getattr(opts, "environment", None),
        app_version=getattr(opts, "app_version", None),
        base_attributes=meta,
        status=status,
    )
    for span in spans[1:]:
        client.enqueue(span)
    client.enqueue(spans[0])
    if flush and not client.flush(timeout=flush_timeout_s):
        _log.warning("the voice.call trace for session %s was queued but not confirmed sent", scorecard["call"].get("session"))
    return spans[0].trace_id


def trace_recording(
    recording_dir: str,
    client: Any = None,
    *,
    recording_uri: Optional[str] = None,
    flush: bool = True,
) -> str:
    """Sends the trace of a recording directory, analysing it first if it has no ``scorecard.json``."""
    from .analysis.scorecard import SCORECARD_NAME, write_scorecard

    path = os.path.join(recording_dir, SCORECARD_NAME)
    if not os.path.exists(path):
        write_scorecard(recording_dir)
    with open(path, "r", encoding="utf-8") as fh:
        scorecard = json.load(fh)
    status = "OK"
    call_path = os.path.join(recording_dir, "call.json")
    if os.path.exists(call_path):
        with open(call_path, "r", encoding="utf-8") as fh:
            call = json.load(fh)
        if isinstance(call, dict) and call.get("ended_by") == "error":
            status = "ERROR"
    uri = recording_uri or "file://" + os.path.abspath(os.path.join(recording_dir, "recording.wav"))
    return send_call_trace(scorecard, client, recording_uri=uri, status=status, flush=flush)
