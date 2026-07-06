"""Patch AssemblyAI's Transcriber.transcribe at the class level.

Billed by audio duration, not tokens — cost is proportional to
``response.audio_duration`` (seconds). Deliberately NOT the length of
``response.audio_url``: a shared OpenLit helper conflates the two (treats any
non-empty "prompt"-like string as character-billed), which silently prices
AssemblyAI transcripts off the length of the audio URL string rather than the
actual audio duration — confirmed by reading their real source. We use
audio_duration only.

AssemblyAI also doesn't raise on a failed transcription — a failed job comes
back as a normal ``Transcript`` with ``.error`` set (and ``.status ==
"error"``), not an exception — so success/failure must be read off the
response, not just caught in a try/except.

Only ``transcribe()`` is patched. ``transcribe_async()`` returns a
``concurrent.futures.Future`` (thread-pool based, not an asyncio coroutine),
a genuinely different concurrency model this wrapper doesn't attempt to
bridge — same scope OpenLit itself covers.

Degrades to a no-op if ``assemblyai`` isn't installed and never raises into
the caller.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..._version import SDK_VERSION
from ...core.client import _new_span_id, _now_iso
from ...core.context import current_span_ctx, get_or_create_ambient_trace_id
from ...core.types import TraceEnvelope
from ._bodies import _safe_body

if TYPE_CHECKING:
    from ...core.client import EvalKitClient

_TEXT_MAX = 8192


def patch_assemblyai(client: "EvalKitClient") -> None:
    """Patch assemblyai.transcriber.Transcriber.transcribe. Idempotent."""
    try:
        from assemblyai.transcriber import Transcriber
    except ImportError:
        return
    try:
        _patch_transcribe(client, Transcriber)
    except Exception:
        pass


def _patch_transcribe(ek: "EvalKitClient", Transcriber: Any) -> None:
    if getattr(getattr(Transcriber, "transcribe", None), "_evalkit", False):
        return
    original = Transcriber.transcribe

    def traced(self: Any, data: Any, config: Any = None, **kwargs: Any) -> Any:
        ctx = current_span_ctx()
        trace_id = ctx.trace_id if ctx else get_or_create_ambient_trace_id()
        parent_span_id = ctx.span_id if ctx else None
        span_id = _new_span_id()
        start_ms = time.time() * 1000
        ts = _now_iso()

        source = data if isinstance(data, str) else getattr(data, "name", None) or "audio file"
        model = "best"
        try:
            model = str(getattr(config, "speech_model", None) or "best")
        except Exception:
            pass
        req_body = _safe_body({"speech_model": model})

        def end(status: str, *, error: Any = None, completion: Optional[str] = None,
                extra: Optional[Dict[str, Any]] = None) -> None:
            try:
                attrs: Dict[str, Any] = {"gen_ai.system": "assemblyai"}
                if req_body:
                    attrs["gen_ai.request.body"] = req_body
                if extra:
                    attrs.update(extra)
                if error is not None:
                    attrs["error.message"] = str(error)
                ek.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call", status=status, status_message=str(error) if error is not None else None,
                    operation="audio.transcription", service_name=ek._opts.service_name,
                    provider="assemblyai", model=model,
                    prompt=f"[{source}]",
                    completion=completion[:_TEXT_MAX] if completion else None,
                    attributes=attrs, latency_ms=int(time.time() * 1000 - start_ms),
                    start_time=ts, end_time=_now_iso(), created_at=ts,
                    environment=ek._opts.environment, app_version=ek._opts.app_version,
                    session_id=ctx.session_id if ctx else None,
                    device_id=(ctx.device_id if ctx else None) or ek._opts.device_id,
                    sdk_version=SDK_VERSION,
                ))
            except Exception:
                pass

        try:
            result = original(self, data, config, **kwargs)
        except Exception as exc:
            end("ERROR", error=exc)
            raise

        error_msg = getattr(result, "error", None)
        if error_msg:
            end("ERROR", error=error_msg)
            return result

        extra: Dict[str, Any] = {}
        duration = getattr(result, "audio_duration", None)
        if duration is not None:
            extra["gen_ai.audio.duration_seconds"] = duration
        end("OK", completion=getattr(result, "text", None), extra=extra)
        return result

    traced._evalkit = True  # type: ignore[attr-defined]
    Transcriber.transcribe = traced
