"""Patch ElevenLabs' TextToSpeechClient.convert at the class level.

Same rationale as ``openai_audio.py`` — TTS returns raw audio bytes with no
usage at all, so cost is proportional to input text length
(``gen_ai.audio.input_characters``), not tokens.

``convert()`` returns an iterator of audio chunks (sync) or an async iterator
(async) — the wrapper only brackets the call itself and passes the
iterator/async-iterator straight through unconsumed; it never buffers or
inspects the audio bytes.

Degrades to a no-op if ``elevenlabs`` isn't installed and never raises into
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


def patch_elevenlabs(client: "EvalKitClient") -> None:
    """Patch elevenlabs.text_to_speech's TextToSpeechClient (sync + async). Idempotent."""
    try:
        from elevenlabs.text_to_speech.client import AsyncTextToSpeechClient, TextToSpeechClient
    except ImportError:
        return

    for cls, is_async in ((TextToSpeechClient, False), (AsyncTextToSpeechClient, True)):
        try:
            _patch_convert(client, cls, is_async)
        except Exception:
            pass


def _patch_convert(ek: "EvalKitClient", Cls: Any, is_async: bool) -> None:
    if getattr(getattr(Cls, "convert", None), "_evalkit", False):
        return
    original = Cls.convert

    def _start(model: str, text: str, req_body: Optional[str]):
        ctx = current_span_ctx()
        trace_id = ctx.trace_id if ctx else get_or_create_ambient_trace_id()
        parent_span_id = ctx.span_id if ctx else None
        span_id = _new_span_id()
        start_ms = time.time() * 1000
        ts = _now_iso()

        def end(status: str = "OK", *, error: Any = None, extra: Optional[Dict[str, Any]] = None) -> None:
            try:
                attrs: Dict[str, Any] = {"gen_ai.system": "elevenlabs"}
                if req_body:
                    attrs["gen_ai.request.body"] = req_body
                if extra:
                    attrs.update(extra)
                if error is not None:
                    attrs["error.message"] = str(error)
                ek.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call", status=status, status_message=str(error) if error is not None else None,
                    operation="audio.speech", service_name=ek._opts.service_name,
                    provider="elevenlabs", model=model,
                    prompt=text[:_TEXT_MAX] if text else None,
                    completion="[audio generated]" if status == "OK" else None,
                    attributes=attrs, latency_ms=int(time.time() * 1000 - start_ms),
                    start_time=ts, end_time=_now_iso(), created_at=ts,
                    environment=ek._opts.environment, app_version=ek._opts.app_version,
                    session_id=ctx.session_id if ctx else None,
                    device_id=(ctx.device_id if ctx else None) or ek._opts.device_id,
                    sdk_version=SDK_VERSION,
                ))
            except Exception:
                pass

        return end

    def _run(self: Any, voice_id: str, *args: Any, **kwargs: Any):
        model = str(kwargs.get("model_id", "unknown"))
        text = kwargs.get("text") or ""
        req_body = _safe_body({k: v for k, v in kwargs.items() if k != "text"})
        end = _start(model, text, req_body)
        extra = {"gen_ai.audio.input_characters": len(text), "gen_ai.request.voice": voice_id}
        try:
            result = original(self, voice_id, *args, **kwargs)
        except Exception as exc:
            end("ERROR", error=exc, extra=extra)
            raise
        end("OK", extra=extra)
        return result

    async def _arun(self: Any, voice_id: str, *args: Any, **kwargs: Any):
        model = str(kwargs.get("model_id", "unknown"))
        text = kwargs.get("text") or ""
        req_body = _safe_body({k: v for k, v in kwargs.items() if k != "text"})
        end = _start(model, text, req_body)
        extra = {"gen_ai.audio.input_characters": len(text), "gen_ai.request.voice": voice_id}
        try:
            result = await original(self, voice_id, *args, **kwargs)
        except Exception as exc:
            end("ERROR", error=exc, extra=extra)
            raise
        end("OK", extra=extra)
        return result

    wrapper = _arun if is_async else _run
    wrapper._evalkit = True  # type: ignore[attr-defined]
    Cls.convert = wrapper
