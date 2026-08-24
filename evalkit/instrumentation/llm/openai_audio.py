from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..._version import SDK_VERSION
from ...core.client import _new_span_id, _now_iso
from ...core.context import current_span_ctx, get_or_create_ambient_trace_id
from ...core.types import TraceEnvelope
from ._bodies import _dump, _safe_body

if TYPE_CHECKING:
    from ...core.client import EvalKitClient

_TEXT_MAX = 8192


def patch_openai_audio(client: "EvalKitClient") -> None:
    try:
        from openai.resources.audio.speech import AsyncSpeech, Speech
        from openai.resources.audio.transcriptions import AsyncTranscriptions, Transcriptions
        from openai.resources.audio.translations import AsyncTranslations, Translations
    except ImportError:
        return

    for cls, is_async in ((Speech, False), (AsyncSpeech, True)):
        try:
            _patch_speech(client, cls, is_async)
        except Exception:
            pass
    for cls, is_async in ((Transcriptions, False), (AsyncTranscriptions, True)):
        try:
            _patch_transcribe(client, cls, "audio.transcription", is_async)
        except Exception:
            pass
    for cls, is_async in ((Translations, False), (AsyncTranslations, True)):
        try:
            _patch_transcribe(client, cls, "audio.translation", is_async)
        except Exception:
            pass


def _start(ek: "EvalKitClient", operation: str, model: str, prompt: str, req_body: Optional[str]):
    ctx = current_span_ctx()
    trace_id = ctx.trace_id if ctx else get_or_create_ambient_trace_id()
    parent_span_id = ctx.span_id if ctx else None
    span_id = _new_span_id()
    start_ms = time.time() * 1000
    ts = _now_iso()

    def end(status: str = "OK", *, error: Any = None, completion: str = "", extra: Optional[Dict[str, Any]] = None,
            tokens_in: Optional[int] = None, tokens_out: Optional[int] = None) -> None:
        try:
            attrs: Dict[str, Any] = {"gen_ai.system": "openai"}
            if req_body:
                attrs["gen_ai.request.body"] = req_body
            if extra:
                attrs.update(extra)
            if error is not None:
                attrs["error.message"] = str(error)
            ek.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status=status, status_message=str(error) if error is not None else None,
                operation=operation, service_name=ek._opts.service_name,
                provider="openai", model=model,
                prompt=prompt[:_TEXT_MAX] if prompt else None,
                completion=completion[:_TEXT_MAX] if completion else None,
                tokens_in=tokens_in, tokens_out=tokens_out,
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


def _patch_speech(ek: "EvalKitClient", Speech: Any, is_async: bool) -> None:
    if getattr(getattr(Speech, "create", None), "_evalkit", False):
        return
    original = Speech.create

    def _run(self: Any, *args: Any, **kwargs: Any):
        model = str(kwargs.get("model", "unknown"))
        input_text = kwargs.get("input") or ""
        req_body = _safe_body({k: v for k, v in kwargs.items() if k != "input"})
        end = _start(ek, "audio.speech", model, input_text, req_body)
        extra = {"gen_ai.audio.input_characters": len(input_text), "gen_ai.request.voice": str(kwargs.get("voice", ""))}
        try:
            result = original(self, *args, **kwargs)
        except Exception as exc:
            end("ERROR", error=exc, extra=extra)
            raise
        end("OK", completion="[audio generated]", extra=extra)
        return result

    async def _arun(self: Any, *args: Any, **kwargs: Any):
        model = str(kwargs.get("model", "unknown"))
        input_text = kwargs.get("input") or ""
        req_body = _safe_body({k: v for k, v in kwargs.items() if k != "input"})
        end = _start(ek, "audio.speech", model, input_text, req_body)
        extra = {"gen_ai.audio.input_characters": len(input_text), "gen_ai.request.voice": str(kwargs.get("voice", ""))}
        try:
            result = await original(self, *args, **kwargs)
        except Exception as exc:
            end("ERROR", error=exc, extra=extra)
            raise
        end("OK", completion="[audio generated]", extra=extra)
        return result

    wrapper = _arun if is_async else _run
    wrapper._evalkit = True
    Speech.create = wrapper


def _extract_transcription_usage(response: Any) -> Dict[str, Any]:
    extra: Dict[str, Any] = {}
    tokens_in = tokens_out = None
    try:
        usage = getattr(response, "usage", None)
        if usage is not None:
            utype = getattr(usage, "type", None)
            if utype == "duration":
                extra["gen_ai.audio.duration_seconds"] = getattr(usage, "seconds", None)
            elif utype == "tokens":
                tokens_in = getattr(usage, "input_tokens", None)
                tokens_out = getattr(usage, "output_tokens", None)
                details = getattr(usage, "input_token_details", None)
                audio_tokens = getattr(details, "audio_tokens", None) if details is not None else None
                if audio_tokens:
                    extra["gen_ai.usage.audio_input_tokens"] = audio_tokens
        duration = getattr(response, "duration", None)
        if duration is not None:
            extra["gen_ai.audio.duration_seconds"] = duration
    except Exception:
        pass
    return {"extra": extra, "tokens_in": tokens_in, "tokens_out": tokens_out}


def _transcription_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    return str(getattr(response, "text", "") or "")


def _patch_transcribe(ek: "EvalKitClient", Cls: Any, operation: str, is_async: bool) -> None:
    if getattr(getattr(Cls, "create", None), "_evalkit", False):
        return
    original = Cls.create

    def _run(self: Any, *args: Any, **kwargs: Any):
        model = str(kwargs.get("model", "unknown"))
        file_obj = kwargs.get("file")
        file_name = getattr(file_obj, "name", None) or "audio file"
        req_body = _safe_body({k: v for k, v in kwargs.items() if k != "file"})
        end = _start(ek, operation, model, f"[{file_name}]", req_body)
        if kwargs.get("stream"):
            return original(self, *args, **kwargs)
        try:
            result = original(self, *args, **kwargs)
        except Exception as exc:
            end("ERROR", error=exc)
            raise
        usage = _extract_transcription_usage(result)
        resp_body = _safe_body(_dump(result)) if not isinstance(result, str) else None
        extra = usage["extra"]
        if resp_body:
            extra["gen_ai.response.body"] = resp_body
        end("OK", completion=_transcription_text(result), extra=extra,
            tokens_in=usage["tokens_in"], tokens_out=usage["tokens_out"])
        return result

    async def _arun(self: Any, *args: Any, **kwargs: Any):
        model = str(kwargs.get("model", "unknown"))
        file_obj = kwargs.get("file")
        file_name = getattr(file_obj, "name", None) or "audio file"
        req_body = _safe_body({k: v for k, v in kwargs.items() if k != "file"})
        end = _start(ek, operation, model, f"[{file_name}]", req_body)
        if kwargs.get("stream"):
            return await original(self, *args, **kwargs)
        try:
            result = await original(self, *args, **kwargs)
        except Exception as exc:
            end("ERROR", error=exc)
            raise
        usage = _extract_transcription_usage(result)
        resp_body = _safe_body(_dump(result)) if not isinstance(result, str) else None
        extra = usage["extra"]
        if resp_body:
            extra["gen_ai.response.body"] = resp_body
        end("OK", completion=_transcription_text(result), extra=extra,
            tokens_in=usage["tokens_in"], tokens_out=usage["tokens_out"])
        return result

    wrapper = _arun if is_async else _run
    wrapper._evalkit = True
    Cls.create = wrapper
