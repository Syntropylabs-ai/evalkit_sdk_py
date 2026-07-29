"""Patch google.genai (new unified Google AI SDK, used by Google ADK) at the class level.

Unlike the instance-level patchers, this patches Models and AsyncModels **classes**
so every Client instance — including those created internally by Google ADK — is traced.

Covers both the unary (`generate_content`) and streaming (`generate_content_stream`)
methods, sync and async, so ADK runs (which fan out across many internal Gemini calls
and sub-agents) are captured regardless of the run mode.

Usage:
    import evalkit
    evalkit.init(...)
    # auto-instrumented by init(); manual call is also available:
    evalkit.patch_google_genai()   # works for ADK, direct genai, etc.
"""
from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..._version import SDK_VERSION
from ...core.client import EvalKitClient, _new_span_id, _now_iso
from ...core.context import (
    SpanContext,
    _ambient_trace_id,
    _current_span, current_span_ctx,
    get_or_create_ambient_trace_id,
)
from ...core.types import SpanEvent, TraceEnvelope
from .._toolcalls import from_google_parts

if TYPE_CHECKING:
    pass


def patch_google_genai(client: EvalKitClient) -> None:
    """
    Patch google.genai.Models and google.genai.AsyncModels at the class level.

    This intercepts every `client.models.generate_content[_stream]()` call across ALL
    google.genai Client instances, including those created by Google ADK internally.
    Call once at startup after evalkit.init(). Safe to call repeatedly (idempotent).
    """
    try:
        from google.genai import models as _gm
    except ImportError:
        return

    _patch_sync_models(client, _gm)
    _patch_async_models(client, _gm)


# ── extraction helpers ────────────────────────────────────────────────────────


def _extract_prompt(contents: Any, config: Any = None) -> str:
    """Convert google.genai contents to a readable prompt string."""
    parts: List[str] = []

    # System instruction from config
    if config is not None:
        si = getattr(config, "system_instruction", None)
        if si:
            if hasattr(si, "parts"):
                for p in si.parts:
                    t = getattr(p, "text", None)
                    if t:
                        parts.append(f"system: {t}")
            elif isinstance(si, str):
                parts.append(f"system: {si}")

    if isinstance(contents, str):
        parts.append(contents)
    elif isinstance(contents, list):
        for item in contents:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                role = item.get("role", "")
                for p in item.get("parts", []):
                    t = p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")
                    if t:
                        parts.append(f"{role}: {t}" if role else t)
            else:
                role = getattr(item, "role", "")
                for p in getattr(item, "parts", []):
                    t = getattr(p, "text", None)
                    if t:
                        parts.append(f"{role}: {t}" if role else t)
    elif contents is not None:
        text = getattr(contents, "text", None)
        if text:
            parts.append(str(text))
        else:
            parts.append(str(contents))

    return "\n".join(parts)


def _extract_completion(response: Any) -> str:
    """Pull text from a generate_content response, joining all non-thought text parts.

    When the model returns only function calls (no text), we return a compact
    JSON summary of the calls so the span is never silently empty. Thinking
    (``part.thought``) parts are excluded here — see ``_extract_thinking``.
    """
    text_parts: List[str] = []
    fn_parts: List[str] = []
    try:
        for candidate in (getattr(response, "candidates", None) or []):
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", []) or []:
                if getattr(part, "thought", False):
                    continue
                t = getattr(part, "text", None)
                if t:
                    text_parts.append(str(t))
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    name = getattr(fc, "name", "?")
                    raw_args = getattr(fc, "args", None) or {}
                    try:
                        args_str = json.dumps(dict(raw_args))
                    except Exception:
                        args_str = str(raw_args)
                    fn_parts.append(f"[tool_call] {name}({args_str})")
    except Exception:
        pass

    if text_parts:
        return "".join(text_parts)
    if fn_parts:
        return "\n".join(fn_parts)
    # Fall back to the .text shortcut.
    try:
        text = getattr(response, "text", None)
        if text:
            return str(text)
    except Exception:
        pass
    return ""


def _extract_thinking(response: Any) -> str:
    """Pull Gemini's extended-thinking text (``part.thought`` parts) out of a
    generate_content response or streaming chunk. Never raises."""
    parts: List[str] = []
    try:
        for candidate in (getattr(response, "candidates", None) or []):
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", []) or []:
                if getattr(part, "thought", False):
                    t = getattr(part, "text", None)
                    if t:
                        parts.append(str(t))
    except Exception:
        pass
    return "".join(parts)


def _extract_tool_events(response: Any, ts: str) -> List[SpanEvent]:
    """Extract function calls from response as SpanEvent list."""
    return from_google_parts(response, ts)


def _usage_tokens(response: Any):
    """Return (input_tokens, output_tokens, reasoning_tokens, cached_tokens) from a response
    or streaming chunk. `cached_tokens` is the subset of input_tokens Gemini served from its
    (much cheaper) context cache — it's already included in input_tokens, never additive.
    Surfaced separately (gen_ai.usage.cached_tokens attribute) so downstream cost math isn't
    forced to price an entire long, mostly-cached prompt at full fresh-input-token rate."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return 0, 0, 0, 0
    tokens_in = getattr(meta, "prompt_token_count", 0) or 0
    tokens_out = getattr(meta, "candidates_token_count", 0) or 0
    reasoning_tokens = getattr(meta, "thoughts_token_count", 0) or 0
    cached_tokens = getattr(meta, "cached_content_token_count", 0) or 0
    return tokens_in, tokens_out, reasoning_tokens, cached_tokens


def _resolve_trace(ek: EvalKitClient):
    """Resolve (trace_id, parent_span_id, session_id, device_id) for a new llm_call.

    Falls back to the context's ambient trace id (NOT a brand-new id) when there
    is no active span, so the many internal Gemini calls an ADK run makes share a
    single trace instead of each fragmenting into its own. Pins the ambient id so
    sequential sibling calls in the same execution context join too.
    """
    ctx_parent = current_span_ctx()
    if ctx_parent is not None:
        trace_id = ctx_parent.trace_id
    else:
        trace_id = get_or_create_ambient_trace_id()
    _ambient_trace_id.set(trace_id)
    parent_span_id = ctx_parent.span_id if ctx_parent else None
    session_id = ctx_parent.session_id if ctx_parent else None
    device_id = (ctx_parent.device_id if ctx_parent else None) or ek._opts.device_id
    return trace_id, parent_span_id, session_id, device_id


def _dump(o: Any) -> Any:
    if isinstance(o, (list, tuple)):
        return [_dump(x) for x in o]
    if hasattr(o, "model_dump"):
        try:
            return o.model_dump(exclude_none=True)
        except Exception:
            return str(o)
    return o


def _safe_body(obj: Any, limit: int = 16384) -> Optional[str]:
    """Best-effort JSON of a request/response object for the trace body. Never raises;
    truncated to keep the span small (the exporter's fitEvent drops it first anyway)."""
    if obj is None:
        return None
    import json as _json
    try:
        s = _json.dumps(obj, default=str)
    except Exception:
        try:
            s = str(obj)
        except Exception:
            return None
    return s if len(s) <= limit else s[:limit] + "…"


def _request_body(contents: Any, config: Any) -> Optional[str]:
    body: Dict[str, Any] = {"contents": _dump(contents)}
    if config is not None:
        body["config"] = _dump(config)
    return _safe_body(body)


def _response_body(result: Any) -> Optional[str]:
    return _safe_body(_dump(result))


def _build_envelope(
    client: EvalKitClient,
    *,
    trace_id: str,
    span_id: str,
    parent_span_id: Optional[str],
    model: str,
    prompt: str,
    completion: str,
    tokens_in: int,
    tokens_out: int,
    latency: int,
    ts: str,
    end_ts: str,
    session_id: Optional[str] = None,
    device_id: Optional[str] = None,
    status: str = "OK",
    status_message: Optional[str] = None,
    events: Optional[List[SpanEvent]] = None,
    attrs: Optional[Dict[str, Any]] = None,
    request_body: Optional[str] = None,
    response_body: Optional[str] = None,
) -> TraceEnvelope:
    base_attrs: Dict[str, Any] = {
        **client._meta_attrs(),
        "gen_ai.request.model": model,
        "gen_ai.response.model": model,
        "gen_ai.usage.total_tokens": tokens_in + tokens_out,
    }
    if request_body:
        base_attrs["gen_ai.request.body"] = request_body
    if response_body:
        base_attrs["gen_ai.response.body"] = response_body
    if attrs:
        base_attrs.update(attrs)

    return TraceEnvelope(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        span_type="llm_call",
        status=status,
        status_message=status_message,
        operation="generate_content",
        service_name=client._opts.service_name,
        provider="google-genai",
        model=model,
        prompt=prompt,
        completion=completion,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency,
        start_time=ts,
        end_time=end_ts,
        created_at=ts,
        events=events or None,
        attributes=base_attrs,
        environment=client._opts.environment,
        app_version=client._opts.app_version,
        session_id=session_id,
        device_id=device_id or client._opts.device_id,
        sdk_version=SDK_VERSION,
    )


# ── sync patchers ──────────────────────────────────────────────────────────────


def _patch_sync_models(ek: EvalKitClient, gm: Any) -> None:
    """Patch google.genai.Models.{generate_content,generate_content_stream}."""
    Models = getattr(gm, "Models", None)
    if Models is None:
        return

    if not getattr(getattr(Models, "generate_content", None), "_evalkit", False):
        original = Models.generate_content

        def traced_generate_content(self: Any, *, model: str, contents: Any, config: Any = None, **kwargs: Any) -> Any:
            trace_id, parent_span_id, session_id, device_id = _resolve_trace(ek)
            span_id = _new_span_id()
            start_ms = time.time() * 1000
            ts = _now_iso()
            prompt = _extract_prompt(contents, config)

            span_ctx = SpanContext(trace_id=trace_id, span_id=span_id, session_id=session_id, device_id=device_id)
            token = _current_span.set(span_ctx)
            try:
                result = original(self, model=model, contents=contents, config=config, **kwargs)
            except Exception as exc:
                end_ts = _now_iso()
                latency = int(time.time() * 1000 - start_ms)
                ek.enqueue(_build_envelope(
                    ek, trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    model=model, prompt=prompt, completion="", tokens_in=0, tokens_out=0,
                    latency=latency, ts=ts, end_ts=end_ts,
                    session_id=session_id, device_id=device_id,
                    status="ERROR", status_message=str(exc),
                ))
                raise
            finally:
                _current_span.reset(token)

            end_ts = _now_iso()
            latency = int(time.time() * 1000 - start_ms)
            completion = _extract_completion(result)
            thinking = _extract_thinking(result)
            events = _extract_tool_events(result, end_ts)
            tokens_in, tokens_out, reasoning_tokens, cached_tokens = _usage_tokens(result)
            extra_attrs: Dict[str, Any] = {}
            if thinking:
                extra_attrs["gen_ai.response.thinking"] = thinking
            if reasoning_tokens:
                extra_attrs["gen_ai.usage.reasoning_tokens"] = reasoning_tokens
            if cached_tokens:
                extra_attrs["gen_ai.usage.cached_tokens"] = cached_tokens

            ek.enqueue(_build_envelope(
                ek, trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                model=model, prompt=prompt, completion=completion,
                tokens_in=tokens_in, tokens_out=tokens_out,
                latency=latency, ts=ts, end_ts=end_ts, events=events,
                session_id=session_id, device_id=device_id,
                request_body=_request_body(contents, config),
                response_body=_response_body(result),
                attrs=extra_attrs or None,
            ))
            return result

        traced_generate_content._evalkit = True  # type: ignore[attr-defined]
        Models.generate_content = traced_generate_content

    if not getattr(getattr(Models, "generate_content_stream", None), "_evalkit", False):
        original_stream = getattr(Models, "generate_content_stream", None)
        if original_stream is not None:
            def traced_generate_content_stream(self: Any, *, model: str, contents: Any, config: Any = None, **kwargs: Any) -> Any:
                trace_id, parent_span_id, session_id, device_id = _resolve_trace(ek)
                span_id = _new_span_id()
                start_ms = time.time() * 1000
                ts = _now_iso()
                prompt = _extract_prompt(contents, config)

                try:
                    stream = original_stream(self, model=model, contents=contents, config=config, **kwargs)
                except Exception as exc:
                    end_ts = _now_iso()
                    ek.enqueue(_build_envelope(
                        ek, trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                        model=model, prompt=prompt, completion="", tokens_in=0, tokens_out=0,
                        latency=int(time.time() * 1000 - start_ms), ts=ts, end_ts=end_ts,
                        session_id=session_id, device_id=device_id,
                        status="ERROR", status_message=str(exc), attrs={"gen_ai.streaming": True},
                    ))
                    raise

                def _gen():
                    parts: List[str] = []
                    thinking_parts: List[str] = []
                    chunk_dumps: List[Any] = []
                    tin = tout = treason = tcached = 0
                    events: List[SpanEvent] = []
                    status = "OK"
                    smsg: Optional[str] = None
                    try:
                        for chunk in stream:
                            try:
                                t = _extract_completion(chunk)
                                if t:
                                    parts.append(t)
                                th = _extract_thinking(chunk)
                                if th:
                                    thinking_parts.append(th)
                                ci, co, cr, cc = _usage_tokens(chunk)
                                if ci:
                                    tin = ci
                                if co:
                                    tout = co
                                if cr:
                                    treason = cr
                                if cc:
                                    tcached = cc
                                evs = _extract_tool_events(chunk, _now_iso())
                                if evs:
                                    events.extend(evs)
                                chunk_dumps.append(_dump(chunk))
                            except Exception:
                                pass
                            yield chunk
                    except Exception as exc:
                        status, smsg = "ERROR", str(exc)
                        raise
                    finally:
                        end_ts = _now_iso()
                        try:
                            extra_attrs: Dict[str, Any] = {"gen_ai.streaming": True}
                            if thinking_parts:
                                extra_attrs["gen_ai.response.thinking"] = "".join(thinking_parts)
                            if treason:
                                extra_attrs["gen_ai.usage.reasoning_tokens"] = treason
                            if tcached:
                                extra_attrs["gen_ai.usage.cached_tokens"] = tcached
                            ek.enqueue(_build_envelope(
                                ek, trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                                model=model, prompt=prompt, completion="".join(parts),
                                tokens_in=tin, tokens_out=tout,
                                latency=int(time.time() * 1000 - start_ms), ts=ts, end_ts=end_ts,
                                events=events or None, session_id=session_id, device_id=device_id,
                                status=status, status_message=smsg, attrs=extra_attrs,
                                request_body=_request_body(contents, config),
                                response_body=_safe_body(chunk_dumps) if chunk_dumps else None,
                            ))
                        except Exception:
                            pass

                return _gen()

            traced_generate_content_stream._evalkit = True  # type: ignore[attr-defined]
            Models.generate_content_stream = traced_generate_content_stream


# ── async patchers ─────────────────────────────────────────────────────────────


def _patch_async_models(ek: EvalKitClient, gm: Any) -> None:
    """Patch google.genai.AsyncModels.{generate_content,generate_content_stream}."""
    AsyncModels = getattr(gm, "AsyncModels", None)
    if AsyncModels is None:
        return

    if not getattr(getattr(AsyncModels, "generate_content", None), "_evalkit", False):
        original = AsyncModels.generate_content

        async def traced_generate_content_async(self: Any, *, model: str, contents: Any, config: Any = None, **kwargs: Any) -> Any:
            trace_id, parent_span_id, session_id, device_id = _resolve_trace(ek)
            span_id = _new_span_id()
            start_ms = time.time() * 1000
            ts = _now_iso()
            prompt = _extract_prompt(contents, config)

            span_ctx = SpanContext(trace_id=trace_id, span_id=span_id, session_id=session_id, device_id=device_id)
            token = _current_span.set(span_ctx)
            try:
                result = await original(self, model=model, contents=contents, config=config, **kwargs)
            except Exception as exc:
                end_ts = _now_iso()
                latency = int(time.time() * 1000 - start_ms)
                ek.enqueue(_build_envelope(
                    ek, trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    model=model, prompt=prompt, completion="", tokens_in=0, tokens_out=0,
                    latency=latency, ts=ts, end_ts=end_ts,
                    session_id=session_id, device_id=device_id,
                    status="ERROR", status_message=str(exc),
                ))
                raise
            finally:
                _current_span.reset(token)

            end_ts = _now_iso()
            latency = int(time.time() * 1000 - start_ms)
            completion = _extract_completion(result)
            thinking = _extract_thinking(result)
            events = _extract_tool_events(result, end_ts)
            tokens_in, tokens_out, reasoning_tokens, cached_tokens = _usage_tokens(result)
            extra_attrs: Dict[str, Any] = {}
            if thinking:
                extra_attrs["gen_ai.response.thinking"] = thinking
            if reasoning_tokens:
                extra_attrs["gen_ai.usage.reasoning_tokens"] = reasoning_tokens
            if cached_tokens:
                extra_attrs["gen_ai.usage.cached_tokens"] = cached_tokens

            ek.enqueue(_build_envelope(
                ek, trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                model=model, prompt=prompt, completion=completion,
                tokens_in=tokens_in, tokens_out=tokens_out,
                latency=latency, ts=ts, end_ts=end_ts, events=events,
                session_id=session_id, device_id=device_id,
                request_body=_request_body(contents, config),
                response_body=_response_body(result),
                attrs=extra_attrs or None,
            ))
            return result

        traced_generate_content_async._evalkit = True  # type: ignore[attr-defined]
        AsyncModels.generate_content = traced_generate_content_async

    if not getattr(getattr(AsyncModels, "generate_content_stream", None), "_evalkit", False):
        original_stream = getattr(AsyncModels, "generate_content_stream", None)
        if original_stream is not None:
            async def traced_generate_content_stream_async(self: Any, *, model: str, contents: Any, config: Any = None, **kwargs: Any) -> Any:
                trace_id, parent_span_id, session_id, device_id = _resolve_trace(ek)
                span_id = _new_span_id()
                start_ms = time.time() * 1000
                ts = _now_iso()
                prompt = _extract_prompt(contents, config)

                try:
                    stream = await original_stream(self, model=model, contents=contents, config=config, **kwargs)
                except Exception as exc:
                    end_ts = _now_iso()
                    ek.enqueue(_build_envelope(
                        ek, trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                        model=model, prompt=prompt, completion="", tokens_in=0, tokens_out=0,
                        latency=int(time.time() * 1000 - start_ms), ts=ts, end_ts=end_ts,
                        session_id=session_id, device_id=device_id,
                        status="ERROR", status_message=str(exc), attrs={"gen_ai.streaming": True},
                    ))
                    raise

                async def _agen():
                    parts: List[str] = []
                    thinking_parts: List[str] = []
                    chunk_dumps: List[Any] = []
                    tin = tout = treason = tcached = 0
                    events: List[SpanEvent] = []
                    status = "OK"
                    smsg: Optional[str] = None
                    try:
                        async for chunk in stream:
                            try:
                                t = _extract_completion(chunk)
                                if t:
                                    parts.append(t)
                                th = _extract_thinking(chunk)
                                if th:
                                    thinking_parts.append(th)
                                ci, co, cr, cc = _usage_tokens(chunk)
                                if ci:
                                    tin = ci
                                if co:
                                    tout = co
                                if cr:
                                    treason = cr
                                if cc:
                                    tcached = cc
                                evs = _extract_tool_events(chunk, _now_iso())
                                if evs:
                                    events.extend(evs)
                                chunk_dumps.append(_dump(chunk))
                            except Exception:
                                pass
                            yield chunk
                    except Exception as exc:
                        status, smsg = "ERROR", str(exc)
                        raise
                    finally:
                        end_ts = _now_iso()
                        try:
                            extra_attrs: Dict[str, Any] = {"gen_ai.streaming": True}
                            if thinking_parts:
                                extra_attrs["gen_ai.response.thinking"] = "".join(thinking_parts)
                            if treason:
                                extra_attrs["gen_ai.usage.reasoning_tokens"] = treason
                            if tcached:
                                extra_attrs["gen_ai.usage.cached_tokens"] = tcached
                            ek.enqueue(_build_envelope(
                                ek, trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                                model=model, prompt=prompt, completion="".join(parts),
                                tokens_in=tin, tokens_out=tout,
                                latency=int(time.time() * 1000 - start_ms), ts=ts, end_ts=end_ts,
                                events=events or None, session_id=session_id, device_id=device_id,
                                status=status, status_message=smsg, attrs=extra_attrs,
                                request_body=_request_body(contents, config),
                                response_body=_safe_body(chunk_dumps) if chunk_dumps else None,
                            ))
                        except Exception:
                            pass

                return _agen()

            traced_generate_content_stream_async._evalkit = True  # type: ignore[attr-defined]
            AsyncModels.generate_content_stream = traced_generate_content_stream_async
