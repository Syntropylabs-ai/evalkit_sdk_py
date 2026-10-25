from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional

from ..._version import SDK_VERSION
from ...core.client import _new_span_id, _now_iso
from ...core.content import message_content_text
from ...core.context import (
    _ambient_trace_id,
    _current_span, child_span_ctx, current_span_ctx,
    get_or_create_ambient_trace_id,
)
from ...core.types import TraceEnvelope
from .._streaming import (
    _StreamState,
    record_stream_open_failure,
    stream_response_attrs,
    stream_tool_events,
    trace_async_stream,
    trace_sync_stream,
)
from .._safe import hook_failed
from .._toolcalls import from_openai_chat
from ._bodies import _dump, _safe_body

if TYPE_CHECKING:
    from ...core.client import EvalKitClient


def _messages_to_prompt(messages: Any) -> str:
    try:
        return "\n".join(
            f"{m.get('role', '')}: {message_content_text(m.get('content'))}"
            for m in (messages if isinstance(messages, (list, tuple)) else ())
            if isinstance(m, dict)
        )
    except Exception:
        return str(messages)[:4096]


def _completion_text(result: Any) -> str:
    try:
        choice = result.choices[0]
        return (getattr(choice.message, "content", None) or "") if choice else ""
    except Exception:
        return ""


def _usage(result: Any):
    usage = getattr(result, "usage", None)
    return (
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
    )


def _cache_usage(result: Any):
    usage = getattr(result, "usage", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    cache_read = getattr(prompt_details, "cached_tokens", 0) or getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return cache_read, cache_creation


def _reasoning_tokens(result: Any) -> int:
    usage = getattr(result, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    return getattr(details, "reasoning_tokens", 0) or 0


def _reasoning_content(result: Any) -> str:
    try:
        choice = result.choices[0]
        return getattr(choice.message, "reasoning_content", None) or ""
    except Exception:
        return ""


def _tool_events(result: Any, ts: str):
    return from_openai_chat(result, ts)


def patch_litellm(client: "EvalKitClient") -> None:
    try:
        import litellm
    except ImportError:
        return

    _patch_sync(client, litellm)
    _patch_async(client, litellm)


def _patch_sync(ek: "EvalKitClient", litellm: Any) -> None:
    fn = getattr(litellm, "completion", None)
    if fn is None or getattr(fn, "_evalkit", False):
        return
    original = fn

    def traced(*args: Any, **kwargs: Any) -> Any:
        try:
            ctx_parent = current_span_ctx()
            trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
            _ambient_trace_id.set(trace_id)
            span_id = _new_span_id()
            parent_span_id = ctx_parent.span_id if ctx_parent else None
            start_ms = time.time() * 1000
            ts = _now_iso()
            model = str(kwargs.get("model") or (args[0] if args else "unknown"))
            prompt = _messages_to_prompt(kwargs.get("messages", []))

            request_body = _safe_body(_dump(kwargs))
        except Exception as exc:
            hook_failed("litellm.completion", exc)
            return original(*args, **kwargs)

        if kwargs.get("stream"):
            state = _StreamState(start_ms)

            def on_done(
                completion: str, tin: int, tout: int,
                thinking: str, reasoning_tokens: int, has_redacted_thinking: bool,
                ttft_ms: Optional[int],
                cache_read_tokens: int = 0, cache_creation_tokens: int = 0,
                error: Optional[BaseException] = None,
            ) -> None:
                stream_attrs = {**ek._meta_attrs(), "gen_ai.request.model": model, "gen_ai.streaming": True}
                if request_body:
                    stream_attrs["gen_ai.request.body"] = request_body
                if reasoning_tokens:
                    stream_attrs["gen_ai.usage.reasoning_tokens"] = reasoning_tokens
                if ttft_ms is not None:
                    stream_attrs["gen_ai.server.time_to_first_token_ms"] = ttft_ms
                if tin or tout:
                    stream_attrs["gen_ai.usage.total_tokens"] = tin + tout
                stream_attrs.update(stream_response_attrs(state))
                end_ts = _now_iso()
                ek.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call", operation="litellm.completion",
                    status="ERROR" if error is not None else "OK",
                    status_message=str(error) if error is not None else None, exception=error,
                    service_name=ek._opts.service_name, provider="litellm", model=model,
                    prompt=prompt, completion=completion, tokens_in=tin, tokens_out=tout,
                    cache_read_tokens=cache_read_tokens or None, cache_creation_tokens=cache_creation_tokens or None,
                    latency_ms=int(time.time() * 1000 - start_ms), start_time=ts, end_time=end_ts,
                    created_at=ts, events=stream_tool_events(state, end_ts) or None,
                    attributes=stream_attrs,
                    environment=ek._opts.environment, app_version=ek._opts.app_version, sdk_version=SDK_VERSION,
                ))
            try:
                raw = original(*args, **kwargs)
            except Exception as exc:
                record_stream_open_failure(on_done, state, exc)
                raise
            return trace_sync_stream(raw, on_done, start_ms=start_ms, state=state)

        span_ctx = child_span_ctx(trace_id, span_id)
        token = _current_span.set(span_ctx)
        try:
            result = original(*args, **kwargs)
        except Exception as exc:
            try:
                ek.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call", status="ERROR", status_message=str(exc), exception=exc, operation="litellm.completion",
                    service_name=ek._opts.service_name, provider="litellm", model=model, prompt=prompt,
                    latency_ms=int(time.time() * 1000 - start_ms), start_time=ts, end_time=_now_iso(), created_at=ts,
                    attributes={**ek._meta_attrs(), "gen_ai.request.model": model,
                                **({"gen_ai.request.body": request_body} if request_body else {})},
                    environment=ek._opts.environment, app_version=ek._opts.app_version, sdk_version=SDK_VERSION,
                ))
            except Exception as hook_exc:
                hook_failed("model call", hook_exc)
            raise
        finally:
            _current_span.reset(token)
        try:

            end_ts = _now_iso()
            tin, tout = _usage(result)
            cache_read_tokens, cache_creation_tokens = _cache_usage(result)
            reasoning_tokens = _reasoning_tokens(result)
            reasoning_content = _reasoning_content(result)
            attrs = {**ek._meta_attrs(), "gen_ai.request.model": model, "gen_ai.usage.total_tokens": tin + tout}
            if reasoning_tokens:
                attrs["gen_ai.usage.reasoning_tokens"] = reasoning_tokens
            if reasoning_content:
                attrs["gen_ai.response.thinking"] = reasoning_content
            if request_body:
                attrs["gen_ai.request.body"] = request_body
            response_body = _safe_body(_dump(result))
            if response_body:
                attrs["gen_ai.response.body"] = response_body
            ek.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="OK", operation="litellm.completion",
                service_name=ek._opts.service_name, provider="litellm", model=model,
                prompt=prompt, completion=_completion_text(result), tokens_in=tin, tokens_out=tout,
                cache_read_tokens=cache_read_tokens or None, cache_creation_tokens=cache_creation_tokens or None,
                latency_ms=int(time.time() * 1000 - start_ms), start_time=ts, end_time=end_ts, created_at=ts,
                events=_tool_events(result, end_ts) or None,
                attributes=attrs,
                environment=ek._opts.environment, app_version=ek._opts.app_version, sdk_version=SDK_VERSION,
            ))
        except Exception as exc:
            hook_failed("model call", exc)
        return result

    traced._evalkit = True
    litellm.completion = traced


def _patch_async(ek: "EvalKitClient", litellm: Any) -> None:
    fn = getattr(litellm, "acompletion", None)
    if fn is None or getattr(fn, "_evalkit", False):
        return
    original = fn

    async def traced_async(*args: Any, **kwargs: Any) -> Any:
        try:
            ctx_parent = current_span_ctx()
            trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
            _ambient_trace_id.set(trace_id)
            span_id = _new_span_id()
            parent_span_id = ctx_parent.span_id if ctx_parent else None
            start_ms = time.time() * 1000
            ts = _now_iso()
            model = str(kwargs.get("model") or (args[0] if args else "unknown"))
            prompt = _messages_to_prompt(kwargs.get("messages", []))

            request_body = _safe_body(_dump(kwargs))
        except Exception as exc:
            hook_failed("litellm.acompletion", exc)
            return await original(*args, **kwargs)

        if kwargs.get("stream"):
            state = _StreamState(start_ms)

            def on_done(
                completion: str, tin: int, tout: int,
                thinking: str, reasoning_tokens: int, has_redacted_thinking: bool,
                ttft_ms: Optional[int],
                cache_read_tokens: int = 0, cache_creation_tokens: int = 0,
                error: Optional[BaseException] = None,
            ) -> None:
                stream_attrs = {**ek._meta_attrs(), "gen_ai.request.model": model, "gen_ai.streaming": True}
                if request_body:
                    stream_attrs["gen_ai.request.body"] = request_body
                if reasoning_tokens:
                    stream_attrs["gen_ai.usage.reasoning_tokens"] = reasoning_tokens
                if ttft_ms is not None:
                    stream_attrs["gen_ai.server.time_to_first_token_ms"] = ttft_ms
                if tin or tout:
                    stream_attrs["gen_ai.usage.total_tokens"] = tin + tout
                stream_attrs.update(stream_response_attrs(state))
                end_ts = _now_iso()
                ek.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call", operation="litellm.acompletion",
                    status="ERROR" if error is not None else "OK",
                    status_message=str(error) if error is not None else None, exception=error,
                    service_name=ek._opts.service_name, provider="litellm", model=model,
                    prompt=prompt, completion=completion, tokens_in=tin, tokens_out=tout,
                    cache_read_tokens=cache_read_tokens or None, cache_creation_tokens=cache_creation_tokens or None,
                    latency_ms=int(time.time() * 1000 - start_ms), start_time=ts, end_time=end_ts,
                    created_at=ts, events=stream_tool_events(state, end_ts) or None,
                    attributes=stream_attrs,
                    environment=ek._opts.environment, app_version=ek._opts.app_version, sdk_version=SDK_VERSION,
                ))
            try:
                raw = await original(*args, **kwargs)
            except Exception as exc:
                record_stream_open_failure(on_done, state, exc)
                raise
            return trace_async_stream(raw, on_done, start_ms=start_ms, state=state)

        span_ctx = child_span_ctx(trace_id, span_id)
        token = _current_span.set(span_ctx)
        try:
            result = await original(*args, **kwargs)
        except Exception as exc:
            try:
                ek.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call", status="ERROR", status_message=str(exc), exception=exc, operation="litellm.acompletion",
                    service_name=ek._opts.service_name, provider="litellm", model=model, prompt=prompt,
                    latency_ms=int(time.time() * 1000 - start_ms), start_time=ts, end_time=_now_iso(), created_at=ts,
                    attributes={**ek._meta_attrs(), "gen_ai.request.model": model,
                                **({"gen_ai.request.body": request_body} if request_body else {})},
                    environment=ek._opts.environment, app_version=ek._opts.app_version, sdk_version=SDK_VERSION,
                ))
            except Exception as hook_exc:
                hook_failed("model call", hook_exc)
            raise
        finally:
            _current_span.reset(token)
        try:

            end_ts = _now_iso()
            tin, tout = _usage(result)
            cache_read_tokens, cache_creation_tokens = _cache_usage(result)
            reasoning_tokens = _reasoning_tokens(result)
            reasoning_content = _reasoning_content(result)
            attrs = {**ek._meta_attrs(), "gen_ai.request.model": model, "gen_ai.usage.total_tokens": tin + tout}
            if reasoning_tokens:
                attrs["gen_ai.usage.reasoning_tokens"] = reasoning_tokens
            if reasoning_content:
                attrs["gen_ai.response.thinking"] = reasoning_content
            if request_body:
                attrs["gen_ai.request.body"] = request_body
            response_body = _safe_body(_dump(result))
            if response_body:
                attrs["gen_ai.response.body"] = response_body
            ek.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="OK", operation="litellm.acompletion",
                service_name=ek._opts.service_name, provider="litellm", model=model,
                prompt=prompt, completion=_completion_text(result), tokens_in=tin, tokens_out=tout,
                cache_read_tokens=cache_read_tokens or None, cache_creation_tokens=cache_creation_tokens or None,
                latency_ms=int(time.time() * 1000 - start_ms), start_time=ts, end_time=end_ts, created_at=ts,
                events=_tool_events(result, end_ts) or None,
                attributes=attrs,
                environment=ek._opts.environment, app_version=ek._opts.app_version, sdk_version=SDK_VERSION,
            ))
        except Exception as exc:
            hook_failed("model call", exc)
        return result

    traced_async._evalkit = True
    litellm.acompletion = traced_async
