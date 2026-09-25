from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from ..._version import SDK_VERSION
from ...core.client import EvalKitClient, _new_span_id, _now_iso
from ...core.context import SpanContext, _current_span, child_span_ctx, current_span_ctx, get_or_create_ambient_trace_id
from ...core.types import SpanEvent, TraceEnvelope
from .._streaming import _AsyncStreamTracer, _StreamState, _SyncStreamTracer
from .._toolcalls import from_openai_chat, from_stream_tool_calls
from ._bodies import _dump, _safe_body

_PROMPT_MAX = 32768
_COMPLETION_MAX = 32768


def _extract_mistral_prompt(messages: Any) -> str:
    parts: List[str] = []
    for m in (messages or []):
        role = m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "")
        content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        if isinstance(content, list):
            content = " ".join(
                (p.get("text", "") if isinstance(p, dict) else getattr(p, "text", ""))
                for p in content
            )
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _extract_mistral_completion(result: Any) -> Tuple[str, str]:
    state = _StreamState()
    try:
        for choice in (getattr(result, "choices", None) or []):
            msg = getattr(choice, "message", None)
            if msg is not None:
                state._add_mistral_content(getattr(msg, "content", None))
    except Exception:
        pass
    return "".join(state.chunks), "".join(state.thinking_chunks)


def _extract_mistral_usage(result: Any):
    usage = getattr(result, "usage", None)
    if usage is None:
        return 0, 0
    tokens_in = getattr(usage, "prompt_tokens", 0) or 0
    tokens_out = getattr(usage, "completion_tokens", 0) or 0
    return tokens_in, tokens_out


def _extract_mistral_tool_events(result: Any, ts: str) -> List[SpanEvent]:
    return from_openai_chat(result, ts)


def patch_mistral(
    client: EvalKitClient,
    mistral_client: Any,
    parent_ctx: Optional[SpanContext] = None,
) -> None:
    chat = getattr(mistral_client, "chat", None)
    if chat is None:
        return
    _patch_mistral_complete(client, chat, parent_ctx)
    _patch_mistral_stream(client, chat, parent_ctx, method="stream", is_async=False)


def patch_async_mistral(
    client: EvalKitClient,
    mistral_client: Any,
    parent_ctx: Optional[SpanContext] = None,
) -> None:
    chat = getattr(mistral_client, "chat", None)
    if chat is None:
        return
    _patch_mistral_complete_async(client, chat, parent_ctx)
    _patch_mistral_stream(client, chat, parent_ctx, method="stream_async", is_async=True)


def patch_mistral_chat(
    client: EvalKitClient,
    chat: Any,
    parent_ctx: Optional[SpanContext] = None,
) -> None:
    _patch_mistral_complete(client, chat, parent_ctx)
    _patch_mistral_complete_async(client, chat, parent_ctx)
    _patch_mistral_stream(client, chat, parent_ctx, method="stream", is_async=False)
    _patch_mistral_stream(client, chat, parent_ctx, method="stream_async", is_async=True)


def _patch_mistral_complete(
    client: EvalKitClient, chat: Any, parent_ctx: Optional[SpanContext],
) -> None:
    complete = getattr(chat, "complete", None)
    if complete is None or getattr(complete, "_evalkit", False):
        return

    original_complete = complete

    def traced_complete(**kwargs: Any) -> Any:
        ctx_parent = parent_ctx or current_span_ctx()
        trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()

        model = kwargs.get("model", "mistral-small")
        messages = kwargs.get("messages", [])
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")
        prompt_text = _extract_mistral_prompt(messages)[:_PROMPT_MAX]
        request_body = _safe_body(_dump(kwargs))

        span_ctx = child_span_ctx(trace_id, span_id)
        token = _current_span.set(span_ctx)
        try:
            result = original_complete(**kwargs)
        except Exception as exc:
            end_ts = _now_iso()
            latency = int(time.time() * 1000 - start_ms)
            client.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="ERROR", status_message=str(exc),
                operation="chat.complete",
                service_name=client._opts.service_name, provider="mistral", model=model,
                prompt=prompt_text, latency_ms=latency, start_time=ts, end_time=end_ts,
                created_at=ts,
                attributes={**client._meta_attrs(), "gen_ai.request.model": model,
                            **({"gen_ai.request.body": request_body} if request_body else {})},
                environment=client._opts.environment, app_version=client._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
            raise
        finally:
            _current_span.reset(token)

        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)
        completion, thinking = _extract_mistral_completion(result)
        tokens_in, tokens_out = _extract_mistral_usage(result)
        events = _extract_mistral_tool_events(result, end_ts)

        attrs: Dict[str, Any] = {
            **client._meta_attrs(),
            "gen_ai.request.model": model,
            "gen_ai.response.model": getattr(result, "model", model),
            "gen_ai.response.id": getattr(result, "id", ""),
            "gen_ai.usage.total_tokens": tokens_in + tokens_out,
        }
        if temperature is not None:
            attrs["gen_ai.request.temperature"] = temperature
        if max_tokens is not None:
            attrs["gen_ai.request.max_tokens"] = max_tokens
        if thinking:
            attrs["gen_ai.response.thinking"] = thinking
        choices = getattr(result, "choices", None) or []
        if choices:
            finish = getattr(choices[0], "finish_reason", None)
            if finish:
                attrs["gen_ai.response.finish_reasons"] = [str(finish)]
        if request_body:
            attrs["gen_ai.request.body"] = request_body
        response_body = _safe_body(_dump(result))
        if response_body:
            attrs["gen_ai.response.body"] = response_body

        client.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call", status="OK",
            operation="chat.complete",
            service_name=client._opts.service_name, provider="mistral", model=model,
            prompt=prompt_text, completion=completion[:_COMPLETION_MAX],
            tokens_in=tokens_in, tokens_out=tokens_out,
            latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
            events=events or None, attributes=attrs,
            environment=client._opts.environment, app_version=client._opts.app_version,
            sdk_version=SDK_VERSION,
        ))
        return result

    traced_complete._evalkit = True
    chat.complete = traced_complete


def _patch_mistral_complete_async(
    client: EvalKitClient, chat: Any, parent_ctx: Optional[SpanContext],
) -> None:
    complete_async = getattr(chat, "complete_async", None)
    if complete_async is None or getattr(complete_async, "_evalkit", False):
        return

    original_complete_async = complete_async

    async def traced_complete_async(**kwargs: Any) -> Any:
        ctx_parent = parent_ctx or current_span_ctx()
        trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()

        model = kwargs.get("model", "mistral-small")
        messages = kwargs.get("messages", [])
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")
        prompt_text = _extract_mistral_prompt(messages)[:_PROMPT_MAX]
        request_body = _safe_body(_dump(kwargs))

        span_ctx = child_span_ctx(trace_id, span_id)
        token = _current_span.set(span_ctx)
        try:
            result = await original_complete_async(**kwargs)
        except Exception as exc:
            end_ts = _now_iso()
            latency = int(time.time() * 1000 - start_ms)
            client.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="ERROR", status_message=str(exc),
                operation="chat.complete_async",
                service_name=client._opts.service_name, provider="mistral", model=model,
                prompt=prompt_text, latency_ms=latency, start_time=ts, end_time=end_ts,
                created_at=ts,
                attributes={**client._meta_attrs(), "gen_ai.request.model": model,
                            **({"gen_ai.request.body": request_body} if request_body else {})},
                environment=client._opts.environment, app_version=client._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
            raise
        finally:
            _current_span.reset(token)

        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)
        completion, thinking = _extract_mistral_completion(result)
        tokens_in, tokens_out = _extract_mistral_usage(result)
        events = _extract_mistral_tool_events(result, end_ts)

        attrs: Dict[str, Any] = {
            **client._meta_attrs(),
            "gen_ai.request.model": model,
            "gen_ai.response.model": getattr(result, "model", model),
            "gen_ai.response.id": getattr(result, "id", ""),
            "gen_ai.usage.total_tokens": tokens_in + tokens_out,
        }
        if temperature is not None:
            attrs["gen_ai.request.temperature"] = temperature
        if max_tokens is not None:
            attrs["gen_ai.request.max_tokens"] = max_tokens
        if thinking:
            attrs["gen_ai.response.thinking"] = thinking
        choices = getattr(result, "choices", None) or []
        if choices:
            finish = getattr(choices[0], "finish_reason", None)
            if finish:
                attrs["gen_ai.response.finish_reasons"] = [str(finish)]
        if request_body:
            attrs["gen_ai.request.body"] = request_body
        response_body = _safe_body(_dump(result))
        if response_body:
            attrs["gen_ai.response.body"] = response_body

        client.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call", status="OK",
            operation="chat.complete_async",
            service_name=client._opts.service_name, provider="mistral", model=model,
            prompt=prompt_text, completion=completion[:_COMPLETION_MAX],
            tokens_in=tokens_in, tokens_out=tokens_out,
            latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
            events=events or None, attributes=attrs,
            environment=client._opts.environment, app_version=client._opts.app_version,
            sdk_version=SDK_VERSION,
        ))
        return result

    traced_complete_async._evalkit = True
    chat.complete_async = traced_complete_async


def _mistral_stream_recorder(
    client: EvalKitClient, *, operation: str, kwargs: Dict[str, Any],
    trace_id: str, span_id: str, parent_span_id: Optional[str],
    start_ms: float, ts: str, state: _StreamState,
):
    model = kwargs.get("model", "mistral-small")
    temperature = kwargs.get("temperature")
    max_tokens = kwargs.get("max_tokens")
    prompt_text = _extract_mistral_prompt(kwargs.get("messages", []))[:_PROMPT_MAX]
    request_body = _safe_body(_dump(kwargs))

    def on_stream_done(
        completion: str, tokens_in: int, tokens_out: int, thinking: str,
        reasoning_tokens: int, has_redacted_thinking: bool, ttft_ms: Optional[int],
        cache_read_tokens: int, cache_creation_tokens: int,
        error: Optional[BaseException],
    ) -> None:
        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)
        attrs: Dict[str, Any] = {
            **client._meta_attrs(),
            "gen_ai.request.model": model,
            "gen_ai.response.model": state.response_model or model,
            "gen_ai.response.id": state.response_id,
            "gen_ai.usage.total_tokens": tokens_in + tokens_out,
            "gen_ai.streaming": True,
        }
        if temperature is not None:
            attrs["gen_ai.request.temperature"] = temperature
        if max_tokens is not None:
            attrs["gen_ai.request.max_tokens"] = max_tokens
        if thinking:
            attrs["gen_ai.response.thinking"] = thinking
        if state.finish_reasons:
            attrs["gen_ai.response.finish_reasons"] = list(state.finish_reasons)
        if ttft_ms is not None:
            attrs["gen_ai.server.time_to_first_token_ms"] = ttft_ms
        if request_body:
            attrs["gen_ai.request.body"] = request_body
        events: List[SpanEvent] = from_stream_tool_calls(state.tool_calls, ts)
        client.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call",
            status="ERROR" if error is not None else "OK",
            status_message=str(error) if error is not None else None,
            operation=operation,
            service_name=client._opts.service_name, provider="mistral", model=model,
            prompt=prompt_text, completion=completion[:_COMPLETION_MAX],
            tokens_in=tokens_in, tokens_out=tokens_out,
            latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
            events=events or None, attributes=attrs,
            environment=client._opts.environment, app_version=client._opts.app_version,
            sdk_version=SDK_VERSION,
        ))

    return on_stream_done


def _patch_mistral_stream(
    client: EvalKitClient, chat: Any, parent_ctx: Optional[SpanContext],
    *, method: str, is_async: bool,
) -> None:
    original_stream = getattr(chat, method, None)
    if original_stream is None or getattr(original_stream, "_evalkit", False):
        return

    operation = f"chat.{method}"

    def prepare(kwargs: Dict[str, Any]):
        ctx_parent = parent_ctx or current_span_ctx()
        trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        state = _StreamState(start_ms)
        on_stream_done = _mistral_stream_recorder(
            client, operation=operation, kwargs=kwargs,
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            start_ms=start_ms, ts=_now_iso(), state=state,
        )
        return child_span_ctx(trace_id, span_id), start_ms, state, on_stream_done

    if is_async:
        async def traced_stream_async(**kwargs: Any) -> Any:
            span_ctx, start_ms, state, on_stream_done = prepare(kwargs)
            token = _current_span.set(span_ctx)
            try:
                stream = await original_stream(**kwargs)
            except Exception as exc:
                on_stream_done("", 0, 0, "", 0, False, None, 0, 0, exc)
                raise
            finally:
                _current_span.reset(token)
            return _AsyncStreamTracer(
                stream, on_stream_done, start_ms=start_ms,
                process=_StreamState.process_mistral_chunk, state=state,
            )

        traced_stream_async._evalkit = True
        setattr(chat, method, traced_stream_async)
        return

    def traced_stream(**kwargs: Any) -> Any:
        span_ctx, start_ms, state, on_stream_done = prepare(kwargs)
        token = _current_span.set(span_ctx)
        try:
            stream = original_stream(**kwargs)
        except Exception as exc:
            on_stream_done("", 0, 0, "", 0, False, None, 0, 0, exc)
            raise
        finally:
            _current_span.reset(token)
        return _SyncStreamTracer(
            stream, on_stream_done, start_ms=start_ms,
            process=_StreamState.process_mistral_chunk, state=state,
        )

    traced_stream._evalkit = True
    setattr(chat, method, traced_stream)
