from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..._version import SDK_VERSION
from ...core.client import EvalKitClient, _new_span_id, _now_iso
from ...core.context import SpanContext, _current_span, current_span_ctx, get_or_create_ambient_trace_id
from ...core.types import SpanEvent, TraceEnvelope
from .._streaming import _AsyncStreamTracer, _StreamState, stream_response_attrs, stream_tool_events
from .._toolcalls import from_anthropic_blocks
from ._anthropic_stream import patch_messages_stream
from ._bodies import _dump, _safe_body
from .anthropic import _build_prompt, _extract_thinking

_PROMPT_MAX = 32768
_COMPLETION_MAX = 32768


def patch_async_anthropic(
    client: EvalKitClient,
    anthropic_client: Any,
    parent_ctx: Optional[SpanContext] = None,
) -> None:
    messages_api = anthropic_client.messages
    patch_messages_stream(client, messages_api, "anthropic", parent_ctx)
    if getattr(messages_api.create, "_evalkit", False):
        return

    original_create = messages_api.create

    async def traced_create(*args: Any, **kwargs: Any) -> Any:
        ctx_parent = parent_ctx or current_span_ctx()
        trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()

        model = kwargs.get("model", "unknown")
        system = kwargs.get("system")
        messages = kwargs.get("messages", [])
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")
        prompt_text = _build_prompt(system, messages)[:_PROMPT_MAX]
        request_body = _safe_body(_dump(kwargs))

        if kwargs.get("stream"):
            state = _StreamState(start_ms)

            def on_stream_done(
                completion: str, tokens_in: int, tokens_out: int,
                thinking: str, reasoning_tokens: int, has_redacted_thinking: bool,
                ttft_ms: Optional[int],
                cache_read_tokens: int = 0, cache_creation_tokens: int = 0,
                error: Optional[BaseException] = None,
            ) -> None:
                end_ts = _now_iso()
                latency = int(time.time() * 1000 - start_ms)
                stream_attrs = {**client._meta_attrs(), "gen_ai.request.model": model, "gen_ai.streaming": True}
                if request_body:
                    stream_attrs["gen_ai.request.body"] = request_body
                if thinking:
                    stream_attrs["gen_ai.response.thinking"] = thinking[:_COMPLETION_MAX]
                if has_redacted_thinking:
                    stream_attrs["gen_ai.response.has_redacted_thinking"] = True
                if ttft_ms is not None:
                    stream_attrs["gen_ai.server.time_to_first_token_ms"] = ttft_ms
                if temperature is not None:
                    stream_attrs["gen_ai.request.temperature"] = temperature
                if max_tokens is not None:
                    stream_attrs["gen_ai.request.max_tokens"] = max_tokens
                stream_attrs.update(stream_response_attrs(state))
                client.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call",
                    status="ERROR" if error is not None else "OK",
                    status_message=str(error) if error is not None else None,
                    operation="messages.create",
                    service_name=client._opts.service_name, provider="anthropic", model=model,
                    prompt=prompt_text, completion=completion[:_COMPLETION_MAX],
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    cache_read_tokens=cache_read_tokens or None, cache_creation_tokens=cache_creation_tokens or None,
                    latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
                    events=stream_tool_events(state, end_ts) or None,
                    attributes=stream_attrs,
                    environment=client._opts.environment, app_version=client._opts.app_version,
                    sdk_version=SDK_VERSION,
                ))
            raw = await original_create(*args, **kwargs)
            return _AsyncStreamTracer(
                raw, on_stream_done, is_anthropic=True, start_ms=start_ms, state=state,
            )

        span_ctx = SpanContext(trace_id=trace_id, span_id=span_id)
        token = _current_span.set(span_ctx)
        try:
            result = await original_create(*args, **kwargs)
        except Exception as exc:
            end_ts = _now_iso()
            latency = int(time.time() * 1000 - start_ms)
            client.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="ERROR", status_message=str(exc),
                operation="messages.create",
                service_name=client._opts.service_name, provider="anthropic", model=model,
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

        text_content = ""
        thinking_text, has_redacted_thinking = _extract_thinking(getattr(result, "content", []))
        for block in getattr(result, "content", []):
            if getattr(block, "type", None) == "text":
                text_content += getattr(block, "text", "")
        events: List[SpanEvent] = from_anthropic_blocks(getattr(result, "content", []), end_ts)

        usage = getattr(result, "usage", None)
        tokens_out = getattr(usage, "output_tokens", 0) or 0
        cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
        tokens_in = (getattr(usage, "input_tokens", 0) or 0) + cache_read_tokens + cache_creation_tokens

        attrs: Dict[str, Any] = {
            **client._meta_attrs(),
            "gen_ai.request.model": model,
            "gen_ai.response.model": getattr(result, "model", model),
            "gen_ai.response.id": getattr(result, "id", ""),
        }
        if temperature is not None:
            attrs["gen_ai.request.temperature"] = temperature
        if max_tokens is not None:
            attrs["gen_ai.request.max_tokens"] = max_tokens
        stop_reason = getattr(result, "stop_reason", None)
        if stop_reason:
            attrs["gen_ai.response.finish_reasons"] = [stop_reason]
        if thinking_text:
            attrs["gen_ai.response.thinking"] = thinking_text[:_COMPLETION_MAX]
        if has_redacted_thinking:
            attrs["gen_ai.response.has_redacted_thinking"] = True
        if request_body:
            attrs["gen_ai.request.body"] = request_body
        response_body = _safe_body(_dump(result))
        if response_body:
            attrs["gen_ai.response.body"] = response_body

        client.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call", status="OK",
            operation="messages.create",
            service_name=client._opts.service_name, provider="anthropic", model=model,
            prompt=prompt_text, completion=text_content[:_COMPLETION_MAX],
            tokens_in=tokens_in, tokens_out=tokens_out,
            cache_read_tokens=cache_read_tokens or None, cache_creation_tokens=cache_creation_tokens or None,
            latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
            events=events or None, attributes=attrs,
            environment=client._opts.environment, app_version=client._opts.app_version,
            sdk_version=SDK_VERSION,
        ))
        return result

    traced_create._evalkit = True
    messages_api.create = traced_create
