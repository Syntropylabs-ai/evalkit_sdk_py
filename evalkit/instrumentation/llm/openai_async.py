from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..._version import SDK_VERSION
from ...core.client import EvalKitClient, _new_span_id, _new_trace_id, _now_iso
from ...core.context import SpanContext, _current_span, get_or_create_ambient_trace_id
from ...core.types import SpanEvent, TraceEnvelope
from .._streaming import _AsyncStreamTracer
from .._toolcalls import from_openai_chat
from .openai import _messages_to_text

_PROMPT_MAX = 32768
_COMPLETION_MAX = 32768


def patch_async_openai(
    client: EvalKitClient,
    openai_client: Any,
    parent_ctx: Optional[SpanContext] = None,
) -> None:
    completions = openai_client.chat.completions
    if getattr(completions.create, "_evalkit", False):
        return

    original_create = completions.create

    async def traced_create(*args: Any, **kwargs: Any) -> Any:
        ctx_parent = parent_ctx or _current_span.get()
        trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()

        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")
        prompt_text = _messages_to_text(messages)[:_PROMPT_MAX]

        if kwargs.get("stream"):
            def on_stream_done(completion: str, tokens_in: int, tokens_out: int) -> None:
                end_ts = _now_iso()
                latency = int(time.time() * 1000 - start_ms)
                client.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call", status="OK",
                    operation="chat.completions.create",
                    service_name=client._opts.service_name, provider="openai", model=model,
                    prompt=prompt_text, completion=completion[:_COMPLETION_MAX],
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
                    attributes={**client._meta_attrs(), "gen_ai.request.model": model, "gen_ai.streaming": True},
                    environment=client._opts.environment, app_version=client._opts.app_version,
                    sdk_version=SDK_VERSION,
                ))
            raw = await original_create(*args, **kwargs)
            return _AsyncStreamTracer(raw, on_stream_done)

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
                operation="chat.completions.create",
                service_name=client._opts.service_name, provider="openai", model=model,
                prompt=prompt_text, latency_ms=latency, start_time=ts, end_time=end_ts,
                created_at=ts, attributes={**client._meta_attrs(), "gen_ai.request.model": model},
                environment=client._opts.environment, app_version=client._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
            raise
        finally:
            _current_span.reset(token)

        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)
        choice = result.choices[0] if result.choices else None
        completion = (choice.message.content or "") if choice else ""
        usage = result.usage
        tokens_in = getattr(usage, "prompt_tokens", 0) or 0
        tokens_out = getattr(usage, "completion_tokens", 0) or 0

        events: List[SpanEvent] = from_openai_chat(result, end_ts)

        attrs: Dict[str, Any] = {
            **client._meta_attrs(),
            "gen_ai.request.model": model,
            "gen_ai.response.model": getattr(result, "model", model),
            "gen_ai.response.id": getattr(result, "id", ""),
            "gen_ai.usage.total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }
        if temperature is not None:
            attrs["gen_ai.request.temperature"] = temperature
        if max_tokens is not None:
            attrs["gen_ai.request.max_tokens"] = max_tokens
        if choice and getattr(choice, "finish_reason", None):
            attrs["gen_ai.response.finish_reasons"] = [choice.finish_reason]

        client.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call", status="OK",
            operation="chat.completions.create",
            service_name=client._opts.service_name, provider="openai", model=model,
            prompt=prompt_text, completion=completion[:_COMPLETION_MAX],
            tokens_in=tokens_in, tokens_out=tokens_out,
            latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
            events=events or None, attributes=attrs,
            environment=client._opts.environment, app_version=client._opts.app_version,
            sdk_version=SDK_VERSION,
        ))
        return result

    traced_create._evalkit = True  # type: ignore[attr-defined]
    completions.create = traced_create
