from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..._version import SDK_VERSION
from ...core.client import EvalKitClient, _new_span_id, _new_trace_id, _now_iso
from ...core.context import SpanContext, _current_span, current_span_ctx, get_or_create_ambient_trace_id
from ...core.types import SpanEvent, TraceEnvelope
from .._streaming import _SyncStreamTracer
from .._toolcalls import from_openai_chat

_PROMPT_MAX = 32768
_COMPLETION_MAX = 32768


def patch_openai(
    client: EvalKitClient,
    openai_client: Any,
    parent_ctx: Optional[SpanContext] = None,
) -> None:
    completions = openai_client.chat.completions
    if getattr(completions.create, "_evalkit", False):
        return

    original_create = completions.create

    def traced_create(*args: Any, **kwargs: Any) -> Any:
        ctx_parent = parent_ctx or current_span_ctx()
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
            return _SyncStreamTracer(original_create(*args, **kwargs), on_stream_done)

        span_ctx = SpanContext(trace_id=trace_id, span_id=span_id)
        token = _current_span.set(span_ctx)
        try:
            result = original_create(*args, **kwargs)
        except Exception as e:
            end_iso = _now_iso()
            latency = int(time.time() * 1000 - start_ms)
            client.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="ERROR", status_message=str(e),
                operation="chat.completions.create",
                service_name=client._opts.service_name, provider="openai", model=model,
                prompt=prompt_text, latency_ms=latency, start_time=ts,
                end_time=end_iso, created_at=ts,
                attributes={**client._meta_attrs(), "gen_ai.request.model": model},
                environment=client._opts.environment, app_version=client._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
            raise
        finally:
            _current_span.reset(token)

        end_iso = _now_iso()
        latency = int(time.time() * 1000 - start_ms)
        choice = result.choices[0] if result.choices else None
        completion = (choice.message.content or "") if choice else ""
        usage = result.usage
        tokens_in = getattr(usage, "prompt_tokens", 0) or 0
        tokens_out = getattr(usage, "completion_tokens", 0) or 0

        events: List[SpanEvent] = from_openai_chat(result, end_iso)

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
            latency_ms=latency, start_time=ts, end_time=end_iso, created_at=ts,
            events=events or None, attributes=attrs,
            environment=client._opts.environment, app_version=client._opts.app_version,
            sdk_version=SDK_VERSION,
        ))
        return result

    traced_create._evalkit = True  # type: ignore[attr-defined]
    completions.create = traced_create


def _messages_to_text(messages: List[Dict[str, Any]]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "")
        content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        if isinstance(content, list):
            content = " ".join(
                (p.get("text", "") if isinstance(p, dict) else getattr(p, "text", ""))
                for p in content
            )
        parts.append(f"{role}: {content}")
    return "\n".join(parts)
