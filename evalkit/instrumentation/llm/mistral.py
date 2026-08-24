from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..._version import SDK_VERSION
from ...core.client import EvalKitClient, _new_span_id, _new_trace_id, _now_iso
from ...core.context import SpanContext, _current_span, current_span_ctx, get_or_create_ambient_trace_id
from ...core.types import SpanEvent, TraceEnvelope
from .._toolcalls import from_openai_chat
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


def _extract_mistral_completion(result: Any) -> str:
    try:
        choices = getattr(result, "choices", None) or []
        if choices:
            msg = getattr(choices[0], "message", None)
            if msg:
                return getattr(msg, "content", "") or ""
    except Exception:
        pass
    return ""


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

        span_ctx = SpanContext(trace_id=trace_id, span_id=span_id)
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
        completion = _extract_mistral_completion(result)
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


def patch_async_mistral(
    client: EvalKitClient,
    mistral_client: Any,
    parent_ctx: Optional[SpanContext] = None,
) -> None:
    chat = getattr(mistral_client, "chat", None)
    if chat is None:
        return
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

        span_ctx = SpanContext(trace_id=trace_id, span_id=span_id)
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
        completion = _extract_mistral_completion(result)
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
