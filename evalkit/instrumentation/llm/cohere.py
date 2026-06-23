from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..._version import SDK_VERSION
from ...core.client import EvalKitClient, _new_span_id, _now_iso
from ...core.context import SpanContext, _current_span, current_span_ctx, get_or_create_ambient_trace_id
from ...core.types import TraceEnvelope
from ._bodies import _dump, _safe_body


def patch_cohere(client: EvalKitClient, cohere_client: Any) -> None:
    """
    Patch CohereClient.chat() to trace every call.
    Supports cohere-ai v2 SDK (CohereClient.chat).
    """
    if getattr(cohere_client.chat, "_evalkit", False):
        return

    original_chat = cohere_client.chat

    def traced_chat(**kwargs: Any) -> Any:
        ctx_parent = current_span_ctx()
        trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()

        model = kwargs.get("model", "command-r")
        messages = kwargs.get("messages", [])
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")

        prompt_parts: List[str] = []
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role", "")
                content = m.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") for p in content if isinstance(p, dict)
                    )
                prompt_parts.append(f"{role}: {content}")
            else:
                role = getattr(m, "role", "")
                content = getattr(m, "content", "")
                prompt_parts.append(f"{role}: {content}")
        prompt_text = "\n".join(prompt_parts)
        request_body = _safe_body(_dump(kwargs))

        span_ctx = SpanContext(trace_id=trace_id, span_id=span_id)
        token = _current_span.set(span_ctx)
        try:
            result = original_chat(**kwargs)
        except Exception as exc:
            end_ts = _now_iso()
            latency = int(time.time() * 1000 - start_ms)
            client.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="ERROR", status_message=str(exc),
                operation="chat",
                service_name=client._opts.service_name, provider="cohere", model=model,
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

        completion = ""
        message = getattr(result, "message", None)
        if message:
            content_list = getattr(message, "content", []) or []
            for block in content_list:
                if getattr(block, "type", None) == "text":
                    completion += getattr(block, "text", "")
        if not completion:
            text = getattr(result, "text", None)
            if text:
                completion = text

        usage = getattr(result, "usage", None)
        billing_tokens = getattr(usage, "billed_units", None) if usage else None
        tokens_in = getattr(billing_tokens, "input_tokens", 0) or 0 if billing_tokens else 0
        tokens_out = getattr(billing_tokens, "output_tokens", 0) or 0 if billing_tokens else 0

        finish_reason = getattr(result, "finish_reason", None)
        response_id = getattr(result, "id", "")

        attrs: Dict[str, Any] = {
            **client._meta_attrs(),
            "gen_ai.request.model": model,
            "gen_ai.response.model": model,
            "gen_ai.response.id": response_id,
            "gen_ai.usage.total_tokens": tokens_in + tokens_out,
        }
        if temperature is not None:
            attrs["gen_ai.request.temperature"] = temperature
        if max_tokens is not None:
            attrs["gen_ai.request.max_tokens"] = max_tokens
        if finish_reason:
            attrs["gen_ai.response.finish_reasons"] = [str(finish_reason)]
        if request_body:
            attrs["gen_ai.request.body"] = request_body
        response_body = _safe_body(_dump(result))
        if response_body:
            attrs["gen_ai.response.body"] = response_body

        client.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call", status="OK",
            operation="chat",
            service_name=client._opts.service_name, provider="cohere", model=model,
            prompt=prompt_text, completion=completion,
            tokens_in=tokens_in, tokens_out=tokens_out,
            latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
            attributes=attrs,
            environment=client._opts.environment, app_version=client._opts.app_version,
            sdk_version=SDK_VERSION,
        ))
        return result

    traced_chat._evalkit = True  # type: ignore[attr-defined]
    cohere_client.chat = traced_chat
