from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from ..._version import SDK_VERSION
from ...core.client import EvalKitClient, _new_span_id, _new_trace_id, _now_iso
from ...core.context import SpanContext, _current_span, current_span_ctx
from ...core.types import TraceEnvelope


def patch_bedrock(client: EvalKitClient, bedrock_runtime_client: Any) -> None:
    """
    Patch BedrockRuntimeClient to trace ConverseCommand calls.
    Intercepts client.converse() which is the unified multi-model Bedrock API.
    """
    if getattr(bedrock_runtime_client, "_evalkit", False):
        return

    original_converse = bedrock_runtime_client.converse

    def traced_converse(**kwargs: Any) -> Any:
        ctx_parent = current_span_ctx()
        trace_id = ctx_parent.trace_id if ctx_parent else _new_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()

        model_id: str = kwargs.get("modelId", "unknown")
        messages = kwargs.get("messages", [])
        system_list = kwargs.get("system", [])
        inference_config = kwargs.get("inferenceConfig", {})
        temperature = inference_config.get("temperature")
        max_tokens = inference_config.get("maxTokens")

        prompt_parts = []
        for s in system_list:
            text = s.get("text", "")
            if text:
                prompt_parts.append(f"system: {text}")
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", [])
            text = " ".join(
                c.get("text", "") for c in content if isinstance(c, dict) and "text" in c
            )
            prompt_parts.append(f"{role}: {text}")
        prompt_text = "\n".join(prompt_parts)

        span_ctx = SpanContext(trace_id=trace_id, span_id=span_id)
        token = _current_span.set(span_ctx)
        try:
            result = original_converse(**kwargs)
        except Exception as exc:
            end_ts = _now_iso()
            latency = int(time.time() * 1000 - start_ms)
            client.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="ERROR", status_message=str(exc),
                operation="converse",
                service_name=client._opts.service_name, provider="bedrock", model=model_id,
                prompt=prompt_text, latency_ms=latency, start_time=ts, end_time=end_ts,
                created_at=ts,
                attributes={**client._meta_attrs(), "gen_ai.request.model": model_id},
                environment=client._opts.environment, app_version=client._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
            raise
        finally:
            _current_span.reset(token)

        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)

        output = result.get("output", {})
        output_msg = output.get("message", {})
        output_content = output_msg.get("content", [])
        completion = " ".join(
            c.get("text", "") for c in output_content if isinstance(c, dict) and "text" in c
        )

        usage = result.get("usage", {})
        tokens_in = usage.get("inputTokens", 0) or 0
        tokens_out = usage.get("outputTokens", 0) or 0
        stop_reason = result.get("stopReason", "")

        attrs: Dict[str, Any] = {
            **client._meta_attrs(),
            "gen_ai.request.model": model_id,
            "gen_ai.response.model": model_id,
            "gen_ai.usage.total_tokens": tokens_in + tokens_out,
        }
        if temperature is not None:
            attrs["gen_ai.request.temperature"] = temperature
        if max_tokens is not None:
            attrs["gen_ai.request.max_tokens"] = max_tokens
        if stop_reason:
            attrs["gen_ai.response.finish_reasons"] = [stop_reason]

        client.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call", status="OK",
            operation="converse",
            service_name=client._opts.service_name, provider="bedrock", model=model_id,
            prompt=prompt_text, completion=completion,
            tokens_in=tokens_in, tokens_out=tokens_out,
            latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
            attributes=attrs,
            environment=client._opts.environment, app_version=client._opts.app_version,
            sdk_version=SDK_VERSION,
        ))
        return result

    traced_converse._evalkit = True  # type: ignore[attr-defined]
    bedrock_runtime_client.converse = traced_converse
    bedrock_runtime_client._evalkit = True
