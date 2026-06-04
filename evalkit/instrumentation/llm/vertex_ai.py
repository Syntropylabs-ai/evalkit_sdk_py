from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..._version import SDK_VERSION
from ...core.client import EvalKitClient, _new_span_id, _new_trace_id, _now_iso
from ...core.context import SpanContext, _current_span
from ...core.types import SpanEvent, TraceEnvelope


# ── Anthropic on Vertex AI ────────────────────────────────────────────────────

def patch_anthropic_vertex(
    client: EvalKitClient,
    anthropic_vertex_client: Any,
    parent_ctx: Optional[SpanContext] = None,
) -> None:
    """
    Patch AnthropicVertex.messages.create (same API surface as direct Anthropic client).
    Import from anthropic-sdk-python package with Vertex AI backend.
    """
    messages_api = anthropic_vertex_client.messages
    if getattr(messages_api.create, "_evalkit", False):
        return

    original_create = messages_api.create

    def traced_create(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return original_create(*args, **kwargs)

        ctx_parent = parent_ctx or _current_span.get()
        trace_id = ctx_parent.trace_id if ctx_parent else _new_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()

        model = kwargs.get("model", "unknown")
        system = kwargs.get("system")
        messages = kwargs.get("messages", [])
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")

        parts: List[str] = []
        if system:
            parts.append(f"system: {system}")
        parts.extend(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)
        prompt_text = "\n".join(parts)

        span_ctx = SpanContext(trace_id=trace_id, span_id=span_id)
        token = _current_span.set(span_ctx)
        try:
            result = original_create(*args, **kwargs)
        except Exception as exc:
            end_ts = _now_iso()
            latency = int(time.time() * 1000 - start_ms)
            client.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="ERROR", status_message=str(exc),
                operation="messages.create",
                service_name=client._opts.service_name, provider="anthropic-vertex", model=model,
                prompt=prompt_text, latency_ms=latency, start_time=ts, end_time=end_ts,
                created_at=ts,
                attributes={**client._meta_attrs(), "gen_ai.request.model": model},
                environment=client._opts.environment, app_version=client._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
            raise
        finally:
            _current_span.reset(token)

        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)

        text_content = ""
        events: List[SpanEvent] = []
        for block in getattr(result, "content", []):
            btype = getattr(block, "type", None)
            if btype == "text":
                text_content += getattr(block, "text", "")
            elif btype == "tool_use":
                events.append(SpanEvent(
                    name="gen_ai.tool.call", timestamp=end_ts,
                    attributes={
                        "gen_ai.tool.call.id": getattr(block, "id", ""),
                        "gen_ai.tool.name": getattr(block, "name", ""),
                        "gen_ai.tool.call.arguments": str(getattr(block, "input", "")),
                    },
                ))

        usage = getattr(result, "usage", None)
        tokens_in = getattr(usage, "input_tokens", 0) or 0
        tokens_out = getattr(usage, "output_tokens", 0) or 0

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

        client.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call", status="OK",
            operation="messages.create",
            service_name=client._opts.service_name, provider="anthropic-vertex", model=model,
            prompt=prompt_text, completion=text_content,
            tokens_in=tokens_in, tokens_out=tokens_out,
            latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
            events=events or None, attributes=attrs,
            environment=client._opts.environment, app_version=client._opts.app_version,
            sdk_version=SDK_VERSION,
        ))
        return result

    traced_create._evalkit = True  # type: ignore[attr-defined]
    messages_api.create = traced_create


# ── Google Cloud Vertex AI GenerativeModel ────────────────────────────────────

def patch_vertex_generative_model(
    client: EvalKitClient,
    model: Any,
) -> None:
    """
    Patch a vertexai.generative_models.GenerativeModel instance to trace generate_content calls.
    """
    if getattr(model.generate_content, "_evalkit", False):
        return

    original = model.generate_content

    def traced_generate_content(contents: Any, **kwargs: Any) -> Any:
        ctx_parent = _current_span.get()
        trace_id = ctx_parent.trace_id if ctx_parent else _new_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()

        model_name = getattr(model, "_model_name", getattr(model, "model_name", "unknown"))
        generation_config = kwargs.get("generation_config", {})
        temperature = None
        max_tokens = None
        if hasattr(generation_config, "temperature"):
            temperature = generation_config.temperature
            max_tokens = getattr(generation_config, "max_output_tokens", None)
        elif isinstance(generation_config, dict):
            temperature = generation_config.get("temperature")
            max_tokens = generation_config.get("max_output_tokens")

        prompt_text = _vertex_contents_to_text(contents)

        span_ctx = SpanContext(trace_id=trace_id, span_id=span_id)
        token = _current_span.set(span_ctx)
        try:
            result = original(contents, **kwargs)
        except Exception as exc:
            end_ts = _now_iso()
            latency = int(time.time() * 1000 - start_ms)
            client.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="ERROR", status_message=str(exc),
                operation="generate_content",
                service_name=client._opts.service_name, provider="vertex-ai", model=model_name,
                prompt=prompt_text, latency_ms=latency, start_time=ts, end_time=end_ts,
                created_at=ts,
                attributes={**client._meta_attrs(), "gen_ai.request.model": model_name},
                environment=client._opts.environment, app_version=client._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
            raise
        finally:
            _current_span.reset(token)

        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)

        completion = ""
        try:
            completion = result.text or ""
        except Exception:
            pass

        usage_metadata = getattr(result, "usage_metadata", None)
        tokens_in = getattr(usage_metadata, "prompt_token_count", 0) or 0
        tokens_out = getattr(usage_metadata, "candidates_token_count", 0) or 0

        attrs: Dict[str, Any] = {
            **client._meta_attrs(),
            "gen_ai.request.model": model_name,
            "gen_ai.response.model": model_name,
            "gen_ai.usage.total_tokens": tokens_in + tokens_out,
        }
        try:
            candidates = getattr(result, "candidates", None) or []
            if candidates:
                finish_reason = getattr(candidates[0], "finish_reason", None)
                if finish_reason is not None:
                    attrs["gen_ai.response.finish_reasons"] = [str(finish_reason)]
        except Exception:
            pass
        if temperature is not None:
            attrs["gen_ai.request.temperature"] = temperature
        if max_tokens is not None:
            attrs["gen_ai.request.max_tokens"] = max_tokens

        client.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call", status="OK",
            operation="generate_content",
            service_name=client._opts.service_name, provider="vertex-ai", model=model_name,
            prompt=prompt_text, completion=completion,
            tokens_in=tokens_in, tokens_out=tokens_out,
            latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
            attributes=attrs,
            environment=client._opts.environment, app_version=client._opts.app_version,
            sdk_version=SDK_VERSION,
        ))
        return result

    traced_generate_content._evalkit = True  # type: ignore[attr-defined]
    model.generate_content = traced_generate_content


def _vertex_contents_to_text(contents: Any) -> str:
    if isinstance(contents, str):
        return contents
    if isinstance(contents, list):
        parts = []
        for item in contents:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(contents)
