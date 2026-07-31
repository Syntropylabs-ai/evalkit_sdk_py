from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ..._version import SDK_VERSION
from ...core.client import EvalKitClient, _new_span_id, _now_iso
from ...core.context import SpanContext, _current_span, current_span_ctx, get_or_create_ambient_trace_id
from ...core.types import TraceEnvelope
from ._bodies import _dump, _safe_body


def patch_google_ai(client: EvalKitClient, model: Any) -> None:
    """
    Patch a google.generativeai.GenerativeModel instance to trace generate_content calls.
    Works with @google/generative-ai Python SDK.
    """
    if getattr(model.generate_content, "_evalkit", False):
        return

    original = model.generate_content

    def traced_generate_content(contents: Any, **kwargs: Any) -> Any:
        ctx_parent = current_span_ctx()
        trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()

        model_name = getattr(model, "model_name", "unknown")
        generation_config = kwargs.get("generation_config", {})
        temperature: Optional[float] = None
        max_tokens: Optional[int] = None
        if hasattr(generation_config, "temperature"):
            temperature = generation_config.temperature
            max_tokens = getattr(generation_config, "max_output_tokens", None)
        elif isinstance(generation_config, dict):
            temperature = generation_config.get("temperature")
            max_tokens = generation_config.get("max_output_tokens")

        prompt_text = _contents_to_text(contents)
        request_body = _safe_body({"contents": _dump(contents), **({"config": _dump(kwargs)} if kwargs else {})})

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
                service_name=client._opts.service_name, provider="google-ai", model=model_name,
                prompt=prompt_text, latency_ms=latency, start_time=ts, end_time=end_ts,
                created_at=ts,
                attributes={**client._meta_attrs(), "gen_ai.request.model": model_name,
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
        try:
            completion = result.text or ""
        except Exception:
            pass

        usage_metadata = getattr(result, "usage_metadata", None)
        tokens_in = getattr(usage_metadata, "prompt_token_count", 0) or 0
        tokens_out = getattr(usage_metadata, "candidates_token_count", 0) or 0
        # Subset of tokens_in Gemini served from its (much cheaper) context
        # cache — already included in tokens_in, never additive. Surfaced
        # separately so downstream cost math isn't forced to price an entire
        # long, mostly-cached prompt at the full fresh-input-token rate.
        cached_tokens = getattr(usage_metadata, "cached_content_token_count", 0) or 0

        safety_ratings = []
        try:
            for candidate in (result.candidates or []):
                for sr in (getattr(candidate, "safety_ratings", []) or []):
                    safety_ratings.append({
                        "category": str(getattr(sr, "category", "")),
                        "probability": str(getattr(sr, "probability", "")),
                    })
        except Exception:
            pass

        attrs: Dict[str, Any] = {
            **client._meta_attrs(),
            "gen_ai.request.model": model_name,
            "gen_ai.response.model": model_name,
            "gen_ai.usage.total_tokens": tokens_in + tokens_out,
        }
        if cached_tokens:
            attrs["gen_ai.usage.cached_tokens"] = cached_tokens
        if temperature is not None:
            attrs["gen_ai.request.temperature"] = temperature
        if max_tokens is not None:
            attrs["gen_ai.request.max_tokens"] = max_tokens
        if safety_ratings:
            import json
            attrs["gen_ai.response.safety_ratings"] = json.dumps(safety_ratings)
        if request_body:
            attrs["gen_ai.request.body"] = request_body
        response_body = _safe_body(_dump(result))
        if response_body:
            attrs["gen_ai.response.body"] = response_body

        client.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call", status="OK",
            operation="generate_content",
            service_name=client._opts.service_name, provider="google-ai", model=model_name,
            prompt=prompt_text, completion=completion,
            tokens_in=tokens_in, tokens_out=tokens_out, cache_read_tokens=cached_tokens or None,
            latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
            attributes=attrs,
            environment=client._opts.environment, app_version=client._opts.app_version,
            sdk_version=SDK_VERSION,
        ))
        return result

    traced_generate_content._evalkit = True  # type: ignore[attr-defined]
    model.generate_content = traced_generate_content


def _contents_to_text(contents: Any) -> str:
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
                else:
                    parts_inner = getattr(item, "parts", [])
                    for p in (parts_inner or []):
                        t = getattr(p, "text", None)
                        if t:
                            parts.append(str(t))
        return "\n".join(parts)
    return str(contents)
