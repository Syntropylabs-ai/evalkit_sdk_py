from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..._version import SDK_VERSION
from ...core.client import EvalKitClient, _new_span_id, _now_iso
from ...core.context import SpanContext, _current_span, current_span_ctx, get_or_create_ambient_trace_id
from ...core.types import SpanEvent, TraceEnvelope
from .._google_usage import google_usage_attrs
from .._streaming import _StreamState, _SyncStreamTracer
from .._toolcalls import from_stream_tool_calls
from ._bodies import _dump, _safe_body


def patch_google_ai(client: EvalKitClient, model: Any) -> None:
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
        streaming = bool(kwargs.get("stream"))

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

        def base_attrs() -> Dict[str, Any]:
            attrs: Dict[str, Any] = {
                **client._meta_attrs(),
                "gen_ai.request.model": model_name,
            }
            if temperature is not None:
                attrs["gen_ai.request.temperature"] = temperature
            if max_tokens is not None:
                attrs["gen_ai.request.max_tokens"] = max_tokens
            if request_body:
                attrs["gen_ai.request.body"] = request_body
            if streaming:
                attrs["gen_ai.streaming"] = True
            return attrs

        def emit(
            state: _StreamState, *, completion: str, thinking: str, response_body: Optional[str],
            safety_ratings: List[Dict[str, str]], ttft_ms: Optional[int],
            error: Optional[BaseException],
        ) -> None:
            end_ts = _now_iso()
            latency = int(time.time() * 1000 - start_ms)
            attrs = base_attrs()
            attrs["gen_ai.response.model"] = state.response_model or model_name
            attrs.update(google_usage_attrs(state.google_usage))
            if thinking:
                attrs["gen_ai.response.thinking"] = thinking
            if state.finish_reasons:
                attrs["gen_ai.response.finish_reasons"] = list(state.finish_reasons)
            if ttft_ms is not None:
                attrs["gen_ai.server.time_to_first_token_ms"] = ttft_ms
            if safety_ratings:
                import json
                attrs["gen_ai.response.safety_ratings"] = json.dumps(safety_ratings)
            if response_body:
                attrs["gen_ai.response.body"] = response_body
            events: List[SpanEvent] = from_stream_tool_calls(state.tool_calls, ts)
            client.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call",
                status="ERROR" if error is not None else "OK",
                status_message=str(error) if error is not None else None,
                operation="generate_content",
                service_name=client._opts.service_name, provider="google-ai", model=model_name,
                prompt=prompt_text, completion=completion,
                tokens_in=state.tokens_in, tokens_out=state.tokens_out,
                cache_read_tokens=state.cache_read_tokens or None,
                latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
                attributes=attrs, events=events or None,
                environment=client._opts.environment, app_version=client._opts.app_version,
                sdk_version=SDK_VERSION,
            ))

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
                attributes=base_attrs(),
                environment=client._opts.environment, app_version=client._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
            raise
        finally:
            _current_span.reset(token)

        if streaming:
            state = _StreamState(start_ms)

            def on_stream_done(
                completion: str, tokens_in: int, tokens_out: int, thinking: str,
                reasoning_tokens: int, has_redacted_thinking: bool, ttft_ms: Optional[int],
                cache_read_tokens: int, cache_creation_tokens: int,
                error: Optional[BaseException],
            ) -> None:
                emit(
                    state, completion=completion, thinking=thinking, response_body="",
                    safety_ratings=[], ttft_ms=ttft_ms, error=error,
                )

            return _SyncStreamTracer(
                result, on_stream_done, start_ms=start_ms,
                process=_StreamState.process_google_legacy_chunk, state=state,
            )

        state = _StreamState()
        try:
            state.process_google_legacy_chunk(result)
        except Exception:
            pass
        emit(
            state,
            completion="".join(state.chunks) or _fallback_text(state, result),
            thinking="".join(state.thinking_chunks),
            response_body=_safe_body(_dump(result)),
            safety_ratings=_safety_ratings(result),
            ttft_ms=None,
            error=None,
        )
        return result

    traced_generate_content._evalkit = True
    model.generate_content = traced_generate_content


def _fallback_text(state: _StreamState, result: Any) -> str:
    if state.thinking_chunks or state.tool_calls:
        return ""
    try:
        return result.text or ""
    except Exception:
        return ""


def _safety_ratings(result: Any) -> List[Dict[str, str]]:
    ratings: List[Dict[str, str]] = []
    try:
        for candidate in (result.candidates or []):
            for sr in (getattr(candidate, "safety_ratings", []) or []):
                ratings.append({
                    "category": str(getattr(sr, "category", "")),
                    "probability": str(getattr(sr, "probability", "")),
                })
    except Exception:
        pass
    return ratings


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
