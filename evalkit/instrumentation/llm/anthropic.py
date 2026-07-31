from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..._version import SDK_VERSION
from ...core.client import EvalKitClient, _new_span_id, _new_trace_id, _now_iso
from ...core.context import SpanContext, _current_span, current_span_ctx, get_or_create_ambient_trace_id
from ...core.types import SpanEvent, TraceEnvelope
from .._streaming import _SyncStreamTracer
from .._toolcalls import from_anthropic_blocks
from ._bodies import _dump, _safe_body

_PROMPT_MAX = 32768
_COMPLETION_MAX = 32768


def patch_anthropic(
    client: EvalKitClient,
    anthropic_client: Any,
    parent_ctx: Optional[SpanContext] = None,
) -> None:
    messages_api = anthropic_client.messages
    if getattr(messages_api.create, "_evalkit", False):
        return

    original_create = messages_api.create

    def traced_create(*args: Any, **kwargs: Any) -> Any:
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
            def on_stream_done(
                completion: str, tokens_in: int, tokens_out: int,
                thinking: str, reasoning_tokens: int, has_redacted_thinking: bool,
                ttft_ms: Optional[int],
                cache_read_tokens: int = 0, cache_creation_tokens: int = 0,
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
                client.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call", status="OK",
                    operation="messages.create",
                    service_name=client._opts.service_name, provider="anthropic", model=model,
                    prompt=prompt_text, completion=completion[:_COMPLETION_MAX],
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    cache_read_tokens=cache_read_tokens or None, cache_creation_tokens=cache_creation_tokens or None,
                    latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
                    attributes=stream_attrs,
                    environment=client._opts.environment, app_version=client._opts.app_version,
                    sdk_version=SDK_VERSION,
                ))
            return _SyncStreamTracer(original_create(*args, **kwargs), on_stream_done, is_anthropic=True, start_ms=start_ms)

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
                operation="messages.create",
                service_name=client._opts.service_name, provider="anthropic", model=model,
                prompt=prompt_text, latency_ms=latency, start_time=ts,
                end_time=end_iso, created_at=ts,
                attributes={**client._meta_attrs(), "gen_ai.request.model": model,
                            **({"gen_ai.request.body": request_body} if request_body else {})},
                environment=client._opts.environment, app_version=client._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
            raise
        finally:
            _current_span.reset(token)

        end_iso = _now_iso()
        latency = int(time.time() * 1000 - start_ms)

        text_content = ""
        thinking_text, has_redacted_thinking = _extract_thinking(getattr(result, "content", []))
        for block in getattr(result, "content", []):
            if getattr(block, "type", None) == "text":
                text_content += getattr(block, "text", "")
        events: List[SpanEvent] = from_anthropic_blocks(getattr(result, "content", []), end_iso)

        usage = getattr(result, "usage", None)
        tokens_out = getattr(usage, "output_tokens", 0) or 0
        cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
        # Unlike OpenAI/Gemini, Anthropic's input_tokens EXCLUDES cache reads/writes —
        # all three are separate, additive counts (confirmed in Anthropic's own docs:
        # "the input count is split across input_tokens, cache_read_input_tokens, and
        # cache_creation_input_tokens"). Normalize tokens_in to the total here so it's
        # inclusive-of-cache like every other provider, and downstream cost math (which
        # subtracts the cache portion back out of tokens_in) works the same way everywhere.
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
            latency_ms=latency, start_time=ts, end_time=end_iso, created_at=ts,
            events=events or None, attributes=attrs,
            environment=client._opts.environment, app_version=client._opts.app_version,
            sdk_version=SDK_VERSION,
        ))
        return result

    traced_create._evalkit = True  # type: ignore[attr-defined]
    messages_api.create = traced_create


def _extract_thinking(blocks: List[Any]) -> "tuple[str, bool]":
    """Pulls extended-thinking text out of an Anthropic response's content blocks.

    Returns (thinking_text, has_redacted_thinking). Redacted blocks carry only an
    opaque encrypted payload (no readable text), so we just flag their presence.
    """
    parts: List[str] = []
    has_redacted = False
    for block in blocks or []:
        btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if btype == "thinking":
            text = block.get("thinking", "") if isinstance(block, dict) else getattr(block, "thinking", "")
            if text:
                parts.append(text)
        elif btype == "redacted_thinking":
            has_redacted = True
    return "".join(parts), has_redacted


def _build_prompt(system: Optional[str], messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    if system:
        parts.append(f"system: {system}")
    for m in messages:
        role = m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "")
        content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        if isinstance(content, list):
            content = " ".join(
                (b.get("text", "") if isinstance(b, dict) else getattr(b, "text", ""))
                for b in content if (b.get("type") if isinstance(b, dict) else getattr(b, "type", "")) == "text"
            )
        parts.append(f"{role}: {content}")
    return "\n".join(parts)
