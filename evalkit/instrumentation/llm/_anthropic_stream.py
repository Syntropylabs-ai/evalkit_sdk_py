from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..._version import SDK_VERSION
from ...core.client import EvalKitClient, _new_span_id, _now_iso
from ...core.context import SpanContext, _current_span, current_span_ctx, get_or_create_ambient_trace_id
from ...core.types import SpanEvent, TraceEnvelope
from .._toolcalls import from_anthropic_blocks
from ._bodies import _dump, _safe_body

_PROMPT_MAX = 32768
_COMPLETION_MAX = 32768


def _snapshot(stream: Any) -> Any:
    try:
        return stream.current_message_snapshot
    except Exception:
        return None


def _final_fields(message: Any, end_ts: str) -> Tuple[str, str, bool, List[SpanEvent], int, int, int, int]:
    from .anthropic import _extract_thinking

    blocks = getattr(message, "content", None) or []
    completion = ""
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            completion += text
    try:
        thinking, has_redacted = _extract_thinking(blocks)
    except Exception:
        thinking, has_redacted = "", False
    try:
        events = from_anthropic_blocks(blocks, end_ts)
    except Exception:
        events = []
    usage = getattr(message, "usage", None)
    tokens_out = getattr(usage, "output_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
    tokens_in = (getattr(usage, "input_tokens", 0) or 0) + cache_read + cache_creation
    return completion, thinking, has_redacted, events, tokens_in, tokens_out, cache_read, cache_creation


class _TracedStreamManager:
    def __init__(
        self,
        invoke: Callable[[], Any],
        span_ctx: SpanContext,
        emit: Callable[..., None],
    ) -> None:
        self._invoke = invoke
        self._span_ctx = span_ctx
        self._emit = emit
        self._mgr: Any = None
        self._underlying: Any = None

    def __enter__(self) -> Any:
        token = _current_span.set(self._span_ctx)
        try:
            self._mgr = self._invoke()
            self._underlying = self._mgr.__enter__()
        except BaseException as exc:
            self._emit(error=exc)
            raise
        finally:
            _current_span.reset(token)
        return self._underlying

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        self._emit(_snapshot(self._underlying), exc_val)
        return self._mgr.__exit__(exc_type, exc_val, exc_tb)

    async def __aenter__(self) -> Any:
        token = _current_span.set(self._span_ctx)
        try:
            self._mgr = self._invoke()
            self._underlying = await self._mgr.__aenter__()
        except BaseException as exc:
            self._emit(error=exc)
            raise
        finally:
            _current_span.reset(token)
        return self._underlying

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        self._emit(_snapshot(self._underlying), exc_val)
        return await self._mgr.__aexit__(exc_type, exc_val, exc_tb)


def patch_messages_stream(
    client: EvalKitClient,
    target: Any,
    provider: str,
    parent_ctx: Optional[SpanContext] = None,
    class_level: bool = False,
) -> None:
    original_stream = getattr(target, "stream", None)
    if original_stream is None or getattr(original_stream, "_evalkit", False):
        return

    def begin(kwargs: Dict[str, Any]) -> Tuple[SpanContext, Callable[..., None]]:
        from .anthropic import _build_prompt

        ctx_parent = parent_ctx or current_span_ctx()
        trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()

        model = kwargs.get("model", "unknown")
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")
        prompt_text = _build_prompt(kwargs.get("system"), kwargs.get("messages") or [])[:_PROMPT_MAX]
        request_body = _safe_body(_dump(kwargs))

        def emit(message: Any = None, error: Optional[BaseException] = None) -> None:
            end_ts = _now_iso()
            completion = ""
            thinking = ""
            has_redacted = False
            events: List[SpanEvent] = []
            tokens_in = tokens_out = cache_read = cache_creation = 0
            if message is not None:
                (completion, thinking, has_redacted, events,
                 tokens_in, tokens_out, cache_read, cache_creation) = _final_fields(message, end_ts)

            attrs: Dict[str, Any] = {
                **client._meta_attrs(),
                "gen_ai.request.model": model,
                "gen_ai.streaming": True,
            }
            if request_body:
                attrs["gen_ai.request.body"] = request_body
            if temperature is not None:
                attrs["gen_ai.request.temperature"] = temperature
            if max_tokens is not None:
                attrs["gen_ai.request.max_tokens"] = max_tokens
            if thinking:
                attrs["gen_ai.response.thinking"] = thinking[:_COMPLETION_MAX]
            if has_redacted:
                attrs["gen_ai.response.has_redacted_thinking"] = True

            client.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call",
                status="ERROR" if error is not None else "OK",
                status_message=str(error) if error is not None else None,
                operation="messages.stream",
                service_name=client._opts.service_name, provider=provider, model=model,
                prompt=prompt_text, completion=completion[:_COMPLETION_MAX],
                tokens_in=tokens_in, tokens_out=tokens_out,
                cache_read_tokens=cache_read or None, cache_creation_tokens=cache_creation or None,
                latency_ms=int(time.time() * 1000 - start_ms),
                start_time=ts, end_time=end_ts, created_at=ts,
                events=events or None, attributes=attrs,
                environment=client._opts.environment, app_version=client._opts.app_version,
                sdk_version=SDK_VERSION,
            ))

        def safe_emit(message: Any = None, error: Optional[BaseException] = None) -> None:
            try:
                emit(message, error)
            except Exception:
                pass

        return SpanContext(trace_id=trace_id, span_id=span_id), safe_emit

    def wrap(invoke: Callable[[], Any], kwargs: Dict[str, Any]) -> Any:
        try:
            span_ctx, emit = begin(kwargs)
        except Exception:
            return invoke()
        return _TracedStreamManager(invoke, span_ctx, emit)

    if class_level:
        def traced_stream(self: Any, *args: Any, **kwargs: Any) -> Any:
            return wrap(lambda: original_stream(self, *args, **kwargs), kwargs)
    else:
        def traced_stream(*args: Any, **kwargs: Any) -> Any:
            return wrap(lambda: original_stream(*args, **kwargs), kwargs)

    traced_stream._evalkit = True
    try:
        target.stream = traced_stream
    except Exception:
        pass
