from __future__ import annotations

import inspect
import time
from typing import Any, Dict, List, Optional

from ..._version import SDK_VERSION
from ...core.client import EvalKitClient, _new_span_id, _now_iso
from ...core.context import SpanContext, _current_span, current_span_ctx, get_or_create_ambient_trace_id
from ...core.types import SpanEvent, TraceEnvelope
from .._streaming import _AsyncStreamTracer, _StreamState, _SyncStreamTracer
from .._toolcalls import from_stream_tool_calls
from ._bodies import _dump, _safe_body


def _message_text(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content if isinstance(content, str) else str(content)


def _cohere_prompt(kwargs: Dict[str, Any]) -> str:
    prompt_parts: List[str] = []
    for m in (kwargs.get("chat_history") or []):
        if isinstance(m, dict):
            prompt_parts.append(f"{m.get('role', '')}: {_message_text(m.get('message', m.get('content', '')))}")
        else:
            role = getattr(m, "role", "")
            text = getattr(m, "message", None)
            if text is None:
                text = getattr(m, "content", "")
            prompt_parts.append(f"{role}: {_message_text(text)}")
    for m in (kwargs.get("messages") or []):
        if isinstance(m, dict):
            prompt_parts.append(f"{m.get('role', '')}: {_message_text(m.get('content', ''))}")
        else:
            prompt_parts.append(f"{getattr(m, 'role', '')}: {_message_text(getattr(m, 'content', ''))}")
    single = kwargs.get("message")
    if single:
        prompt_parts.append(f"user: {_message_text(single)}")
    return "\n".join(prompt_parts)


def _traced_kwargs(args: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if not args or "message" in kwargs or "messages" in kwargs:
        return kwargs
    return {**kwargs, "message": args[0]}


def _cohere_unary_state(result: Any) -> _StreamState:
    state = _StreamState()
    try:
        message = getattr(result, "message", None)
        if message is not None:
            for block in (getattr(message, "content", None) or []):
                if getattr(block, "type", None) == "thinking":
                    state._add_thinking(getattr(block, "thinking", None))
                else:
                    state._add_text(getattr(block, "text", None))
            state._add_thinking(getattr(message, "tool_plan", None))
            for call in (getattr(message, "tool_calls", None) or []):
                entry = state._new_tool()
                entry["id"] = str(getattr(call, "id", "") or "")
                function = getattr(call, "function", None)
                if function is not None:
                    entry["name"] = str(getattr(function, "name", "") or "")
                    entry["arguments"] = getattr(function, "arguments", "") or ""
        else:
            for call in (getattr(result, "tool_calls", None) or []):
                entry = state._new_tool()
                entry["name"] = str(getattr(call, "name", "") or "")
                entry["arguments"] = getattr(call, "parameters", None) or {}
        if not state.chunks:
            state._add_text(getattr(result, "text", None))
        state._add_finish_reason(getattr(result, "finish_reason", None))
        state.response_id = str(
            getattr(result, "id", None) or getattr(result, "response_id", None) or ""
        )
        state._apply_cohere_usage(getattr(result, "usage", None))
        state._apply_cohere_usage(getattr(result, "meta", None))
    except Exception:
        pass
    return state


def patch_cohere(client: EvalKitClient, cohere_client: Any) -> None:
    chat = getattr(cohere_client, "chat", None)
    if inspect.iscoroutinefunction(chat):
        _patch_cohere_chat_async(client, cohere_client)
    else:
        _patch_cohere_chat(client, cohere_client)
    _patch_cohere_chat_stream(client, cohere_client)


def _cohere_recorder(
    client: EvalKitClient, *, operation: str, kwargs: Dict[str, Any],
    trace_id: str, span_id: str, parent_span_id: Optional[str],
    start_ms: float, ts: str, streaming: bool,
):
    model = kwargs.get("model", "command-r")
    temperature = kwargs.get("temperature")
    max_tokens = kwargs.get("max_tokens")
    prompt_text = _cohere_prompt(kwargs)
    request_body = _safe_body(_dump(kwargs))

    def emit(
        state: _StreamState, *, completion: str, thinking: str, tokens_in: int, tokens_out: int,
        cache_read_tokens: int, ttft_ms: Optional[int], response_body: Optional[str],
        error: Optional[BaseException],
    ) -> None:
        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)
        attrs: Dict[str, Any] = {
            **client._meta_attrs(),
            "gen_ai.request.model": model,
            "gen_ai.response.model": model,
            "gen_ai.response.id": state.response_id,
            "gen_ai.usage.total_tokens": tokens_in + tokens_out,
        }
        if streaming:
            attrs["gen_ai.streaming"] = True
        if temperature is not None:
            attrs["gen_ai.request.temperature"] = temperature
        if max_tokens is not None:
            attrs["gen_ai.request.max_tokens"] = max_tokens
        if state.finish_reasons:
            attrs["gen_ai.response.finish_reasons"] = list(state.finish_reasons)
        if thinking:
            attrs["gen_ai.response.thinking"] = thinking
        if ttft_ms is not None:
            attrs["gen_ai.server.time_to_first_token_ms"] = ttft_ms
        if request_body:
            attrs["gen_ai.request.body"] = request_body
        if response_body:
            attrs["gen_ai.response.body"] = response_body
        events: List[SpanEvent] = from_stream_tool_calls(state.tool_calls, ts)
        client.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call",
            status="ERROR" if error is not None else "OK",
            status_message=str(error) if error is not None else None,
            operation=operation,
            service_name=client._opts.service_name, provider="cohere", model=model,
            prompt=prompt_text, completion=completion,
            tokens_in=tokens_in, tokens_out=tokens_out,
            cache_read_tokens=cache_read_tokens or None,
            latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
            attributes=attrs, events=events or None,
            environment=client._opts.environment, app_version=client._opts.app_version,
            sdk_version=SDK_VERSION,
        ))

    return emit


def _cohere_span_start(kwargs: Dict[str, Any], *, operation: str, client: EvalKitClient, streaming: bool):
    ctx_parent = current_span_ctx()
    trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
    span_id = _new_span_id()
    parent_span_id = ctx_parent.span_id if ctx_parent else None
    start_ms = time.time() * 1000
    emit = _cohere_recorder(
        client, operation=operation, kwargs=kwargs,
        trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
        start_ms=start_ms, ts=_now_iso(), streaming=streaming,
    )
    return SpanContext(trace_id=trace_id, span_id=span_id), start_ms, emit


def _emit_unary_error(emit, exc: BaseException) -> None:
    emit(
        _StreamState(), completion="", thinking="", tokens_in=0, tokens_out=0,
        cache_read_tokens=0, ttft_ms=None, response_body=None, error=exc,
    )


def _emit_unary_result(emit, result: Any) -> None:
    state = _cohere_unary_state(result)
    emit(
        state,
        completion="".join(state.chunks),
        thinking="".join(state.thinking_chunks),
        tokens_in=state.tokens_in,
        tokens_out=state.tokens_out,
        cache_read_tokens=state.cache_read_tokens,
        ttft_ms=None,
        response_body=_safe_body(_dump(result)),
        error=None,
    )


def _patch_cohere_chat(client: EvalKitClient, cohere_client: Any) -> None:
    original_chat = getattr(cohere_client, "chat", None)
    if original_chat is None or getattr(original_chat, "_evalkit", False):
        return

    def traced_chat(*args: Any, **kwargs: Any) -> Any:
        span_ctx, _start_ms, emit = _cohere_span_start(
            _traced_kwargs(args, kwargs), operation="chat", client=client, streaming=False,
        )
        token = _current_span.set(span_ctx)
        try:
            result = original_chat(*args, **kwargs)
        except Exception as exc:
            _emit_unary_error(emit, exc)
            raise
        finally:
            _current_span.reset(token)
        _emit_unary_result(emit, result)
        return result

    traced_chat._evalkit = True
    cohere_client.chat = traced_chat


def _patch_cohere_chat_async(client: EvalKitClient, cohere_client: Any) -> None:
    original_chat = getattr(cohere_client, "chat", None)
    if original_chat is None or getattr(original_chat, "_evalkit", False):
        return

    async def traced_chat(*args: Any, **kwargs: Any) -> Any:
        span_ctx, _start_ms, emit = _cohere_span_start(
            _traced_kwargs(args, kwargs), operation="chat", client=client, streaming=False,
        )
        token = _current_span.set(span_ctx)
        try:
            result = await original_chat(*args, **kwargs)
        except Exception as exc:
            _emit_unary_error(emit, exc)
            raise
        finally:
            _current_span.reset(token)
        _emit_unary_result(emit, result)
        return result

    traced_chat._evalkit = True
    cohere_client.chat = traced_chat


def _patch_cohere_chat_stream(client: EvalKitClient, cohere_client: Any) -> None:
    original_stream = getattr(cohere_client, "chat_stream", None)
    if original_stream is None or getattr(original_stream, "_evalkit", False):
        return

    def traced_chat_stream(*args: Any, **kwargs: Any) -> Any:
        span_ctx, start_ms, emit = _cohere_span_start(
            _traced_kwargs(args, kwargs), operation="chat_stream", client=client, streaming=True,
        )
        state = _StreamState(start_ms)

        def on_stream_done(
            completion: str, tokens_in: int, tokens_out: int, thinking: str,
            reasoning_tokens: int, has_redacted_thinking: bool, ttft_ms: Optional[int],
            cache_read_tokens: int, cache_creation_tokens: int,
            error: Optional[BaseException],
        ) -> None:
            emit(
                state, completion=completion, thinking=thinking,
                tokens_in=tokens_in, tokens_out=tokens_out,
                cache_read_tokens=cache_read_tokens, ttft_ms=ttft_ms,
                response_body=None, error=error,
            )

        token = _current_span.set(span_ctx)
        try:
            stream = original_stream(*args, **kwargs)
        except Exception as exc:
            on_stream_done("", 0, 0, "", 0, False, None, 0, 0, exc)
            raise
        finally:
            _current_span.reset(token)

        tracer = _AsyncStreamTracer if hasattr(stream, "__aiter__") else _SyncStreamTracer
        return tracer(
            stream, on_stream_done, start_ms=start_ms,
            process=_StreamState.process_cohere_chunk, state=state,
        )

    traced_chat_stream._evalkit = True
    cohere_client.chat_stream = traced_chat_stream
