from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ._streaming import (
    _AsyncStreamTracer,
    _StreamState,
    _SyncStreamTracer,
    stream_response_attrs,
    stream_tool_events,
    with_stream_usage_warning,
)
from ._toolcalls import from_anthropic_blocks, from_openai_chat, from_openai_responses
from .llm.anthropic import _extract_thinking
from ..core.client import _exc_attrs
from ..core.content import message_content_text

if TYPE_CHECKING:
    from ..core.client import EvalKitClient


# The request settings, besides the model and the system prompt, that tell one agent
# configuration from another. The trace service fingerprints agent versions from them.
_SAMPLING_PARAMS = {
    "temperature": "gen_ai.request.temperature",
    "top_p": "gen_ai.request.top_p",
    "max_tokens": "gen_ai.request.max_tokens",
    "max_completion_tokens": "gen_ai.request.max_tokens",
    "max_output_tokens": "gen_ai.request.max_tokens",
}


def _request_attrs(kwargs: dict) -> dict:
    """The declared tool names and sampling parameters of a chat request."""
    attrs: dict = {}
    tools = kwargs.get("tools")
    if isinstance(tools, (list, tuple)):
        names = [name for name in map(_tool_name, tools) if name]
        if names:
            attrs["gen_ai.request.tools"] = names
    for param, key in _SAMPLING_PARAMS.items():
        value = kwargs.get(param)
        # The clients' NOT_GIVEN sentinels and booleans are not settings.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            attrs[key] = value
    return attrs


def _tool_name(tool) -> str | None:
    """OpenAI chat nests the name under ``function``; Anthropic and Responses name the tool
    at the top, and a built-in tool (web search, code interpreter) has only its type."""
    def field(obj, key):
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    function = field(tool, "function")
    name = (function is not None and field(function, "name")) or field(tool, "name") or field(tool, "type")
    return name if isinstance(name, str) and name else None


def _system_text(system) -> str:
    """Anthropic's ``system`` as text; a list of blocks is how prompt caching passes it."""
    if not isinstance(system, (list, tuple)):
        return str(system)
    texts = [block.get("text") if isinstance(block, dict) else getattr(block, "text", None) for block in system]
    return "\n".join(t for t in texts if isinstance(t, str))


def _try_patch_openai(ek: "EvalKitClient") -> None:
    try:
        import openai.resources.chat.completions.completions as _oc
    except ImportError:
        try:
            import openai.resources.chat.completions as _oc
        except ImportError:
            return
    try:
        _patch_openai_class_sync(ek, _oc.Completions)
    except Exception:
        pass
    try:
        _patch_openai_class_async(ek, _oc.AsyncCompletions)
    except Exception:
        pass


def _patch_openai_class_sync(ek: "EvalKitClient", cls, provider: str = "openai") -> None:
    if getattr(cls.create, "_evalkit", False):
        return
    import time
    from ..core.client import _new_span_id, _now_iso
    from ..core.context import _current_span, child_span_ctx, current_span_ctx, get_or_create_ambient_trace_id
    from ..core.types import SpanEvent, TraceEnvelope
    from .._version import SDK_VERSION

    original = cls.create

    def traced(self, *args, **kwargs):
        ctx_parent = current_span_ctx()
        trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        prompt = "\n".join(f"{m.get('role','')}: {message_content_text(m.get('content'))}" for m in messages)

        if kwargs.get("stream"):
            state = _StreamState(start_ms)

            def on_stream_done(
                completion: str, tokens_in: int, tokens_out: int,
                thinking: str, reasoning_tokens: int, has_redacted_thinking: bool,
                ttft_ms=None, cache_read_tokens: int = 0, cache_creation_tokens: int = 0,
                error=None,
            ) -> None:
                end_ts = _now_iso()
                latency = int(time.time() * 1000 - start_ms)
                stream_attrs = {**ek._meta_attrs(), "gen_ai.request.model": model, **_request_attrs(kwargs), "gen_ai.streaming": True}
                if reasoning_tokens:
                    stream_attrs["gen_ai.usage.reasoning_tokens"] = reasoning_tokens
                if ttft_ms is not None:
                    stream_attrs["gen_ai.server.time_to_first_token_ms"] = ttft_ms
                if tokens_in or tokens_out:
                    stream_attrs["gen_ai.usage.total_tokens"] = tokens_in + tokens_out
                stream_attrs.update(stream_response_attrs(state))
                ek.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call",
                    status="ERROR" if error is not None else "OK",
                    status_message=str(error) if error is not None else None,
                    operation="chat.completions.create",
                    service_name=ek._opts.service_name, provider=provider, model=model,
                    prompt=prompt, completion=completion,
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    cache_read_tokens=cache_read_tokens or None, cache_creation_tokens=cache_creation_tokens or None,
                    latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
                    events=stream_tool_events(state, end_ts) or None,
                    attributes=stream_attrs,
                    environment=ek._opts.environment, app_version=ek._opts.app_version,
                    sdk_version=SDK_VERSION,
                ))
            wrapped_done = with_stream_usage_warning(on_stream_done, kwargs, provider == "openai")
            return _SyncStreamTracer(
                original(self, *args, **kwargs), wrapped_done, start_ms=start_ms, state=state,
            )

        span_ctx = child_span_ctx(trace_id, span_id)
        token = _current_span.set(span_ctx)
        try:
            result = original(self, *args, **kwargs)
        except Exception as exc:
            end_ts = _now_iso()
            ek.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="ERROR", status_message=str(exc),
                operation="chat.completions.create",
                service_name=ek._opts.service_name, provider=provider, model=model,
                prompt=prompt, latency_ms=int(time.time() * 1000 - start_ms),
                start_time=ts, end_time=_now_iso(), created_at=ts,
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model, **_request_attrs(kwargs), **_exc_attrs(exc)},
                environment=ek._opts.environment, app_version=ek._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
            raise
        finally:
            _current_span.reset(token)
        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)

        if not hasattr(result, "choices"):
            return result
        try:
            choice = result.choices[0] if result.choices else None
            completion = (choice.message.content or "") if choice else ""
            usage = getattr(result, "usage", None)
            tokens_in = getattr(usage, "prompt_tokens", 0) or 0
            tokens_out = getattr(usage, "completion_tokens", 0) or 0
            completion_details = getattr(usage, "completion_tokens_details", None)
            reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) or 0
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            cache_read_tokens = getattr(prompt_details, "cached_tokens", 0) or 0
            events = from_openai_chat(result, end_ts)
            attrs = {**ek._meta_attrs(), "gen_ai.request.model": model, **_request_attrs(kwargs),
                     "gen_ai.response.model": getattr(result, "model", model),
                     "gen_ai.usage.total_tokens": getattr(usage, "total_tokens", 0) or 0}
            if reasoning_tokens:
                attrs["gen_ai.usage.reasoning_tokens"] = reasoning_tokens
            ek.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="OK",
                operation="chat.completions.create",
                service_name=ek._opts.service_name, provider=provider, model=model,
                prompt=prompt, completion=completion,
                tokens_in=tokens_in, tokens_out=tokens_out,
                cache_read_tokens=cache_read_tokens or None,
                latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
                events=events or None,
                attributes=attrs,
                environment=ek._opts.environment, app_version=ek._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
        except Exception:
            pass
        return result

    traced._evalkit = True
    cls.create = traced


def _patch_openai_class_async(ek: "EvalKitClient", cls, provider: str = "openai") -> None:
    if getattr(cls.create, "_evalkit", False):
        return
    import time
    from ..core.client import _new_span_id, _now_iso
    from ..core.context import _current_span, child_span_ctx, current_span_ctx, get_or_create_ambient_trace_id
    from ..core.types import SpanEvent, TraceEnvelope
    from .._version import SDK_VERSION

    original = cls.create

    async def traced_async(self, *args, **kwargs):
        ctx_parent = current_span_ctx()
        trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        prompt = "\n".join(f"{m.get('role','')}: {message_content_text(m.get('content'))}" for m in messages)

        if kwargs.get("stream"):
            state = _StreamState(start_ms)

            def on_stream_done(
                completion: str, tokens_in: int, tokens_out: int,
                thinking: str, reasoning_tokens: int, has_redacted_thinking: bool,
                ttft_ms=None, cache_read_tokens: int = 0, cache_creation_tokens: int = 0,
                error=None,
            ) -> None:
                end_ts = _now_iso()
                latency = int(time.time() * 1000 - start_ms)
                stream_attrs = {**ek._meta_attrs(), "gen_ai.request.model": model, **_request_attrs(kwargs), "gen_ai.streaming": True}
                if reasoning_tokens:
                    stream_attrs["gen_ai.usage.reasoning_tokens"] = reasoning_tokens
                if ttft_ms is not None:
                    stream_attrs["gen_ai.server.time_to_first_token_ms"] = ttft_ms
                if tokens_in or tokens_out:
                    stream_attrs["gen_ai.usage.total_tokens"] = tokens_in + tokens_out
                stream_attrs.update(stream_response_attrs(state))
                ek.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call",
                    status="ERROR" if error is not None else "OK",
                    status_message=str(error) if error is not None else None,
                    operation="chat.completions.create",
                    service_name=ek._opts.service_name, provider=provider, model=model,
                    prompt=prompt, completion=completion,
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    cache_read_tokens=cache_read_tokens or None, cache_creation_tokens=cache_creation_tokens or None,
                    latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
                    events=stream_tool_events(state, end_ts) or None,
                    attributes=stream_attrs,
                    environment=ek._opts.environment, app_version=ek._opts.app_version,
                    sdk_version=SDK_VERSION,
                ))
            wrapped_done = with_stream_usage_warning(on_stream_done, kwargs, provider == "openai")
            raw = await original(self, *args, **kwargs)
            return _AsyncStreamTracer(raw, wrapped_done, start_ms=start_ms, state=state)

        span_ctx = child_span_ctx(trace_id, span_id)
        token = _current_span.set(span_ctx)
        try:
            result = await original(self, *args, **kwargs)
        except Exception as exc:
            end_ts = _now_iso()
            ek.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="ERROR", status_message=str(exc),
                operation="chat.completions.create",
                service_name=ek._opts.service_name, provider=provider, model=model,
                prompt=prompt, latency_ms=int(time.time() * 1000 - start_ms),
                start_time=ts, end_time=end_ts, created_at=ts,
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model, **_request_attrs(kwargs), **_exc_attrs(exc)},
                environment=ek._opts.environment, app_version=ek._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
            raise
        finally:
            _current_span.reset(token)
        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)
        if not hasattr(result, "choices"):
            return result
        try:
            choice = result.choices[0] if result.choices else None
            completion = (choice.message.content or "") if choice else ""
            usage = getattr(result, "usage", None)
            tokens_in = getattr(usage, "prompt_tokens", 0) or 0
            tokens_out = getattr(usage, "completion_tokens", 0) or 0
            completion_details = getattr(usage, "completion_tokens_details", None)
            reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) or 0
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            cache_read_tokens = getattr(prompt_details, "cached_tokens", 0) or 0
            events = from_openai_chat(result, end_ts)
            attrs = {**ek._meta_attrs(), "gen_ai.request.model": model, **_request_attrs(kwargs),
                     "gen_ai.response.model": getattr(result, "model", model),
                     "gen_ai.usage.total_tokens": getattr(usage, "total_tokens", 0) or 0}
            if reasoning_tokens:
                attrs["gen_ai.usage.reasoning_tokens"] = reasoning_tokens
            ek.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="OK",
                operation="chat.completions.create",
                service_name=ek._opts.service_name, provider=provider, model=model,
                prompt=prompt, completion=completion,
                tokens_in=tokens_in, tokens_out=tokens_out,
                cache_read_tokens=cache_read_tokens or None,
                latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
                events=events or None,
                attributes=attrs,
                environment=ek._opts.environment, app_version=ek._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
        except Exception:
            pass
        return result

    traced_async._evalkit = True
    cls.create = traced_async


def _try_patch_anthropic(ek: "EvalKitClient") -> None:
    try:
        import anthropic.resources.messages as _am
    except ImportError:
        return
    try:
        _patch_anthropic_class_sync(ek, _am.Messages)
    except Exception:
        pass
    try:
        _patch_anthropic_class_async(ek, _am.AsyncMessages)
    except Exception:
        pass
    try:
        _patch_anthropic_stream_sync(ek, _am.Messages)
    except Exception:
        pass
    try:
        _patch_anthropic_stream_async(ek, _am.AsyncMessages)
    except Exception:
        pass


def _patch_anthropic_stream_sync(ek: "EvalKitClient", cls) -> None:
    from .llm._anthropic_stream import patch_messages_stream

    patch_messages_stream(ek, cls, "anthropic", class_level=True)


def _patch_anthropic_stream_async(ek: "EvalKitClient", cls) -> None:
    from .llm._anthropic_stream import patch_messages_stream

    patch_messages_stream(ek, cls, "anthropic", class_level=True)


def _patch_anthropic_class_sync(ek: "EvalKitClient", cls) -> None:
    if getattr(cls.create, "_evalkit", False):
        return
    import time
    from ..core.client import _new_span_id, _now_iso
    from ..core.context import _current_span, child_span_ctx, current_span_ctx, get_or_create_ambient_trace_id
    from ..core.types import SpanEvent, TraceEnvelope
    from .._version import SDK_VERSION

    original = cls.create

    def traced(self, *args, **kwargs):
        ctx_parent = current_span_ctx()
        trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()
        model = kwargs.get("model", "unknown")
        system = kwargs.get("system")
        messages = kwargs.get("messages", [])
        parts = ([f"system: {_system_text(system)}"] if system else []) + [
            f"{m.get('role','')}: {message_content_text(m.get('content'))}" for m in messages
        ]
        prompt = "\n".join(parts)

        if kwargs.get("stream"):
            state = _StreamState(start_ms)

            def on_stream_done(
                completion: str, tokens_in: int, tokens_out: int,
                thinking: str, reasoning_tokens: int, has_redacted_thinking: bool,
                ttft_ms=None, cache_read_tokens: int = 0, cache_creation_tokens: int = 0,
                error=None,
            ) -> None:
                end_ts = _now_iso()
                latency = int(time.time() * 1000 - start_ms)
                stream_attrs = {**ek._meta_attrs(), "gen_ai.request.model": model, **_request_attrs(kwargs), "gen_ai.streaming": True}
                if thinking:
                    stream_attrs["gen_ai.response.thinking"] = thinking[:32768]
                if has_redacted_thinking:
                    stream_attrs["gen_ai.response.has_redacted_thinking"] = True
                if ttft_ms is not None:
                    stream_attrs["gen_ai.server.time_to_first_token_ms"] = ttft_ms
                stream_attrs.update(stream_response_attrs(state))
                ek.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call",
                    status="ERROR" if error is not None else "OK",
                    status_message=str(error) if error is not None else None,
                    operation="messages.create",
                    service_name=ek._opts.service_name, provider="anthropic", model=model,
                    prompt=prompt, completion=completion,
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    cache_read_tokens=cache_read_tokens or None, cache_creation_tokens=cache_creation_tokens or None,
                    latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
                    events=stream_tool_events(state, end_ts) or None,
                    attributes=stream_attrs,
                    environment=ek._opts.environment, app_version=ek._opts.app_version,
                    sdk_version=SDK_VERSION,
                ))
            return _SyncStreamTracer(
                original(self, *args, **kwargs), on_stream_done,
                is_anthropic=True, start_ms=start_ms, state=state,
            )

        span_ctx = child_span_ctx(trace_id, span_id)
        token = _current_span.set(span_ctx)
        try:
            result = original(self, *args, **kwargs)
        except Exception as exc:
            end_ts = _now_iso()
            ek.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="ERROR", status_message=str(exc),
                operation="messages.create",
                service_name=ek._opts.service_name, provider="anthropic", model=model,
                prompt=prompt, latency_ms=int(time.time() * 1000 - start_ms),
                start_time=ts, end_time=end_ts, created_at=ts,
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model, **_request_attrs(kwargs), **_exc_attrs(exc)},
                environment=ek._opts.environment, app_version=ek._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
            raise
        finally:
            _current_span.reset(token)
        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)
        content = getattr(result, "content", [])
        completion = next((b.text for b in content if hasattr(b, "text")), "")
        thinking_text, has_redacted_thinking = _extract_thinking(content)
        usage = getattr(result, "usage", None)
        tokens_out = getattr(usage, "output_tokens", 0) or 0
        cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
        tokens_in = (getattr(usage, "input_tokens", 0) or 0) + cache_read_tokens + cache_creation_tokens
        events = from_anthropic_blocks(content, end_ts)
        stop_reason = getattr(result, "stop_reason", None)
        attrs = {
            **ek._meta_attrs(),
            "gen_ai.request.model": model,
            **_request_attrs(kwargs),
            "gen_ai.response.model": getattr(result, "model", model),
            "gen_ai.response.id": getattr(result, "id", ""),
            "gen_ai.usage.total_tokens": tokens_in + tokens_out,
        }
        if stop_reason:
            attrs["gen_ai.response.finish_reasons"] = [stop_reason]
        if thinking_text:
            attrs["gen_ai.response.thinking"] = thinking_text[:32768]
        if has_redacted_thinking:
            attrs["gen_ai.response.has_redacted_thinking"] = True
        ek.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call", status="OK",
            operation="messages.create",
            service_name=ek._opts.service_name, provider="anthropic", model=model,
            prompt=prompt, completion=completion,
            tokens_in=tokens_in, tokens_out=tokens_out,
            cache_read_tokens=cache_read_tokens or None, cache_creation_tokens=cache_creation_tokens or None,
            latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
            events=events or None,
            attributes=attrs,
            environment=ek._opts.environment, app_version=ek._opts.app_version,
            sdk_version=SDK_VERSION,
        ))
        return result

    traced._evalkit = True
    cls.create = traced


def _patch_anthropic_class_async(ek: "EvalKitClient", cls) -> None:
    if getattr(cls.create, "_evalkit", False):
        return
    import time
    from ..core.client import _new_span_id, _now_iso
    from ..core.context import _current_span, child_span_ctx, current_span_ctx, get_or_create_ambient_trace_id
    from ..core.types import SpanEvent, TraceEnvelope
    from .._version import SDK_VERSION

    original = cls.create

    async def traced_async(self, *args, **kwargs):
        ctx_parent = current_span_ctx()
        trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()
        model = kwargs.get("model", "unknown")
        system = kwargs.get("system")
        messages = kwargs.get("messages", [])
        parts = ([f"system: {_system_text(system)}"] if system else []) + [
            f"{m.get('role','')}: {message_content_text(m.get('content'))}" for m in messages
        ]
        prompt = "\n".join(parts)

        if kwargs.get("stream"):
            state = _StreamState(start_ms)

            def on_stream_done(
                completion: str, tokens_in: int, tokens_out: int,
                thinking: str, reasoning_tokens: int, has_redacted_thinking: bool,
                ttft_ms=None, cache_read_tokens: int = 0, cache_creation_tokens: int = 0,
                error=None,
            ) -> None:
                end_ts = _now_iso()
                latency = int(time.time() * 1000 - start_ms)
                stream_attrs = {**ek._meta_attrs(), "gen_ai.request.model": model, **_request_attrs(kwargs), "gen_ai.streaming": True}
                if thinking:
                    stream_attrs["gen_ai.response.thinking"] = thinking[:32768]
                if has_redacted_thinking:
                    stream_attrs["gen_ai.response.has_redacted_thinking"] = True
                if ttft_ms is not None:
                    stream_attrs["gen_ai.server.time_to_first_token_ms"] = ttft_ms
                stream_attrs.update(stream_response_attrs(state))
                ek.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call",
                    status="ERROR" if error is not None else "OK",
                    status_message=str(error) if error is not None else None,
                    operation="messages.create",
                    service_name=ek._opts.service_name, provider="anthropic", model=model,
                    prompt=prompt, completion=completion,
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    cache_read_tokens=cache_read_tokens or None, cache_creation_tokens=cache_creation_tokens or None,
                    latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
                    events=stream_tool_events(state, end_ts) or None,
                    attributes=stream_attrs,
                    environment=ek._opts.environment, app_version=ek._opts.app_version,
                    sdk_version=SDK_VERSION,
                ))
            raw = await original(self, *args, **kwargs)
            return _AsyncStreamTracer(
                raw, on_stream_done, is_anthropic=True, start_ms=start_ms, state=state,
            )

        span_ctx = child_span_ctx(trace_id, span_id)
        token = _current_span.set(span_ctx)
        try:
            result = await original(self, *args, **kwargs)
        except Exception as exc:
            end_ts = _now_iso()
            ek.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="ERROR", status_message=str(exc),
                operation="messages.create",
                service_name=ek._opts.service_name, provider="anthropic", model=model,
                prompt=prompt, latency_ms=int(time.time() * 1000 - start_ms),
                start_time=ts, end_time=end_ts, created_at=ts,
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model, **_request_attrs(kwargs), **_exc_attrs(exc)},
                environment=ek._opts.environment, app_version=ek._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
            raise
        finally:
            _current_span.reset(token)
        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)
        content = getattr(result, "content", [])
        completion = next((b.text for b in content if hasattr(b, "text")), "")
        thinking_text, has_redacted_thinking = _extract_thinking(content)
        usage = getattr(result, "usage", None)
        tokens_out = getattr(usage, "output_tokens", 0) or 0
        cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
        tokens_in = (getattr(usage, "input_tokens", 0) or 0) + cache_read_tokens + cache_creation_tokens
        events = from_anthropic_blocks(content, end_ts)
        stop_reason = getattr(result, "stop_reason", None)
        attrs = {
            **ek._meta_attrs(),
            "gen_ai.request.model": model,
            **_request_attrs(kwargs),
            "gen_ai.response.model": getattr(result, "model", model),
            "gen_ai.response.id": getattr(result, "id", ""),
            "gen_ai.usage.total_tokens": tokens_in + tokens_out,
        }
        if stop_reason:
            attrs["gen_ai.response.finish_reasons"] = [stop_reason]
        if thinking_text:
            attrs["gen_ai.response.thinking"] = thinking_text[:32768]
        if has_redacted_thinking:
            attrs["gen_ai.response.has_redacted_thinking"] = True
        ek.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call", status="OK",
            operation="messages.create",
            service_name=ek._opts.service_name, provider="anthropic", model=model,
            prompt=prompt, completion=completion,
            tokens_in=tokens_in, tokens_out=tokens_out,
            cache_read_tokens=cache_read_tokens or None, cache_creation_tokens=cache_creation_tokens or None,
            latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
            events=events or None,
            attributes=attrs,
            environment=ek._opts.environment, app_version=ek._opts.app_version,
            sdk_version=SDK_VERSION,
        ))
        return result

    traced_async._evalkit = True
    cls.create = traced_async


def _try_patch_google_genai(ek: "EvalKitClient") -> None:
    try:
        from google.genai import models as _gm
    except ImportError:
        return
    try:
        from .llm.google_genai import patch_google_genai
        patch_google_genai(ek)
    except Exception:
        pass


def _try_patch_google_adk(ek: "EvalKitClient") -> None:
    try:
        import google.adk
    except ImportError:
        return
    try:
        from .llm.google_adk import patch_google_adk
        patch_google_adk(ek)
    except Exception:
        pass


def _lazy_instance_patch(cls, names, upgrade) -> None:
    import inspect

    originals = {}
    for name in names:
        fn = getattr(cls, name, None)
        if fn is None:
            continue
        if getattr(fn, "_evalkit", False):
            return
        originals[name] = fn
    if not originals:
        return

    def resolve(instance, name):
        bound = instance.__dict__.get(name)
        if bound is not None and getattr(bound, "_evalkit", False):
            return bound
        for other, fn in originals.items():
            instance.__dict__[other] = fn.__get__(instance, type(instance))
        upgrade(instance)
        bound = instance.__dict__.get(name)
        return bound if getattr(bound, "_evalkit", False) else None

    def make_sync(name, original):
        def traced(self, *args, **kwargs):
            try:
                bound = resolve(self, name)
            except Exception:
                bound = None
            if bound is not None:
                return bound(*args, **kwargs)
            return original(self, *args, **kwargs)

        traced._evalkit = True
        return traced

    def make_async(name, original):
        async def traced(self, *args, **kwargs):
            try:
                bound = resolve(self, name)
            except Exception:
                bound = None
            if bound is not None:
                return await bound(*args, **kwargs)
            return await original(self, *args, **kwargs)

        traced._evalkit = True
        return traced

    for name, original in originals.items():
        make = make_async if inspect.iscoroutinefunction(original) else make_sync
        setattr(cls, name, make(name, original))


def _try_patch_google_ai(ek: "EvalKitClient") -> None:
    try:
        import google.generativeai as _genai
    except ImportError:
        return
    try:
        _patch_google_genai_legacy_class(ek, _genai.GenerativeModel)
    except Exception:
        pass


def _patch_google_genai_legacy_class(ek: "EvalKitClient", cls) -> None:
    from .llm.google_ai import patch_google_ai

    _lazy_instance_patch(cls, ("generate_content",), lambda model: patch_google_ai(ek, model))


_COHERE_CLIENT_ATTRS = ("Client", "AsyncClient", "ClientV2", "AsyncClientV2")
_COHERE_METHODS = ("chat", "chat_stream")


def _try_patch_cohere(ek: "EvalKitClient") -> None:
    try:
        import cohere as _co
    except ImportError:
        return
    for attr in _COHERE_CLIENT_ATTRS:
        try:
            cls = getattr(_co, attr, None)
            if cls is not None:
                _patch_cohere_class(ek, cls)
        except Exception:
            pass


def _patch_cohere_class(ek: "EvalKitClient", cls) -> None:
    from .llm.cohere import patch_cohere

    _lazy_instance_patch(cls, _COHERE_METHODS, lambda co: patch_cohere(ek, co))


def _responses_tool_events(output_items: list, ts: str) -> list:
    return from_openai_responses(output_items, ts)


def _try_patch_openai_responses(ek: "EvalKitClient") -> None:
    try:
        from openai.resources.responses.responses import Responses, AsyncResponses
    except ImportError:
        return
    try:
        _patch_openai_responses_sync(ek, Responses)
    except Exception:
        pass
    try:
        _patch_openai_responses_async(ek, AsyncResponses)
    except Exception:
        pass


def _patch_openai_responses_sync(ek: "EvalKitClient", cls) -> None:
    if getattr(cls.create, "_evalkit", False):
        return
    import time
    from ..core.client import _new_span_id, _now_iso
    from ..core.context import _current_span, child_span_ctx, current_span_ctx, get_or_create_ambient_trace_id
    from ..core.types import SpanEvent, TraceEnvelope
    from .._version import SDK_VERSION

    original = cls.create

    def traced(self, *args, **kwargs):
        ctx_parent = current_span_ctx()
        trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()
        model = kwargs.get("model", "unknown")
        raw_input = kwargs.get("input", "")
        instructions = kwargs.get("instructions") or ""
        if isinstance(raw_input, str):
            input_text = raw_input
        elif isinstance(raw_input, list):
            parts = []
            for item in raw_input:
                if isinstance(item, dict):
                    role = item.get("role", "")
                    content = item.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            (c.get("text", "") if isinstance(c, dict) else getattr(c, "text", ""))
                            for c in content
                        )
                    parts.append(f"{role}: {content}" if role else str(content))
                else:
                    parts.append(str(item))
            input_text = "\n".join(parts)
        else:
            input_text = str(raw_input)
        prompt = (f"system: {instructions}\n" if instructions else "") + input_text

        if kwargs.get("stream"):
            chunks: list[str] = []
            tokens_in_box = [0]
            tokens_out_box = [0]
            tool_events_box: list[SpanEvent] = []
            done_box = [False]

            span_ctx = child_span_ctx(trace_id, span_id)
            token = _current_span.set(span_ctx)
            try:
                raw_stream = original(self, *args, **kwargs)
            except Exception as exc:
                _current_span.reset(token)
                end_ts = _now_iso()
                ek.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call", status="ERROR", status_message=str(exc),
                    operation="responses.create",
                    service_name=ek._opts.service_name, provider="openai", model=model,
                    prompt=prompt[:32768], latency_ms=int(time.time() * 1000 - start_ms),
                    start_time=ts, end_time=end_ts, created_at=ts,
                    attributes={**ek._meta_attrs(), "gen_ai.request.model": model, **_request_attrs(kwargs), **_exc_attrs(exc)},
                    environment=ek._opts.environment, app_version=ek._opts.app_version,
                    sdk_version=SDK_VERSION,
                ))
                raise
            _current_span.reset(token)

            def _emit_responses_span() -> None:
                if done_box[0]:
                    return
                done_box[0] = True
                end_ts2 = _now_iso()
                latency2 = int(time.time() * 1000 - start_ms)
                ek.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call", status="OK",
                    operation="responses.create",
                    service_name=ek._opts.service_name, provider="openai", model=model,
                    prompt=prompt[:32768], completion="".join(chunks)[:32768],
                    tokens_in=tokens_in_box[0], tokens_out=tokens_out_box[0],
                    latency_ms=latency2, start_time=ts, end_time=end_ts2, created_at=ts,
                    events=tool_events_box or None,
                    attributes={**ek._meta_attrs(), "gen_ai.request.model": model, **_request_attrs(kwargs),
                                "gen_ai.streaming": True,
                                "gen_ai.usage.total_tokens": tokens_in_box[0] + tokens_out_box[0]},
                    environment=ek._opts.environment, app_version=ek._opts.app_version,
                    sdk_version=SDK_VERSION,
                ))

            class _ResponsesStreamWrapper:
                def __iter__(self_inner):
                    try:
                        for event in raw_stream:
                            etype = getattr(event, "type", "")
                            if etype == "response.output_text.delta":
                                delta = getattr(event, "delta", "")
                                if delta:
                                    chunks.append(delta)
                            elif etype in ("response.completed", "response.done"):
                                resp = getattr(event, "response", None)
                                if resp:
                                    usage = getattr(resp, "usage", None)
                                    tokens_in_box[0] = getattr(usage, "input_tokens", 0) or 0
                                    tokens_out_box[0] = getattr(usage, "output_tokens", 0) or 0
                                    end_ts3 = _now_iso()
                                    tool_events_box.extend(
                                        from_openai_responses(list(getattr(resp, "output", []) or []), end_ts3)
                                    )
                            yield event
                    finally:
                        _emit_responses_span()

                def __enter__(self_inner):
                    if hasattr(raw_stream, "__enter__"):
                        raw_stream.__enter__()
                    return self_inner

                def __exit__(self_inner, exc_type, exc_val, exc_tb):
                    _emit_responses_span()
                    if hasattr(raw_stream, "__exit__"):
                        return raw_stream.__exit__(exc_type, exc_val, exc_tb)

                def __getattr__(self_inner, name):
                    return getattr(raw_stream, name)

            return _ResponsesStreamWrapper()

        span_ctx = child_span_ctx(trace_id, span_id)
        token = _current_span.set(span_ctx)
        try:
            result = original(self, *args, **kwargs)
        except Exception as exc:
            end_ts = _now_iso()
            ek.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="ERROR", status_message=str(exc),
                operation="responses.create",
                service_name=ek._opts.service_name, provider="openai", model=model,
                prompt=prompt[:32768], latency_ms=int(time.time() * 1000 - start_ms),
                start_time=ts, end_time=end_ts, created_at=ts,
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model, **_request_attrs(kwargs), **_exc_attrs(exc)},
                environment=ek._opts.environment, app_version=ek._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
            raise
        finally:
            _current_span.reset(token)
        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)
        try:
            completion = result.output_text or ""
            usage = getattr(result, "usage", None)
            tokens_in = getattr(usage, "input_tokens", 0) or 0
            tokens_out = getattr(usage, "output_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", 0) or 0
            output_items = list(getattr(result, "output", []) or [])
            events = _responses_tool_events(output_items, end_ts)
            ek.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="OK",
                operation="responses.create",
                service_name=ek._opts.service_name, provider="openai", model=model,
                prompt=prompt[:32768], completion=completion[:32768],
                tokens_in=tokens_in, tokens_out=tokens_out,
                latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
                events=events or None,
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model, **_request_attrs(kwargs),
                            "gen_ai.response.model": getattr(result, "model", model),
                            "gen_ai.response.id": getattr(result, "id", ""),
                            "gen_ai.usage.total_tokens": total_tokens},
                environment=ek._opts.environment, app_version=ek._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
        except Exception:
            pass
        return result

    traced._evalkit = True
    cls.create = traced


def _patch_openai_responses_async(ek: "EvalKitClient", cls) -> None:
    if getattr(cls.create, "_evalkit", False):
        return
    import time
    from ..core.client import _new_span_id, _now_iso
    from ..core.context import _current_span, child_span_ctx, current_span_ctx, get_or_create_ambient_trace_id
    from ..core.types import SpanEvent, TraceEnvelope
    from .._version import SDK_VERSION

    original = cls.create

    async def traced_async(self, *args, **kwargs):
        ctx_parent = current_span_ctx()
        trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()
        model = kwargs.get("model", "unknown")
        raw_input = kwargs.get("input", "")
        instructions = kwargs.get("instructions") or ""
        if isinstance(raw_input, str):
            input_text = raw_input
        elif isinstance(raw_input, list):
            parts = []
            for item in raw_input:
                if isinstance(item, dict):
                    role = item.get("role", "")
                    content = item.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            (c.get("text", "") if isinstance(c, dict) else getattr(c, "text", ""))
                            for c in content
                        )
                    parts.append(f"{role}: {content}" if role else str(content))
                else:
                    parts.append(str(item))
            input_text = "\n".join(parts)
        else:
            input_text = str(raw_input)
        prompt = (f"system: {instructions}\n" if instructions else "") + input_text

        span_ctx = child_span_ctx(trace_id, span_id)
        token = _current_span.set(span_ctx)
        try:
            result = await original(self, *args, **kwargs)
        except Exception as exc:
            end_ts = _now_iso()
            ek.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="ERROR", status_message=str(exc),
                operation="responses.create",
                service_name=ek._opts.service_name, provider="openai", model=model,
                prompt=prompt[:32768], latency_ms=int(time.time() * 1000 - start_ms),
                start_time=ts, end_time=end_ts, created_at=ts,
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model, **_request_attrs(kwargs), **_exc_attrs(exc)},
                environment=ek._opts.environment, app_version=ek._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
            raise
        finally:
            _current_span.reset(token)
        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)
        try:
            completion = result.output_text or ""
            usage = getattr(result, "usage", None)
            tokens_in = getattr(usage, "input_tokens", 0) or 0
            tokens_out = getattr(usage, "output_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", 0) or 0
            output_items = list(getattr(result, "output", []) or [])
            events = _responses_tool_events(output_items, end_ts)
            ek.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="OK",
                operation="responses.create",
                service_name=ek._opts.service_name, provider="openai", model=model,
                prompt=prompt[:32768], completion=completion[:32768],
                tokens_in=tokens_in, tokens_out=tokens_out,
                latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
                events=events or None,
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model, **_request_attrs(kwargs),
                            "gen_ai.response.model": getattr(result, "model", model),
                            "gen_ai.response.id": getattr(result, "id", ""),
                            "gen_ai.usage.total_tokens": total_tokens},
                environment=ek._opts.environment, app_version=ek._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
        except Exception:
            pass
        return result

    traced_async._evalkit = True
    cls.create = traced_async


def _try_patch_groq(ek: "EvalKitClient") -> None:
    try:
        import groq.resources.chat.completions as _gc
    except ImportError:
        return
    try:
        _patch_openai_class_sync(ek, _gc.Completions, provider="groq")
    except Exception:
        pass
    try:
        _patch_openai_class_async(ek, _gc.AsyncCompletions, provider="groq")
    except Exception:
        pass


def _try_patch_litellm(ek: "EvalKitClient") -> None:
    try:
        import litellm
    except ImportError:
        return
    try:
        from .llm.litellm import patch_litellm
        patch_litellm(ek)
    except Exception:
        pass


def _try_patch_ollama(ek: "EvalKitClient") -> None:
    try:
        import ollama
    except ImportError:
        return
    try:
        from .llm.ollama import patch_ollama
        patch_ollama(ek)
    except Exception:
        pass


def _try_patch_pinecone(ek: "EvalKitClient") -> None:
    try:
        import pinecone
    except ImportError:
        return
    try:
        from .vectordb.pinecone import patch_pinecone
        patch_pinecone(ek)
    except Exception:
        pass


def _try_patch_qdrant(ek: "EvalKitClient") -> None:
    try:
        import qdrant_client
    except ImportError:
        return
    try:
        from .vectordb.qdrant import patch_qdrant
        patch_qdrant(ek)
    except Exception:
        pass


def _try_patch_chroma(ek: "EvalKitClient") -> None:
    try:
        import chromadb
    except ImportError:
        return
    try:
        from .vectordb.chroma import patch_chroma
        patch_chroma(ek)
    except Exception:
        pass


def _try_patch_weaviate(ek: "EvalKitClient") -> None:
    try:
        import weaviate
    except ImportError:
        return
    try:
        from .vectordb.weaviate import patch_weaviate
        patch_weaviate(ek)
    except Exception:
        pass


def _try_patch_llamaindex(ek: "EvalKitClient") -> None:
    try:
        import llama_index.core
    except ImportError:
        return
    try:
        from .llm.llamaindex import patch_llamaindex
        patch_llamaindex(ek)
    except Exception:
        pass


def _try_patch_openai_audio(ek: "EvalKitClient") -> None:
    try:
        import openai
    except ImportError:
        return
    try:
        from .llm.openai_audio import patch_openai_audio
        patch_openai_audio(ek)
    except Exception:
        pass


def _try_patch_elevenlabs(ek: "EvalKitClient") -> None:
    try:
        import elevenlabs
    except ImportError:
        return
    try:
        from .llm.elevenlabs import patch_elevenlabs
        patch_elevenlabs(ek)
    except Exception:
        pass


def _try_patch_assemblyai(ek: "EvalKitClient") -> None:
    try:
        import assemblyai
    except ImportError:
        return
    try:
        from .llm.assemblyai import patch_assemblyai
        patch_assemblyai(ek)
    except Exception:
        pass


def _try_patch_bedrock(ek: "EvalKitClient") -> None:
    try:
        import botocore.client as _bc
    except ImportError:
        return
    try:
        _patch_bedrock_class(ek, _bc.BaseClient)
        _mark_owned("opentelemetry.instrumentation.botocore.bedrock-runtime")
    except Exception:
        pass


def _patch_bedrock_class(ek: "EvalKitClient", cls) -> None:
    if getattr(getattr(cls, "_make_api_call", None), "_evalkit_bedrock", False):
        return
    from .llm.bedrock import handles as _bedrock_handles, trace_bedrock_call

    original = cls._make_api_call

    def traced(self, operation_name, api_params):
        service = getattr(getattr(self, "meta", None), "service_model", None)
        service_name = getattr(service, "service_name", "") if service else ""
        if not _bedrock_handles(service_name, operation_name) or getattr(self, "_evalkit", False):
            return original(self, operation_name, api_params)
        return trace_bedrock_call(
            ek, service_name, operation_name, api_params,
            lambda: original(self, operation_name, api_params),
        )

    traced._evalkit_bedrock = True
    cls._make_api_call = traced


def _try_patch_aiobotocore(ek: "EvalKitClient") -> None:
    try:
        import aiobotocore.client as _abc
    except ImportError:
        return
    try:
        _patch_aio_bedrock_class(ek, _abc.AioBaseClient)
    except Exception:
        pass


def _patch_aio_bedrock_class(ek: "EvalKitClient", cls) -> None:
    if getattr(getattr(cls, "_make_api_call", None), "_evalkit_bedrock", False):
        return
    from .llm.bedrock import handles as _bedrock_handles, trace_bedrock_call_async

    original = cls._make_api_call

    async def traced(self, operation_name, api_params):
        service = getattr(getattr(self, "meta", None), "service_model", None)
        service_name = getattr(service, "service_name", "") if service else ""
        if not _bedrock_handles(service_name, operation_name) or getattr(self, "_evalkit", False):
            return await original(self, operation_name, api_params)
        return await trace_bedrock_call_async(
            ek, service_name, operation_name, api_params,
            lambda: original(self, operation_name, api_params),
        )

    traced._evalkit_bedrock = True
    cls._make_api_call = traced


_MISTRAL_CHAT_MODULES = ("mistralai.client.chat", "mistralai.chat")
_MISTRAL_CHAT_METHODS = ("complete", "complete_async", "stream", "stream_async")


def _resolve_mistral_chat_cls(mistral_cls):
    import importlib

    mapping = getattr(mistral_cls, "_sub_sdk_map", None)
    if isinstance(mapping, dict):
        entry = mapping.get("chat")
        if entry:
            try:
                cls = getattr(importlib.import_module(entry[0]), entry[1], None)
            except ImportError:
                cls = None
            if cls is not None:
                return cls
    chat = getattr(mistral_cls, "chat", None)
    if chat is not None and not isinstance(chat, type):
        getter = getattr(chat, "fget", None)
        if getter is None:
            return type(chat)
    for path in _MISTRAL_CHAT_MODULES:
        try:
            module = importlib.import_module(path)
        except ImportError:
            continue
        cls = getattr(module, "Chat", None)
        if cls is not None:
            return cls
    return None


def _try_patch_mistral(ek: "EvalKitClient") -> None:
    try:
        from mistralai import Mistral as _Mistral
    except ImportError:
        return
    try:
        chat_cls = _resolve_mistral_chat_cls(_Mistral)
        if chat_cls is not None:
            _patch_mistral_chat_class(ek, chat_cls)
    except Exception:
        pass


def _patch_mistral_chat_class(ek: "EvalKitClient", chat_cls) -> None:
    from .llm.mistral import patch_mistral_chat

    _lazy_instance_patch(chat_cls, _MISTRAL_CHAT_METHODS, lambda chat: patch_mistral_chat(ek, chat))


def _mark_owned(*scopes: str) -> None:
    try:
        from ..core.otel_provider import mark_owned_scope
        mark_owned_scope(*scopes)
    except Exception:
        pass


def _try_patch_databases(ek: "EvalKitClient") -> None:
    import importlib
    _otel_scopes = {
        "redis": ("opentelemetry.instrumentation.redis",),
        "pymongo": ("opentelemetry.instrumentation.pymongo",),
        "asyncpg": ("opentelemetry.instrumentation.asyncpg",),
        "psycopg": ("opentelemetry.instrumentation.psycopg",
                    "opentelemetry.instrumentation.psycopg2"),
        "pymysql": ("opentelemetry.instrumentation.pymysql",),
    }
    try:
        from .db._common import set_stmt_max
        set_stmt_max(ek._opts.max_body_bytes)
    except Exception:
        pass
    for mod_name, fn_name in (
        ("redis", "patch_redis"),
        ("pymongo", "patch_pymongo"),
        ("asyncpg", "patch_asyncpg"),
        ("psycopg", "patch_psycopg"),
        ("pymysql", "patch_pymysql"),
    ):
        try:
            mod = importlib.import_module(f".db.{mod_name}", __package__)
            getattr(mod, fn_name)(ek)
            _mark_owned(*_otel_scopes[mod_name])
        except Exception:
            pass


def _try_patch_langchain(ek: "EvalKitClient") -> None:
    try:
        import langchain_core
    except ImportError:
        return
    try:
        from .llm.langchain import patch_langchain
        patch_langchain(ek)
    except Exception:
        pass


def setup_auto_instrumentation(
    client: "EvalKitClient",
    logging: bool = True,
    requests: bool = True,
    httpx: bool = True,
    aiohttp: bool = True,
) -> None:
    try:
        from .runtime.threads import patch_threads
        patch_threads()
    except Exception:
        pass

    if logging:
        try:
            from .runtime.logging import install_logging_handler
            install_logging_handler(client)
        except Exception:
            pass
    if requests:
        try:
            from .http.requests import patch_requests
            patch_requests(client)
            _mark_owned("opentelemetry.instrumentation.requests")
        except Exception:
            pass
    if httpx:
        try:
            from .http.httpx import patch_httpx
            patch_httpx(client)
            _mark_owned("opentelemetry.instrumentation.httpx")
        except Exception:
            pass
    if aiohttp:
        try:
            from .http.aiohttp import patch_aiohttp
            patch_aiohttp(client)
            _mark_owned("opentelemetry.instrumentation.aiohttp_client")
        except Exception:
            pass

    _try_patch_openai(client)
    _try_patch_openai_responses(client)
    _try_patch_anthropic(client)
    _try_patch_google_genai(client)
    _try_patch_google_ai(client)
    _try_patch_google_adk(client)
    _try_patch_cohere(client)
    _try_patch_bedrock(client)
    _try_patch_aiobotocore(client)
    _try_patch_mistral(client)
    _try_patch_groq(client)
    _try_patch_litellm(client)
    _try_patch_ollama(client)
    _try_patch_pinecone(client)
    _try_patch_qdrant(client)
    _try_patch_chroma(client)
    _try_patch_weaviate(client)
    _try_patch_llamaindex(client)
    _try_patch_openai_audio(client)
    _try_patch_elevenlabs(client)
    _try_patch_assemblyai(client)
    _try_patch_langchain(client)
    _try_patch_databases(client)
    _try_patch_claude_agent_sdk(client)
    _try_patch_google_live(client)
    _try_patch_openai_realtime(client)
    try:
        from .mcp import patch_mcp
        patch_mcp(client)
    except Exception:
        pass


def _try_patch_claude_agent_sdk(ek: "EvalKitClient") -> None:
    try:
        from .llm.claude_agent_sdk import patch_claude_agent_sdk
        patch_claude_agent_sdk(ek)
    except Exception:
        pass


def _try_patch_google_live(ek: "EvalKitClient") -> None:
    try:
        from google.genai import live as _live
    except ImportError:
        return
    try:
        from .llm.google_live import patch_google_genai_live
        patch_google_genai_live(ek)
    except Exception:
        pass


def _try_patch_openai_realtime(ek: "EvalKitClient") -> None:
    try:
        import openai
    except ImportError:
        return
    try:
        from .llm.openai_realtime import patch_openai_realtime
        patch_openai_realtime(ek)
    except Exception:
        pass
