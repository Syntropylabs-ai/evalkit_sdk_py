from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ._streaming import _AsyncStreamTracer, _SyncStreamTracer
from ._toolcalls import from_anthropic_blocks, from_openai_chat, from_openai_responses
from .llm.anthropic import _extract_thinking
from .llm.bedrock import _extract_reasoning as _extract_bedrock_reasoning
from ..core.client import _exc_attrs

if TYPE_CHECKING:
    from ..core.client import EvalKitClient


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
    from ..core.context import SpanContext, _current_span, current_span_ctx, get_or_create_ambient_trace_id
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
        prompt = "\n".join(f"{m.get('role','')}: {m.get('content','')}" for m in messages)

        if kwargs.get("stream"):
            def on_stream_done(
                completion: str, tokens_in: int, tokens_out: int,
                thinking: str, reasoning_tokens: int, has_redacted_thinking: bool,
                ttft_ms=None, cache_read_tokens: int = 0, cache_creation_tokens: int = 0,
            ) -> None:
                end_ts = _now_iso()
                latency = int(time.time() * 1000 - start_ms)
                stream_attrs = {**ek._meta_attrs(), "gen_ai.request.model": model, "gen_ai.streaming": True}
                if reasoning_tokens:
                    stream_attrs["gen_ai.usage.reasoning_tokens"] = reasoning_tokens
                if ttft_ms is not None:
                    stream_attrs["gen_ai.server.time_to_first_token_ms"] = ttft_ms
                ek.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call", status="OK",
                    operation="chat.completions.create",
                    service_name=ek._opts.service_name, provider=provider, model=model,
                    prompt=prompt, completion=completion,
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    cache_read_tokens=cache_read_tokens or None, cache_creation_tokens=cache_creation_tokens or None,
                    latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
                    attributes=stream_attrs,
                    environment=ek._opts.environment, app_version=ek._opts.app_version,
                    sdk_version=SDK_VERSION,
                ))
            return _SyncStreamTracer(original(self, *args, **kwargs), on_stream_done, start_ms=start_ms)

        span_ctx = SpanContext(trace_id=trace_id, span_id=span_id)
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
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model, **_exc_attrs(exc)},
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
            attrs = {**ek._meta_attrs(), "gen_ai.request.model": model,
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
    from ..core.context import SpanContext, _current_span, current_span_ctx, get_or_create_ambient_trace_id
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
        prompt = "\n".join(f"{m.get('role','')}: {m.get('content','')}" for m in messages)

        if kwargs.get("stream"):
            def on_stream_done(
                completion: str, tokens_in: int, tokens_out: int,
                thinking: str, reasoning_tokens: int, has_redacted_thinking: bool,
                ttft_ms=None, cache_read_tokens: int = 0, cache_creation_tokens: int = 0,
            ) -> None:
                end_ts = _now_iso()
                latency = int(time.time() * 1000 - start_ms)
                stream_attrs = {**ek._meta_attrs(), "gen_ai.request.model": model, "gen_ai.streaming": True}
                if reasoning_tokens:
                    stream_attrs["gen_ai.usage.reasoning_tokens"] = reasoning_tokens
                if ttft_ms is not None:
                    stream_attrs["gen_ai.server.time_to_first_token_ms"] = ttft_ms
                ek.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call", status="OK",
                    operation="chat.completions.create",
                    service_name=ek._opts.service_name, provider=provider, model=model,
                    prompt=prompt, completion=completion,
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    cache_read_tokens=cache_read_tokens or None, cache_creation_tokens=cache_creation_tokens or None,
                    latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
                    attributes=stream_attrs,
                    environment=ek._opts.environment, app_version=ek._opts.app_version,
                    sdk_version=SDK_VERSION,
                ))
            raw = await original(self, *args, **kwargs)
            return _AsyncStreamTracer(raw, on_stream_done, start_ms=start_ms)

        span_ctx = SpanContext(trace_id=trace_id, span_id=span_id)
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
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model, **_exc_attrs(exc)},
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
            attrs = {**ek._meta_attrs(), "gen_ai.request.model": model,
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
    original_stream = getattr(cls, "stream", None)
    if original_stream is None or getattr(original_stream, "_evalkit", False):
        return
    import json as _json
    import time
    from ..core.client import _new_span_id, _now_iso
    from ..core.context import SpanContext, _current_span, current_span_ctx, get_or_create_ambient_trace_id
    from ..core.types import SpanEvent, TraceEnvelope
    from .._version import SDK_VERSION

    def traced_stream(self, *args, **kwargs):
        ctx_parent = current_span_ctx()
        trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()
        model = kwargs.get("model", "unknown")
        system = kwargs.get("system")
        messages_arg = kwargs.get("messages", [])
        parts = ([f"system: {system}"] if system else []) + [
            f"{(m.get('role','') if isinstance(m, dict) else getattr(m,'role',''))}: "
            f"{(m.get('content','') if isinstance(m, dict) else getattr(m,'content',''))}"
            for m in messages_arg
        ]
        prompt = "\n".join(parts)

        class _TracedStreamManager:
            def __enter__(self_inner):
                span_ctx = SpanContext(trace_id=trace_id, span_id=span_id)
                self_inner._token = _current_span.set(span_ctx)
                self_inner._mgr = original_stream(self, *args, **kwargs)
                self_inner._underlying = self_inner._mgr.__enter__()
                _current_span.reset(self_inner._token)
                self_inner._token = None
                return self_inner._underlying

            def __exit__(self_inner, exc_type, exc_val, exc_tb):
                result = self_inner._mgr.__exit__(exc_type, exc_val, exc_tb)
                end_ts = _now_iso()
                latency = int(time.time() * 1000 - start_ms)
                completion = ""
                thinking_text = ""
                has_redacted_thinking = False
                tokens_in = tokens_out = 0
                events = []
                try:
                    final_msg = self_inner._underlying.get_final_message()
                    content = getattr(final_msg, "content", [])
                    for block in content:
                        if hasattr(block, "text"):
                            completion += block.text
                    thinking_text, has_redacted_thinking = _extract_thinking(content)
                    events = from_anthropic_blocks(content, end_ts)
                    usage = getattr(final_msg, "usage", None)
                    tokens_in = getattr(usage, "input_tokens", 0) or 0
                    tokens_out = getattr(usage, "output_tokens", 0) or 0
                except Exception:
                    pass
                status = "ERROR" if exc_type else "OK"
                stream_attrs = {**ek._meta_attrs(), "gen_ai.request.model": model, "gen_ai.streaming": True}
                if thinking_text:
                    stream_attrs["gen_ai.response.thinking"] = thinking_text[:32768]
                if has_redacted_thinking:
                    stream_attrs["gen_ai.response.has_redacted_thinking"] = True
                ek.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call", status=status,
                    operation="messages.stream",
                    service_name=ek._opts.service_name, provider="anthropic", model=model,
                    prompt=prompt[:32768], completion=completion[:32768],
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
                    events=events or None,
                    attributes=stream_attrs,
                    environment=ek._opts.environment, app_version=ek._opts.app_version,
                    sdk_version=SDK_VERSION,
                ))
                return result

        return _TracedStreamManager()

    traced_stream._evalkit = True
    cls.stream = traced_stream


def _patch_anthropic_stream_async(ek: "EvalKitClient", cls) -> None:
    original_stream = getattr(cls, "stream", None)
    if original_stream is None or getattr(original_stream, "_evalkit", False):
        return
    import json as _json
    import time
    from ..core.client import _new_span_id, _now_iso
    from ..core.context import SpanContext, _current_span, current_span_ctx, get_or_create_ambient_trace_id
    from ..core.types import SpanEvent, TraceEnvelope
    from .._version import SDK_VERSION

    def traced_stream_async(self, *args, **kwargs):
        ctx_parent = current_span_ctx()
        trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()
        model = kwargs.get("model", "unknown")
        system = kwargs.get("system")
        messages_arg = kwargs.get("messages", [])
        parts = ([f"system: {system}"] if system else []) + [
            f"{(m.get('role','') if isinstance(m, dict) else getattr(m,'role',''))}: "
            f"{(m.get('content','') if isinstance(m, dict) else getattr(m,'content',''))}"
            for m in messages_arg
        ]
        prompt = "\n".join(parts)

        class _TracedAsyncStreamManager:
            async def __aenter__(self_inner):
                span_ctx = SpanContext(trace_id=trace_id, span_id=span_id)
                self_inner._token = _current_span.set(span_ctx)
                self_inner._mgr = original_stream(self, *args, **kwargs)
                self_inner._underlying = await self_inner._mgr.__aenter__()
                _current_span.reset(self_inner._token)
                self_inner._token = None
                return self_inner._underlying

            async def __aexit__(self_inner, exc_type, exc_val, exc_tb):
                result = await self_inner._mgr.__aexit__(exc_type, exc_val, exc_tb)
                end_ts = _now_iso()
                latency = int(time.time() * 1000 - start_ms)
                completion = ""
                thinking_text = ""
                has_redacted_thinking = False
                tokens_in = tokens_out = 0
                events = []
                try:
                    final_msg = await self_inner._underlying.get_final_message()
                    content = getattr(final_msg, "content", [])
                    for block in content:
                        if hasattr(block, "text"):
                            completion += block.text
                    thinking_text, has_redacted_thinking = _extract_thinking(content)
                    events = from_anthropic_blocks(content, end_ts)
                    usage = getattr(final_msg, "usage", None)
                    tokens_in = getattr(usage, "input_tokens", 0) or 0
                    tokens_out = getattr(usage, "output_tokens", 0) or 0
                except Exception:
                    pass
                status = "ERROR" if exc_type else "OK"
                stream_attrs = {**ek._meta_attrs(), "gen_ai.request.model": model, "gen_ai.streaming": True}
                if thinking_text:
                    stream_attrs["gen_ai.response.thinking"] = thinking_text[:32768]
                if has_redacted_thinking:
                    stream_attrs["gen_ai.response.has_redacted_thinking"] = True
                ek.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call", status=status,
                    operation="messages.stream",
                    service_name=ek._opts.service_name, provider="anthropic", model=model,
                    prompt=prompt[:32768], completion=completion[:32768],
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
                    events=events or None,
                    attributes=stream_attrs,
                    environment=ek._opts.environment, app_version=ek._opts.app_version,
                    sdk_version=SDK_VERSION,
                ))
                return result

        return _TracedAsyncStreamManager()

    traced_stream_async._evalkit = True
    cls.stream = traced_stream_async


def _patch_anthropic_class_sync(ek: "EvalKitClient", cls) -> None:
    if getattr(cls.create, "_evalkit", False):
        return
    import time
    from ..core.client import _new_span_id, _now_iso
    from ..core.context import SpanContext, _current_span, current_span_ctx, get_or_create_ambient_trace_id
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
        parts = ([f"system: {system}"] if system else []) + [
            f"{m.get('role','')}: {m.get('content','')}" for m in messages
        ]
        prompt = "\n".join(parts)

        if kwargs.get("stream"):
            def on_stream_done(
                completion: str, tokens_in: int, tokens_out: int,
                thinking: str, reasoning_tokens: int, has_redacted_thinking: bool,
                ttft_ms=None, cache_read_tokens: int = 0, cache_creation_tokens: int = 0,
            ) -> None:
                end_ts = _now_iso()
                latency = int(time.time() * 1000 - start_ms)
                stream_attrs = {**ek._meta_attrs(), "gen_ai.request.model": model, "gen_ai.streaming": True}
                if thinking:
                    stream_attrs["gen_ai.response.thinking"] = thinking[:32768]
                if has_redacted_thinking:
                    stream_attrs["gen_ai.response.has_redacted_thinking"] = True
                if ttft_ms is not None:
                    stream_attrs["gen_ai.server.time_to_first_token_ms"] = ttft_ms
                ek.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call", status="OK",
                    operation="messages.create",
                    service_name=ek._opts.service_name, provider="anthropic", model=model,
                    prompt=prompt, completion=completion,
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    cache_read_tokens=cache_read_tokens or None, cache_creation_tokens=cache_creation_tokens or None,
                    latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
                    attributes=stream_attrs,
                    environment=ek._opts.environment, app_version=ek._opts.app_version,
                    sdk_version=SDK_VERSION,
                ))
            return _SyncStreamTracer(original(self, *args, **kwargs), on_stream_done, is_anthropic=True, start_ms=start_ms)

        span_ctx = SpanContext(trace_id=trace_id, span_id=span_id)
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
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model, **_exc_attrs(exc)},
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
    from ..core.context import SpanContext, _current_span, current_span_ctx, get_or_create_ambient_trace_id
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
        parts = ([f"system: {system}"] if system else []) + [
            f"{m.get('role','')}: {m.get('content','')}" for m in messages
        ]
        prompt = "\n".join(parts)

        if kwargs.get("stream"):
            def on_stream_done(
                completion: str, tokens_in: int, tokens_out: int,
                thinking: str, reasoning_tokens: int, has_redacted_thinking: bool,
                ttft_ms=None, cache_read_tokens: int = 0, cache_creation_tokens: int = 0,
            ) -> None:
                end_ts = _now_iso()
                latency = int(time.time() * 1000 - start_ms)
                stream_attrs = {**ek._meta_attrs(), "gen_ai.request.model": model, "gen_ai.streaming": True}
                if thinking:
                    stream_attrs["gen_ai.response.thinking"] = thinking[:32768]
                if has_redacted_thinking:
                    stream_attrs["gen_ai.response.has_redacted_thinking"] = True
                if ttft_ms is not None:
                    stream_attrs["gen_ai.server.time_to_first_token_ms"] = ttft_ms
                ek.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call", status="OK",
                    operation="messages.create",
                    service_name=ek._opts.service_name, provider="anthropic", model=model,
                    prompt=prompt, completion=completion,
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    cache_read_tokens=cache_read_tokens or None, cache_creation_tokens=cache_creation_tokens or None,
                    latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
                    attributes=stream_attrs,
                    environment=ek._opts.environment, app_version=ek._opts.app_version,
                    sdk_version=SDK_VERSION,
                ))
            raw = await original(self, *args, **kwargs)
            return _AsyncStreamTracer(raw, on_stream_done, is_anthropic=True, start_ms=start_ms)

        span_ctx = SpanContext(trace_id=trace_id, span_id=span_id)
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
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model, **_exc_attrs(exc)},
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


def _try_patch_google_ai(ek: "EvalKitClient") -> None:
    try:
        import google.generativeai as _genai
    except ImportError:
        return
    try:
        cls = _genai.GenerativeModel
        if not getattr(getattr(cls, "generate_content", None), "_evalkit", False):
            _patch_google_genai_legacy_class(ek, cls)
    except Exception:
        pass


def _patch_google_genai_legacy_class(ek: "EvalKitClient", cls) -> None:
    if getattr(getattr(cls, "generate_content", None), "_evalkit", False):
        return
    import time
    from ..core.client import _new_span_id, _now_iso
    from ..core.context import SpanContext, _current_span, current_span_ctx, get_or_create_ambient_trace_id
    from ..core.types import TraceEnvelope
    from .._version import SDK_VERSION

    original = cls.generate_content

    def traced(self, *args, **kwargs):
        ctx_parent = current_span_ctx()
        trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()
        model_name = getattr(self, "model_name", "unknown")
        contents = args[0] if args else kwargs.get("contents", "")
        prompt = str(contents)[:2000]
        span_ctx = SpanContext(trace_id=trace_id, span_id=span_id)
        token = _current_span.set(span_ctx)
        try:
            result = original(self, *args, **kwargs)
        except Exception as exc:
            end_ts = _now_iso()
            ek.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="ERROR", status_message=str(exc),
                operation="generate_content",
                service_name=ek._opts.service_name, provider="google-ai", model=model_name,
                prompt=prompt, latency_ms=int(time.time() * 1000 - start_ms),
                start_time=ts, end_time=end_ts, created_at=ts,
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model_name, **_exc_attrs(exc)},
                environment=ek._opts.environment, app_version=ek._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
            raise
        finally:
            _current_span.reset(token)
        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)
        try:
            completion = result.text or ""
        except Exception:
            completion = ""
        ek.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call", status="OK",
            operation="generate_content",
            service_name=ek._opts.service_name, provider="google-ai", model=model_name,
            prompt=prompt, completion=completion,
            latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
            attributes={**ek._meta_attrs(), "gen_ai.request.model": model_name},
            environment=ek._opts.environment, app_version=ek._opts.app_version,
            sdk_version=SDK_VERSION,
        ))
        return result

    traced._evalkit = True
    cls.generate_content = traced


def _try_patch_cohere(ek: "EvalKitClient") -> None:
    try:
        import cohere as _co
    except ImportError:
        return
    try:
        cls = _co.Client
        if not getattr(getattr(cls, "chat", None), "_evalkit", False):
            _patch_cohere_class(ek, cls)
    except Exception:
        pass


def _patch_cohere_class(ek: "EvalKitClient", cls) -> None:
    if getattr(getattr(cls, "chat", None), "_evalkit", False):
        return
    import time
    from ..core.client import _new_span_id, _now_iso
    from ..core.context import SpanContext, _current_span, current_span_ctx, get_or_create_ambient_trace_id
    from ..core.types import TraceEnvelope
    from .._version import SDK_VERSION

    original = cls.chat

    def traced(self, *args, **kwargs):
        ctx_parent = current_span_ctx()
        trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()
        model = kwargs.get("model", "command-r-plus")
        message = kwargs.get("message", args[0] if args else "")
        span_ctx = SpanContext(trace_id=trace_id, span_id=span_id)
        token = _current_span.set(span_ctx)
        try:
            result = original(self, *args, **kwargs)
        except Exception as exc:
            end_ts = _now_iso()
            ek.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="ERROR", status_message=str(exc),
                operation="chat",
                service_name=ek._opts.service_name, provider="cohere", model=model,
                prompt=str(message)[:2000], latency_ms=int(time.time() * 1000 - start_ms),
                start_time=ts, end_time=end_ts, created_at=ts,
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model, **_exc_attrs(exc)},
                environment=ek._opts.environment, app_version=ek._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
            raise
        finally:
            _current_span.reset(token)
        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)
        completion = getattr(result, "text", "") or ""
        usage = getattr(result, "usage", None)
        tokens_in = getattr(getattr(usage, "tokens", None), "input_tokens", 0) or 0
        tokens_out = getattr(getattr(usage, "tokens", None), "output_tokens", 0) or 0
        ek.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call", status="OK",
            operation="chat",
            service_name=ek._opts.service_name, provider="cohere", model=model,
            prompt=str(message)[:2000], completion=completion,
            tokens_in=tokens_in, tokens_out=tokens_out,
            latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
            attributes={**ek._meta_attrs(), "gen_ai.request.model": model},
            environment=ek._opts.environment, app_version=ek._opts.app_version,
            sdk_version=SDK_VERSION,
        ))
        return result

    traced._evalkit = True
    cls.chat = traced


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
    from ..core.context import SpanContext, _current_span, current_span_ctx, get_or_create_ambient_trace_id
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

            span_ctx = SpanContext(trace_id=trace_id, span_id=span_id)
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
                    attributes={**ek._meta_attrs(), "gen_ai.request.model": model, **_exc_attrs(exc)},
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
                    attributes={**ek._meta_attrs(), "gen_ai.request.model": model,
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

        span_ctx = SpanContext(trace_id=trace_id, span_id=span_id)
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
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model, **_exc_attrs(exc)},
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
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model,
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
    from ..core.context import SpanContext, _current_span, current_span_ctx, get_or_create_ambient_trace_id
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

        span_ctx = SpanContext(trace_id=trace_id, span_id=span_id)
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
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model, **_exc_attrs(exc)},
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
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model,
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
    except Exception:
        pass


def _patch_bedrock_class(ek: "EvalKitClient", cls) -> None:
    if getattr(getattr(cls, "_make_api_call", None), "_evalkit_bedrock", False):
        return
    import time
    from ..core.client import _new_span_id, _now_iso
    from ..core.context import SpanContext, _current_span, current_span_ctx, get_or_create_ambient_trace_id
    from ..core.types import TraceEnvelope
    from .._version import SDK_VERSION

    original = cls._make_api_call

    def traced(self, operation_name, api_params):
        service = getattr(getattr(self, "meta", None), "service_model", None)
        service_name = getattr(service, "service_name", "") if service else ""
        if service_name != "bedrock-runtime" or operation_name != "Converse":
            return original(self, operation_name, api_params)

        ctx_parent = current_span_ctx()
        trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()
        model = api_params.get("modelId", "unknown")
        messages = api_params.get("messages", [])
        prompt_parts = [
            f"{m.get('role','')}: {' '.join(c.get('text','') for c in m.get('content',[]) if isinstance(c,dict))}"
            for m in messages
        ]
        prompt = "\n".join(prompt_parts)[:2000]
        span_ctx = SpanContext(trace_id=trace_id, span_id=span_id)
        token = _current_span.set(span_ctx)
        try:
            result = original(self, operation_name, api_params)
        except Exception as exc:
            end_ts = _now_iso()
            ek.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="ERROR", status_message=str(exc),
                operation="Converse",
                service_name=ek._opts.service_name, provider="bedrock", model=model,
                prompt=prompt, latency_ms=int(time.time() * 1000 - start_ms),
                start_time=ts, end_time=end_ts, created_at=ts,
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model, **_exc_attrs(exc)},
                environment=ek._opts.environment, app_version=ek._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
            raise
        finally:
            _current_span.reset(token)
        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)
        out_msg = (result.get("output", {}) or {}).get("message", {}) or {}
        content = out_msg.get("content", []) or []
        completion = next((c.get("text", "") for c in content if isinstance(c, dict) and "text" in c), "")
        thinking_text, has_redacted_thinking = _extract_bedrock_reasoning(content)
        usage = result.get("usage", {}) or {}
        tokens_in = usage.get("inputTokens", 0) or 0
        tokens_out = usage.get("outputTokens", 0) or 0
        attrs = {**ek._meta_attrs(), "gen_ai.request.model": model,
                 "gen_ai.usage.total_tokens": tokens_in + tokens_out}
        if thinking_text:
            attrs["gen_ai.response.thinking"] = thinking_text
        if has_redacted_thinking:
            attrs["gen_ai.response.has_redacted_thinking"] = True
        ek.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call", status="OK",
            operation="Converse",
            service_name=ek._opts.service_name, provider="bedrock", model=model,
            prompt=prompt, completion=completion,
            tokens_in=tokens_in, tokens_out=tokens_out,
            latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
            attributes=attrs,
            environment=ek._opts.environment, app_version=ek._opts.app_version,
            sdk_version=SDK_VERSION,
        ))
        return result

    traced._evalkit_bedrock = True
    cls._make_api_call = traced


def _try_patch_mistral(ek: "EvalKitClient") -> None:
    try:
        from mistralai import Mistral as _Mistral
    except ImportError:
        return
    try:
        from .llm.mistral import patch_mistral, patch_async_mistral
        cls = _Mistral
        chat = getattr(cls, "chat", None)
        if chat is None:
            return
        _patch_mistral_class_sync(ek, cls)
        _patch_mistral_class_async(ek, cls)
    except Exception:
        pass


def _patch_mistral_class_sync(ek: "EvalKitClient", cls) -> None:
    from .llm.mistral import (
        _extract_mistral_completion, _extract_mistral_prompt,
        _extract_mistral_tool_events, _extract_mistral_usage,
    )
    import time
    from ..core.client import _new_span_id, _now_iso
    from ..core.context import SpanContext, _current_span, current_span_ctx, get_or_create_ambient_trace_id
    from ..core.types import TraceEnvelope
    from .._version import SDK_VERSION

    try:
        chat_cls = type(cls.chat.fget(cls()) if hasattr(cls.chat, "fget") else cls.chat)
    except Exception:
        return

    complete_fn = getattr(chat_cls, "complete", None)
    if complete_fn is None or getattr(complete_fn, "_evalkit", False):
        return

    original = complete_fn

    def traced(self, **kwargs):
        ctx_parent = current_span_ctx()
        trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()
        model = kwargs.get("model", "mistral-small")
        prompt_text = _extract_mistral_prompt(kwargs.get("messages", []))[:32768]
        span_ctx = SpanContext(trace_id=trace_id, span_id=span_id)
        token = _current_span.set(span_ctx)
        try:
            result = original(self, **kwargs)
        except Exception as exc:
            end_ts = _now_iso()
            ek.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="ERROR", status_message=str(exc),
                operation="chat.complete", service_name=ek._opts.service_name,
                provider="mistral", model=model, prompt=prompt_text,
                latency_ms=int(time.time() * 1000 - start_ms),
                start_time=ts, end_time=end_ts, created_at=ts,
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model, **_exc_attrs(exc)},
                environment=ek._opts.environment, app_version=ek._opts.app_version,
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
        ek.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call", status="OK", operation="chat.complete",
            service_name=ek._opts.service_name, provider="mistral", model=model,
            prompt=prompt_text, completion=completion[:32768],
            tokens_in=tokens_in, tokens_out=tokens_out,
            latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
            events=events or None,
            attributes={**ek._meta_attrs(), "gen_ai.request.model": model,
                        "gen_ai.usage.total_tokens": tokens_in + tokens_out},
            environment=ek._opts.environment, app_version=ek._opts.app_version,
            sdk_version=SDK_VERSION,
        ))
        return result

    traced._evalkit = True
    chat_cls.complete = traced


def _patch_mistral_class_async(ek: "EvalKitClient", cls) -> None:
    pass


def _try_patch_databases(ek: "EvalKitClient") -> None:
    import importlib
    import importlib.util
    try:
        from ..core.otel_provider import otel_active
        bridge = otel_active()
    except Exception:
        bridge = False
    _otel_pkg = {
        "redis": "opentelemetry.instrumentation.redis",
        "pymongo": "opentelemetry.instrumentation.pymongo",
        "asyncpg": "opentelemetry.instrumentation.asyncpg",
        "psycopg": "opentelemetry.instrumentation.psycopg",
        "pymysql": "opentelemetry.instrumentation.pymysql",
    }
    for mod_name, fn_name in (
        ("redis", "patch_redis"),
        ("pymongo", "patch_pymongo"),
        ("asyncpg", "patch_asyncpg"),
        ("psycopg", "patch_psycopg"),
        ("pymysql", "patch_pymysql"),
    ):
        try:
            if bridge and importlib.util.find_spec(_otel_pkg[mod_name]) is not None:
                continue
        except Exception:
            pass
        try:
            mod = importlib.import_module(f".db.{mod_name}", __package__)
            getattr(mod, fn_name)(ek)
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
    import importlib.util as _ilu

    try:
        from ..core.otel_provider import otel_active
        _bridge = otel_active()
    except Exception:
        _bridge = False

    def _otel_covers(pkg: str) -> bool:
        if not _bridge:
            return False
        try:
            return _ilu.find_spec(pkg) is not None
        except Exception:
            return False

    if requests and not _otel_covers("opentelemetry.instrumentation.requests"):
        try:
            from .http.requests import patch_requests
            patch_requests(client)
        except Exception:
            pass
    if httpx and not _otel_covers("opentelemetry.instrumentation.httpx"):
        try:
            from .http.httpx import patch_httpx
            patch_httpx(client)
        except Exception:
            pass
    if aiohttp and not _otel_covers("opentelemetry.instrumentation.aiohttp_client"):
        try:
            from .http.aiohttp import patch_aiohttp
            patch_aiohttp(client)
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
