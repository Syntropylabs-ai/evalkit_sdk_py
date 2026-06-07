"""LiteLLM instrumentation.

LiteLLM is a very common aggregator in AI apps (`litellm.completion(model="gpt-4o"...)`).
For most providers LiteLLM issues its own HTTP requests rather than going through the
provider SDKs, so without a dedicated patch those calls show up only as generic
``http_call`` spans with no prompt/completion/tokens. This patches the module-level
``completion`` / ``acompletion`` functions so they emit rich ``llm_call`` spans.

Degrades to a no-op if litellm is not installed and never raises into the caller.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ..._version import SDK_VERSION
from ...core.client import _exc_attrs, _new_span_id, _now_iso
from ...core.context import (
    SpanContext,
    _ambient_trace_id,
    _current_span,
    get_or_create_ambient_trace_id,
)
from ...core.types import TraceEnvelope
from .._streaming import _AsyncStreamTracer, _SyncStreamTracer
from .._toolcalls import from_openai_chat

if TYPE_CHECKING:
    from ...core.client import EvalKitClient


def _messages_to_prompt(messages: Any) -> str:
    try:
        return "\n".join(
            f"{m.get('role', '')}: {m.get('content', '')}"
            for m in (messages or [])
            if isinstance(m, dict)
        )
    except Exception:
        return str(messages)[:4096]


def _completion_text(result: Any) -> str:
    try:
        choice = result.choices[0]
        return (getattr(choice.message, "content", None) or "") if choice else ""
    except Exception:
        return ""


def _usage(result: Any):
    usage = getattr(result, "usage", None)
    return (
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
    )


def _tool_events(result: Any, ts: str):
    return from_openai_chat(result, ts)


def patch_litellm(client: "EvalKitClient") -> None:
    try:
        import litellm  # type: ignore
    except ImportError:
        return

    _patch_sync(client, litellm)
    _patch_async(client, litellm)


def _patch_sync(ek: "EvalKitClient", litellm: Any) -> None:
    fn = getattr(litellm, "completion", None)
    if fn is None or getattr(fn, "_evalkit", False):
        return
    original = fn

    def traced(*args: Any, **kwargs: Any) -> Any:
        ctx_parent = _current_span.get()
        trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
        _ambient_trace_id.set(trace_id)
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()
        model = str(kwargs.get("model") or (args[0] if args else "unknown"))
        prompt = _messages_to_prompt(kwargs.get("messages", []))

        if kwargs.get("stream"):
            def on_done(completion: str, tin: int, tout: int) -> None:
                ek.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call", status="OK", operation="litellm.completion",
                    service_name=ek._opts.service_name, provider="litellm", model=model,
                    prompt=prompt, completion=completion, tokens_in=tin, tokens_out=tout,
                    latency_ms=int(time.time() * 1000 - start_ms), start_time=ts, end_time=_now_iso(),
                    created_at=ts, attributes={**ek._meta_attrs(), "gen_ai.request.model": model, "gen_ai.streaming": True},
                    environment=ek._opts.environment, app_version=ek._opts.app_version, sdk_version=SDK_VERSION,
                ))
            return _SyncStreamTracer(original(*args, **kwargs), on_done)

        span_ctx = SpanContext(trace_id=trace_id, span_id=span_id)
        token = _current_span.set(span_ctx)
        try:
            result = original(*args, **kwargs)
        except Exception as exc:
            ek.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="ERROR", status_message=str(exc), operation="litellm.completion",
                service_name=ek._opts.service_name, provider="litellm", model=model, prompt=prompt,
                latency_ms=int(time.time() * 1000 - start_ms), start_time=ts, end_time=_now_iso(), created_at=ts,
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model, **_exc_attrs(exc)},
                environment=ek._opts.environment, app_version=ek._opts.app_version, sdk_version=SDK_VERSION,
            ))
            raise
        finally:
            _current_span.reset(token)

        end_ts = _now_iso()
        tin, tout = _usage(result)
        attrs = {**ek._meta_attrs(), "gen_ai.request.model": model, "gen_ai.usage.total_tokens": tin + tout}
        ek.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call", status="OK", operation="litellm.completion",
            service_name=ek._opts.service_name, provider="litellm", model=model,
            prompt=prompt, completion=_completion_text(result), tokens_in=tin, tokens_out=tout,
            latency_ms=int(time.time() * 1000 - start_ms), start_time=ts, end_time=end_ts, created_at=ts,
            events=_tool_events(result, end_ts) or None,
            attributes=attrs,
            environment=ek._opts.environment, app_version=ek._opts.app_version, sdk_version=SDK_VERSION,
        ))
        return result

    traced._evalkit = True  # type: ignore[attr-defined]
    litellm.completion = traced


def _patch_async(ek: "EvalKitClient", litellm: Any) -> None:
    fn = getattr(litellm, "acompletion", None)
    if fn is None or getattr(fn, "_evalkit", False):
        return
    original = fn

    async def traced_async(*args: Any, **kwargs: Any) -> Any:
        ctx_parent = _current_span.get()
        trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
        _ambient_trace_id.set(trace_id)
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()
        model = str(kwargs.get("model") or (args[0] if args else "unknown"))
        prompt = _messages_to_prompt(kwargs.get("messages", []))

        if kwargs.get("stream"):
            def on_done(completion: str, tin: int, tout: int) -> None:
                ek.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call", status="OK", operation="litellm.acompletion",
                    service_name=ek._opts.service_name, provider="litellm", model=model,
                    prompt=prompt, completion=completion, tokens_in=tin, tokens_out=tout,
                    latency_ms=int(time.time() * 1000 - start_ms), start_time=ts, end_time=_now_iso(),
                    created_at=ts, attributes={**ek._meta_attrs(), "gen_ai.request.model": model, "gen_ai.streaming": True},
                    environment=ek._opts.environment, app_version=ek._opts.app_version, sdk_version=SDK_VERSION,
                ))
            raw = await original(*args, **kwargs)
            return _AsyncStreamTracer(raw, on_done)

        span_ctx = SpanContext(trace_id=trace_id, span_id=span_id)
        token = _current_span.set(span_ctx)
        try:
            result = await original(*args, **kwargs)
        except Exception as exc:
            ek.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="ERROR", status_message=str(exc), operation="litellm.acompletion",
                service_name=ek._opts.service_name, provider="litellm", model=model, prompt=prompt,
                latency_ms=int(time.time() * 1000 - start_ms), start_time=ts, end_time=_now_iso(), created_at=ts,
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model, **_exc_attrs(exc)},
                environment=ek._opts.environment, app_version=ek._opts.app_version, sdk_version=SDK_VERSION,
            ))
            raise
        finally:
            _current_span.reset(token)

        end_ts = _now_iso()
        tin, tout = _usage(result)
        attrs = {**ek._meta_attrs(), "gen_ai.request.model": model, "gen_ai.usage.total_tokens": tin + tout}
        ek.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call", status="OK", operation="litellm.acompletion",
            service_name=ek._opts.service_name, provider="litellm", model=model,
            prompt=prompt, completion=_completion_text(result), tokens_in=tin, tokens_out=tout,
            latency_ms=int(time.time() * 1000 - start_ms), start_time=ts, end_time=end_ts, created_at=ts,
            events=_tool_events(result, end_ts) or None,
            attributes=attrs,
            environment=ek._opts.environment, app_version=ek._opts.app_version, sdk_version=SDK_VERSION,
        ))
        return result

    traced_async._evalkit = True  # type: ignore[attr-defined]
    litellm.acompletion = traced_async
