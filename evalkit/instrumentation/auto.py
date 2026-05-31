from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ._streaming import _AsyncStreamTracer, _SyncStreamTracer

if TYPE_CHECKING:
    from ..core.client import EvalKitClient


# ── OpenAI ────────────────────────────────────────────────────────────────────

def _try_patch_openai(ek: "EvalKitClient") -> None:
    try:
        import openai.resources.chat.completions.completions as _oc
    except ImportError:
        try:
            import openai.resources.chat.completions as _oc  # type: ignore[no-redef]
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


def _patch_openai_class_sync(ek: "EvalKitClient", cls) -> None:
    if getattr(cls.create, "_evalkit", False):
        return
    import time
    from ..core.client import _new_span_id, _now_iso
    from ..core.context import SpanContext, _current_span, get_or_create_ambient_trace_id
    from ..core.types import SpanEvent, TraceEnvelope
    from .._version import SDK_VERSION

    original = cls.create

    def traced(self, *args, **kwargs):
        ctx_parent = _current_span.get()
        trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        prompt = "\n".join(f"{m.get('role','')}: {m.get('content','')}" for m in messages)

        if kwargs.get("stream"):
            def on_stream_done(completion: str, tokens_in: int, tokens_out: int) -> None:
                end_ts = _now_iso()
                latency = int(time.time() * 1000 - start_ms)
                ek.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call", status="OK",
                    operation="chat.completions.create",
                    service_name=ek._opts.service_name, provider="openai", model=model,
                    prompt=prompt, completion=completion,
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
                    attributes={**ek._meta_attrs(), "gen_ai.request.model": model, "gen_ai.streaming": True},
                    environment=ek._opts.environment, app_version=ek._opts.app_version,
                    sdk_version=SDK_VERSION,
                ))
            return _SyncStreamTracer(original(self, *args, **kwargs), on_stream_done)

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
                service_name=ek._opts.service_name, provider="openai", model=model,
                prompt=prompt, latency_ms=int(time.time() * 1000 - start_ms),
                start_time=ts, end_time=_now_iso(), created_at=ts,
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model},
                environment=ek._opts.environment, app_version=ek._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
            raise
        finally:
            _current_span.reset(token)
        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)
        choice = result.choices[0] if result.choices else None
        completion = (choice.message.content or "") if choice else ""
        usage = getattr(result, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", 0) or 0
        tokens_out = getattr(usage, "completion_tokens", 0) or 0
        events = []
        if choice and getattr(choice.message, "tool_calls", None):
            for tc in choice.message.tool_calls:
                events.append(SpanEvent(
                    name="gen_ai.tool.call", timestamp=end_ts,
                    attributes={"gen_ai.tool.name": tc.function.name,
                                "gen_ai.tool.call.id": getattr(tc, "id", ""),
                                "gen_ai.tool.call.arguments": tc.function.arguments},
                ))
        ek.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call", status="OK",
            operation="chat.completions.create",
            service_name=ek._opts.service_name, provider="openai", model=model,
            prompt=prompt, completion=completion,
            tokens_in=tokens_in, tokens_out=tokens_out,
            latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
            events=events or None,
            attributes={**ek._meta_attrs(), "gen_ai.request.model": model,
                        "gen_ai.response.model": getattr(result, "model", model),
                        "gen_ai.usage.total_tokens": getattr(usage, "total_tokens", 0) or 0},
            environment=ek._opts.environment, app_version=ek._opts.app_version,
            sdk_version=SDK_VERSION,
        ))
        return result

    traced._evalkit = True  # type: ignore[attr-defined]
    cls.create = traced


def _patch_openai_class_async(ek: "EvalKitClient", cls) -> None:
    if getattr(cls.create, "_evalkit", False):
        return
    import time
    from ..core.client import _new_span_id, _now_iso
    from ..core.context import SpanContext, _current_span, get_or_create_ambient_trace_id
    from ..core.types import SpanEvent, TraceEnvelope
    from .._version import SDK_VERSION

    original = cls.create

    async def traced_async(self, *args, **kwargs):
        ctx_parent = _current_span.get()
        trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        start_ms = time.time() * 1000
        ts = _now_iso()
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        prompt = "\n".join(f"{m.get('role','')}: {m.get('content','')}" for m in messages)

        if kwargs.get("stream"):
            def on_stream_done(completion: str, tokens_in: int, tokens_out: int) -> None:
                end_ts = _now_iso()
                latency = int(time.time() * 1000 - start_ms)
                ek.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call", status="OK",
                    operation="chat.completions.create",
                    service_name=ek._opts.service_name, provider="openai", model=model,
                    prompt=prompt, completion=completion,
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
                    attributes={**ek._meta_attrs(), "gen_ai.request.model": model, "gen_ai.streaming": True},
                    environment=ek._opts.environment, app_version=ek._opts.app_version,
                    sdk_version=SDK_VERSION,
                ))
            raw = await original(self, *args, **kwargs)
            return _AsyncStreamTracer(raw, on_stream_done)

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
                service_name=ek._opts.service_name, provider="openai", model=model,
                prompt=prompt, latency_ms=int(time.time() * 1000 - start_ms),
                start_time=ts, end_time=end_ts, created_at=ts,
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model},
                environment=ek._opts.environment, app_version=ek._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
            raise
        finally:
            _current_span.reset(token)
        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)
        choice = result.choices[0] if result.choices else None
        completion = (choice.message.content or "") if choice else ""
        usage = getattr(result, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", 0) or 0
        tokens_out = getattr(usage, "completion_tokens", 0) or 0
        events = []
        if choice and getattr(choice.message, "tool_calls", None):
            for tc in choice.message.tool_calls:
                events.append(SpanEvent(
                    name="gen_ai.tool.call", timestamp=end_ts,
                    attributes={"gen_ai.tool.name": tc.function.name,
                                "gen_ai.tool.call.arguments": tc.function.arguments,
                                "gen_ai.tool.call.id": getattr(tc, "id", "")},
                ))
        ek.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call", status="OK",
            operation="chat.completions.create",
            service_name=ek._opts.service_name, provider="openai", model=model,
            prompt=prompt, completion=completion,
            tokens_in=tokens_in, tokens_out=tokens_out,
            latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
            events=events or None,
            attributes={**ek._meta_attrs(), "gen_ai.request.model": model,
                        "gen_ai.response.model": getattr(result, "model", model),
                        "gen_ai.usage.total_tokens": getattr(usage, "total_tokens", 0) or 0},
            environment=ek._opts.environment, app_version=ek._opts.app_version,
            sdk_version=SDK_VERSION,
        ))
        return result

    traced_async._evalkit = True  # type: ignore[attr-defined]
    cls.create = traced_async


# ── Anthropic ─────────────────────────────────────────────────────────────────

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


def _patch_anthropic_class_sync(ek: "EvalKitClient", cls) -> None:
    if getattr(cls.create, "_evalkit", False):
        return
    import time
    from ..core.client import _new_span_id, _now_iso
    from ..core.context import SpanContext, _current_span, get_or_create_ambient_trace_id
    from ..core.types import SpanEvent, TraceEnvelope
    from .._version import SDK_VERSION

    original = cls.create

    def traced(self, *args, **kwargs):
        ctx_parent = _current_span.get()
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
            def on_stream_done(completion: str, tokens_in: int, tokens_out: int) -> None:
                end_ts = _now_iso()
                latency = int(time.time() * 1000 - start_ms)
                ek.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call", status="OK",
                    operation="messages.create",
                    service_name=ek._opts.service_name, provider="anthropic", model=model,
                    prompt=prompt, completion=completion,
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
                    attributes={**ek._meta_attrs(), "gen_ai.request.model": model, "gen_ai.streaming": True},
                    environment=ek._opts.environment, app_version=ek._opts.app_version,
                    sdk_version=SDK_VERSION,
                ))
            return _SyncStreamTracer(original(self, *args, **kwargs), on_stream_done, is_anthropic=True)

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
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model},
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
        usage = getattr(result, "usage", None)
        tokens_in = getattr(usage, "input_tokens", 0) or 0
        tokens_out = getattr(usage, "output_tokens", 0) or 0
        import json as _json
        events = []
        for block in content:
            if getattr(block, "type", None) == "tool_use":
                events.append(SpanEvent(
                    name="gen_ai.tool.call", timestamp=end_ts,
                    attributes={"gen_ai.tool.name": block.name,
                                "gen_ai.tool.call.id": getattr(block, "id", ""),
                                "gen_ai.tool.call.arguments": _json.dumps(getattr(block, "input", {}))},
                ))
        ek.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call", status="OK",
            operation="messages.create",
            service_name=ek._opts.service_name, provider="anthropic", model=model,
            prompt=prompt, completion=completion,
            tokens_in=tokens_in, tokens_out=tokens_out,
            latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
            events=events or None,
            attributes={**ek._meta_attrs(), "gen_ai.request.model": model,
                        "gen_ai.response.model": getattr(result, "model", model),
                        "gen_ai.usage.total_tokens": tokens_in + tokens_out},
            environment=ek._opts.environment, app_version=ek._opts.app_version,
            sdk_version=SDK_VERSION,
        ))
        return result

    traced._evalkit = True  # type: ignore[attr-defined]
    cls.create = traced


def _patch_anthropic_class_async(ek: "EvalKitClient", cls) -> None:
    if getattr(cls.create, "_evalkit", False):
        return
    import time
    from ..core.client import _new_span_id, _now_iso
    from ..core.context import SpanContext, _current_span, get_or_create_ambient_trace_id
    from ..core.types import SpanEvent, TraceEnvelope
    from .._version import SDK_VERSION

    original = cls.create

    async def traced_async(self, *args, **kwargs):
        ctx_parent = _current_span.get()
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
            def on_stream_done(completion: str, tokens_in: int, tokens_out: int) -> None:
                end_ts = _now_iso()
                latency = int(time.time() * 1000 - start_ms)
                ek.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="llm_call", status="OK",
                    operation="messages.create",
                    service_name=ek._opts.service_name, provider="anthropic", model=model,
                    prompt=prompt, completion=completion,
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
                    attributes={**ek._meta_attrs(), "gen_ai.request.model": model, "gen_ai.streaming": True},
                    environment=ek._opts.environment, app_version=ek._opts.app_version,
                    sdk_version=SDK_VERSION,
                ))
            raw = await original(self, *args, **kwargs)
            return _AsyncStreamTracer(raw, on_stream_done, is_anthropic=True)

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
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model},
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
        usage = getattr(result, "usage", None)
        tokens_in = getattr(usage, "input_tokens", 0) or 0
        tokens_out = getattr(usage, "output_tokens", 0) or 0
        import json as _json
        events = []
        for block in content:
            if getattr(block, "type", None) == "tool_use":
                events.append(SpanEvent(
                    name="gen_ai.tool.call", timestamp=end_ts,
                    attributes={"gen_ai.tool.name": block.name,
                                "gen_ai.tool.call.id": getattr(block, "id", ""),
                                "gen_ai.tool.call.arguments": _json.dumps(getattr(block, "input", {}))},
                ))
        ek.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call", status="OK",
            operation="messages.create",
            service_name=ek._opts.service_name, provider="anthropic", model=model,
            prompt=prompt, completion=completion,
            tokens_in=tokens_in, tokens_out=tokens_out,
            latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
            events=events or None,
            attributes={**ek._meta_attrs(), "gen_ai.request.model": model,
                        "gen_ai.response.model": getattr(result, "model", model),
                        "gen_ai.usage.total_tokens": tokens_in + tokens_out},
            environment=ek._opts.environment, app_version=ek._opts.app_version,
            sdk_version=SDK_VERSION,
        ))
        return result

    traced_async._evalkit = True  # type: ignore[attr-defined]
    cls.create = traced_async


# ── Google GenAI (google.genai — used by ADK) ────────────────────────────────

def _try_patch_google_genai(ek: "EvalKitClient") -> None:
    try:
        from google.genai import models as _gm  # noqa: F401
    except ImportError:
        return
    try:
        from .llm.google_genai import patch_google_genai
        patch_google_genai(ek)
    except Exception:
        pass


# ── Google Generative AI (legacy google.generativeai) ───────────────────────

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
    from ..core.context import SpanContext, _current_span, get_or_create_ambient_trace_id
    from ..core.types import TraceEnvelope
    from .._version import SDK_VERSION

    original = cls.generate_content

    def traced(self, *args, **kwargs):
        ctx_parent = _current_span.get()
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
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model_name},
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

    traced._evalkit = True  # type: ignore[attr-defined]
    cls.generate_content = traced


# ── Cohere ────────────────────────────────────────────────────────────────────

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
    from ..core.context import SpanContext, _current_span, get_or_create_ambient_trace_id
    from ..core.types import TraceEnvelope
    from .._version import SDK_VERSION

    original = cls.chat

    def traced(self, *args, **kwargs):
        ctx_parent = _current_span.get()
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
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model},
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

    traced._evalkit = True  # type: ignore[attr-defined]
    cls.chat = traced


# ── AWS Bedrock ───────────────────────────────────────────────────────────────

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
    from ..core.context import SpanContext, _current_span, get_or_create_ambient_trace_id
    from ..core.types import TraceEnvelope
    from .._version import SDK_VERSION

    original = cls._make_api_call

    def traced(self, operation_name, api_params):
        service = getattr(getattr(self, "meta", None), "service_model", None)
        service_name = getattr(service, "service_name", "") if service else ""
        if service_name != "bedrock-runtime" or operation_name != "Converse":
            return original(self, operation_name, api_params)

        ctx_parent = _current_span.get()
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
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model},
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
        usage = result.get("usage", {}) or {}
        tokens_in = usage.get("inputTokens", 0) or 0
        tokens_out = usage.get("outputTokens", 0) or 0
        ek.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call", status="OK",
            operation="Converse",
            service_name=ek._opts.service_name, provider="bedrock", model=model,
            prompt=prompt, completion=completion,
            tokens_in=tokens_in, tokens_out=tokens_out,
            latency_ms=latency, start_time=ts, end_time=end_ts, created_at=ts,
            attributes={**ek._meta_attrs(), "gen_ai.request.model": model,
                        "gen_ai.usage.total_tokens": tokens_in + tokens_out},
            environment=ek._opts.environment, app_version=ek._opts.app_version,
            sdk_version=SDK_VERSION,
        ))
        return result

    traced._evalkit_bedrock = True  # type: ignore[attr-defined]
    cls._make_api_call = traced


# ── MistralAI ────────────────────────────────────────────────────────────────

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
    from ..core.context import SpanContext, _current_span, get_or_create_ambient_trace_id
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
        ctx_parent = _current_span.get()
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
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model},
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

    traced._evalkit = True  # type: ignore[attr-defined]
    chat_cls.complete = traced


def _patch_mistral_class_async(ek: "EvalKitClient", cls) -> None:
    pass  # async Mistral patched via instance patcher (patch_async_mistral_client)


# ── Main entry point ──────────────────────────────────────────────────────────

def setup_auto_instrumentation(
    client: "EvalKitClient",
    logging: bool = True,
    requests: bool = True,
    httpx: bool = True,
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
        except Exception:
            pass
    if httpx:
        try:
            from .http.httpx import patch_httpx
            patch_httpx(client)
        except Exception:
            pass

    _try_patch_openai(client)
    _try_patch_anthropic(client)
    _try_patch_google_genai(client)
    _try_patch_google_ai(client)
    _try_patch_cohere(client)
    _try_patch_bedrock(client)
    _try_patch_mistral(client)
