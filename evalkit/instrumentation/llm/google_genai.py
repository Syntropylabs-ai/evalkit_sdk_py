"""Patch google.genai (new unified Google AI SDK, used by Google ADK) at the class level.

Unlike the instance-level patchers, this patches Models and AsyncModels **classes**
so every Client instance — including those created internally by Google ADK — is traced.

Usage:
    import evalkit
    evalkit.init(...)
    evalkit.patch_google_genai()   # call once; works for ADK, direct genai, etc.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..._version import SDK_VERSION
from ...core.client import EvalKitClient, _new_span_id, _new_trace_id, _now_iso
from ...core.context import SpanContext, _current_span
from ...core.types import SpanEvent, TraceEnvelope

if TYPE_CHECKING:
    pass


def patch_google_genai(client: EvalKitClient) -> None:
    """
    Patch google.genai.Models and google.genai.AsyncModels at the class level.

    This intercepts every `client.models.generate_content()` call across ALL
    google.genai Client instances, including those created by Google ADK internally.
    Call once at startup after evalkit.init().
    """
    try:
        from google.genai import models as _gm
    except ImportError:
        return

    _patch_sync_models(client, _gm)
    _patch_async_models(client, _gm)


def _extract_prompt(contents: Any, config: Any = None) -> str:
    """Convert google.genai contents to a readable prompt string."""
    parts: List[str] = []

    # System instruction from config
    if config is not None:
        si = getattr(config, "system_instruction", None)
        if si:
            if hasattr(si, "parts"):
                for p in si.parts:
                    t = getattr(p, "text", None)
                    if t:
                        parts.append(f"system: {t}")
            elif isinstance(si, str):
                parts.append(f"system: {si}")

    if isinstance(contents, str):
        parts.append(contents)
    elif isinstance(contents, list):
        for item in contents:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                role = item.get("role", "")
                for p in item.get("parts", []):
                    t = p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")
                    if t:
                        parts.append(f"{role}: {t}" if role else t)
            else:
                role = getattr(item, "role", "")
                for p in getattr(item, "parts", []):
                    t = getattr(p, "text", None)
                    if t:
                        parts.append(f"{role}: {t}" if role else t)
    elif contents is not None:
        text = getattr(contents, "text", None)
        if text:
            parts.append(str(text))
        else:
            parts.append(str(contents))

    return "\n".join(parts)


def _extract_completion(response: Any) -> str:
    """Pull text from a generate_content response."""
    try:
        # New google.genai response: response.text shortcut
        text = getattr(response, "text", None)
        if text:
            return str(text)
    except Exception:
        pass
    try:
        # Iterate candidates → content → parts
        for candidate in (getattr(response, "candidates", None) or []):
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", []):
                t = getattr(part, "text", None)
                if t:
                    return str(t)
    except Exception:
        pass
    return ""


def _extract_tool_events(response: Any, ts: str) -> List[SpanEvent]:
    """Extract function calls from response as SpanEvent list."""
    events: List[SpanEvent] = []
    try:
        for candidate in (getattr(response, "candidates", None) or []):
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", []):
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    raw_args = getattr(fc, "args", None) or {}
                    try:
                        args_json = json.dumps(dict(raw_args))
                    except Exception:
                        args_json = str(raw_args)
                    events.append(SpanEvent(
                        name="gen_ai.tool.call",
                        timestamp=ts,
                        attributes={
                            "gen_ai.tool.name": getattr(fc, "name", ""),
                            "gen_ai.tool.call.arguments": args_json,
                            "gen_ai.tool.call.id": f"call_{id(fc):016x}",
                        },
                    ))
    except Exception:
        pass
    return events


def _usage_tokens(response: Any):
    """Return (input_tokens, output_tokens) from response."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return 0, 0
    tokens_in = getattr(meta, "prompt_token_count", 0) or 0
    tokens_out = getattr(meta, "candidates_token_count", 0) or 0
    return tokens_in, tokens_out


def _build_envelope(
    client: EvalKitClient,
    *,
    trace_id: str,
    span_id: str,
    parent_span_id: Optional[str],
    model: str,
    prompt: str,
    completion: str,
    tokens_in: int,
    tokens_out: int,
    latency: int,
    ts: str,
    end_ts: str,
    session_id: Optional[str] = None,
    device_id: Optional[str] = None,
    status: str = "OK",
    status_message: Optional[str] = None,
    events: Optional[List[SpanEvent]] = None,
    attrs: Optional[Dict[str, Any]] = None,
) -> TraceEnvelope:
    base_attrs: Dict[str, Any] = {
        **client._meta_attrs(),
        "gen_ai.request.model": model,
        "gen_ai.response.model": model,
        "gen_ai.usage.total_tokens": tokens_in + tokens_out,
    }
    if attrs:
        base_attrs.update(attrs)

    return TraceEnvelope(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        span_type="llm_call",
        status=status,
        status_message=status_message,
        operation="generate_content",
        service_name=client._opts.service_name,
        provider="google-genai",
        model=model,
        prompt=prompt,
        completion=completion,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency,
        start_time=ts,
        end_time=end_ts,
        created_at=ts,
        events=events or None,
        attributes=base_attrs,
        environment=client._opts.environment,
        app_version=client._opts.app_version,
        session_id=session_id,
        device_id=device_id or client._opts.device_id,
        sdk_version=SDK_VERSION,
    )


def _patch_sync_models(ek: EvalKitClient, gm: Any) -> None:
    """Patch google.genai.Models.generate_content at the class level."""
    Models = getattr(gm, "Models", None)
    if Models is None:
        return
    if getattr(Models.generate_content, "_evalkit", False):
        return

    original = Models.generate_content

    def traced_generate_content(self: Any, *, model: str, contents: Any, config: Any = None, **kwargs: Any) -> Any:
        ctx_parent = _current_span.get()
        trace_id = ctx_parent.trace_id if ctx_parent else _new_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        session_id = ctx_parent.session_id if ctx_parent else None
        device_id = (ctx_parent.device_id if ctx_parent else None) or ek._opts.device_id
        start_ms = time.time() * 1000
        ts = _now_iso()
        prompt = _extract_prompt(contents, config)

        span_ctx = SpanContext(trace_id=trace_id, span_id=span_id, session_id=session_id, device_id=device_id)
        token = _current_span.set(span_ctx)
        try:
            result = original(self, model=model, contents=contents, config=config, **kwargs)
        except Exception as exc:
            end_ts = _now_iso()
            latency = int(time.time() * 1000 - start_ms)
            ek.enqueue(_build_envelope(
                ek, trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                model=model, prompt=prompt, completion="", tokens_in=0, tokens_out=0,
                latency=latency, ts=ts, end_ts=end_ts,
                session_id=session_id, device_id=device_id,
                status="ERROR", status_message=str(exc),
            ))
            raise
        finally:
            _current_span.reset(token)

        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)
        completion = _extract_completion(result)
        events = _extract_tool_events(result, end_ts)
        tokens_in, tokens_out = _usage_tokens(result)

        ek.enqueue(_build_envelope(
            ek, trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            model=model, prompt=prompt, completion=completion,
            tokens_in=tokens_in, tokens_out=tokens_out,
            latency=latency, ts=ts, end_ts=end_ts, events=events,
            session_id=session_id, device_id=device_id,
        ))
        return result

    traced_generate_content._evalkit = True  # type: ignore[attr-defined]
    Models.generate_content = traced_generate_content


def _patch_async_models(ek: EvalKitClient, gm: Any) -> None:
    """Patch google.genai.AsyncModels.generate_content at the class level."""
    AsyncModels = getattr(gm, "AsyncModels", None)
    if AsyncModels is None:
        return
    if getattr(getattr(AsyncModels, "generate_content", None), "_evalkit", False):
        return

    original = AsyncModels.generate_content

    async def traced_generate_content_async(self: Any, *, model: str, contents: Any, config: Any = None, **kwargs: Any) -> Any:
        ctx_parent = _current_span.get()
        trace_id = ctx_parent.trace_id if ctx_parent else _new_trace_id()
        span_id = _new_span_id()
        parent_span_id = ctx_parent.span_id if ctx_parent else None
        session_id = ctx_parent.session_id if ctx_parent else None
        device_id = (ctx_parent.device_id if ctx_parent else None) or ek._opts.device_id
        start_ms = time.time() * 1000
        ts = _now_iso()
        prompt = _extract_prompt(contents, config)

        span_ctx = SpanContext(trace_id=trace_id, span_id=span_id, session_id=session_id, device_id=device_id)
        token = _current_span.set(span_ctx)
        try:
            result = await original(self, model=model, contents=contents, config=config, **kwargs)
        except Exception as exc:
            end_ts = _now_iso()
            latency = int(time.time() * 1000 - start_ms)
            ek.enqueue(_build_envelope(
                ek, trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                model=model, prompt=prompt, completion="", tokens_in=0, tokens_out=0,
                latency=latency, ts=ts, end_ts=end_ts,
                session_id=session_id, device_id=device_id,
                status="ERROR", status_message=str(exc),
            ))
            raise
        finally:
            _current_span.reset(token)

        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)
        completion = _extract_completion(result)
        events = _extract_tool_events(result, end_ts)
        tokens_in, tokens_out = _usage_tokens(result)

        ek.enqueue(_build_envelope(
            ek, trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            model=model, prompt=prompt, completion=completion,
            tokens_in=tokens_in, tokens_out=tokens_out,
            latency=latency, ts=ts, end_ts=end_ts, events=events,
            session_id=session_id, device_id=device_id,
        ))
        return result

    traced_generate_content_async._evalkit = True  # type: ignore[attr-defined]
    AsyncModels.generate_content = traced_generate_content_async
