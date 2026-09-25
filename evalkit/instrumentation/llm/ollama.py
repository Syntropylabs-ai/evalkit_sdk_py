from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, List, Optional

from ..._version import SDK_VERSION
from ...core.client import _new_span_id, _now_iso
from ...core.context import (
    _ambient_trace_id,
    _current_span, child_span_ctx, current_span_ctx,
    get_or_create_ambient_trace_id,
)
from ...core.types import SpanEvent, TraceEnvelope
from ._bodies import _dump, _safe_body

if TYPE_CHECKING:
    from ...core.client import EvalKitClient


def _messages_to_prompt(messages: Any) -> str:
    parts: List[str] = []
    for m in messages or []:
        role = m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "")
        content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _tool_calls_from_message(message: Any) -> List[SpanEvent]:
    events: List[SpanEvent] = []
    try:
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls is None and isinstance(message, dict):
            tool_calls = message.get("tool_calls")
        for tc in tool_calls or []:
            fn = getattr(tc, "function", None)
            if fn is None and isinstance(tc, dict):
                fn = tc.get("function")
            if fn is None:
                continue
            name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else None)
            if not name:
                continue
            args = getattr(fn, "arguments", None)
            if args is None and isinstance(fn, dict):
                args = fn.get("arguments")
            try:
                args_str = json.dumps(dict(args)) if args else "{}"
            except Exception:
                args_str = str(args)
            events.append(SpanEvent(
                name="gen_ai.tool.call", timestamp=_now_iso(),
                attributes={"gen_ai.tool.name": str(name), "gen_ai.tool.call.arguments": args_str},
            ))
    except Exception:
        pass
    return events


def _resolve_trace(ek: "EvalKitClient"):
    ctx_parent = current_span_ctx()
    trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
    _ambient_trace_id.set(trace_id)
    parent_span_id = ctx_parent.span_id if ctx_parent else None
    session_id = ctx_parent.session_id if ctx_parent else None
    device_id = (ctx_parent.device_id if ctx_parent else None) or ek._opts.device_id
    return trace_id, parent_span_id, session_id, device_id


def patch_ollama(client: "EvalKitClient") -> None:
    try:
        import ollama as _ollama
    except ImportError:
        return
    try:
        _patch_sync(client, getattr(_ollama, "Client", None))
    except Exception:
        pass
    try:
        _patch_async(client, getattr(_ollama, "AsyncClient", None))
    except Exception:
        pass


def _patch_sync(ek: "EvalKitClient", Client: Any) -> None:
    if Client is None or getattr(getattr(Client, "chat", None), "_evalkit", False):
        return
    original = Client.chat

    def traced_chat(self: Any, model: str = "", messages: Any = None, *, stream: bool = False, **kwargs: Any) -> Any:
        trace_id, parent_span_id, session_id, device_id = _resolve_trace(ek)
        span_id = _new_span_id()
        start_ms = time.time() * 1000
        ts = _now_iso()
        prompt = _messages_to_prompt(messages)
        req_body = _safe_body(_dump({"model": model, "messages": messages, **kwargs}))

        if stream:
            def _gen():
                parts: List[str] = []
                thinking_parts: List[str] = []
                tokens_in = tokens_out = 0
                ttft_ms: Optional[int] = None
                events: List[SpanEvent] = []
                status, smsg = "OK", None
                try:
                    for chunk in original(self, model=model, messages=messages, stream=True, **kwargs):
                        try:
                            msg = getattr(chunk, "message", None)
                            c = getattr(msg, "content", None) if msg is not None else None
                            if c:
                                if ttft_ms is None:
                                    ttft_ms = int(time.time() * 1000 - start_ms)
                                parts.append(c)
                            th = getattr(msg, "thinking", None) if msg is not None else None
                            if th:
                                thinking_parts.append(th)
                            if getattr(chunk, "done", False):
                                tokens_in = getattr(chunk, "prompt_eval_count", 0) or 0
                                tokens_out = getattr(chunk, "eval_count", 0) or 0
                            if msg is not None:
                                events.extend(_tool_calls_from_message(msg))
                        except Exception:
                            pass
                        yield chunk
                except Exception as exc:
                    status, smsg = "ERROR", str(exc)
                    raise
                finally:
                    end_ts = _now_iso()
                    attrs = {**ek._meta_attrs(), "gen_ai.request.model": model, "gen_ai.streaming": True}
                    if thinking_parts:
                        attrs["gen_ai.response.thinking"] = "".join(thinking_parts)
                    if ttft_ms is not None and status == "OK":
                        attrs["gen_ai.server.time_to_first_token_ms"] = ttft_ms
                    if req_body:
                        attrs["gen_ai.request.body"] = req_body
                    try:
                        ek.enqueue(TraceEnvelope(
                            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                            span_type="llm_call", status=status, status_message=smsg,
                            operation="chat", service_name=ek._opts.service_name,
                            provider="ollama", model=model,
                            prompt=prompt, completion="".join(parts),
                            tokens_in=tokens_in, tokens_out=tokens_out,
                            latency_ms=int(time.time() * 1000 - start_ms), start_time=ts, end_time=end_ts, created_at=ts,
                            events=events or None, attributes=attrs,
                            session_id=session_id, device_id=device_id,
                            environment=ek._opts.environment, app_version=ek._opts.app_version,
                            sdk_version=SDK_VERSION,
                        ))
                    except Exception:
                        pass
            return _gen()

        span_ctx = replace(child_span_ctx(trace_id, span_id), session_id=session_id, device_id=device_id)
        token = _current_span.set(span_ctx)
        try:
            result = original(self, model=model, messages=messages, stream=False, **kwargs)
        except Exception as exc:
            end_ts = _now_iso()
            ek.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="ERROR", status_message=str(exc),
                operation="chat", service_name=ek._opts.service_name, provider="ollama", model=model,
                prompt=prompt, latency_ms=int(time.time() * 1000 - start_ms), start_time=ts, end_time=end_ts, created_at=ts,
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model,
                            **({"gen_ai.request.body": req_body} if req_body else {})},
                session_id=session_id, device_id=device_id,
                environment=ek._opts.environment, app_version=ek._opts.app_version, sdk_version=SDK_VERSION,
            ))
            raise
        finally:
            _current_span.reset(token)

        end_ts = _now_iso()
        message = getattr(result, "message", None)
        completion = (getattr(message, "content", "") or "") if message is not None else ""
        thinking = getattr(message, "thinking", None) if message is not None else None
        tokens_in = getattr(result, "prompt_eval_count", 0) or 0
        tokens_out = getattr(result, "eval_count", 0) or 0
        events = _tool_calls_from_message(message) if message is not None else []

        attrs = {**ek._meta_attrs(), "gen_ai.request.model": model,
                 "gen_ai.response.model": getattr(result, "model", model)}
        if thinking:
            attrs["gen_ai.response.thinking"] = thinking
        done_reason = getattr(result, "done_reason", None)
        if done_reason:
            attrs["gen_ai.response.finish_reasons"] = [done_reason]
        if req_body:
            attrs["gen_ai.request.body"] = req_body
        resp_body = _safe_body(_dump(result))
        if resp_body:
            attrs["gen_ai.response.body"] = resp_body

        ek.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call", status="OK", operation="chat",
            service_name=ek._opts.service_name, provider="ollama", model=model,
            prompt=prompt, completion=completion, tokens_in=tokens_in, tokens_out=tokens_out,
            latency_ms=int(time.time() * 1000 - start_ms), start_time=ts, end_time=end_ts, created_at=ts,
            events=events or None, attributes=attrs,
            session_id=session_id, device_id=device_id,
            environment=ek._opts.environment, app_version=ek._opts.app_version, sdk_version=SDK_VERSION,
        ))
        return result

    traced_chat._evalkit = True
    Client.chat = traced_chat


def _patch_async(ek: "EvalKitClient", AsyncClient: Any) -> None:
    if AsyncClient is None or getattr(getattr(AsyncClient, "chat", None), "_evalkit", False):
        return
    original = AsyncClient.chat

    async def traced_chat(self: Any, model: str = "", messages: Any = None, *, stream: bool = False, **kwargs: Any) -> Any:
        trace_id, parent_span_id, session_id, device_id = _resolve_trace(ek)
        span_id = _new_span_id()
        start_ms = time.time() * 1000
        ts = _now_iso()
        prompt = _messages_to_prompt(messages)
        req_body = _safe_body(_dump({"model": model, "messages": messages, **kwargs}))

        if stream:
            async def _agen():
                parts: List[str] = []
                thinking_parts: List[str] = []
                tokens_in = tokens_out = 0
                ttft_ms: Optional[int] = None
                events: List[SpanEvent] = []
                status, smsg = "OK", None
                try:
                    stream_obj = await original(self, model=model, messages=messages, stream=True, **kwargs)
                    async for chunk in stream_obj:
                        try:
                            msg = getattr(chunk, "message", None)
                            c = getattr(msg, "content", None) if msg is not None else None
                            if c:
                                if ttft_ms is None:
                                    ttft_ms = int(time.time() * 1000 - start_ms)
                                parts.append(c)
                            th = getattr(msg, "thinking", None) if msg is not None else None
                            if th:
                                thinking_parts.append(th)
                            if getattr(chunk, "done", False):
                                tokens_in = getattr(chunk, "prompt_eval_count", 0) or 0
                                tokens_out = getattr(chunk, "eval_count", 0) or 0
                            if msg is not None:
                                events.extend(_tool_calls_from_message(msg))
                        except Exception:
                            pass
                        yield chunk
                except Exception as exc:
                    status, smsg = "ERROR", str(exc)
                    raise
                finally:
                    end_ts = _now_iso()
                    attrs = {**ek._meta_attrs(), "gen_ai.request.model": model, "gen_ai.streaming": True}
                    if thinking_parts:
                        attrs["gen_ai.response.thinking"] = "".join(thinking_parts)
                    if ttft_ms is not None and status == "OK":
                        attrs["gen_ai.server.time_to_first_token_ms"] = ttft_ms
                    if req_body:
                        attrs["gen_ai.request.body"] = req_body
                    try:
                        ek.enqueue(TraceEnvelope(
                            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                            span_type="llm_call", status=status, status_message=smsg,
                            operation="chat", service_name=ek._opts.service_name,
                            provider="ollama", model=model,
                            prompt=prompt, completion="".join(parts),
                            tokens_in=tokens_in, tokens_out=tokens_out,
                            latency_ms=int(time.time() * 1000 - start_ms), start_time=ts, end_time=end_ts, created_at=ts,
                            events=events or None, attributes=attrs,
                            session_id=session_id, device_id=device_id,
                            environment=ek._opts.environment, app_version=ek._opts.app_version,
                            sdk_version=SDK_VERSION,
                        ))
                    except Exception:
                        pass
            return _agen()

        span_ctx = replace(child_span_ctx(trace_id, span_id), session_id=session_id, device_id=device_id)
        token = _current_span.set(span_ctx)
        try:
            result = await original(self, model=model, messages=messages, stream=False, **kwargs)
        except Exception as exc:
            end_ts = _now_iso()
            ek.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="llm_call", status="ERROR", status_message=str(exc),
                operation="chat", service_name=ek._opts.service_name, provider="ollama", model=model,
                prompt=prompt, latency_ms=int(time.time() * 1000 - start_ms), start_time=ts, end_time=end_ts, created_at=ts,
                attributes={**ek._meta_attrs(), "gen_ai.request.model": model,
                            **({"gen_ai.request.body": req_body} if req_body else {})},
                session_id=session_id, device_id=device_id,
                environment=ek._opts.environment, app_version=ek._opts.app_version, sdk_version=SDK_VERSION,
            ))
            raise
        finally:
            _current_span.reset(token)

        end_ts = _now_iso()
        message = getattr(result, "message", None)
        completion = (getattr(message, "content", "") or "") if message is not None else ""
        thinking = getattr(message, "thinking", None) if message is not None else None
        tokens_in = getattr(result, "prompt_eval_count", 0) or 0
        tokens_out = getattr(result, "eval_count", 0) or 0
        events = _tool_calls_from_message(message) if message is not None else []

        attrs = {**ek._meta_attrs(), "gen_ai.request.model": model,
                 "gen_ai.response.model": getattr(result, "model", model)}
        if thinking:
            attrs["gen_ai.response.thinking"] = thinking
        done_reason = getattr(result, "done_reason", None)
        if done_reason:
            attrs["gen_ai.response.finish_reasons"] = [done_reason]
        if req_body:
            attrs["gen_ai.request.body"] = req_body
        resp_body = _safe_body(_dump(result))
        if resp_body:
            attrs["gen_ai.response.body"] = resp_body

        ek.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="llm_call", status="OK", operation="chat",
            service_name=ek._opts.service_name, provider="ollama", model=model,
            prompt=prompt, completion=completion, tokens_in=tokens_in, tokens_out=tokens_out,
            latency_ms=int(time.time() * 1000 - start_ms), start_time=ts, end_time=end_ts, created_at=ts,
            events=events or None, attributes=attrs,
            session_id=session_id, device_id=device_id,
            environment=ek._opts.environment, app_version=ek._opts.app_version, sdk_version=SDK_VERSION,
        ))
        return result

    traced_chat._evalkit = True
    AsyncClient.chat = traced_chat
