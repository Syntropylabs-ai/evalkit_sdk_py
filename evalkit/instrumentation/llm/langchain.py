from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..._version import SDK_VERSION
from ...core.bounded import BoundedTTLMap
from ...core.client import _new_span_id, _now_iso
from ...core.content import message_content_text
from ...core.context import _ambient_trace_id, _current_span, current_span_ctx, get_or_create_ambient_trace_id
from ...core.types import SpanEvent, TraceEnvelope
from ._bodies import _dump, _safe_body

if TYPE_CHECKING:
    from ...core.client import EvalKitClient

_log = logging.getLogger("evalkit")

_ATTR_MAX = 4096

_registered = False


def _truncate(s: Any, limit: int = _ATTR_MAX) -> str:
    try:
        text = s if isinstance(s, str) else str(s)
    except Exception:
        return "<unserializable>"
    return text if len(text) <= limit else text[:limit] + "…"


def _provider_from_serialized(serialized: Any) -> Optional[str]:
    try:
        ident = (serialized or {}).get("id")
        if isinstance(ident, list) and ident:
            for token in ("openai", "anthropic", "google", "vertexai", "cohere", "mistral", "bedrock", "ollama", "groq"):
                if token in ident:
                    return token if token != "vertexai" else "vertex-ai"
            return str(ident[-1])
    except Exception:
        pass
    return None


def _model_from(serialized: Any, kwargs: Dict[str, Any]) -> str:
    inv = (kwargs or {}).get("invocation_params") or {}
    for key in ("model", "model_name", "model_id", "deployment_name"):
        v = inv.get(key)
        if v:
            return str(v)
    skw = (serialized or {}).get("kwargs") or {}
    for key in ("model", "model_name", "model_id", "deployment_name"):
        v = skw.get(key)
        if v:
            return str(v)
    return "unknown"


def _langgraph_attrs(metadata: Any) -> Dict[str, Any]:
    """Where in a LangGraph run a span sits, from the metadata LangGraph gives every run inside a node.

    The triggers go out as a compact JSON array, the same string the TypeScript SDK writes for them.
    """
    out: Dict[str, Any] = {}
    try:
        if not isinstance(metadata, Mapping):
            return out
        node = metadata.get("langgraph_node")
        if node:
            out["langgraph.node"] = str(node)
        step = metadata.get("langgraph_step")
        if isinstance(step, int) and not isinstance(step, bool):
            out["langgraph.step"] = step
        triggers = metadata.get("langgraph_triggers")
        if isinstance(triggers, str):
            triggers = [triggers]
        if isinstance(triggers, (list, tuple)) and triggers:
            out["langgraph.triggers"] = json.dumps(
                [str(t) for t in triggers], separators=(",", ":"), ensure_ascii=False
            )
        namespace = metadata.get("langgraph_checkpoint_ns")
        if namespace:
            out["langgraph.checkpoint_ns"] = str(namespace)
    except Exception:
        return {}
    return out


def _messages_to_prompt(messages: Any) -> str:
    parts: List[str] = []
    try:
        batch = messages[0] if messages else []
        for m in batch:
            role = getattr(m, "type", None) or getattr(m, "role", None) or m.__class__.__name__
            parts.append(f"{role}: {message_content_text(getattr(m, 'content', ''))}")
    except Exception:
        return _truncate(messages)
    return _truncate("\n".join(parts))


def _llm_result_completion(response: Any) -> str:
    try:
        gens = getattr(response, "generations", None) or []
        out: List[str] = []
        for batch in gens:
            for g in batch:
                text = getattr(g, "text", None)
                if text:
                    out.append(str(text))
                    continue
                msg = getattr(g, "message", None)
                if msg is not None:
                    out.append(str(getattr(msg, "content", "")))
        return _truncate("\n".join(p for p in out if p))
    except Exception:
        return ""


def _llm_result_tokens(response: Any):
    tokens_in = tokens_out = 0
    try:
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        tokens_in = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        tokens_out = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    except Exception:
        pass
    if not (tokens_in or tokens_out):
        try:
            for batch in (getattr(response, "generations", None) or []):
                for g in batch:
                    msg = getattr(g, "message", None)
                    um = getattr(msg, "usage_metadata", None) if msg is not None else None
                    if um:
                        tokens_in = um.get("input_tokens", 0) or tokens_in
                        tokens_out = um.get("output_tokens", 0) or tokens_out
        except Exception:
            pass
    return int(tokens_in or 0), int(tokens_out or 0)


def _build_handler(client: "EvalKitClient"):
    try:
        from langchain_core.callbacks.base import BaseCallbackHandler
    except Exception:
        return None

    class EvalKitLangChainHandler(BaseCallbackHandler):

        raise_error = False

        def __init__(self) -> None:
            # A run whose end callback never comes (a chain cancelled mid-way, a callback manager that
            # swallowed the error) must not keep its record for the life of the process.
            self._runs: BoundedTTLMap[str, Dict[str, Any]] = BoundedTTLMap()
            self._lock = Lock()

        def _start(
            self,
            run_id: Any,
            parent_run_id: Any,
            span_type: str,
            operation: str,
            *,
            attrs: Optional[Dict[str, Any]] = None,
            model: Optional[str] = None,
            provider: Optional[str] = None,
            prompt: Optional[str] = None,
        ) -> None:
            try:
                parent = None
                if parent_run_id is not None:
                    with self._lock:
                        parent = self._runs.get(str(parent_run_id))
                if parent:
                    trace_id = parent["trace_id"]
                    parent_span_id = parent["span_id"]
                    session_id = parent.get("session_id")
                    device_id = parent.get("device_id")
                else:
                    ctx = current_span_ctx()
                    if ctx is not None:
                        trace_id = ctx.trace_id
                        parent_span_id = ctx.span_id
                        session_id = ctx.session_id
                        device_id = ctx.device_id
                    else:
                        trace_id = get_or_create_ambient_trace_id()
                        parent_span_id = None
                        session_id = None
                        device_id = None
                    _ambient_trace_id.set(trace_id)

                rec = {
                    "trace_id": trace_id,
                    "span_id": _new_span_id(),
                    "parent_span_id": parent_span_id,
                    "session_id": session_id,
                    "device_id": device_id or client._opts.device_id,
                    "span_type": span_type,
                    "operation": operation,
                    "model": model,
                    "provider": provider,
                    "prompt": prompt,
                    "attrs": attrs or {},
                    "start_ms": time.time() * 1000,
                    "start_iso": _now_iso(),
                }
                with self._lock:
                    self._runs[str(run_id)] = rec
            except Exception as exc:
                _log.debug("[evalkit] langchain start failed: %s", exc)

        def _end(
            self,
            run_id: Any,
            *,
            status: str = "OK",
            status_message: Optional[str] = None,
            completion: Optional[str] = None,
            tokens_in: Optional[int] = None,
            tokens_out: Optional[int] = None,
            extra_attrs: Optional[Dict[str, Any]] = None,
            events: Optional[List[SpanEvent]] = None,
            exception: Optional[BaseException] = None,
        ) -> None:
            try:
                with self._lock:
                    rec = self._runs.pop(str(run_id), None)
                if rec is None:
                    return
                end_iso = _now_iso()
                latency = int(time.time() * 1000 - rec["start_ms"])
                attrs: Dict[str, Any] = {**client._meta_attrs(), **rec["attrs"]}
                if extra_attrs:
                    attrs.update(extra_attrs)
                if rec.get("model"):
                    attrs.setdefault("gen_ai.request.model", rec["model"])
                client.enqueue(TraceEnvelope(
                    trace_id=rec["trace_id"],
                    span_id=rec["span_id"],
                    parent_span_id=rec["parent_span_id"],
                    span_type=rec["span_type"],
                    status=status,
                    status_message=status_message,
                    operation=rec["operation"],
                    service_name=client._opts.service_name,
                    provider=rec.get("provider"),
                    model=rec.get("model"),
                    prompt=rec.get("prompt"),
                    completion=completion,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    latency_ms=latency,
                    start_time=rec["start_iso"],
                    end_time=end_iso,
                    created_at=rec["start_iso"],
                    events=events or None,
                    attributes=attrs or None,
                    environment=client._opts.environment,
                    app_version=client._opts.app_version,
                    session_id=rec.get("session_id"),
                    device_id=rec.get("device_id"),
                    sdk_version=SDK_VERSION,
                    exception=exception,
                ))
            except Exception as exc:
                _log.debug("[evalkit] langchain end failed: %s", exc)

        def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
            name = None
            if isinstance(serialized, dict):
                name = serialized.get("name") or (serialized.get("id") or [None])[-1]
            name = name or (kwargs.get("name")) or "chain"
            attrs = {"evalkit.framework": "langchain", "langchain.run_type": "chain"}
            if tags:
                attrs["langchain.tags"] = list(tags)
            attrs.update(_langgraph_attrs(metadata))
            self._start(run_id, parent_run_id, "function_call", str(name), attrs=attrs)

        def on_chain_end(self, outputs, *, run_id, **kwargs):
            self._end(run_id, status="OK")

        def on_chain_error(self, error, *, run_id, **kwargs):
            self._end(run_id, status="ERROR", status_message=str(error),
                      exception=error if isinstance(error, BaseException) else None)

        def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
            model = _model_from(serialized, kwargs)
            provider = _provider_from_serialized(serialized)
            prompt = _truncate("\n".join(str(p) for p in (prompts or [])))
            attrs = {"evalkit.framework": "langchain", "gen_ai.request.model": model}
            attrs.update(_langgraph_attrs(kwargs.get("metadata")))
            request_body = _safe_body({"prompts": _dump(prompts), "invocation_params": _dump(kwargs.get("invocation_params"))})
            if request_body:
                attrs["gen_ai.request.body"] = request_body
            self._start(run_id, parent_run_id, "llm_call", "llm.generate",
                        attrs=attrs, model=model, provider=provider, prompt=prompt)

        def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
            model = _model_from(serialized, kwargs)
            provider = _provider_from_serialized(serialized)
            prompt = _messages_to_prompt(messages)
            attrs = {"evalkit.framework": "langchain", "gen_ai.request.model": model}
            attrs.update(_langgraph_attrs(kwargs.get("metadata")))
            request_body = _safe_body({"messages": _dump(messages), "invocation_params": _dump(kwargs.get("invocation_params"))})
            if request_body:
                attrs["gen_ai.request.body"] = request_body
            self._start(run_id, parent_run_id, "llm_call", "chat.generate",
                        attrs=attrs, model=model, provider=provider, prompt=prompt)

        def on_llm_end(self, response, *, run_id, **kwargs):
            completion = _llm_result_completion(response)
            tokens_in, tokens_out = _llm_result_tokens(response)
            extra: Dict[str, Any] = {}
            if tokens_in or tokens_out:
                extra["gen_ai.usage.total_tokens"] = tokens_in + tokens_out
            response_body = _safe_body(_dump(response))
            if response_body:
                extra["gen_ai.response.body"] = response_body
            self._end(run_id, status="OK", completion=completion,
                      tokens_in=tokens_in or None, tokens_out=tokens_out or None,
                      extra_attrs=extra or None)

        def on_llm_error(self, error, *, run_id, **kwargs):
            self._end(run_id, status="ERROR", status_message=str(error),
                      exception=error if isinstance(error, BaseException) else None)

        def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
            name = None
            if isinstance(serialized, dict):
                name = serialized.get("name") or (serialized.get("id") or [None])[-1]
            name = name or kwargs.get("name") or "tool"
            tool_call_id = kwargs.get("tool_call_id") or ""
            attrs: Dict[str, Any] = {
                "evalkit.framework": "langchain",
                "tool.name": str(name),
                "gen_ai.tool.name": str(name),
                "tool.arguments": _truncate(input_str),
            }
            if tool_call_id:
                attrs["gen_ai.tool.call.id"] = tool_call_id
            attrs.update(_langgraph_attrs(kwargs.get("metadata")))
            self._start(run_id, parent_run_id, "tool_call", str(name), attrs=attrs)

        def on_tool_end(self, output, *, run_id, **kwargs):
            if hasattr(output, "content"):
                raw = output.content
            else:
                raw = output
            if raw is None or raw == "" or raw == [] or raw == {}:
                result = "(empty)"
            else:
                result = _truncate(raw)
            self._end(run_id, status="OK", extra_attrs={"tool.result": result})

        def on_tool_error(self, error, *, run_id, **kwargs):
            self._end(run_id, status="ERROR", status_message=str(error),
                      exception=error if isinstance(error, BaseException) else None)

        def on_retriever_start(self, serialized, query, *, run_id, parent_run_id=None, **kwargs):
            name = None
            if isinstance(serialized, dict):
                name = serialized.get("name") or (serialized.get("id") or [None])[-1]
            attrs = {"evalkit.framework": "langchain", "retriever.query": _truncate(query)}
            attrs.update(_langgraph_attrs(kwargs.get("metadata")))
            self._start(run_id, parent_run_id, "db_query", str(name or "retriever"), attrs=attrs)

        def on_retriever_end(self, documents, *, run_id, **kwargs):
            try:
                count = len(documents) if documents is not None else 0
            except Exception:
                count = 0
            self._end(run_id, status="OK", extra_attrs={"retriever.document_count": count})

        def on_retriever_error(self, error, *, run_id, **kwargs):
            self._end(run_id, status="ERROR", status_message=str(error),
                      exception=error if isinstance(error, BaseException) else None)

    return EvalKitLangChainHandler()


def patch_langchain(client: "EvalKitClient") -> Optional[Any]:
    global _registered

    handler = _build_handler(client)
    if handler is None:
        return None

    if _registered:
        return handler

    try:
        from langchain_core.callbacks.base import BaseCallbackManager
    except Exception:
        return handler

    try:
        if getattr(BaseCallbackManager.__init__, "_evalkit", False):
            _registered = True
            return handler

        original_init = BaseCallbackManager.__init__
        handler_type = type(handler)

        def patched_init(self, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            try:
                if any(isinstance(h, handler_type) for h in self.inheritable_handlers):
                    return
                self.add_handler(handler, True)
            except Exception as exc:
                _log.debug("[evalkit] langchain handler injection failed: %s", exc)

        patched_init._evalkit = True
        BaseCallbackManager.__init__ = patched_init
        _registered = True
    except Exception as exc:
        _log.debug("[evalkit] langchain global registration failed: %s", exc)

    return handler
