"""LangChain / LangGraph instrumentation.

LangGraph is built on LangChain Core, so a single LangChain ``BaseCallbackHandler``
captures every chain/node, LLM, chat model, tool, and retriever run in a LangGraph
graph. We register the handler **globally** via ``register_configure_hook`` so that —
true to ``init()`` does everything — the user never has to pass ``callbacks=[...]``.

The handler builds the span tree from LangChain's own ``run_id`` / ``parent_run_id``
graph (independent of ContextVar timing across async tasks), and nests the whole
graph under any active evalkit trace (e.g. an HTTP request or a simulation scenario).

Nothing here imports LangChain at module top level; everything degrades to a no-op
if LangChain is not installed, and no callback may ever raise into the host graph.
"""
from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..._version import SDK_VERSION
from ...core.client import _new_span_id, _now_iso
from ...core.context import _ambient_trace_id, _current_span, get_or_create_ambient_trace_id
from ...core.types import SpanEvent, TraceEnvelope

if TYPE_CHECKING:
    from ...core.client import EvalKitClient

_log = logging.getLogger("evalkit")

_ATTR_MAX = 4096

# Module-level singletons so global registration happens exactly once.
_registered = False
_handler_var: Optional[ContextVar] = None


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
            # e.g. ["langchain","chat_models","openai","ChatOpenAI"]
            for token in ("openai", "anthropic", "google", "vertexai", "cohere", "mistral", "bedrock", "ollama", "groq"):
                if token in ident:
                    return token if token != "vertexai" else "vertex-ai"
            return str(ident[-1])
    except Exception:
        pass
    return None


def _model_from(serialized: Any, kwargs: Dict[str, Any]) -> str:
    # invocation_params is the most reliable source at call time.
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


def _messages_to_prompt(messages: Any) -> str:
    # messages: List[List[BaseMessage]] (one inner list per generation batch).
    parts: List[str] = []
    try:
        batch = messages[0] if messages else []
        for m in batch:
            role = getattr(m, "type", None) or getattr(m, "role", None) or m.__class__.__name__
            content = getattr(m, "content", "")
            parts.append(f"{role}: {content}")
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
        # Newer LangChain attaches usage_metadata to the AIMessage.
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
        """Maps LangChain/LangGraph runs to evalkit spans. Never raises."""

        raise_error = False

        def __init__(self) -> None:
            self._runs: Dict[str, Dict[str, Any]] = {}
            self._lock = Lock()

        # ── span lifecycle ────────────────────────────────────────────────
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
                    ctx = _current_span.get()
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
                    span_type=rec["span_type"],  # type: ignore[arg-type]
                    status=status,  # type: ignore[arg-type]
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
                ))
            except Exception as exc:
                _log.debug("[evalkit] langchain end failed: %s", exc)

        # ── chains / graph nodes ──────────────────────────────────────────
        def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
            name = None
            if isinstance(serialized, dict):
                name = serialized.get("name") or (serialized.get("id") or [None])[-1]
            name = name or (kwargs.get("name")) or "chain"
            attrs = {"evalkit.framework": "langchain", "langchain.run_type": "chain"}
            if tags:
                attrs["langchain.tags"] = list(tags)
            self._start(run_id, parent_run_id, "function_call", str(name), attrs=attrs)

        def on_chain_end(self, outputs, *, run_id, **kwargs):
            self._end(run_id, status="OK")

        def on_chain_error(self, error, *, run_id, **kwargs):
            self._end(run_id, status="ERROR", status_message=str(error))

        # ── LLMs / chat models ────────────────────────────────────────────
        def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
            model = _model_from(serialized, kwargs)
            provider = _provider_from_serialized(serialized)
            prompt = _truncate("\n".join(str(p) for p in (prompts or [])))
            self._start(run_id, parent_run_id, "llm_call", "llm.generate",
                        attrs={"evalkit.framework": "langchain", "gen_ai.request.model": model},
                        model=model, provider=provider, prompt=prompt)

        def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
            model = _model_from(serialized, kwargs)
            provider = _provider_from_serialized(serialized)
            prompt = _messages_to_prompt(messages)
            self._start(run_id, parent_run_id, "llm_call", "chat.generate",
                        attrs={"evalkit.framework": "langchain", "gen_ai.request.model": model},
                        model=model, provider=provider, prompt=prompt)

        def on_llm_end(self, response, *, run_id, **kwargs):
            completion = _llm_result_completion(response)
            tokens_in, tokens_out = _llm_result_tokens(response)
            extra = {"gen_ai.usage.total_tokens": tokens_in + tokens_out} if (tokens_in or tokens_out) else None
            self._end(run_id, status="OK", completion=completion,
                      tokens_in=tokens_in or None, tokens_out=tokens_out or None,
                      extra_attrs=extra)

        def on_llm_error(self, error, *, run_id, **kwargs):
            self._end(run_id, status="ERROR", status_message=str(error))

        # ── tools ─────────────────────────────────────────────────────────
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
            self._start(run_id, parent_run_id, "tool_call", str(name), attrs=attrs)

        def on_tool_end(self, output, *, run_id, **kwargs):
            # output may be a ToolMessage (newer LangChain) or a raw string/value
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
            self._end(run_id, status="ERROR", status_message=str(error))

        # ── retrievers ────────────────────────────────────────────────────
        def on_retriever_start(self, serialized, query, *, run_id, parent_run_id=None, **kwargs):
            name = None
            if isinstance(serialized, dict):
                name = serialized.get("name") or (serialized.get("id") or [None])[-1]
            self._start(run_id, parent_run_id, "db_query", str(name or "retriever"),
                        attrs={"evalkit.framework": "langchain", "retriever.query": _truncate(query)})

        def on_retriever_end(self, documents, *, run_id, **kwargs):
            try:
                count = len(documents) if documents is not None else 0
            except Exception:
                count = 0
            self._end(run_id, status="OK", extra_attrs={"retriever.document_count": count})

        def on_retriever_error(self, error, *, run_id, **kwargs):
            self._end(run_id, status="ERROR", status_message=str(error))

    return EvalKitLangChainHandler()


def patch_langchain(client: "EvalKitClient") -> Optional[Any]:
    """Register a global evalkit callback handler for LangChain / LangGraph.

    Idempotent and dependency-free: if LangChain is not installed (or the global
    hook API is unavailable) this is a silent no-op. Returns the handler instance
    (also usable manually via ``callbacks=[handler]``) or ``None``.
    """
    global _registered, _handler_var

    handler = _build_handler(client)
    if handler is None:
        return None

    try:
        from langchain_core.callbacks.manager import register_configure_hook
    except Exception:
        # Older/newer LangChain without the hook — still return the handler so it
        # can be attached manually.
        return handler

    try:
        if not _registered or _handler_var is None:
            _handler_var = ContextVar("evalkit_langchain_handler", default=None)
            register_configure_hook(_handler_var, True)
            _registered = True
        _handler_var.set(handler)  # type: ignore[union-attr]
    except Exception as exc:
        _log.debug("[evalkit] langchain global registration failed: %s", exc)

    return handler
