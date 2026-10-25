from __future__ import annotations

import logging
import time
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..._version import SDK_VERSION
from ...core.client import _new_span_id, _now_iso
from ...core.context import _llm_call_owner, current_span_ctx, get_or_create_ambient_trace_id
from ...core.types import TraceEnvelope
from ._bodies import _dump, _safe_body

if TYPE_CHECKING:
    from ...core.client import EvalKitClient

_log = logging.getLogger("evalkit")

_ATTR_MAX = 4096
_registered = False

_PROVIDER_TOKENS = (
    "openai", "anthropic", "gemini", "bedrock", "cohere", "mistral",
    "ollama", "groq", "vertex", "azure", "huggingface",
)


def _truncate(s: Any, limit: int = _ATTR_MAX) -> str:
    try:
        text = s if isinstance(s, str) else str(s)
    except Exception:
        return "<unserializable>"
    return text if len(text) <= limit else text[:limit] + "…"


def _provider_from_class_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    lname = name.lower()
    for token in _PROVIDER_TOKENS:
        if token in lname:
            return "google" if token == "gemini" else token
    return None


def _model_from_dict(model_dict: Any) -> Optional[str]:
    md = model_dict or {}
    if not isinstance(md, dict):
        return None
    for key in ("model", "model_name", "model_id", "deployment_name", "engine"):
        v = md.get(key)
        if v:
            return str(v)
    return None


def _messages_to_prompt(messages: Any) -> str:
    try:
        parts: List[str] = []
        for m in messages or []:
            role = getattr(getattr(m, "role", None), "value", None) or getattr(m, "role", None) or "user"
            parts.append(f"{role}: {getattr(m, 'content', '') or ''}")
        return _truncate("\n".join(parts))
    except Exception:
        return _truncate(messages)


def _query_str(q: Any) -> str:
    try:
        if hasattr(q, "query_str"):
            return _truncate(q.query_str)
        return _truncate(q)
    except Exception:
        return ""


def _nodes_summary(nodes: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        for n in nodes or []:
            out.append({
                "id": getattr(n, "node_id", None) or getattr(n, "id_", None),
                "score": getattr(n, "score", None),
            })
    except Exception:
        pass
    return out


def _usage_from(obj: Any) -> Tuple[int, int]:
    if obj is None:
        return 0, 0
    if isinstance(obj, dict):
        tin = obj.get("prompt_tokens") or obj.get("input_tokens") or 0
        tout = obj.get("completion_tokens") or obj.get("output_tokens") or 0
    else:
        tin = getattr(obj, "prompt_tokens", None) or getattr(obj, "input_tokens", None) or 0
        tout = getattr(obj, "completion_tokens", None) or getattr(obj, "output_tokens", None) or 0
    return int(tin or 0), int(tout or 0)


def _extract_tokens(response: Any) -> Tuple[int, int]:
    try:
        ak = getattr(response, "additional_kwargs", None) or {}
        if isinstance(ak, dict):
            tin, tout = _usage_from(ak.get("usage"))
            if not (tin or tout):
                tin, tout = _usage_from(ak)
            if tin or tout:
                return tin, tout
    except Exception:
        pass
    try:
        raw = getattr(response, "raw", None)
        usage = raw.get("usage") if isinstance(raw, dict) else getattr(raw, "usage", None)
        return _usage_from(usage)
    except Exception:
        return 0, 0


class _Tracker:

    def __init__(self, client: "EvalKitClient") -> None:
        self.client = client
        self._spans: Dict[str, Dict[str, Any]] = {}
        self._pending: List[str] = []
        self._lock = Lock()

    def on_span_enter(self, id_: str, parent_id: Optional[str], instance: Any = None) -> None:
        with self._lock:
            parent_span_id = None
            trace_id = None
            session_id = None
            device_id = None
            walk = parent_id
            found = False
            while walk:
                prec = self._spans.get(walk)
                if prec is None:
                    break
                if prec.get("span_type"):
                    parent_span_id = prec["span_id"]
                    trace_id = prec["trace_id"]
                    session_id = prec.get("session_id")
                    device_id = prec.get("device_id")
                    found = True
                    break
                walk = prec.get("parent_id")
            if not found:
                ctx = current_span_ctx()
                if ctx is not None:
                    trace_id = ctx.trace_id
                    parent_span_id = ctx.span_id
                    session_id = ctx.session_id
                    device_id = ctx.device_id
                else:
                    trace_id = get_or_create_ambient_trace_id()

            attrs: Dict[str, Any] = {"evalkit.framework": "llama_index"}
            provider = None
            if instance is not None:
                cls_name = type(instance).__name__
                attrs["llama_index.class"] = cls_name
                provider = _provider_from_class_name(cls_name)

            self._spans[id_] = {
                "key": id_,
                "parent_id": parent_id,
                "span_id": _new_span_id(),
                "parent_span_id": parent_span_id,
                "trace_id": trace_id,
                "session_id": session_id,
                "device_id": device_id or self.client._opts.device_id,
                "span_type": None,
                "operation": None,
                "model": None,
                "provider": provider,
                "prompt": None,
                "completion": None,
                "tokens_in": None,
                "tokens_out": None,
                "attrs": attrs,
                "start_ms": time.time() * 1000,
                "start_iso": _now_iso(),
            }

    def on_span_exit(self, id_: str) -> None:
        with self._lock:
            rec = self._spans.get(id_)
            if rec is None:
                return
            # A streamed chat leaves its span as soon as the generator is handed back; the request
            # itself, and LlamaIndex's end event, come while the caller reads it. The span is sent
            # when the provider's record of that request or the end event arrives.
            streaming = rec.get("model_call") and not rec.get("ended") and not rec.get("absorbed")
            if streaming:
                rec["exited"] = True
                self._pending.append(id_)
                while len(self._pending) > _MAX_PENDING_STREAMS:
                    self._spans.pop(self._pending.pop(0), None)
                return
            self._spans.pop(id_, None)
        self._release_model_call(rec)
        if not rec.get("span_type"):
            return
        try:
            self._emit(rec, "OK")
        except Exception as exc:
            _log.debug("[evalkit] llama_index span exit failed: %s", exc)

    def _finish_stream(self, rec: Dict[str, Any], span_key: Optional[str]) -> None:
        with self._lock:
            if rec.get("emitted") or not rec.get("exited"):
                return
            if span_key is not None:
                self._spans.pop(span_key, None)
                if span_key in self._pending:
                    self._pending.remove(span_key)
        self._release_model_call(rec)
        try:
            self._emit(rec, "OK")
        except Exception as exc:
            _log.debug("[evalkit] llama_index stream end failed: %s", exc)

    def on_span_drop(self, id_: str, err: Any) -> None:
        with self._lock:
            rec = self._spans.pop(id_, None)
        if rec is None:
            return
        self._release_model_call(rec)
        if not rec.get("span_type"):
            return
        try:
            self._emit(rec, "ERROR", status_message=str(err) if err else None,
                       exception=err if isinstance(err, BaseException) else None)
        except Exception as exc:
            _log.debug("[evalkit] llama_index span drop failed: %s", exc)

    def _own_model_call(self, rec: Dict[str, Any]) -> None:
        """While this LLM span runs, take the SDK's record of the provider request it makes.

        The provider's own instrumentation (the OpenAI or Anthropic patch) records the request with
        its exact tokens, cache counts and timing; LlamaIndex reports the same call again. The call
        goes out once, as the provider's span with this span's framework attributes and place in
        the tree.
        """
        def absorb(envelope: TraceEnvelope) -> bool:
            with self._lock:
                if rec.get("absorbed") is not None or rec.get("emitted"):
                    return False
                rec["absorbed"] = envelope
                exited = rec.get("exited")
            if exited:
                self._finish_stream(rec, rec.get("key"))
            return True

        rec["model_call"] = True
        try:
            rec["owner_token"] = _llm_call_owner.set(absorb)
        except Exception:
            pass

    def _release_model_call(self, rec: Dict[str, Any]) -> None:
        token = rec.pop("owner_token", None)
        if token is None:
            return
        try:
            _llm_call_owner.reset(token)
        except Exception:
            # Ended in another context than the one it started in; the owner left there refuses
            # every envelope once this span is emitted.
            pass

    def _emit(
        self, rec: Dict[str, Any], status: str, status_message: Optional[str] = None,
        exception: Optional[BaseException] = None,
    ) -> None:
        with self._lock:
            rec["emitted"] = True
            absorbed = rec.get("absorbed")
        if absorbed is not None:
            self._emit_provider_call(rec, absorbed, status, status_message, exception)
            return
        end_iso = _now_iso()
        latency = int(time.time() * 1000 - rec["start_ms"])
        attrs = {**self.client._meta_attrs(), **rec["attrs"]}
        if rec.get("model"):
            attrs.setdefault("gen_ai.request.model", rec["model"])
        self.client.enqueue(TraceEnvelope(
            trace_id=rec["trace_id"],
            span_id=rec["span_id"],
            parent_span_id=rec["parent_span_id"],
            span_type=rec["span_type"],
            status=status,
            status_message=status_message,
            operation=rec["operation"] or rec["span_type"],
            service_name=self.client._opts.service_name,
            provider=rec.get("provider"),
            model=rec.get("model"),
            prompt=rec.get("prompt"),
            completion=rec.get("completion"),
            tokens_in=rec.get("tokens_in"),
            tokens_out=rec.get("tokens_out"),
            latency_ms=latency,
            start_time=rec["start_iso"],
            end_time=end_iso,
            created_at=rec["start_iso"],
            attributes=attrs or None,
            environment=self.client._opts.environment,
            app_version=self.client._opts.app_version,
            session_id=rec.get("session_id"),
            device_id=rec.get("device_id"),
            sdk_version=SDK_VERSION,
            exception=exception,
        ))

    def _emit_provider_call(
        self, rec: Dict[str, Any], envelope: TraceEnvelope, status: str, status_message: Optional[str],
        exception: Optional[BaseException] = None,
    ) -> None:
        attrs = dict(envelope.attributes or {})
        for key, value in rec["attrs"].items():
            if key == "evalkit.framework" or key.startswith("llama_index."):
                attrs.setdefault(key, value)
        envelope.attributes = attrs
        if rec.get("parent_span_id") and rec.get("trace_id") == envelope.trace_id:
            envelope.parent_span_id = rec["parent_span_id"]
        if status == "ERROR" and envelope.status != "ERROR":
            envelope.status = "ERROR"
            envelope.status_message = status_message or envelope.status_message
        if exception is not None and envelope.exception is None:
            envelope.exception = exception
        self.client.enqueue(envelope)

    def on_event(self, event: Any) -> None:
        try:
            name = event.class_name()
        except Exception:
            return
        handler = self._EVENT_HANDLERS.get(name)
        if handler is None:
            return
        with self._lock:
            try:
                handler(self, event)
            except Exception as exc:
                _log.debug("[evalkit] llama_index event handling failed (%s): %s", name, exc)
            rec = self._spans.get(getattr(event, "span_id", None) or "")
            done = rec is not None and rec.get("ended") and rec.get("exited") and not rec.get("emitted")
        if done:
            self._finish_stream(rec, rec.get("key"))

    def _on_llm_chat_start(self, event: Any) -> None:
        rec = self._spans.get(event.span_id)
        if rec is None:
            return
        model = _model_from_dict(getattr(event, "model_dict", None))
        rec["span_type"] = "llm_call"
        rec["operation"] = "chat"
        self._own_model_call(rec)
        if model:
            rec["model"] = model
        rec["prompt"] = _messages_to_prompt(getattr(event, "messages", None))
        req_body = _safe_body({"messages": _dump(getattr(event, "messages", None))})
        if req_body:
            rec["attrs"]["gen_ai.request.body"] = req_body

    def _on_llm_chat_end(self, event: Any) -> None:
        rec = self._spans.get(event.span_id)
        if rec is None:
            return
        rec["ended"] = True
        resp = getattr(event, "response", None)
        if resp is None:
            return
        msg = getattr(resp, "message", None)
        rec["completion"] = _truncate(getattr(msg, "content", "") or "")
        tin, tout = _extract_tokens(resp)
        if tin:
            rec["tokens_in"] = tin
        if tout:
            rec["tokens_out"] = tout
        body = _safe_body(_dump(resp))
        if body:
            rec["attrs"]["gen_ai.response.body"] = body

    def _on_llm_completion_start(self, event: Any) -> None:
        rec = self._spans.get(event.span_id)
        if rec is None:
            return
        model = _model_from_dict(getattr(event, "model_dict", None))
        rec["span_type"] = "llm_call"
        rec["operation"] = "complete"
        self._own_model_call(rec)
        if model:
            rec["model"] = model
        rec["prompt"] = _truncate(getattr(event, "prompt", "") or "")
        req_body = _safe_body({
            "prompt": getattr(event, "prompt", None),
            "additional_kwargs": _dump(getattr(event, "additional_kwargs", None)),
        })
        if req_body:
            rec["attrs"]["gen_ai.request.body"] = req_body

    def _on_llm_completion_end(self, event: Any) -> None:
        rec = self._spans.get(event.span_id)
        if rec is None:
            return
        rec["ended"] = True
        resp = getattr(event, "response", None)
        if resp is None:
            return
        rec["completion"] = _truncate(getattr(resp, "text", "") or "")
        tin, tout = _extract_tokens(resp)
        if tin:
            rec["tokens_in"] = tin
        if tout:
            rec["tokens_out"] = tout
        body = _safe_body(_dump(resp))
        if body:
            rec["attrs"]["gen_ai.response.body"] = body

    def _on_embedding_start(self, event: Any) -> None:
        rec = self._spans.get(event.span_id)
        if rec is None:
            return
        model = _model_from_dict(getattr(event, "model_dict", None))
        rec["span_type"] = "llm_call"
        rec["operation"] = "embedding"
        if model:
            rec["model"] = model

    def _on_embedding_end(self, event: Any) -> None:
        rec = self._spans.get(event.span_id)
        if rec is None:
            return
        chunks = getattr(event, "chunks", None) or []
        embeddings = getattr(event, "embeddings", None) or []
        rec["attrs"]["evalkit.embedding.chunk_count"] = len(chunks)
        if embeddings:
            try:
                rec["attrs"]["evalkit.embedding.dimensions"] = len(embeddings[0])
            except Exception:
                pass
        preview = _truncate("\n".join(str(c) for c in chunks[:5]))
        if preview:
            rec["prompt"] = preview

    def _on_retrieval_start(self, event: Any) -> None:
        rec = self._spans.get(event.span_id)
        if rec is None:
            return
        rec["span_type"] = "db_query"
        rec["operation"] = "retrieve"
        rec["prompt"] = _query_str(getattr(event, "str_or_query_bundle", None))

    def _on_retrieval_end(self, event: Any) -> None:
        rec = self._spans.get(event.span_id)
        if rec is None:
            return
        nodes = getattr(event, "nodes", None) or []
        matches = _nodes_summary(nodes)
        rec["attrs"]["db.vector.match_count"] = len(matches)
        body = _safe_body({"matches": matches})
        if body:
            rec["attrs"]["gen_ai.response.body"] = body

    def _on_query_start(self, event: Any) -> None:
        rec = self._spans.get(event.span_id)
        if rec is None:
            return
        rec["span_type"] = "function_call"
        rec["operation"] = "query"
        rec["prompt"] = _query_str(getattr(event, "query", None))

    def _on_query_end(self, event: Any) -> None:
        rec = self._spans.get(event.span_id)
        if rec is None:
            return
        resp = getattr(event, "response", None)
        text = getattr(resp, "response", None) if resp is not None else None
        rec["completion"] = _truncate(text if text is not None else (resp or ""))
        source_nodes = getattr(resp, "source_nodes", None) or []
        rec["attrs"]["rag.source_node_count"] = len(source_nodes)

    def _on_rerank_start(self, event: Any) -> None:
        rec = self._spans.get(event.span_id)
        if rec is None:
            return
        rec["span_type"] = "function_call"
        rec["operation"] = "rerank"
        model_name = getattr(event, "model_name", None)
        if model_name:
            rec["model"] = model_name
            rec["attrs"]["rerank.model"] = model_name
        top_n = getattr(event, "top_n", None)
        if top_n is not None:
            rec["attrs"]["rerank.top_n"] = top_n
        nodes = getattr(event, "nodes", None) or []
        rec["attrs"]["rerank.input_count"] = len(nodes)
        q = getattr(event, "query", None)
        if q is not None:
            rec["prompt"] = _query_str(q)

    def _on_rerank_end(self, event: Any) -> None:
        rec = self._spans.get(event.span_id)
        if rec is None:
            return
        nodes = getattr(event, "nodes", None) or []
        rec["attrs"]["rerank.output_count"] = len(nodes)

    _EVENT_HANDLERS: Dict[str, Any] = {}


# Streamed chats whose span was left before their request ended; a stream the caller abandons with
# no request recorded never ends, so only this many are kept waiting.
_MAX_PENDING_STREAMS = 256

_Tracker._EVENT_HANDLERS = {
    "LLMChatStartEvent": _Tracker._on_llm_chat_start,
    "LLMChatEndEvent": _Tracker._on_llm_chat_end,
    "LLMCompletionStartEvent": _Tracker._on_llm_completion_start,
    "LLMCompletionEndEvent": _Tracker._on_llm_completion_end,
    "EmbeddingStartEvent": _Tracker._on_embedding_start,
    "EmbeddingEndEvent": _Tracker._on_embedding_end,
    "RetrievalStartEvent": _Tracker._on_retrieval_start,
    "RetrievalEndEvent": _Tracker._on_retrieval_end,
    "QueryStartEvent": _Tracker._on_query_start,
    "QueryEndEvent": _Tracker._on_query_end,
    "ReRankStartEvent": _Tracker._on_rerank_start,
    "ReRankEndEvent": _Tracker._on_rerank_end,
}


def patch_llamaindex(client: "EvalKitClient") -> Optional[_Tracker]:
    global _registered
    if _registered:
        return None

    try:
        from llama_index.core.instrumentation import get_dispatcher
        from llama_index.core.instrumentation.event_handlers import BaseEventHandler
        from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
    except Exception:
        return None

    tracker = _Tracker(client)

    class _EvalKitEventHandler(BaseEventHandler):
        @classmethod
        def class_name(cls) -> str:
            return "EvalKitEventHandler"

        def handle(self, event: Any, **kwargs: Any) -> Any:
            try:
                tracker.on_event(event)
            except Exception:
                pass

    class _EvalKitSpanHandler(BaseSpanHandler):
        @classmethod
        def class_name(cls) -> str:
            return "EvalKitSpanHandler"

        def new_span(self, id_: str, bound_args: Any, instance: Any = None,
                     parent_span_id: Optional[str] = None, tags: Any = None, **kwargs: Any) -> None:
            try:
                tracker.on_span_enter(id_, parent_span_id, instance)
            except Exception:
                pass
            return None

        def prepare_to_exit_span(self, id_: str, bound_args: Any, instance: Any = None,
                                  result: Any = None, **kwargs: Any) -> None:
            try:
                tracker.on_span_exit(id_)
            except Exception:
                pass
            return None

        def prepare_to_drop_span(self, id_: str, bound_args: Any, instance: Any = None,
                                  err: Any = None, **kwargs: Any) -> None:
            try:
                tracker.on_span_drop(id_, err)
            except Exception:
                pass
            return None

    try:
        dispatcher = get_dispatcher()
        dispatcher.add_event_handler(_EvalKitEventHandler())
        dispatcher.add_span_handler(_EvalKitSpanHandler())
    except Exception as exc:
        _log.debug("[evalkit] llama_index registration failed: %s", exc)
        return None

    _registered = True
    return tracker
