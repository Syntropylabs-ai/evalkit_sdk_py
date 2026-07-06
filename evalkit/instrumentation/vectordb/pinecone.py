"""Patch ``pinecone.Index.query``/``.upsert`` at the class level.

RAG pipelines retrieve context from a vector DB before generating a response —
without this, that lookup is invisible (only the downstream LLM call gets
traced), which makes it impossible to tell whether a bad answer came from
retrieval (wrong/no matches) or generation (bad prompt, bad model). Emits
``db_query`` spans using the same shared helper every other DB integration in
this SDK uses (``instrumentation/db/_common.py``), with vector-specific
attributes: namespace, top_k, match count/scores. Never captures raw embedding
vectors — they're large and not useful for a trace.

Degrades to a no-op if ``pinecone`` isn't installed and never raises into the
caller.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..db._common import start_db
from ..llm._bodies import _safe_body

if TYPE_CHECKING:
    from ...core.client import EvalKitClient


def _matches_summary(response: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        for m in getattr(response, "matches", None) or []:
            out.append({"id": getattr(m, "id", None), "score": getattr(m, "score", None)})
    except Exception:
        pass
    return out


def patch_pinecone(client: "EvalKitClient") -> None:
    """Patch ``pinecone.data.index.Index`` (aka ``pinecone.Index``). Idempotent."""
    try:
        from pinecone import Index
    except ImportError:
        return
    try:
        _patch_query(client, Index)
    except Exception:
        pass
    try:
        _patch_upsert(client, Index)
    except Exception:
        pass


def _patch_query(ek: "EvalKitClient", Index: Any) -> None:
    if getattr(getattr(Index, "query", None), "_evalkit", False):
        return
    original = Index.query

    def traced_query(self: Any, *args: Any, **kwargs: Any) -> Any:
        namespace = kwargs.get("namespace") or ""
        top_k = kwargs.get("top_k")
        end = start_db(ek, "pinecone", operation="query")
        try:
            result = original(self, *args, **kwargs)
        except Exception as exc:
            end("ERROR", error=exc)
            raise

        matches = _matches_summary(result)
        extra: Dict[str, Any] = {
            "db.vector.namespace": namespace,
            "db.vector.match_count": len(matches),
        }
        if top_k is not None:
            extra["db.vector.top_k"] = top_k
        req_body = _safe_body({k: v for k, v in kwargs.items() if k != "vector"})
        if req_body:
            extra["gen_ai.request.body"] = req_body
        resp_body = _safe_body({"matches": matches, "namespace": getattr(result, "namespace", namespace)})
        if resp_body:
            extra["gen_ai.response.body"] = resp_body
        end("OK", extra=extra, rows=len(matches))
        return result

    traced_query._evalkit = True  # type: ignore[attr-defined]
    Index.query = traced_query


def _patch_upsert(ek: "EvalKitClient", Index: Any) -> None:
    if getattr(getattr(Index, "upsert", None), "_evalkit", False):
        return
    original = Index.upsert

    def traced_upsert(self: Any, *args: Any, **kwargs: Any) -> Any:
        namespace = kwargs.get("namespace") or ""
        vectors = kwargs.get("vectors")
        vector_count: Optional[int] = None
        try:
            vector_count = len(vectors) if vectors is not None else None
        except Exception:
            vector_count = None
        end = start_db(ek, "pinecone", operation="upsert")
        try:
            result = original(self, *args, **kwargs)
        except Exception as exc:
            end("ERROR", error=exc)
            raise

        extra: Dict[str, Any] = {"db.vector.namespace": namespace}
        if vector_count is not None:
            extra["db.vector.count"] = vector_count
        upserted = getattr(result, "upserted_count", None)
        end("OK", extra=extra, rows=upserted)
        return result

    traced_upsert._evalkit = True  # type: ignore[attr-defined]
    Index.upsert = traced_upsert
