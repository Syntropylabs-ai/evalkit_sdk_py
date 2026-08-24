from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..db._common import start_db
from ..llm._bodies import _safe_body

if TYPE_CHECKING:
    from ...core.client import EvalKitClient


def _query_result_summary(result: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        ids = (result or {}).get("ids") or []
        distances = (result or {}).get("distances") or []
        for batch_idx, batch_ids in enumerate(ids):
            batch_distances = distances[batch_idx] if batch_idx < len(distances) else []
            for i, doc_id in enumerate(batch_ids):
                dist = batch_distances[i] if i < len(batch_distances) else None
                out.append({"id": doc_id, "distance": dist})
    except Exception:
        pass
    return out


def patch_chroma(client: "EvalKitClient") -> None:
    try:
        from chromadb.api.models.Collection import Collection
    except ImportError:
        return
    try:
        _patch_query(client, Collection)
    except Exception:
        pass
    try:
        _patch_add(client, Collection)
    except Exception:
        pass
    try:
        _patch_upsert(client, Collection)
    except Exception:
        pass


def _patch_query(ek: "EvalKitClient", Collection: Any) -> None:
    if getattr(getattr(Collection, "query", None), "_evalkit", False):
        return
    original = Collection.query

    def traced_query(self: Any, *args: Any, **kwargs: Any) -> Any:
        n_results = kwargs.get("n_results", 10)
        collection_name = getattr(self, "name", None)
        end = start_db(ek, "chroma", operation="query")
        try:
            result = original(self, *args, **kwargs)
        except Exception as exc:
            end("ERROR", error=exc)
            raise

        matches = _query_result_summary(result)
        extra: Dict[str, Any] = {
            "db.vector.top_k": n_results,
            "db.vector.match_count": len(matches),
        }
        if collection_name:
            extra["db.vector.collection"] = collection_name
        req_body = _safe_body({k: v for k, v in kwargs.items() if k not in ("query_embeddings", "query_texts")})
        if req_body:
            extra["gen_ai.request.body"] = req_body
        resp_body = _safe_body({"matches": matches})
        if resp_body:
            extra["gen_ai.response.body"] = resp_body
        end("OK", extra=extra, rows=len(matches))
        return result

    traced_query._evalkit = True
    Collection.query = traced_query


def _patch_add(ek: "EvalKitClient", Collection: Any) -> None:
    if getattr(getattr(Collection, "add", None), "_evalkit", False):
        return
    original = Collection.add

    def traced_add(self: Any, ids: Any, *args: Any, **kwargs: Any) -> Any:
        collection_name = getattr(self, "name", None)
        count: Optional[int] = None
        try:
            count = 1 if isinstance(ids, str) else len(ids)
        except Exception:
            count = None
        end = start_db(ek, "chroma", operation="add")
        try:
            result = original(self, ids, *args, **kwargs)
        except Exception as exc:
            end("ERROR", error=exc)
            raise

        extra: Dict[str, Any] = {}
        if collection_name:
            extra["db.vector.collection"] = collection_name
        if count is not None:
            extra["db.vector.count"] = count
        end("OK", extra=extra, rows=count)
        return result

    traced_add._evalkit = True
    Collection.add = traced_add


def _patch_upsert(ek: "EvalKitClient", Collection: Any) -> None:
    if getattr(getattr(Collection, "upsert", None), "_evalkit", False):
        return
    original = Collection.upsert

    def traced_upsert(self: Any, ids: Any, *args: Any, **kwargs: Any) -> Any:
        collection_name = getattr(self, "name", None)
        count: Optional[int] = None
        try:
            count = 1 if isinstance(ids, str) else len(ids)
        except Exception:
            count = None
        end = start_db(ek, "chroma", operation="upsert")
        try:
            result = original(self, ids, *args, **kwargs)
        except Exception as exc:
            end("ERROR", error=exc)
            raise

        extra: Dict[str, Any] = {}
        if collection_name:
            extra["db.vector.collection"] = collection_name
        if count is not None:
            extra["db.vector.count"] = count
        end("OK", extra=extra, rows=count)
        return result

    traced_upsert._evalkit = True
    Collection.upsert = traced_upsert
