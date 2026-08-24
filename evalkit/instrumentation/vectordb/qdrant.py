from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..db._common import start_db
from ..llm._bodies import _safe_body

if TYPE_CHECKING:
    from ...core.client import EvalKitClient


def _points_summary(response: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        points = getattr(response, "points", None)
        if points is None:
            points = response
        for p in points or []:
            out.append({"id": getattr(p, "id", None), "score": getattr(p, "score", None)})
    except Exception:
        pass
    return out


def patch_qdrant(client: "EvalKitClient") -> None:
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        return
    try:
        _patch_query(client, QdrantClient)
    except Exception:
        pass
    try:
        _patch_upsert(client, QdrantClient)
    except Exception:
        pass


def _patch_query(ek: "EvalKitClient", QdrantClient: Any) -> None:
    if getattr(getattr(QdrantClient, "query_points", None), "_evalkit", False):
        return
    original = QdrantClient.query_points

    def traced_query_points(self: Any, collection_name: str, *args: Any, **kwargs: Any) -> Any:
        limit = kwargs.get("limit", 10)
        end = start_db(ek, "qdrant", operation="query")
        try:
            result = original(self, collection_name, *args, **kwargs)
        except Exception as exc:
            end("ERROR", error=exc)
            raise

        points = _points_summary(result)
        extra: Dict[str, Any] = {
            "db.vector.collection": collection_name,
            "db.vector.top_k": limit,
            "db.vector.match_count": len(points),
        }
        req_body = _safe_body({k: v for k, v in kwargs.items() if k != "query"})
        if req_body:
            extra["gen_ai.request.body"] = req_body
        resp_body = _safe_body({"points": points})
        if resp_body:
            extra["gen_ai.response.body"] = resp_body
        end("OK", extra=extra, rows=len(points))
        return result

    traced_query_points._evalkit = True
    QdrantClient.query_points = traced_query_points


def _patch_upsert(ek: "EvalKitClient", QdrantClient: Any) -> None:
    if getattr(getattr(QdrantClient, "upsert", None), "_evalkit", False):
        return
    original = QdrantClient.upsert

    def traced_upsert(self: Any, collection_name: str, points: Any, *args: Any, **kwargs: Any) -> Any:
        point_count: Optional[int] = None
        try:
            point_count = len(points)
        except Exception:
            point_count = None
        end = start_db(ek, "qdrant", operation="upsert")
        try:
            result = original(self, collection_name, points, *args, **kwargs)
        except Exception as exc:
            end("ERROR", error=exc)
            raise

        extra: Dict[str, Any] = {"db.vector.collection": collection_name}
        if point_count is not None:
            extra["db.vector.count"] = point_count
        status = getattr(result, "status", None)
        if status is not None:
            extra["db.vector.status"] = str(status)
        end("OK", extra=extra, rows=point_count)
        return result

    traced_upsert._evalkit = True
    QdrantClient.upsert = traced_upsert
