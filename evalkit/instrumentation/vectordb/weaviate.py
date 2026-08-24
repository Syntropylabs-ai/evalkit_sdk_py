from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..db._common import start_db
from ..llm._bodies import _safe_body

if TYPE_CHECKING:
    from ...core.client import EvalKitClient


def _objects_summary(result: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        for o in getattr(result, "objects", None) or []:
            meta = getattr(o, "metadata", None)
            entry: Dict[str, Any] = {"id": str(getattr(o, "uuid", None))}
            if meta is not None:
                for key in ("distance", "certainty", "score"):
                    v = getattr(meta, key, None)
                    if v is not None:
                        entry[key] = v
            out.append(entry)
    except Exception:
        pass
    return out


def patch_weaviate(client: "EvalKitClient") -> None:
    try:
        import weaviate
    except ImportError:
        return

    try:
        from weaviate.collections.queries.near_vector.query.sync import _NearVectorQuery
        _patch_near_vector_sync(client, _NearVectorQuery)
    except Exception:
        pass
    try:
        from weaviate.collections.queries.near_vector.query.async_ import _NearVectorQueryAsync
        _patch_near_vector_async(client, _NearVectorQueryAsync)
    except Exception:
        pass
    try:
        from weaviate.collections.data.sync import _DataCollection
        _patch_data_sync(client, _DataCollection)
    except Exception:
        pass
    try:
        from weaviate.collections.data.async_ import _DataCollectionAsync
        _patch_data_async(client, _DataCollectionAsync)
    except Exception:
        pass


def _query_extra(self: Any, near_vector: Any, kwargs: Dict[str, Any], result: Any) -> Dict[str, Any]:
    collection_name = getattr(self, "_name", None)
    limit = kwargs.get("limit")
    matches = _objects_summary(result)
    extra: Dict[str, Any] = {"db.vector.match_count": len(matches)}
    if collection_name:
        extra["db.vector.collection"] = collection_name
    if limit is not None:
        extra["db.vector.top_k"] = limit
    req_body = _safe_body({k: v for k, v in kwargs.items() if k != "near_vector"})
    if req_body:
        extra["gen_ai.request.body"] = req_body
    resp_body = _safe_body({"objects": matches})
    if resp_body:
        extra["gen_ai.response.body"] = resp_body
    return extra


def _patch_near_vector_sync(ek: "EvalKitClient", QueryCls: Any) -> None:
    if getattr(getattr(QueryCls, "near_vector", None), "_evalkit", False):
        return
    original = QueryCls.near_vector

    def traced_near_vector(self: Any, near_vector: Any = None, **kwargs: Any) -> Any:
        end = start_db(ek, "weaviate", operation="query")
        try:
            result = original(self, near_vector, **kwargs)
        except Exception as exc:
            end("ERROR", error=exc)
            raise
        end("OK", extra=_query_extra(self, near_vector, kwargs, result))
        return result

    traced_near_vector._evalkit = True
    QueryCls.near_vector = traced_near_vector


def _patch_near_vector_async(ek: "EvalKitClient", QueryCls: Any) -> None:
    if getattr(getattr(QueryCls, "near_vector", None), "_evalkit", False):
        return
    original = QueryCls.near_vector

    def traced_near_vector(self: Any, near_vector: Any = None, **kwargs: Any) -> Any:
        end = start_db(ek, "weaviate", operation="query")

        async def _run() -> Any:
            try:
                result = await original(self, near_vector, **kwargs)
            except Exception as exc:
                end("ERROR", error=exc)
                raise
            end("OK", extra=_query_extra(self, near_vector, kwargs, result))
            return result

        return _run()

    traced_near_vector._evalkit = True
    QueryCls.near_vector = traced_near_vector


def _patch_data_sync(ek: "EvalKitClient", DataCls: Any) -> None:
    _patch_insert(ek, DataCls, is_async=False)
    _patch_insert_many(ek, DataCls, is_async=False)


def _patch_data_async(ek: "EvalKitClient", DataCls: Any) -> None:
    _patch_insert(ek, DataCls, is_async=True)
    _patch_insert_many(ek, DataCls, is_async=True)


def _patch_insert(ek: "EvalKitClient", DataCls: Any, *, is_async: bool) -> None:
    if getattr(getattr(DataCls, "insert", None), "_evalkit", False):
        return
    original = DataCls.insert

    def _extra(self: Any) -> Dict[str, Any]:
        extra: Dict[str, Any] = {"db.vector.count": 1}
        collection_name = getattr(self, "_name", None)
        if collection_name:
            extra["db.vector.collection"] = collection_name
        return extra

    if is_async:
        def traced_insert(self: Any, *args: Any, **kwargs: Any) -> Any:
            end = start_db(ek, "weaviate", operation="insert")

            async def _run() -> Any:
                try:
                    result = await original(self, *args, **kwargs)
                except Exception as exc:
                    end("ERROR", error=exc)
                    raise
                end("OK", extra=_extra(self), rows=1)
                return result

            return _run()
    else:
        def traced_insert(self: Any, *args: Any, **kwargs: Any) -> Any:
            end = start_db(ek, "weaviate", operation="insert")
            try:
                result = original(self, *args, **kwargs)
            except Exception as exc:
                end("ERROR", error=exc)
                raise
            end("OK", extra=_extra(self), rows=1)
            return result

    traced_insert._evalkit = True
    DataCls.insert = traced_insert


def _patch_insert_many(ek: "EvalKitClient", DataCls: Any, *, is_async: bool) -> None:
    if getattr(getattr(DataCls, "insert_many", None), "_evalkit", False):
        return
    original = DataCls.insert_many

    def _extra(self: Any, objects: Any) -> Dict[str, Any]:
        extra: Dict[str, Any] = {}
        try:
            extra["db.vector.count"] = len(objects)
        except Exception:
            pass
        collection_name = getattr(self, "_name", None)
        if collection_name:
            extra["db.vector.collection"] = collection_name
        return extra

    if is_async:
        def traced_insert_many(self: Any, objects: Any, *args: Any, **kwargs: Any) -> Any:
            end = start_db(ek, "weaviate", operation="insert_many")

            async def _run() -> Any:
                try:
                    result = await original(self, objects, *args, **kwargs)
                except Exception as exc:
                    end("ERROR", error=exc)
                    raise
                extra = _extra(self, objects)
                failed = getattr(result, "has_errors", None)
                end("ERROR" if failed else "OK", extra=extra, rows=extra.get("db.vector.count"))
                return result

            return _run()
    else:
        def traced_insert_many(self: Any, objects: Any, *args: Any, **kwargs: Any) -> Any:
            end = start_db(ek, "weaviate", operation="insert_many")
            try:
                result = original(self, objects, *args, **kwargs)
            except Exception as exc:
                end("ERROR", error=exc)
                raise
            extra = _extra(self, objects)
            failed = getattr(result, "has_errors", None)
            end("ERROR" if failed else "OK", extra=extra, rows=extra.get("db.vector.count"))
            return result

    traced_insert_many._evalkit = True
    DataCls.insert_many = traced_insert_many
