from __future__ import annotations

from typing import Any

from ._common import start_db

_METHODS = [
    "insert_one", "insert_many", "update_one", "update_many",
    "delete_one", "delete_many", "find_one", "find_one_and_update",
    "find_one_and_delete", "find_one_and_replace", "replace_one",
    "count_documents", "aggregate", "distinct", "bulk_write",
]


def patch_pymongo(client: Any) -> None:
    try:
        from pymongo.collection import Collection
        for name in _METHODS:
            _wrap_sync(client, Collection, name)
    except ImportError:
        pass
    except Exception:
        pass
    try:
        from motor.motor_asyncio import AsyncIOMotorCollection
        for name in _METHODS:
            _wrap_async(client, AsyncIOMotorCollection, name)
    except ImportError:
        pass
    except Exception:
        pass


def _coll_name(self: Any) -> str:
    return str(getattr(self, "name", None) or "collection")


def _wrap_sync(client: Any, cls: Any, name: str) -> None:
    orig = getattr(cls, name, None)
    if orig is None or getattr(orig, "_evalkit", False):
        return

    def traced(self: Any, *args: Any, **kwargs: Any) -> Any:
        end = start_db(client, "mongodb", operation=name, table=_coll_name(self))
        try:
            r = orig(self, *args, **kwargs)
        except Exception as e:
            end("ERROR", error=e)
            raise
        end("OK")
        return r

    try:
        traced._evalkit = True
        setattr(cls, name, traced)
    except Exception:
        pass


def _wrap_async(client: Any, cls: Any, name: str) -> None:
    orig = getattr(cls, name, None)
    if orig is None or getattr(orig, "_evalkit", False):
        return

    async def traced(self: Any, *args: Any, **kwargs: Any) -> Any:
        end = start_db(client, "mongodb", operation=name, table=_coll_name(self))
        try:
            r = await orig(self, *args, **kwargs)
        except Exception as e:
            end("ERROR", error=e)
            raise
        end("OK")
        return r

    try:
        traced._evalkit = True
        setattr(cls, name, traced)
    except Exception:
        pass
