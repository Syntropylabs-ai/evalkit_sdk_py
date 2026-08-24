from __future__ import annotations

from typing import Any

from ._common import start_db


def _stmt(query: Any) -> str:
    try:
        return query.decode() if isinstance(query, (bytes, bytearray)) else str(query)
    except Exception:
        return ""


def patch_psycopg(client: Any) -> None:
    try:
        from psycopg import Cursor, AsyncCursor
    except ImportError:
        return
    _wrap_sync(client, Cursor)
    _wrap_async(client, AsyncCursor)


def _wrap_sync(client: Any, cls: Any) -> None:
    orig = getattr(cls, "execute", None)
    if orig is None or getattr(orig, "_evalkit", False):
        return

    def traced(self: Any, query: Any, params: Any = None, **kwargs: Any) -> Any:
        end = start_db(client, "postgresql", statement=_stmt(query))
        try:
            r = orig(self, query, params, **kwargs)
        except Exception as e:
            end("ERROR", error=e)
            raise
        end("OK", rows=getattr(self, "rowcount", None))
        return r

    traced._evalkit = True
    cls.execute = traced


def _wrap_async(client: Any, cls: Any) -> None:
    orig = getattr(cls, "execute", None)
    if orig is None or getattr(orig, "_evalkit", False):
        return

    async def traced(self: Any, query: Any, params: Any = None, **kwargs: Any) -> Any:
        end = start_db(client, "postgresql", statement=_stmt(query))
        try:
            r = await orig(self, query, params, **kwargs)
        except Exception as e:
            end("ERROR", error=e)
            raise
        end("OK", rows=getattr(self, "rowcount", None))
        return r

    traced._evalkit = True
    cls.execute = traced
