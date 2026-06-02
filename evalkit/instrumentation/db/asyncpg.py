"""asyncpg instrumentation (async Postgres, also used by Tortoise ORM) → db_query spans."""
from __future__ import annotations

from typing import Any

from ._common import start_db

_METHODS = ["execute", "executemany", "fetch", "fetchrow", "fetchval"]


def patch_asyncpg(client: Any) -> None:
    try:
        from asyncpg.connection import Connection
    except ImportError:
        return
    for name in _METHODS:
        _wrap(client, Connection, name)


def _wrap(client: Any, cls: Any, name: str) -> None:
    orig = getattr(cls, name, None)
    if orig is None or getattr(orig, "_evalkit", False):
        return

    async def traced(self: Any, query: Any = "", *args: Any, **kwargs: Any) -> Any:
        stmt = query if isinstance(query, str) else None
        end = start_db(client, "postgresql", statement=stmt,
                       operation=None if stmt else name.upper())
        try:
            r = await orig(self, query, *args, **kwargs)
        except Exception as e:
            end("ERROR", error=e)
            raise
        end("OK")
        return r

    try:
        traced._evalkit = True  # type: ignore[attr-defined]
        setattr(cls, name, traced)
    except Exception:
        pass
