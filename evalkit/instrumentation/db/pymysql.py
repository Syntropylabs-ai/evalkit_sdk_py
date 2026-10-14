from __future__ import annotations

from typing import Any

from ._common import orm_records_statement, start_db


def patch_pymysql(client: Any) -> None:
    try:
        import pymysql.cursors
    except ImportError:
        return
    cls = getattr(pymysql.cursors, "Cursor", None)
    if cls is None:
        return
    orig = getattr(cls, "execute", None)
    if orig is None or getattr(orig, "_evalkit", False):
        return

    def traced(self: Any, query: Any, args: Any = None) -> Any:
        if orm_records_statement():
            return orig(self, query, args)
        end = start_db(client, "mysql", statement=str(query))
        try:
            r = orig(self, query, args)
        except Exception as e:
            end("ERROR", error=e)
            raise
        end("OK", rows=r if isinstance(r, int) else None)
        return r

    traced._evalkit = True
    cls.execute = traced
