"""redis-py instrumentation (sync + asyncio) → db_query spans."""
from __future__ import annotations

from typing import Any

from ._common import start_db


def _stmt(args) -> str:
    try:
        return " ".join(str(a) for a in args[:8])
    except Exception:
        return ""


def patch_redis(client: Any) -> None:
    try:
        import redis  # noqa: F401
    except ImportError:
        return

    # Sync client
    try:
        from redis.client import Redis as SyncRedis
        _patch_sync(client, SyncRedis)
    except Exception:
        pass
    # Async client (redis.asyncio)
    try:
        from redis.asyncio.client import Redis as AsyncRedis
        _patch_async(client, AsyncRedis)
    except Exception:
        pass


def _patch_sync(client: Any, cls: Any) -> None:
    orig = getattr(cls, "execute_command", None)
    if orig is None or getattr(orig, "_evalkit", False):
        return

    def traced(self: Any, *args: Any, **kwargs: Any) -> Any:
        cmd = str(args[0]) if args else "CMD"
        end = start_db(client, "redis", operation=cmd, statement=_stmt(args))
        try:
            r = orig(self, *args, **kwargs)
        except Exception as e:
            end("ERROR", error=e)
            raise
        end("OK")
        return r

    traced._evalkit = True  # type: ignore[attr-defined]
    cls.execute_command = traced


def _patch_async(client: Any, cls: Any) -> None:
    orig = getattr(cls, "execute_command", None)
    if orig is None or getattr(orig, "_evalkit", False):
        return

    async def traced(self: Any, *args: Any, **kwargs: Any) -> Any:
        cmd = str(args[0]) if args else "CMD"
        end = start_db(client, "redis", operation=cmd, statement=_stmt(args))
        try:
            r = await orig(self, *args, **kwargs)
        except Exception as e:
            end("ERROR", error=e)
            raise
        end("OK")
        return r

    traced._evalkit = True  # type: ignore[attr-defined]
    cls.execute_command = traced
