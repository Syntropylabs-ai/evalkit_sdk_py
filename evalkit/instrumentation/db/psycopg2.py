from __future__ import annotations

import os
import threading
from typing import Any, Callable, Dict, Optional, Tuple

from .._safe import hook_failed
from ._common import orm_records_statement, start_db

# psycopg2's connection and cursor are C types whose methods cannot be replaced, so tracing
# goes through the factories psycopg2 itself offers: connect() builds the connection from a
# subclass that hands every cursor() a traced subclass of the cursor class it would have used.
# The cursor keeps its class (RealDictCursor stays a RealDictCursor) and every method; only
# execute and executemany are observed on the way through.

_SYSTEM = "postgresql"
_client: Any = None
_lock = threading.Lock()
_cursor_classes: Dict[type, type] = {}
_connection_classes: Dict[type, type] = {}
_mixin: Optional[type] = None


def _reset_lock_after_fork() -> None:
    global _lock
    _lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_lock_after_fork)


def _stmt(query: Any, cursor: Any) -> str:
    try:
        if isinstance(query, (bytes, bytearray)):
            return bytes(query).decode("utf-8", errors="replace")
        as_string = getattr(query, "as_string", None)
        if callable(as_string):
            # A psycopg2.sql composition, rendered the way the server receives it.
            return str(as_string(cursor))
        return str(query)
    except Exception:
        return ""


def _start(query: Any, cursor: Any) -> Optional[Callable[..., None]]:
    client = _client
    if client is None:
        return None
    try:
        if orm_records_statement():
            return None
        return start_db(client, _SYSTEM, statement=_stmt(query, cursor))
    except Exception as exc:
        hook_failed("psycopg2", exc)
        return None


def _end(end: Optional[Callable[..., None]], status: str, cursor: Any = None, error: Any = None) -> None:
    if end is None:
        return
    try:
        rows = getattr(cursor, "rowcount", None) if error is None else None
        end(status, error=error, rows=rows if isinstance(rows, int) and rows >= 0 else None)
    except Exception as exc:
        hook_failed("psycopg2", exc)


def traced_cursor_class(base: type) -> type:
    """A subclass of the cursor class `base` whose execute and executemany are recorded."""
    if getattr(base, "_evalkit_traced_cursor", False):
        return base
    with _lock:
        traced = _cursor_classes.get(base)
        if traced is not None:
            return traced
        base_execute = base.execute
        base_executemany = base.executemany

        def execute(self: Any, query: Any, *args: Any, **kwargs: Any) -> Any:
            end = _start(query, self)
            try:
                result = base_execute(self, query, *args, **kwargs)
            except Exception as exc:
                _end(end, "ERROR", error=exc)
                raise
            _end(end, "OK", cursor=self)
            return result

        def executemany(self: Any, query: Any, *args: Any, **kwargs: Any) -> Any:
            end = _start(query, self)
            try:
                result = base_executemany(self, query, *args, **kwargs)
            except Exception as exc:
                _end(end, "ERROR", error=exc)
                raise
            _end(end, "OK", cursor=self)
            return result

        traced = type(base.__name__, (base,), {
            "__module__": base.__module__,
            "__qualname__": getattr(base, "__qualname__", base.__name__),
            "__doc__": base.__doc__,
            "execute": execute,
            "executemany": executemany,
            "_evalkit_traced_cursor": True,
        })
        _cursor_classes[base] = traced
        return traced


def _traced_factory_args(conn: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any], ext: Any) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    # connection.cursor(name=None, cursor_factory=None, withhold=False, scrollable=None)
    positional = len(args) >= 2
    factory = args[1] if positional else kwargs.get("cursor_factory")
    if factory is None:
        factory = getattr(conn, "cursor_factory", None) or ext.cursor
    if not (isinstance(factory, type) and issubclass(factory, ext.cursor)):
        # A factory function rather than a cursor class: used as given, untraced.
        return args, kwargs
    traced = traced_cursor_class(factory)
    if positional:
        return (args[0], traced) + tuple(args[2:]), kwargs
    return args, {**kwargs, "cursor_factory": traced}


def _connection_mixin(ext: Any) -> type:
    global _mixin
    if _mixin is not None:
        return _mixin
    base_cursor = ext.connection.cursor

    def cursor(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            args, kwargs = _traced_factory_args(self, args, kwargs, ext)
        except Exception as exc:
            hook_failed("psycopg2", exc)
        return base_cursor(self, *args, **kwargs)

    # Placed just above psycopg2's own connection in the method order, so it sees the cursor
    # factory a connection subclass such as DictConnection picks before passing the call down.
    _mixin = type("connection", (ext.connection,), {
        "__module__": ext.connection.__module__,
        "cursor": cursor,
        "_evalkit_traced_connection": True,
    })
    return _mixin


def traced_connection_class(base: type, ext: Any) -> type:
    if getattr(base, "_evalkit_traced_connection", False):
        return base
    mixin = _connection_mixin(ext)
    if base is ext.connection:
        return mixin
    with _lock:
        traced = _connection_classes.get(base)
        if traced is None:
            traced = type(base.__name__, (base, mixin), {
                "__module__": base.__module__,
                "__qualname__": getattr(base, "__qualname__", base.__name__),
                "__doc__": base.__doc__,
            })
            _connection_classes[base] = traced
        return traced


def _connection_attrs(conn: Any) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    try:
        info = getattr(conn, "info", None)
        dbname = getattr(info, "dbname", None)
        host = getattr(info, "host", None)
        port = getattr(info, "port", None)
        if dbname:
            attrs["db.name"] = str(dbname)
        if host:
            attrs["server.address"] = str(host)
        if isinstance(port, int) and port > 0:
            attrs["server.port"] = port
    except Exception:
        pass
    return attrs


def patch_psycopg2(client: Any) -> bool:
    """Trace psycopg2: each connect() and each cursor's execute and executemany, as db_query spans.

    Connections made through psycopg2.connect after this call are traced, including those of
    psycopg2.pool and of frameworks that call psycopg2.connect. Returns False when psycopg2 is not
    installed or could not be patched.
    """
    global _client
    try:
        import psycopg2
        import psycopg2.extensions as ext
    except ImportError:
        return False
    _client = client
    original = getattr(psycopg2, "connect", None)
    if original is None:
        return False
    if getattr(original, "_evalkit", False):
        return True
    try:
        _connection_mixin(ext)
    except Exception as exc:
        hook_failed("psycopg2", exc)
        return False

    def connect(dsn: Any = None, connection_factory: Any = None, cursor_factory: Any = None, **kwargs: Any) -> Any:
        try:
            factory = connection_factory
            if factory is None or (isinstance(factory, type) and issubclass(factory, ext.connection)):
                factory = traced_connection_class(factory or ext.connection, ext)
            end = start_db(_client, _SYSTEM, operation="CONNECT") if _client is not None else None
        except Exception as exc:
            hook_failed("psycopg2", exc)
            return original(dsn, connection_factory=connection_factory, cursor_factory=cursor_factory, **kwargs)
        try:
            conn = original(dsn, connection_factory=factory, cursor_factory=cursor_factory, **kwargs)
        except Exception as exc:
            _end(end, "ERROR", error=exc)
            raise
        if end is not None:
            try:
                end("OK", extra=_connection_attrs(conn))
            except Exception as exc:
                hook_failed("psycopg2", exc)
        return conn

    try:
        connect.__doc__ = getattr(original, "__doc__", None)
        connect.__wrapped__ = original  # type: ignore[attr-defined]
        connect._evalkit = True  # type: ignore[attr-defined]
        psycopg2.connect = connect
    except Exception as exc:
        hook_failed("psycopg2", exc)
        return False
    return True
