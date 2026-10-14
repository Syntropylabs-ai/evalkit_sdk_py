"""The queue goes out before an ASGI application reports its lifespan shutdown complete.

A `uvicorn --workers` or `--reload` child imports the application after uvicorn took SIGTERM and
SIGINT over, so the handler `init()` chains to is uvicorn's own. Once stopped, uvicorn puts back the
handler it found, the default one, and raises the captured signal again: the process ends right
after the shutdown report, without running atexit. The lifespan shutdown is the last moment the
application runs, after the requests in flight finished and its own shutdown hooks ran.
"""
from __future__ import annotations

import asyncio
import functools
import weakref
from typing import Any, Callable, List

_SHUTDOWN_REPLIES = ("lifespan.shutdown.complete", "lifespan.shutdown.failed")

# Clients handed to an integration explicitly; a patch made on a framework's class serves every app
# of that class in the process, so it flushes each of them and the one init() made.
_explicit_clients: "weakref.WeakSet[Any]" = weakref.WeakSet()


def register_client(client: Any) -> None:
    if client is not None:
        try:
            _explicit_clients.add(client)
        except Exception:
            pass


def registered_clients() -> List[Any]:
    clients = list(_explicit_clients)
    try:
        import evalkit as _ek

        instance = _ek._get_instance()
    except Exception:
        instance = None
    if instance is not None and all(c is not instance for c in clients):
        clients.append(instance)
    return clients


async def flush_for_shutdown(get_clients: Callable[[], Any]) -> None:
    """Send each client's queue within the exit budget, off the event loop; never raises."""
    try:
        clients = list(get_clients())
    except Exception:
        return
    for client in clients:
        try:
            await asyncio.get_running_loop().run_in_executor(None, client.flush_before_exit)
        except Exception:
            pass


def flush_before_shutdown_reply(send: Callable, get_clients: Callable[[], Any]) -> Callable:
    """`send` for a lifespan scope: the flush runs before the shutdown reply goes to the server."""
    async def send_after_flush(message: Any) -> None:
        if isinstance(message, dict) and message.get("type") in _SHUTDOWN_REPLIES:
            await flush_for_shutdown(get_clients)
        await send(message)

    return send_after_flush


def _is_lifespan(scope: Any) -> bool:
    return isinstance(scope, dict) and scope.get("type") == "lifespan"


def flush_on_lifespan_shutdown(app_cls: type, get_clients: Callable[[], Any]) -> bool:
    """Make an ASGI application class flush before its lifespan shutdown reply.

    For a framework whose middleware never sees the lifespan scope: Litestar answers lifespan in its
    own `__call__`, before any middleware. Everything else passes through as it came. The wrapper is
    a coroutine function, as servers check for ASGI 3. Returns whether the class is wrapped.
    """
    original = app_cls.__dict__.get("__call__")
    if original is None or not callable(original):
        return False
    if getattr(original, "_evalkit_lifespan", False):
        return True

    @functools.wraps(original)
    async def __call__(self: Any, scope: Any, receive: Callable, send: Callable) -> Any:
        if _is_lifespan(scope):
            try:
                send = flush_before_shutdown_reply(send, get_clients)
            except Exception:
                pass
        return await original(self, scope, receive, send)

    __call__._evalkit_lifespan = True  # type: ignore[attr-defined]
    try:
        setattr(app_cls, "__call__", __call__)
    except Exception:
        return False
    return True


async def _answer_lifespan(receive: Callable, send: Callable, get_clients: Callable[[], Any]) -> None:
    """The lifespan protocol for an application that has none: start at once, flush at shutdown."""
    while True:
        message = await receive()
        kind = message.get("type") if isinstance(message, dict) else None
        if kind == "lifespan.startup":
            await send({"type": "lifespan.startup.complete"})
        elif kind == "lifespan.shutdown":
            await flush_for_shutdown(get_clients)
            await send({"type": "lifespan.shutdown.complete"})
            return
        else:
            return


def answer_lifespan_when_refused(app_cls: type, get_clients: Callable[[], Any]) -> bool:
    """Give an ASGI application class that refuses the lifespan scope a lifespan with the flush.

    Django's ASGI handler raises for any scope but HTTP, and the server then runs without lifespan,
    so nothing of the application runs once the server starts stopping. The scope still goes to the
    application first: when it reads or sends a lifespan message itself it keeps the protocol, with
    the flush before its shutdown reply; only a refusal raised before any message is answered here.
    """
    original = app_cls.__dict__.get("__call__")
    if original is None or not callable(original):
        return False
    if getattr(original, "_evalkit_lifespan", False):
        return True

    @functools.wraps(original)
    async def __call__(self: Any, scope: Any, receive: Callable, send: Callable) -> Any:
        if not _is_lifespan(scope):
            return await original(self, scope, receive, send)
        exchanged = []

        async def receive_seen() -> Any:
            exchanged.append(True)
            return await receive()

        async def send_seen(message: Any) -> None:
            exchanged.append(True)
            await send(message)

        try:
            return await original(self, scope, receive_seen, flush_before_shutdown_reply(send_seen, get_clients))
        except ValueError:
            if exchanged:
                raise
        await _answer_lifespan(receive, send, get_clients)

    __call__._evalkit_lifespan = True  # type: ignore[attr-defined]
    try:
        setattr(app_cls, "__call__", __call__)
    except Exception:
        return False
    return True
