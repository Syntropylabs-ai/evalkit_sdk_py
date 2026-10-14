"""Health and readiness probes reach the app untraced.

Load balancers and orchestrators poll them every few seconds, so each would be a trace of its own that
says nothing about the app. A probe is told by whole path segment (/health, /healthz, /health/live,
/readyz, /livez), so /healthcare and /api/health-records are the app's own routes and are recorded: the
rule outgoing calls are skipped by, and the TypeScript SDK's. While a probe is answered, nothing it does
is recorded either; work it starts that runs on after the response is traced as usual.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator

from ..core.context import end_suppression, suppress_tracing
from ..instrumentation.http._skip import is_health_check


def is_probe(path: Any) -> bool:
    try:
        return isinstance(path, str) and bool(path) and is_health_check(path)
    except Exception:
        return False


def _suppress() -> Any:
    try:
        return suppress_tracing()
    except Exception:
        return None, None


@contextmanager
def probe_untraced() -> Iterator[None]:
    """Hold tracing off while a WSGI framework answers a probe."""
    token, held = _suppress()
    try:
        yield
    finally:
        end_suppression(token, held)


async def answer_probe_untraced(app: Callable, scope: Any, receive: Callable, send: Callable) -> None:
    """Run an ASGI app for a probe, untraced until its response is complete.

    Starlette runs a response's background tasks after the last body message, inside the same call,
    so tracing comes back on as soon as that message is sent.
    """
    token, held = _suppress()

    async def send_until_answered(message: Any) -> None:
        await send(message)
        try:
            if (held is not None and message.get("type") == "http.response.body"
                    and not message.get("more_body", False)):
                held.active = False
        except Exception:
            pass

    try:
        await app(scope, receive, send_until_answered if held is not None else send)
    finally:
        end_suppression(token, held)
