"""Exceptions nothing in the application caught, recorded as error spans when ``capture_unhandled=True``.

Each hook records the exception and then hands it to whatever hook was installed before, so the
application's own handler, or Python's default output, runs exactly as it would have. A hook that
fails to record never stops the one it chains to.
"""
from __future__ import annotations

import logging
import sys
import threading
from typing import Any, Optional

_log = logging.getLogger("evalkit")

_installed = False
_client: Any = None


def _record(exc: Optional[BaseException], handler: str) -> None:
    client = _client
    if client is None or exc is None or not isinstance(exc, BaseException):
        return
    if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
        return
    try:
        from .._version import SDK_VERSION
        from .client import _new_span_id, _now_iso
        from .context import current_span_ctx, get_or_create_ambient_ctx
        from .errors import type_name
        from .types import TraceEnvelope

        ctx = current_span_ctx() or get_or_create_ambient_ctx()
        ts = _now_iso()
        attrs = {**client._meta_attrs(), "evalkit.exception.handler": handler, "evalkit.exception.unhandled": True}
        client.enqueue(TraceEnvelope(
            trace_id=ctx.trace_id,
            span_id=_new_span_id(),
            parent_span_id=ctx.span_id if current_span_ctx() is not None else None,
            span_type="log",
            status="ERROR",
            operation=f"unhandled {type_name(exc)}",
            service_name=client._opts.service_name,
            attributes=attrs,
            latency_ms=0,
            start_time=ts,
            end_time=ts,
            created_at=ts,
            environment=client._opts.environment,
            app_version=client._opts.app_version,
            sdk_version=SDK_VERSION,
            exception=exc,
            exception_escaped=True,
        ))
    except Exception as hook_exc:
        _log.debug("recording an unhandled exception failed: %s", hook_exc)


def _install_sys_hook() -> None:
    previous = sys.excepthook

    def excepthook(exc_type: Any, exc: Any, tb: Any) -> Any:
        try:
            _record(exc, "sys.excepthook")
            # The process ends after this hook; the exit flush sends the span.
        except Exception:
            pass
        return previous(exc_type, exc, tb)

    excepthook._evalkit = True  # type: ignore[attr-defined]
    excepthook.__wrapped__ = previous  # type: ignore[attr-defined]
    sys.excepthook = excepthook


def _install_thread_hook() -> None:
    previous = getattr(threading, "excepthook", None)
    if previous is None:
        return

    def excepthook(args: Any) -> Any:
        try:
            _record(getattr(args, "exc_value", None), "threading.excepthook")
        except Exception:
            pass
        return previous(args)

    excepthook._evalkit = True  # type: ignore[attr-defined]
    excepthook.__wrapped__ = previous  # type: ignore[attr-defined]
    threading.excepthook = excepthook


def _install_asyncio_hook() -> None:
    # Patched on the loop class, so every loop the application makes (asyncio.run makes a new one) is
    # covered, and the loop's own handler, set before or after, still runs: call_exception_handler is
    # what calls it.
    try:
        import asyncio.base_events as base_events
    except Exception:
        return
    loop_cls = base_events.BaseEventLoop
    original = loop_cls.call_exception_handler
    if getattr(original, "_evalkit", False):
        return

    def call_exception_handler(self: Any, context: Any) -> Any:
        try:
            exc = context.get("exception") if isinstance(context, dict) else None
            _record(exc, "asyncio")
        except Exception:
            pass
        return original(self, context)

    call_exception_handler._evalkit = True  # type: ignore[attr-defined]
    call_exception_handler.__wrapped__ = original  # type: ignore[attr-defined]
    loop_cls.call_exception_handler = call_exception_handler  # type: ignore[method-assign]


def install(client: Any) -> None:
    """Record uncaught exceptions of the main thread, of other threads and of asyncio tasks."""
    global _installed, _client
    _client = client
    if _installed:
        return
    _installed = True
    for step in (_install_sys_hook, _install_thread_hook, _install_asyncio_hook):
        try:
            step()
        except Exception as exc:
            _log.debug("installing an unhandled-exception hook failed: %s", exc)
