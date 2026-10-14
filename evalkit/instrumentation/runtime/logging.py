from __future__ import annotations

import logging
import traceback as tb_module
from contextvars import ContextVar
from typing import Any, Optional

from ..._version import SDK_VERSION
from ...core.client import EvalKitClient, _new_span_id, _now_iso
from ...core.context import _current_span, current_span_ctx, get_or_create_ambient_ctx
from ...core.types import TraceEnvelope


class EvalKitLogFilter(logging.Filter):

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = current_span_ctx() or get_or_create_ambient_ctx()
        record.trace_id = ctx.trace_id
        record.span_id  = ctx.span_id
        return True


class EvalKitLoggingHandler(logging.Handler):

    _LEVEL_MAP = {
        logging.DEBUG:    "DEBUG",
        logging.INFO:     "INFO",
        logging.WARNING:  "WARN",
        logging.ERROR:    "ERROR",
        logging.CRITICAL: "CRITICAL",
    }

    def __init__(self, client: EvalKitClient, level: int = logging.DEBUG) -> None:
        super().__init__(level)
        self._client = client

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.record(record)
        except Exception:
            self.handleError(record)

    def record(self, record: logging.LogRecord) -> None:
        """Enqueue the record as a log span, once; raises what went wrong."""
        if getattr(record, _CAPTURED, False):
            return
        setattr(record, _CAPTURED, True)
        ctx = current_span_ctx() or get_or_create_ambient_ctx()
        level = self._LEVEL_MAP.get(record.levelno, "INFO")
        ts = _now_iso()
        span_id = _new_span_id()

        attrs = {
            "log.level":    level,
            "log.message":  record.getMessage(),
            "log.logger":   record.name,
            "log.filename": record.filename,
            "log.lineno":   record.lineno,
            "log.funcname": record.funcName,
            **self._client._meta_attrs(),
        }

        if record.exc_info and record.exc_info[0] is not None:
            attrs["error.type"]    = record.exc_info[0].__name__
            attrs["error.message"] = str(record.exc_info[1])
            attrs["error.stack"]   = "".join(tb_module.format_exception(*record.exc_info))

        status = "ERROR" if record.levelno >= logging.ERROR else "OK"

        self._client.enqueue(TraceEnvelope(
            trace_id=ctx.trace_id, span_id=span_id, parent_span_id=ctx.span_id,
            span_type="log", status=status,
            operation=f"log.{level}",
            service_name=self._client._opts.service_name,
            attributes=attrs, latency_ms=0, start_time=ts, end_time=ts, created_at=ts,
            environment=self._client._opts.environment,
            app_version=self._client._opts.app_version, sdk_version=SDK_VERSION,
        ))


# Set on a record once it became a log span, so the record-level capture and an
# EvalKitLoggingHandler a caller added themselves never record it twice.
_CAPTURED = "_evalkit_captured"
_capturing: ContextVar[bool] = ContextVar("evalkit_log_capturing", default=False)
_capture_handler: Optional[EvalKitLoggingHandler] = None


def _ensure_handler_filter(handler: logging.Handler) -> None:
    if not any(isinstance(f, EvalKitLogFilter) for f in handler.filters):
        handler.addFilter(EvalKitLogFilter())


def _stamp_trace_ids(record: logging.LogRecord) -> None:
    # For formats that ask for %(trace_id)s or %(span_id)s; a value the caller passed in
    # extra= is kept.
    if hasattr(record, "trace_id") and hasattr(record, "span_id"):
        return
    ctx = current_span_ctx() or get_or_create_ambient_ctx()
    if not hasattr(record, "trace_id"):
        record.trace_id = ctx.trace_id
    if not hasattr(record, "span_id"):
        record.span_id = ctx.span_id


def _reaches_root(logger: logging.Logger) -> bool:
    current: Any = logger
    while current is not None and current is not logging.root:
        if not current.propagate:
            return False
        current = current.parent
    return current is logging.root


def install_log_capture(client: EvalKitClient) -> None:
    """Record every log record that reaches the root logger as a log span, as a handler on the
    root logger would, without being one.

    The application's handlers, formatters, filters and levels stay as they are, logging.basicConfig()
    called after init() still configures the root logger, and an app with no handlers keeps
    Python's own last-resort output. Records also carry trace_id and span_id, for a format that
    asks for them.
    """
    global _capture_handler
    _capture_handler = EvalKitLoggingHandler(client)
    original = logging.Logger.callHandlers
    if getattr(original, "_evalkit", False):
        return

    def callHandlers(self: logging.Logger, record: logging.LogRecord) -> None:
        try:
            _stamp_trace_ids(record)
        except Exception:
            pass
        original(self, record)
        handler = _capture_handler
        if handler is None or _capturing.get():
            return
        token = None
        try:
            if record.levelno >= handler.level and _reaches_root(self):
                # A record logged while one is being recorded (by the export path itself) is
                # left out rather than recorded recursively.
                token = _capturing.set(True)
                handler.record(record)
        except Exception:
            pass
        finally:
            if token is not None:
                _capturing.reset(token)

    callHandlers._evalkit = True  # type: ignore[attr-defined]
    callHandlers.__wrapped__ = original  # type: ignore[attr-defined]
    logging.Logger.callHandlers = callHandlers  # type: ignore[method-assign]


def install_logging_handler(
    client: EvalKitClient,
    logger_name: Optional[str] = None,
    level: int = logging.DEBUG,
) -> EvalKitLoggingHandler:
    """Add an EvalKitLoggingHandler to one logger and return it.

    Only the handler is added: the logger's other handlers, their formatters and the logger's
    level are left as the application set them.
    """
    logger = logging.getLogger(logger_name)
    existing = next((h for h in logger.handlers if isinstance(h, EvalKitLoggingHandler)), None)
    if existing is not None:
        return existing
    ek_handler = EvalKitLoggingHandler(client, level=level)
    _ensure_handler_filter(ek_handler)
    logger.addHandler(ek_handler)
    return ek_handler
