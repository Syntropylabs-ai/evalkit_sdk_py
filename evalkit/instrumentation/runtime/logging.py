from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any, Optional

from ..._version import SDK_VERSION
from ...core.bounded import RateLimiter
from ...core.client import EvalKitClient, _new_span_id, _now_iso
from ...core.context import _current_span, current_span_ctx, get_or_create_ambient_ctx
from ...core.limits import TRUNCATED_PREFIX, cut_middle
from ...core.types import TraceEnvelope

# The most of one log message a log span keeps, in UTF-8 bytes; the start and the end are kept.
MAX_LOG_MESSAGE_BYTES = 8 * 1024

# The SDK's own loggers: what it says about exporting is not the application's to read in its traces,
# and recording it would feed a failing export with spans about its failure.
_OWN_LOGGERS = ("evalkit", "evalkit_cli")


def _is_own_logger(name: Any) -> bool:
    if not isinstance(name, str):
        return False
    return any(name == own or name.startswith(own + ".") for own in _OWN_LOGGERS)


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

    def __init__(self, client: EvalKitClient, level: int = logging.DEBUG, rate_limit: Optional[float] = None) -> None:
        super().__init__(level)
        self._client = client
        if rate_limit is None:
            rate_limit = getattr(getattr(client, "_opts", None), "log_rate_limit", 50.0)
        # A logging storm (a loop that logs on every pass) records the first records of each second.
        self._limiter = RateLimiter(rate_limit)

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
        if _is_own_logger(record.name):
            return
        if not self._limiter.allow():
            try:
                self._client._health.count("dropped_rate_limited")
            except Exception:
                pass
            return
        ctx = current_span_ctx() or get_or_create_ambient_ctx()
        level = self._LEVEL_MAP.get(record.levelno, "INFO")
        ts = _now_iso()
        span_id = _new_span_id()

        message = record.getMessage()
        attrs = {
            "log.level":    level,
            "log.message":  message,
            "log.logger":   record.name,
            "log.filename": record.filename,
            "log.lineno":   record.lineno,
            "log.funcname": record.funcName,
            **self._client._meta_attrs(),
        }
        if isinstance(message, str) and len(message) > MAX_LOG_MESSAGE_BYTES // 4:
            cut, original = cut_middle(message, MAX_LOG_MESSAGE_BYTES)
            if original is not None:
                attrs["log.message"] = cut
                attrs[TRUNCATED_PREFIX + "log.message"] = original

        exception = None
        if record.exc_info and record.exc_info[0] is not None:
            exception = record.exc_info[1]
            from ...core.errors import ExceptionRecord

            try:
                described = ExceptionRecord(exception) if isinstance(exception, BaseException) else None
            except Exception:
                described = None
            attrs["error.message"] = described.message if described else str(record.exc_info[1])
            if described is not None:
                attrs["error.stack"] = described.stacktrace

        status = "ERROR" if record.levelno >= logging.ERROR else "OK"
        envelope = TraceEnvelope(
            trace_id=ctx.trace_id, span_id=span_id, parent_span_id=ctx.span_id,
            span_type="log", status=status,
            operation=f"log.{level}",
            service_name=self._client._opts.service_name,
            attributes=attrs, latency_ms=0, start_time=ts, end_time=ts, created_at=ts,
            environment=self._client._opts.environment,
            app_version=self._client._opts.app_version, sdk_version=SDK_VERSION,
        )
        if isinstance(exception, BaseException):
            from ...core.errors import apply_exception

            # Logged, so handled where it was logged: it did not escape. A warning that carries an
            # exception stays a warning; the record's level, not the exception, says it failed.
            apply_exception(envelope, exception, escaped=False, timestamp=ts, mark_error=status == "ERROR")
        self._client.enqueue(envelope)


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

    Only records at ``log_level`` (WARNING unless set lower) and up, at most ``log_rate_limit`` a
    second, and none from the SDK's own loggers.

    The application's handlers, formatters, filters and levels stay as they are, logging.basicConfig()
    called after init() still configures the root logger, and an app with no handlers keeps
    Python's own last-resort output. Records also carry trace_id and span_id, for a format that
    asks for them.
    """
    global _capture_handler
    level = getattr(getattr(client, "_opts", None), "log_level", logging.WARNING)
    _capture_handler = EvalKitLoggingHandler(client, level=level if isinstance(level, int) else logging.WARNING)
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
            if record.levelno >= handler.level and not _is_own_logger(record.name) and _reaches_root(self):
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
