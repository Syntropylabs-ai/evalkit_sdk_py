from __future__ import annotations

import logging
import traceback as tb_module
from typing import Optional

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
        ctx = current_span_ctx() or get_or_create_ambient_ctx()

        try:
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
        except Exception:
            self.handleError(record)


_TRACE_FMT = "%(asctime)s %(levelname)s [%(trace_id)s] %(name)s: %(message)s"
_TRACE_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def _ensure_handler_filter(handler: logging.Handler) -> None:
    if not any(isinstance(f, EvalKitLogFilter) for f in handler.filters):
        handler.addFilter(EvalKitLogFilter())


def install_logging_handler(
    client: EvalKitClient,
    logger_name: Optional[str] = None,
    level: int = logging.DEBUG,
) -> EvalKitLoggingHandler:
    logger = logging.getLogger(logger_name)

    ek_handler = EvalKitLoggingHandler(client, level=level)
    _ensure_handler_filter(ek_handler)
    if not any(isinstance(h, EvalKitLoggingHandler) for h in logger.handlers):
        logger.addHandler(ek_handler)

    stream_handlers = [
        h for h in logger.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, EvalKitLoggingHandler)
    ]
    if not stream_handlers:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_TRACE_FMT, datefmt=_TRACE_DATEFMT))
        _ensure_handler_filter(console)
        logger.addHandler(console)
        if not logger.level:
            logger.setLevel(logging.INFO)
    else:
        for h in stream_handlers:
            _ensure_handler_filter(h)
            fmt_str = getattr(h.formatter, "_fmt", None) if h.formatter else None
            if not fmt_str or "trace_id" not in fmt_str:
                h.setFormatter(logging.Formatter(_TRACE_FMT, datefmt=_TRACE_DATEFMT))

    return ek_handler
