from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, Optional

from ..._version import SDK_VERSION
from ...core.client import _new_span_id, _now_iso
from ...core.context import _current_span, current_span_ctx, get_or_create_ambient_trace_id
from ...core.types import TraceEnvelope

_STMT_MAX = 2048


def set_stmt_max(limit: Any) -> None:
    global _STMT_MAX
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return
    if value > 0:
        _STMT_MAX = value


def stmt_max() -> int:
    return _STMT_MAX


def op_and_table(statement: Optional[str]):
    op = "QUERY"
    table = ""
    if statement:
        m = re.match(r"^\s*(\w+)", statement)
        if m:
            op = m.group(1).upper()
        t = re.search(r"(?:FROM|INTO|UPDATE|JOIN)\s+[\"'`]?(\w+)", statement, re.IGNORECASE)
        if t:
            table = t.group(1)
    return op, table


def start_db(
    client: Any,
    system: str,
    *,
    statement: Optional[str] = None,
    operation: Optional[str] = None,
    table: Optional[str] = None,
) -> Callable[..., None]:
    ctx = current_span_ctx()
    trace_id = ctx.trace_id if ctx else get_or_create_ambient_trace_id()
    parent_span_id = ctx.span_id if ctx else None
    span_id = _new_span_id()
    start_ms = time.time() * 1000
    ts = _now_iso()

    if statement is not None and operation is None:
        operation, t2 = op_and_table(statement)
        table = table or t2

    if table and operation:
        name = f"{operation} {table}"
    elif operation:
        name = f"{system} {operation}"
    else:
        name = system

    def end(status: str = "OK", error: Any = None, extra: Optional[Dict[str, Any]] = None, rows: Any = None) -> None:
        try:
            attrs: Dict[str, Any] = {"db.system": system, **client._meta_attrs()}
            if operation:
                attrs["db.operation"] = operation
            if statement:
                attrs["db.statement"] = str(statement)[:_STMT_MAX]
            if table:
                attrs["db.sql.table"] = table
            if rows is not None and isinstance(rows, int):
                attrs["db.rows_affected"] = rows
            if error is not None:
                attrs["error.message"] = str(error)
            if extra:
                attrs.update(extra)
            client.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="db_query", status=status,
                status_message=str(error) if error is not None else None,
                operation=name, service_name=client._opts.service_name,
                attributes=attrs, latency_ms=int(time.time() * 1000 - start_ms),
                start_time=ts, end_time=_now_iso(), created_at=ts,
                environment=client._opts.environment, app_version=client._opts.app_version,
                session_id=ctx.session_id if ctx else None,
                device_id=(ctx.device_id if ctx else None) or client._opts.device_id,
                sdk_version=SDK_VERSION,
            ))
        except Exception:
            pass

    return end
