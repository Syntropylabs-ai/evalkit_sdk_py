from __future__ import annotations

import re
import time
from typing import Any, Optional

from ..._version import SDK_VERSION
from ...core.client import EvalKitClient, _new_span_id, _now_iso
from ...core.context import _current_span, current_span_ctx
from ...core.types import TraceEnvelope

_BODY_MAX = 2048


def patch_sqlalchemy(client: EvalKitClient, engine: Any) -> None:
    """
    Instrument a SQLAlchemy Engine to trace every query execution.
    Uses SQLAlchemy's core event system (before_cursor_execute / after_cursor_execute).
    Works with both Core queries and ORM sessions.
    """
    try:
        from sqlalchemy import event
    except ImportError:
        raise RuntimeError("SQLAlchemy is not installed. pip install sqlalchemy")

    if getattr(engine, "_evalkit_instrumented", False):
        return

    _start_times: dict = {}

    @event.listens_for(engine, "before_cursor_execute")
    def before_execute(
        conn: Any, cursor: Any, statement: str, parameters: Any,
        context: Any, executemany: bool,
    ) -> None:
        ctx = current_span_ctx()
        if ctx is None:
            return
        span_id = _new_span_id()
        _start_times[id(cursor)] = (time.time() * 1000, span_id, _now_iso(), ctx)

    @event.listens_for(engine, "after_cursor_execute")
    def after_execute(
        conn: Any, cursor: Any, statement: str, parameters: Any,
        context: Any, executemany: bool,
    ) -> None:
        entry = _start_times.pop(id(cursor), None)
        if entry is None:
            return
        start_ms, span_id, start_ts, parent_ctx = entry

        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)

        op_match = re.match(r"^\s*(\w+)", statement)
        op = op_match.group(1).upper() if op_match else "QUERY"
        table_match = re.search(r"(?:FROM|INTO|UPDATE|JOIN)\s+[\"']?(\w+)[\"']?", statement, re.IGNORECASE)
        table = table_match.group(1) if table_match else ""
        span_name = f"{op} {table}" if table else f"sql {op}"

        try:
            rows_affected = cursor.rowcount if cursor.rowcount is not None else 0
        except Exception:
            rows_affected = 0

        try:
            db_name = conn.engine.url.database or ""
        except Exception:
            db_name = ""

        try:
            dialect = conn.engine.dialect.name
        except Exception:
            dialect = "sql"

        params_str: Optional[str] = None
        if parameters:
            try:
                import json
                params_str = json.dumps(
                    parameters if not isinstance(parameters, tuple) else list(parameters)
                )[:1024]
            except Exception:
                params_str = str(parameters)[:1024]

        attrs: dict = {
            "db.system": dialect,
            "db.operation": op,
            "db.statement": statement[:_BODY_MAX],
            **client._meta_attrs(),
        }
        if table:
            attrs["db.sql.table"] = table
        if db_name:
            attrs["db.name"] = db_name
        if params_str:
            attrs["db.params"] = params_str
        if rows_affected >= 0:
            attrs["db.rows_affected"] = rows_affected

        client.enqueue(TraceEnvelope(
            trace_id=parent_ctx.trace_id, span_id=span_id, parent_span_id=parent_ctx.span_id,
            span_type="db_query", status="OK",
            operation=span_name, service_name=client._opts.service_name,
            attributes=attrs, latency_ms=latency, start_time=start_ts, end_time=end_ts,
            created_at=start_ts, environment=client._opts.environment,
            app_version=client._opts.app_version, sdk_version=SDK_VERSION,
        ))

    @event.listens_for(engine, "handle_error")
    def handle_error(exception_context: Any) -> None:
        cursor = getattr(exception_context, "cursor", None)
        if cursor is None:
            return
        entry = _start_times.pop(id(cursor), None)
        if entry is None:
            return
        start_ms, span_id, start_ts, parent_ctx = entry

        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)
        err = exception_context.original_exception

        client.enqueue(TraceEnvelope(
            trace_id=parent_ctx.trace_id, span_id=span_id, parent_span_id=parent_ctx.span_id,
            span_type="db_query", status="ERROR", status_message=str(err),
            operation="sql query", service_name=client._opts.service_name,
            attributes={"error.message": str(err), **client._meta_attrs()},
            latency_ms=latency, start_time=start_ts, end_time=end_ts, created_at=start_ts,
            environment=client._opts.environment, app_version=client._opts.app_version,
            sdk_version=SDK_VERSION,
        ))

    engine._evalkit_instrumented = True


def patch_sqlalchemy_async(client: EvalKitClient, async_engine: Any) -> None:
    """
    Instrument an async SQLAlchemy engine (created with create_async_engine).

    Delegates to patch_sqlalchemy on the underlying sync engine — SQLAlchemy's
    cursor-level events fire synchronously inside greenlet_spawn even for async
    drivers, so this captures all queries including those from AsyncSession.
    """
    if getattr(async_engine, "_evalkit_instrumented", False):
        return
    sync_engine = getattr(async_engine, "sync_engine", None)
    if sync_engine is None:
        return
    patch_sqlalchemy(client, sync_engine)
    async_engine._evalkit_instrumented = True
