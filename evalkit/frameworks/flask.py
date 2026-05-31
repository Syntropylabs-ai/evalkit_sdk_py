from __future__ import annotations

from typing import Any

from .._version import SDK_VERSION
from ..core.client import EvalKitClient
from ..core.context import _current_span
from ..utils.headers import parse_traceparent, sanitize_headers


def instrument_flask(app: Any, client: EvalKitClient) -> None:
    try:
        from flask import g, request as flask_request
    except ImportError:
        raise RuntimeError("Flask is not installed. pip install flask")

    @app.before_request
    def _evalkit_before() -> None:
        try:
            upstream_trace_id, upstream_span_id = None, None
            tp_header = flask_request.headers.get("traceparent", "")
            if tp_header:
                upstream_trace_id, upstream_span_id = parse_traceparent(tp_header)

            body = flask_request.get_json(silent=True) or flask_request.get_data(as_text=True) or None
            safe_headers = sanitize_headers(flask_request.headers)

            trace_id, end, ctx = client.start_http_trace(
                f"{flask_request.method} {flask_request.path}",
                method=flask_request.method,
                url=flask_request.url,
                headers=safe_headers,
                request_body=body,
                upstream_trace_id=upstream_trace_id,
                upstream_span_id=upstream_span_id,
            )
            g._ek_end = end
            g._ek_ctx = ctx
            g._ek_trace_id = trace_id
            g._ek_token = _current_span.set(ctx)
        except Exception:
            pass

    @app.after_request
    def _evalkit_after(response: Any) -> Any:
        try:
            end = getattr(g, "_ek_end", None)
            trace_id = getattr(g, "_ek_trace_id", None)
            ctx = getattr(g, "_ek_ctx", None)
            if end:
                end(
                    "ERROR" if response.status_code >= 400 else "OK",
                    status_code=response.status_code,
                )
            if trace_id:
                response.headers["X-Trace-Id"] = trace_id
            if ctx:
                response.headers["traceparent"] = f"00-{trace_id}-{ctx.span_id}-01"
        except Exception:
            pass
        return response

    @app.teardown_request
    def _evalkit_teardown(exc: Any) -> None:
        try:
            token = getattr(g, "_ek_token", None)
            if token is not None:
                _current_span.reset(token)
        except Exception:
            pass
