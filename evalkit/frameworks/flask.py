from __future__ import annotations

from typing import Any

from ..core.client import EvalKitClient, release_ambient_trace
from ..core.context import _current_span, end_suppression, suppress_tracing
from ..utils.headers import parse_traceparent, sanitize_headers
from ._bodies import capped_prefix, is_binary, is_event_stream
from ._probe import is_ignored

_BODY_MAX = 64 * 1024


def _request_body(request: Any) -> Any:
    """The request body, when the request says it is small enough to keep and is not an upload.

    Read only under that size: Werkzeug holds whatever it reads in memory for the rest of the request,
    so reading an upload, or a chunked body of unknown length, to record it would hold all of it.
    """
    length = getattr(request, "content_length", None)
    if length is None or length <= 0 or length > _BODY_MAX:
        return None
    if is_binary(getattr(request, "content_type", None)):
        return None
    return request.get_json(silent=True) or request.get_data(as_text=True) or None


def _response_body(response: Any) -> Any:
    """What of the response body is already in memory, up to the limit; never a streamed body."""
    if getattr(response, "direct_passthrough", False) or getattr(response, "is_streamed", False):
        return None
    mimetype = getattr(response, "mimetype", None)
    if is_event_stream(mimetype) or is_binary(mimetype):
        return None
    prefix = capped_prefix(getattr(response, "response", None), _BODY_MAX)
    if prefix is None:
        return None
    return prefix.decode("utf-8", errors="replace") or None


def flask_route(request: Any) -> "str | None":
    """The matched URL rule (`/orders/<int:order_id>`), or None when no rule matched (a 404)."""
    try:
        rule = getattr(getattr(request, "url_rule", None), "rule", None)
    except Exception:
        return None
    return rule if isinstance(rule, str) and rule else None


def instrument_flask(app: Any, client: EvalKitClient) -> None:
    global _BODY_MAX
    _BODY_MAX = client._opts.max_body_bytes
    try:
        from flask import g, request as flask_request
    except ImportError:
        raise RuntimeError("Flask is not installed. pip install flask")

    @app.before_request
    def _evalkit_before() -> None:
        try:
            # Flask has matched the URL rule by now, so a pattern can match its template too.
            if is_ignored(flask_request.method, flask_request.path, route=flask_route(flask_request)):
                g._ek_probe = suppress_tracing()
                return
            upstream_trace_id, upstream_span_id = None, None
            tp_header = flask_request.headers.get("traceparent", "")
            if tp_header:
                upstream_trace_id, upstream_span_id = parse_traceparent(tp_header)

            body = _request_body(flask_request)
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
            g._ek_ended = False
            g._ek_token = _current_span.set(ctx)
        except Exception:
            pass

    @app.after_request
    def _evalkit_after(response: Any) -> Any:
        try:
            end = getattr(g, "_ek_end", None)
            trace_id = getattr(g, "_ek_trace_id", None)
            ctx = getattr(g, "_ek_ctx", None)
            if end and not getattr(g, "_ek_ended", False):
                resp_headers = None
                resp_body: Any = None
                try:
                    resp_headers = sanitize_headers(dict(response.headers)) or None
                except Exception:
                    pass
                try:
                    resp_body = _response_body(response)
                except Exception:
                    pass
                fields = dict(
                    status_code=response.status_code,
                    response_body=resp_body,
                    response_headers=resp_headers,
                    route=flask_route(flask_request),
                )
                if response.status_code >= 500:
                    # Flask answers an unhandled exception with a 500 that runs through here before
                    # teardown is handed the exception, so the span ends there, with it.
                    g._ek_pending = fields
                else:
                    end("ERROR" if response.status_code >= 400 else "OK", **fields)
                    g._ek_ended = True
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
            end = getattr(g, "_ek_end", None)
            if end and not getattr(g, "_ek_ended", False):
                fields = getattr(g, "_ek_pending", None) or dict(
                    status_code=500, route=flask_route(flask_request),
                )
                try:
                    # The view raised and Flask answered 500 itself: the exception did not get past
                    # the request, but it ended the view, which is what the span is read for.
                    end("ERROR", exception=exc if isinstance(exc, BaseException) else None,
                        exception_escaped=False, **fields)
                except Exception:
                    pass
                g._ek_ended = True
        except Exception:
            pass
        finally:
            try:
                token = getattr(g, "_ek_token", None)
                if token is not None:
                    _current_span.reset(token)
            except Exception:
                pass
            try:
                release_ambient_trace(getattr(g, "_ek_trace_id", None))
            except Exception:
                pass
            try:
                probe = getattr(g, "_ek_probe", None)
                if probe is not None:
                    end_suppression(*probe)
            except Exception:
                pass
