from __future__ import annotations

import contextlib
import json
import time
from typing import Any, Dict, Optional

from ..._version import SDK_VERSION
from ...core.client import EvalKitClient, _new_span_id, _now_iso
from ...core.context import SpanContext, _current_span, current_span_ctx, _in_http_span, get_or_create_ambient_trace_id
from ...core.types import TraceEnvelope
from ...utils.headers import client_span_name, make_traceparent, sanitize_headers
from ._stream_body import attach_requests, new_capture
from ._skip import should_skip as _should_skip
from .._safe import hook_failed

_BODY_MAX = 64 * 1024
_STREAM_BODIES = True


def _otel_http_suppressed() -> Any:
    """Keep OpenTelemetry's urllib3 instrumentation out of a call this wrapper records itself.

    requests sends through urllib3, so with that instrumentation installed the same call would be
    recorded twice, and its traceparent would replace the one set here with an id that never
    exports. urllib3 used directly (botocore, boto3) is still traced: the suppression lasts only
    for this call.
    """
    try:
        from opentelemetry.instrumentation.utils import suppress_http_instrumentation
    except Exception:
        return contextlib.nullcontext()
    try:
        return suppress_http_instrumentation()
    except Exception:
        return contextlib.nullcontext()


def patch_requests(client: EvalKitClient) -> None:
    global _BODY_MAX, _STREAM_BODIES
    _BODY_MAX = client._opts.max_body_bytes
    _STREAM_BODIES = client._opts.capture_stream_bodies
    try:
        import requests
    except ImportError:
        return

    Session = requests.Session
    if getattr(Session.request, "_evalkit", False):
        return

    original = Session.request

    def traced_request(self: Any, method: str, url: str, *args: Any, **kwargs: Any) -> Any:
        # Positional params, data or headers are passed on exactly as given, untraced: they cannot
        # be told apart without re-binding the call, and the call matters more than its span.
        if args or _in_http_span.get():
            return original(self, method, url, *args, **kwargs)
        if _should_skip(url, method):
            # A call left out (the export itself, a health probe, an ignored URL, a call made while
            # tracing is held off) is left out at every layer: urllib3's own instrumentation would
            # otherwise record the call this wrapper skips.
            with _otel_http_suppressed():
                return original(self, method, url, **kwargs)

        try:
            ctx = current_span_ctx()
            if ctx is None:
                trace_id = get_or_create_ambient_trace_id()
                parent_span_id: Optional[str] = None
            else:
                trace_id = ctx.trace_id
                parent_span_id = ctx.span_id

            span_id = _new_span_id()
            start_ms = time.time() * 1000
            ts = _now_iso()

            # The caller's own headers dict is left as it was; the traceparent goes on a copy.
            call_kwargs = dict(kwargs)
            headers: Any = call_kwargs.pop("headers", None) or {}
            if isinstance(headers, dict):
                headers = {**headers, "traceparent": make_traceparent(trace_id, span_id)}
            call_kwargs["headers"] = headers

            req_headers_sanitized = sanitize_headers(headers)

            req_body: Optional[str] = None
            if "json" in kwargs and kwargs["json"] is not None:
                try:
                    req_body = json.dumps(kwargs["json"])[:_BODY_MAX]
                except Exception:
                    pass
            elif "data" in kwargs and kwargs["data"] is not None:
                d = kwargs["data"]
                req_body = (d if isinstance(d, str) else str(d))[:_BODY_MAX]
        except Exception as exc:
            hook_failed("requests", exc)
            return original(self, method, url, *args, **kwargs)
        kwargs = call_kwargs

        _http_token = _in_http_span.set(True)
        try:
            with _otel_http_suppressed():
                resp = original(self, method, url, **kwargs)
        except Exception as e:
            _in_http_span.reset(_http_token)
            try:
                end_ts = _now_iso()
                latency = int(time.time() * 1000 - start_ms)
                attrs: Dict[str, Any] = {
                    "http.method": method.upper(),
                    "http.url": url,
                    "error.message": str(e),
                    **client._meta_attrs(),
                }
                if req_headers_sanitized:
                    attrs["http.request.headers"] = json.dumps(req_headers_sanitized)
                if req_body:
                    attrs["http.request.body"] = req_body
                client.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="http_call", status="ERROR", status_message=str(e), exception=e,
                    span_kind="CLIENT", operation=client_span_name(method),
                    service_name=client._opts.service_name,
                    attributes=attrs, latency_ms=latency, start_time=ts, end_time=end_ts,
                    created_at=ts, environment=client._opts.environment,
                    app_version=client._opts.app_version, sdk_version=SDK_VERSION,
                ))
            except Exception as hook_exc:
                hook_failed("requests", hook_exc)
            raise
        _in_http_span.reset(_http_token)
        try:
            end_ts = _now_iso()
            latency = int(time.time() * 1000 - start_ms)
            status_code = resp.status_code

            attrs = {
                "http.method": method.upper(),
                "http.url": url,
                "http.status_code": status_code,
                **client._meta_attrs(),
            }
            if req_headers_sanitized:
                attrs["http.request.headers"] = json.dumps(req_headers_sanitized)
            if req_body:
                attrs["http.request.body"] = req_body
            resp_headers = sanitize_headers(resp.headers)
            if resp_headers:
                attrs["http.response.headers"] = json.dumps(resp_headers)

            streaming = bool(kwargs.get("stream"))
            if streaming and _STREAM_BODIES and _BODY_MAX > 0:
                attrs["http.server.time_to_headers_ms"] = latency

                def emit(capture: Any, error: Optional[BaseException]) -> None:
                    stream_attrs = dict(attrs)
                    stream_attrs.update(capture.attrs())
                    status = "ERROR" if status_code >= 400 else "OK"
                    message = None
                    if error is not None:
                        status = "ERROR"
                        message = str(error)
                        stream_attrs["error.message"] = message
                    client.enqueue(TraceEnvelope(
                        trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                        span_type="http_call", status=status, status_message=message, exception=error,
                        span_kind="CLIENT", operation=client_span_name(method),
                        service_name=client._opts.service_name,
                        attributes=stream_attrs,
                        latency_ms=int(time.time() * 1000 - start_ms),
                        start_time=ts, end_time=_now_iso(),
                        created_at=ts, environment=client._opts.environment,
                        app_version=client._opts.app_version, sdk_version=SDK_VERSION,
                    ))

                capture = new_capture(_BODY_MAX, start_ms, emit)
                if attach_requests(resp, capture):
                    return resp
                capture.finish()
                return resp

            if not streaming:
                try:
                    rb = resp.text
                    if rb:
                        attrs["http.response.body"] = rb[:_BODY_MAX]
                except Exception:
                    pass

            client.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="http_call", status="ERROR" if status_code >= 400 else "OK",
                span_kind="CLIENT", operation=client_span_name(method),
                service_name=client._opts.service_name,
                attributes=attrs, latency_ms=latency, start_time=ts, end_time=end_ts,
                created_at=ts, environment=client._opts.environment,
                app_version=client._opts.app_version, sdk_version=SDK_VERSION,
            ))
        except Exception as exc:
            hook_failed("requests", exc)
        return resp

    traced_request._evalkit = True
    Session.request = traced_request
