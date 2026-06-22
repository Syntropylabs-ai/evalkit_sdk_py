from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from ..._version import SDK_VERSION
from ...core.client import EvalKitClient, _new_span_id, _now_iso
from ...core.context import SpanContext, _current_span, current_span_ctx, _in_http_span, get_or_create_ambient_trace_id
from ...core.types import TraceEnvelope
from ...utils.headers import make_traceparent, path_from_url, sanitize_headers

_SKIP = {"/v1/ingest", "/health", "/healthz"}
_BODY_MAX = 10 * 1024 * 1024  # default; overridden from client options at patch time


def _should_skip(url: str) -> bool:
    return any(pat in url for pat in _SKIP)


def patch_requests(client: EvalKitClient) -> None:
    """Patch requests.Session.request to trace every outgoing HTTP call."""
    global _BODY_MAX
    _BODY_MAX = client._opts.max_body_bytes
    try:
        import requests
    except ImportError:
        return

    Session = requests.Session
    if getattr(Session.request, "_evalkit", False):
        return

    original = Session.request

    def traced_request(self: Any, method: str, url: str, **kwargs: Any) -> Any:
        if _should_skip(url):
            return original(self, method, url, **kwargs)

        # Skip if an outer HTTP wrapper is already tracing this call (one client
        # delegating to another), to avoid a duplicate http_call span.
        if _in_http_span.get():
            return original(self, method, url, **kwargs)

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

        headers: Any = kwargs.pop("headers", None) or {}
        if isinstance(headers, dict):
            headers["traceparent"] = make_traceparent(trace_id, span_id)
        kwargs["headers"] = headers

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

        _http_token = _in_http_span.set(True)
        try:
            resp = original(self, method, url, **kwargs)
        except Exception as e:
            _in_http_span.reset(_http_token)
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
                span_type="http_call", status="ERROR", status_message=str(e),
                operation=f"{method.upper()} {path_from_url(url)}",
                service_name=client._opts.service_name,
                attributes=attrs, latency_ms=latency, start_time=ts, end_time=end_ts,
                created_at=ts, environment=client._opts.environment,
                app_version=client._opts.app_version, sdk_version=SDK_VERSION,
            ))
            raise
        _in_http_span.reset(_http_token)

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
        if not kwargs.get("stream"):
            try:
                rb = resp.text
                if rb:
                    attrs["http.response.body"] = rb[:_BODY_MAX]
            except Exception:
                pass

        client.enqueue(TraceEnvelope(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
            span_type="http_call", status="ERROR" if status_code >= 400 else "OK",
            operation=f"{method.upper()} {path_from_url(url)}",
            service_name=client._opts.service_name,
            attributes=attrs, latency_ms=latency, start_time=ts, end_time=end_ts,
            created_at=ts, environment=client._opts.environment,
            app_version=client._opts.app_version, sdk_version=SDK_VERSION,
        ))
        return resp

    traced_request._evalkit = True  # type: ignore[attr-defined]
    Session.request = traced_request  # type: ignore[method-assign]
