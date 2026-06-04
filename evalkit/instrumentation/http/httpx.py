from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from ..._version import SDK_VERSION
from ...core.client import EvalKitClient, _new_span_id, _now_iso
from ...core.context import _current_span, get_or_create_ambient_trace_id
from ...core.types import TraceEnvelope
from ...utils.headers import make_traceparent, path_from_url, sanitize_headers

_SKIP = {"/v1/ingest", "/health", "/healthz"}
_BODY_MAX = 8192


def _should_skip(url: str) -> bool:
    return any(pat in url for pat in _SKIP)


def _req_body(request: Any) -> Optional[str]:
    try:
        raw = request.content
        if raw:
            return raw[:_BODY_MAX].decode("utf-8", errors="replace")
    except Exception:
        pass
    return None


def _resp_body(resp: Any) -> Optional[str]:
    # Only read the body if httpx has already buffered it (non-streaming response).
    # For a live stream is_stream_consumed=False — touching .content would consume
    # the caller's stream, so we skip it and return None instead.
    try:
        is_streaming = getattr(resp, "is_stream_consumed", None) is False
        if is_streaming:
            return None
        raw = resp.content
        if raw:
            return raw[:_BODY_MAX].decode("utf-8", errors="replace")
    except Exception:
        pass
    return None


def patch_httpx(client: EvalKitClient) -> None:
    try:
        import httpx
    except ImportError:
        return

    if not getattr(httpx.Client.send, "_evalkit", False):
        _orig_sync = httpx.Client.send

        def traced_sync_send(self: Any, request: Any, **kwargs: Any) -> Any:
            url = str(request.url)
            if _should_skip(url):
                return _orig_sync(self, request, **kwargs)

            ctx = _current_span.get()
            trace_id = ctx.trace_id if ctx else get_or_create_ambient_trace_id()
            parent_span_id = ctx.span_id if ctx else None
            span_id = _new_span_id()
            start_ms = time.time() * 1000
            ts = _now_iso()
            method = request.method.upper()

            try:
                request.headers["traceparent"] = make_traceparent(trace_id, span_id)
            except Exception:
                pass
            req_headers = sanitize_headers(dict(request.headers))

            try:
                resp = _orig_sync(self, request, **kwargs)
            except Exception as e:
                end_ts = _now_iso()
                _emit_http_span(
                    client, trace_id, parent_span_id, span_id, method, url, None, str(e),
                    ts, end_ts, int(time.time() * 1000 - start_ms),
                    req_headers=req_headers, req_body=_req_body(request),
                )
                raise

            end_ts = _now_iso()
            resp_headers = sanitize_headers(dict(resp.headers))
            _emit_http_span(
                client, trace_id, parent_span_id, span_id, method, url, resp.status_code, None,
                ts, end_ts, int(time.time() * 1000 - start_ms),
                req_headers=req_headers, resp_headers=resp_headers,
                req_body=_req_body(request), resp_body=_resp_body(resp),
            )
            return resp

        traced_sync_send._evalkit = True  # type: ignore[attr-defined]
        httpx.Client.send = traced_sync_send  # type: ignore[method-assign]

    if not getattr(httpx.AsyncClient.send, "_evalkit", False):
        _orig_async = httpx.AsyncClient.send

        async def traced_async_send(self: Any, request: Any, **kwargs: Any) -> Any:
            url = str(request.url)
            if _should_skip(url):
                return await _orig_async(self, request, **kwargs)

            ctx = _current_span.get()
            trace_id = ctx.trace_id if ctx else get_or_create_ambient_trace_id()
            parent_span_id = ctx.span_id if ctx else None
            span_id = _new_span_id()
            start_ms = time.time() * 1000
            ts = _now_iso()
            method = request.method.upper()

            try:
                request.headers["traceparent"] = make_traceparent(trace_id, span_id)
            except Exception:
                pass
            req_headers = sanitize_headers(dict(request.headers))

            try:
                resp = await _orig_async(self, request, **kwargs)
            except Exception as e:
                end_ts = _now_iso()
                _emit_http_span(
                    client, trace_id, parent_span_id, span_id, method, url, None, str(e),
                    ts, end_ts, int(time.time() * 1000 - start_ms),
                    req_headers=req_headers, req_body=_req_body(request),
                )
                raise

            end_ts = _now_iso()
            resp_headers = sanitize_headers(dict(resp.headers))
            _emit_http_span(
                client, trace_id, parent_span_id, span_id, method, url, resp.status_code, None,
                ts, end_ts, int(time.time() * 1000 - start_ms),
                req_headers=req_headers, resp_headers=resp_headers,
                req_body=_req_body(request), resp_body=_resp_body(resp),
            )
            return resp

        traced_async_send._evalkit = True  # type: ignore[attr-defined]
        httpx.AsyncClient.send = traced_async_send  # type: ignore[method-assign]


def _emit_http_span(
    client: EvalKitClient,
    trace_id: str,
    parent_span_id: Optional[str],
    span_id: str,
    method: str,
    url: str,
    status_code: Any,
    error_msg: Any,
    start_ts: str,
    end_ts: str,
    latency: int,
    req_headers: Optional[Dict[str, str]] = None,
    resp_headers: Optional[Dict[str, str]] = None,
    req_body: Optional[str] = None,
    resp_body: Optional[str] = None,
) -> None:
    attrs: Dict[str, Any] = {"http.method": method, "http.url": url, **client._meta_attrs()}
    status = "OK"
    if req_headers:
        attrs["http.request.headers"] = json.dumps(req_headers)
    if req_body:
        attrs["http.request.body"] = req_body
    if status_code is not None:
        attrs["http.status_code"] = status_code
        status = "ERROR" if status_code >= 400 else "OK"
    if resp_headers:
        attrs["http.response.headers"] = json.dumps(resp_headers)
    if resp_body:
        attrs["http.response.body"] = resp_body
    if error_msg:
        attrs["error.message"] = error_msg
        status = "ERROR"

    client.enqueue(TraceEnvelope(
        trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
        span_type="http_call", status=status,
        operation=f"{method} {path_from_url(url)}",
        service_name=client._opts.service_name,
        attributes=attrs, latency_ms=latency, start_time=start_ts, end_time=end_ts,
        created_at=start_ts, environment=client._opts.environment,
        app_version=client._opts.app_version, sdk_version=SDK_VERSION,
    ))
