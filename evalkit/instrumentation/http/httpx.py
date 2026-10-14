from __future__ import annotations

import importlib
import json
import time
from typing import Any, Dict, List, Optional

from ..._version import SDK_VERSION
from ...core.client import EvalKitClient, _new_span_id, _now_iso
from ...core.context import _current_span, current_span_ctx, _in_http_span, get_or_create_ambient_trace_id
from ...core.types import TraceEnvelope
from ...utils.headers import client_span_name, make_traceparent, sanitize_headers
from ._stream_body import attach_httpx_async, attach_httpx_sync, new_capture
from ._skip import should_skip as _should_skip
from .._safe import hook_failed

_BODY_MAX = 10 * 1024 * 1024
_STREAM_BODIES = True


def _req_body(request: Any) -> Optional[str]:
    try:
        raw = request.content
        if raw:
            return raw[:_BODY_MAX].decode("utf-8", errors="replace")
    except Exception:
        pass
    return None


def _is_streaming(resp: Any) -> bool:
    try:
        return not hasattr(resp, "_content")
    except Exception:
        return False


def _resp_body(resp: Any) -> Optional[str]:
    try:
        if _is_streaming(resp):
            return None
        raw = resp.content
        if raw:
            return raw[:_BODY_MAX].decode("utf-8", errors="replace")
    except Exception:
        pass
    return None


def _emit_streamed_span(
    client: EvalKitClient, resp: Any, is_async: bool, start_ms: float, **span: Any
) -> bool:
    if not _STREAM_BODIES or _BODY_MAX <= 0:
        return False

    def emit(capture: Any, error: Optional[BaseException]) -> None:
        _emit_http_span(
            client, latency=int(time.time() * 1000 - start_ms), end_ts=_now_iso(),
            error_msg=str(error) if error is not None else None,
            stream_attrs=capture.attrs(), **span,
        )

    capture = new_capture(_BODY_MAX, start_ms, emit)
    attach = attach_httpx_async if is_async else attach_httpx_sync
    if attach(resp, capture):
        return True
    capture.finish()
    return True


def patch_httpx(client: EvalKitClient) -> List[str]:
    """Trace httpx and httpx2 clients; returns the names of the modules patched.

    httpx2 is httpx's successor, published as a package of its own with the same client API, and
    provider SDKs can send through either, so a call is recorded whichever one carries it.
    """
    global _BODY_MAX, _STREAM_BODIES
    _BODY_MAX = client._opts.max_body_bytes
    _STREAM_BODIES = client._opts.capture_stream_bodies
    patched: List[str] = []
    for name in ("httpx", "httpx2"):
        try:
            module = importlib.import_module(name)
        except Exception:
            continue
        try:
            _patch_clients(client, module)
            patched.append(name)
        except Exception as exc:
            hook_failed(name, exc)
    return patched


def _patch_clients(client: EvalKitClient, httpx: Any) -> None:
    if not getattr(httpx.Client.send, "_evalkit", False):
        _orig_sync = httpx.Client.send

        def traced_sync_send(self: Any, request: Any, **kwargs: Any) -> Any:
            try:
                url = str(request.url)
                skip = _should_skip(url) or _in_http_span.get()
            except Exception:
                skip = True
            if skip:
                return _orig_sync(self, request, **kwargs)

            try:
                ctx = current_span_ctx()
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
            except Exception as exc:
                hook_failed("httpx", exc)
                return _orig_sync(self, request, **kwargs)

            _http_token = _in_http_span.set(True)
            try:
                resp = _orig_sync(self, request, **kwargs)
            except Exception as e:
                _in_http_span.reset(_http_token)
                try:
                    end_ts = _now_iso()
                    _emit_http_span(
                        client, trace_id, parent_span_id, span_id, method, url, None, str(e),
                        ts, end_ts, int(time.time() * 1000 - start_ms),
                        req_headers=req_headers, req_body=_req_body(request),
                    )
                except Exception as hook_exc:
                    hook_failed("httpx", hook_exc)
                raise
            _in_http_span.reset(_http_token)
            try:
                end_ts = _now_iso()
                latency = int(time.time() * 1000 - start_ms)
                resp_headers = sanitize_headers(dict(resp.headers))
                span = dict(
                    trace_id=trace_id, parent_span_id=parent_span_id, span_id=span_id,
                    method=method, url=url, status_code=resp.status_code, start_ts=ts,
                    req_headers=req_headers, resp_headers=resp_headers,
                    req_body=_req_body(request),
                )
                if _is_streaming(resp) and _emit_streamed_span(
                    client, resp, False, start_ms, time_to_headers_ms=latency, **span
                ):
                    return resp
                _emit_http_span(
                    client, error_msg=None, end_ts=end_ts, latency=latency,
                    resp_body=_resp_body(resp), **span,
                )
            except Exception as exc:
                hook_failed("httpx", exc)
            return resp

        traced_sync_send._evalkit = True
        httpx.Client.send = traced_sync_send

    if not getattr(httpx.AsyncClient.send, "_evalkit", False):
        _orig_async = httpx.AsyncClient.send

        async def traced_async_send(self: Any, request: Any, **kwargs: Any) -> Any:
            try:
                url = str(request.url)
                skip = _should_skip(url) or _in_http_span.get()
            except Exception:
                skip = True
            if skip:
                return await _orig_async(self, request, **kwargs)

            try:
                ctx = current_span_ctx()
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
            except Exception as exc:
                hook_failed("httpx", exc)
                return await _orig_async(self, request, **kwargs)

            _http_token = _in_http_span.set(True)
            try:
                resp = await _orig_async(self, request, **kwargs)
            except Exception as e:
                _in_http_span.reset(_http_token)
                try:
                    end_ts = _now_iso()
                    _emit_http_span(
                        client, trace_id, parent_span_id, span_id, method, url, None, str(e),
                        ts, end_ts, int(time.time() * 1000 - start_ms),
                        req_headers=req_headers, req_body=_req_body(request),
                    )
                except Exception as hook_exc:
                    hook_failed("httpx", hook_exc)
                raise
            _in_http_span.reset(_http_token)
            try:
                end_ts = _now_iso()
                latency = int(time.time() * 1000 - start_ms)
                resp_headers = sanitize_headers(dict(resp.headers))
                span = dict(
                    trace_id=trace_id, parent_span_id=parent_span_id, span_id=span_id,
                    method=method, url=url, status_code=resp.status_code, start_ts=ts,
                    req_headers=req_headers, resp_headers=resp_headers,
                    req_body=_req_body(request),
                )
                if _is_streaming(resp) and _emit_streamed_span(
                    client, resp, True, start_ms, time_to_headers_ms=latency, **span
                ):
                    return resp
                _emit_http_span(
                    client, error_msg=None, end_ts=end_ts, latency=latency,
                    resp_body=_resp_body(resp), **span,
                )
            except Exception as exc:
                hook_failed("httpx", exc)
            return resp

        traced_async_send._evalkit = True
        httpx.AsyncClient.send = traced_async_send


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
    stream_attrs: Optional[Dict[str, Any]] = None,
    time_to_headers_ms: Optional[int] = None,
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
    if time_to_headers_ms is not None:
        attrs["http.server.time_to_headers_ms"] = time_to_headers_ms
    if stream_attrs:
        attrs.update(stream_attrs)
    if error_msg:
        attrs["error.message"] = error_msg
        status = "ERROR"

    client.enqueue(TraceEnvelope(
        trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
        span_type="http_call", status=status, status_message=error_msg or None,
        span_kind="CLIENT", operation=client_span_name(method),
        service_name=client._opts.service_name,
        attributes=attrs, latency_ms=latency, start_time=start_ts, end_time=end_ts,
        created_at=start_ts, environment=client._opts.environment,
        app_version=client._opts.app_version, sdk_version=SDK_VERSION,
    ))
