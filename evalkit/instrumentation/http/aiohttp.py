from __future__ import annotations

import json
import time
from typing import Any, Optional

from ..._version import SDK_VERSION
from ...core.client import EvalKitClient, _new_span_id, _now_iso
from ...core.context import _current_span, current_span_ctx, _in_http_span, get_or_create_ambient_trace_id
from ...core.types import TraceEnvelope
from ...utils.headers import client_span_name, make_traceparent, sanitize_headers
from ._stream_body import attach_aiohttp, new_capture
from ._skip import should_skip as _should_skip
from .._safe import hook_failed

_BODY_MAX = 64 * 1024
_STREAM_BODIES = True


def patch_aiohttp(client: EvalKitClient) -> None:
    global _BODY_MAX, _STREAM_BODIES
    _BODY_MAX = client._opts.max_body_bytes
    _STREAM_BODIES = client._opts.capture_stream_bodies
    try:
        import aiohttp
    except ImportError:
        return

    if getattr(aiohttp.ClientSession._request, "_evalkit", False):
        return

    orig_request = aiohttp.ClientSession._request

    async def traced_request(
        self: Any,
        method: str,
        str_or_url: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        try:
            url = str(str_or_url)
            skip = bool(args) or _should_skip(url, method) or _in_http_span.get()
        except Exception:
            skip = True
        if skip:
            return await orig_request(self, method, str_or_url, *args, **kwargs)

        try:
            ctx = current_span_ctx()
            trace_id = ctx.trace_id if ctx else get_or_create_ambient_trace_id()
            parent_span_id: Optional[str] = ctx.span_id if ctx else None
            span_id = _new_span_id()
            start_ms = time.time() * 1000
            ts = _now_iso()

            # The caller's own headers dict is left as it was; the traceparent goes on a copy.
            call_kwargs = dict(kwargs)
            headers: Any = call_kwargs.pop("headers", None) or {}
            if isinstance(headers, dict):
                headers = {**headers, "traceparent": make_traceparent(trace_id, span_id)}
            call_kwargs["headers"] = headers

            req_headers_sanitized = sanitize_headers(dict(headers) if hasattr(headers, "items") else {})

            req_body: Optional[str] = None
            if "json" in kwargs and kwargs["json"] is not None:
                try:
                    req_body = json.dumps(kwargs["json"])[:_BODY_MAX]
                except Exception:
                    pass
            elif "data" in kwargs and kwargs["data"] is not None:
                d = kwargs["data"]
                if isinstance(d, (str, bytes)):
                    req_body = (d if isinstance(d, str) else d.decode("utf-8", errors="replace"))[:_BODY_MAX]
        except Exception as exc:
            hook_failed("aiohttp", exc)
            return await orig_request(self, method, str_or_url, *args, **kwargs)
        kwargs = call_kwargs

        _http_token = _in_http_span.set(True)
        try:
            resp = await orig_request(self, method, str_or_url, **kwargs)
        except Exception as e:
            _in_http_span.reset(_http_token)
            try:
                end_ts = _now_iso()
                latency = int(time.time() * 1000 - start_ms)
                attrs: dict = {
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
                hook_failed("aiohttp", hook_exc)
            raise
        _in_http_span.reset(_http_token)
        try:
            end_ts = _now_iso()
            latency = int(time.time() * 1000 - start_ms)
            status_code = resp.status

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
            try:
                resp_headers = sanitize_headers(dict(resp.headers))
                if resp_headers:
                    attrs["http.response.headers"] = json.dumps(resp_headers)
            except Exception:
                pass

            if _STREAM_BODIES and _BODY_MAX > 0:
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
                if attach_aiohttp(resp, capture):
                    return resp
                capture.finish()
                return resp

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
            hook_failed("aiohttp", exc)
        return resp

    traced_request._evalkit = True
    aiohttp.ClientSession._request = traced_request
