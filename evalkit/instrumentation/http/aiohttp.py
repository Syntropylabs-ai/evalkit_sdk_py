from __future__ import annotations

import json
import time
from typing import Any, Optional

from ..._version import SDK_VERSION
from ...core.client import EvalKitClient, _new_span_id, _now_iso
from ...core.context import _current_span, get_or_create_ambient_trace_id
from ...core.types import TraceEnvelope
from ...utils.headers import make_traceparent, path_from_url, sanitize_headers

_BODY_MAX = 10 * 1024 * 1024  # default; overridden from client options at patch time
_SKIP = {"/v1/ingest", "/health", "/healthz"}


def _should_skip(url: str) -> bool:
    return any(pat in url for pat in _SKIP)


def patch_aiohttp(client: EvalKitClient) -> None:
    """Patch aiohttp.ClientSession._request to trace every outgoing HTTP call."""
    global _BODY_MAX
    _BODY_MAX = client._opts.max_body_bytes
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
        **kwargs: Any,
    ) -> Any:
        url = str(str_or_url)
        if _should_skip(url):
            return await orig_request(self, method, str_or_url, **kwargs)

        ctx = _current_span.get()
        trace_id = ctx.trace_id if ctx else get_or_create_ambient_trace_id()
        parent_span_id: Optional[str] = ctx.span_id if ctx else None
        span_id = _new_span_id()
        start_ms = time.time() * 1000
        ts = _now_iso()

        headers: Any = kwargs.pop("headers", None) or {}
        if isinstance(headers, dict):
            headers["traceparent"] = make_traceparent(trace_id, span_id)
        kwargs["headers"] = headers

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

        try:
            resp = await orig_request(self, method, str_or_url, **kwargs)
        except Exception as e:
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
                span_type="http_call", status="ERROR", status_message=str(e),
                operation=f"{method.upper()} {path_from_url(url)}",
                service_name=client._opts.service_name,
                attributes=attrs, latency_ms=latency, start_time=ts, end_time=end_ts,
                created_at=ts, environment=client._opts.environment,
                app_version=client._opts.app_version, sdk_version=SDK_VERSION,
            ))
            raise

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
    aiohttp.ClientSession._request = traced_request  # type: ignore[method-assign]
