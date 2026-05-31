from __future__ import annotations

import json
from typing import Any, Callable

from ..core.client import EvalKitClient
from ..core.context import _current_span
from ..utils.headers import parse_traceparent

_BODY_MAX = 4096

_SENSITIVE_META = {
    "HTTP_AUTHORIZATION", "HTTP_COOKIE", "HTTP_X_API_KEY",
    "HTTP_X_AUTH_TOKEN", "HTTP_PROXY_AUTHORIZATION", "HTTP_X_SECRET",
    "HTTP_TOKEN", "HTTP_PASSWORD", "HTTP_X_SUBSCRIPTION_KEY",
}


def _sanitize_meta(meta: dict) -> dict:
    out = {}
    for k, v in meta.items():
        if not k.startswith("HTTP_"):
            continue
        if k in _SENSITIVE_META:
            continue
        header_name = k[5:].replace("_", "-").lower()
        out[header_name] = str(v)
    return out


class EvalKitDjangoMiddleware:
    def __init__(self, get_response: Callable) -> None:
        self.get_response = get_response
        self._client: "EvalKitClient | None" = None

    def _get_client(self) -> EvalKitClient:
        if self._client is None:
            import evalkit as _ek
            self._client = _ek._get_instance()
        return self._client

    def __call__(self, request: Any) -> Any:
        client = self._get_client()

        method = request.method or "GET"
        path = request.path or "/"
        url = request.build_absolute_uri()

        safe_headers = _sanitize_meta(request.META)

        upstream_trace_id, upstream_span_id = None, None
        tp = request.META.get("HTTP_TRACEPARENT", "")
        if tp:
            upstream_trace_id, upstream_span_id = parse_traceparent(tp)

        body_json: Any = None
        if method in ("POST", "PUT", "PATCH"):
            try:
                body_json = json.loads(request.body[:_BODY_MAX])
            except Exception:
                try:
                    body_text = request.body[:_BODY_MAX].decode("utf-8", errors="replace")
                    body_json = body_text or None
                except Exception:
                    pass

        trace_id, end, ctx = client.start_http_trace(
            f"{method} {path}",
            method=method,
            url=url,
            headers=safe_headers,
            request_body=body_json,
            upstream_trace_id=upstream_trace_id,
            upstream_span_id=upstream_span_id,
        )
        token = _current_span.set(ctx)
        request._evalkit_trace_id = trace_id

        try:
            response = self.get_response(request)
            sc = response.status_code
            end("ERROR" if sc >= 400 else "OK", status_code=sc)
            response["X-Trace-Id"] = trace_id
            response["traceparent"] = f"00-{trace_id}-{ctx.span_id}-01"
            return response
        except Exception:
            end("ERROR")
            raise
        finally:
            _current_span.reset(token)

    def process_exception(self, request: Any, exception: Exception) -> None:
        pass
