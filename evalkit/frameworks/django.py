from __future__ import annotations

import json
from typing import Any, Callable

from ..core.client import EvalKitClient
from ..core.context import _current_span
from ..utils.headers import parse_traceparent

_BODY_MAX = 10 * 1024 * 1024  # default; overridden from client options per request

_SENSITIVE_META = {
    "HTTP_AUTHORIZATION", "HTTP_COOKIE", "HTTP_X_API_KEY",
    "HTTP_X_AUTH_TOKEN", "HTTP_PROXY_AUTHORIZATION", "HTTP_X_SECRET",
    "HTTP_TOKEN", "HTTP_PASSWORD", "HTTP_X_SUBSCRIPTION_KEY",
}


_SENSITIVE_RESP = {"set-cookie", "authorization", "x-api-key", "x-subscription-key"}


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


def _sanitize_response_headers(response: Any) -> dict:
    out = {}
    try:
        for k, v in response.items():  # HttpResponse supports .items()
            if str(k).lower() in _SENSITIVE_RESP:
                continue
            out[str(k)] = str(v)
    except Exception:
        pass
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
        global _BODY_MAX
        # Tracing setup must never break the request — fall back to untraced.
        try:
            client = self._get_client()
            _BODY_MAX = client._opts.max_body_bytes

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
        except Exception:
            return self.get_response(request)

        try:
            response = self.get_response(request)
            sc = response.status_code
            resp_headers = None
            resp_body: Any = None
            try:
                resp_headers = _sanitize_response_headers(response) or None
            except Exception:
                pass
            try:
                if not getattr(response, "streaming", False):
                    content = getattr(response, "content", b"") or b""
                    if content:
                        resp_body = content[:_BODY_MAX].decode("utf-8", errors="replace") or None
            except Exception:
                pass
            try:
                end("ERROR" if sc >= 400 else "OK", status_code=sc,
                    response_body=resp_body, response_headers=resp_headers)
            except Exception:
                pass
            try:
                response["X-Trace-Id"] = trace_id
                response["traceparent"] = f"00-{trace_id}-{ctx.span_id}-01"
            except Exception:
                pass
            return response
        except Exception:
            try:
                end("ERROR", status_code=500)
            except Exception:
                pass
            raise
        finally:
            try:
                _current_span.reset(token)
            except Exception:
                pass

    def process_exception(self, request: Any, exception: Exception) -> None:
        pass
