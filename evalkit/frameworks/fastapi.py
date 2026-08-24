from __future__ import annotations

import json
from typing import Any, Callable

from ..core.client import EvalKitClient
from ..core.context import SpanContext, _current_span
from ..utils.headers import parse_traceparent, sanitize_headers

_BODY_MAX = 10 * 1024 * 1024


class EvalKitMiddleware:

    def __init__(self, app: Any, client: "EvalKitClient | None" = None) -> None:
        self._app = app
        self._client = client

    def _get_client(self) -> EvalKitClient:
        if self._client:
            return self._client
        import evalkit as _ek
        return _ek._get_instance()

    async def __call__(self, scope: Any, receive: Callable, send: Callable) -> None:
        global _BODY_MAX
        if scope["type"] not in ("http",):
            await self._app(scope, receive, send)
            return

        try:
            client = self._get_client()
        except Exception:
            await self._app(scope, receive, send)
            return
        _BODY_MAX = client._opts.max_body_bytes
        method = scope.get("method", "GET")
        path = scope.get("path", "/")
        query = scope.get("query_string", b"").decode()
        server = scope.get("server") or ("localhost", 80)
        url = f"{scope.get('scheme', 'http')}://{server[0]}{path}"
        if query:
            url = f"{url}?{query}"

        raw_headers = {k.decode(): v.decode() for k, v in scope.get("headers", [])}
        safe_headers = sanitize_headers(raw_headers)

        upstream_trace_id, upstream_span_id = None, None
        tp = raw_headers.get("traceparent", "")
        if tp:
            upstream_trace_id, upstream_span_id = parse_traceparent(tp)

        _chunks: list[bytes] = []
        _msgs: list[Any] = []
        _total_len = 0
        while True:
            _msg = await receive()
            _msgs.append(_msg)
            _chunk = _msg.get("body", b"")
            if _chunk and _total_len < _BODY_MAX:
                _take = _BODY_MAX - _total_len
                _chunks.append(_chunk[:_take])
                _total_len += len(_chunk)
            if not _msg.get("more_body", False):
                break
        total_body = b"".join(_chunks)

        _msg_idx = 0

        async def receive_replay() -> Any:
            nonlocal _msg_idx
            if _msg_idx < len(_msgs):
                m = _msgs[_msg_idx]
                _msg_idx += 1
                return m
            return await receive()

        body_json: Any = None
        if total_body:
            try:
                body_json = json.loads(total_body[:_BODY_MAX].decode("utf-8", errors="replace"))
            except Exception:
                body_json = total_body[:_BODY_MAX].decode("utf-8", errors="replace") or None

        try:
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
        except Exception:
            await self._app(scope, receive_replay, send)
            return

        status_code_holder: list[int] = [200]
        response_headers_holder: list[dict] = [{}]
        response_body_holder: list[bytes] = [b""]

        async def send_wrapper(message: Any) -> None:
            try:
                msg_type = message.get("type")
                if msg_type == "http.response.start":
                    status_code_holder[0] = message.get("status", 200)
                    raw_resp_headers = {
                        k.decode(): v.decode()
                        for k, v in message.get("headers", [])
                    }
                    response_headers_holder[0] = raw_resp_headers
                    headers = list(message.get("headers", []))
                    headers.append((b"x-trace-id", trace_id.encode()))
                    headers.append((
                        b"traceparent",
                        f"00-{trace_id}-{ctx.span_id}-01".encode(),
                    ))
                    message = {**message, "headers": headers}
                elif msg_type == "http.response.body":
                    if len(response_body_holder[0]) < _BODY_MAX:
                        response_body_holder[0] += message.get("body", b"")
            except Exception:
                pass
            await send(message)

        try:
            await self._app(scope, receive_replay, send_wrapper)
            sc = status_code_holder[0]
            resp_body: Any = None
            raw_resp = response_body_holder[0][:_BODY_MAX]
            if raw_resp:
                try:
                    resp_body = json.loads(raw_resp.decode("utf-8", errors="replace"))
                except Exception:
                    resp_body = raw_resp.decode("utf-8", errors="replace") or None
            try:
                end(
                    "ERROR" if sc >= 400 else "OK",
                    status_code=sc,
                    response_body=resp_body,
                    response_headers=response_headers_holder[0] or None,
                )
            except Exception:
                pass
        except Exception:
            try:
                end("ERROR")
            except Exception:
                pass
            raise
        finally:
            try:
                _current_span.reset(token)
            except Exception:
                pass
