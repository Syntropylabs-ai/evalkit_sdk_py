from __future__ import annotations

import json
from typing import Any, Callable

from ..core.client import EvalKitClient
from ..core.context import _current_span
from ..utils.headers import parse_traceparent, sanitize_headers

_BODY_MAX = 4096


def create_litestar_middleware(client: "EvalKitClient | None" = None) -> Any:
    try:
        from litestar.middleware.base import AbstractMiddleware
    except ImportError:
        raise RuntimeError("Litestar is not installed. pip install litestar")

    def _get_client() -> EvalKitClient:
        if client is not None:
            return client
        import evalkit as _ek
        return _ek._get_instance()

    class EvalKitLitestarMiddleware(AbstractMiddleware):
        async def __call__(self, scope: Any, receive: Callable, send: Callable) -> None:
            if scope.get("type") != "http":
                await self.app(scope, receive, send)
                return

            ek = _get_client()
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

            first_msg = await receive()
            first_body = first_msg.get("body", b"")
            first_msg_consumed = False

            async def receive_replay() -> Any:
                nonlocal first_msg_consumed
                if not first_msg_consumed:
                    first_msg_consumed = True
                    return first_msg
                return await receive()

            body_json: Any = None
            if first_body:
                try:
                    body_json = json.loads(first_body[:_BODY_MAX].decode("utf-8", errors="replace"))
                except Exception:
                    body_json = first_body[:_BODY_MAX].decode("utf-8", errors="replace") or None

            try:
                trace_id, end, ctx = ek.start_http_trace(
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
                # Tracing setup failed — run untraced, replaying the consumed body.
                await self.app(scope, receive_replay, send)
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
                    pass  # span bookkeeping must never corrupt the response
                await send(message)

            try:
                await self.app(scope, receive_replay, send_wrapper)
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

    return EvalKitLitestarMiddleware
