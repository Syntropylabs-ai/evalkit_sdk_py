from __future__ import annotations

from typing import Any, Callable

from ..core.client import EvalKitClient, release_ambient_trace
from ..core.context import _current_span
from ..utils.headers import parse_traceparent, sanitize_headers
from ._bodies import RequestBodyTap, ResponseBodyTap
from .fastapi import _failed_status
from ._lifespan import flush_on_lifespan_shutdown, register_client, registered_clients
from ._probe import answer_probe_untraced, is_ignored

_BODY_MAX = 64 * 1024


def _flush_on_litestar_shutdown(client: "EvalKitClient | None") -> None:
    # Litestar answers the lifespan scope in its own __call__, before any middleware, so the flush
    # the worker needs before reporting its shutdown complete is made on the application class.
    register_client(client)
    try:
        from litestar import Litestar

        flush_on_lifespan_shutdown(Litestar, registered_clients)
    except Exception:
        pass


def create_litestar_middleware(client: "EvalKitClient | None" = None) -> Any:
    try:
        from litestar.middleware.base import AbstractMiddleware
    except ImportError:
        raise RuntimeError("Litestar is not installed. pip install litestar")
    _flush_on_litestar_shutdown(client)

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
            # Litestar runs its middleware after routing, so the matched route's template is known here.
            if is_ignored(scope.get("method"), scope.get("path"), route=scope.get("path_template")):
                await answer_probe_untraced(self.app, scope, receive, send)
                return

            ek = _get_client()
            if not getattr(ek, "enabled", True):
                await self.app(scope, receive, send)
                return
            global _BODY_MAX
            _BODY_MAX = ek._opts.max_body_bytes
            method = scope.get("method", "GET")
            path = scope.get("path", "/")
            query = scope.get("query_string", b"").decode("utf-8", "replace")
            server = scope.get("server") or ("localhost", 80)
            url = f"{scope.get('scheme', 'http')}://{server[0]}{path}"
            if query:
                url = f"{url}?{query}"

            raw_headers = {
                k.decode("utf-8", "replace"): v.decode("utf-8", "replace") for k, v in scope.get("headers", [])
            }
            safe_headers = sanitize_headers(raw_headers)

            upstream_trace_id, upstream_span_id = None, None
            tp = raw_headers.get("traceparent", "")
            if tp:
                upstream_trace_id, upstream_span_id = parse_traceparent(tp)

            # Kept from what the app reads, as it reads it; see _bodies.
            request_tap = RequestBodyTap(receive, _BODY_MAX, raw_headers)
            response_tap = ResponseBodyTap(_BODY_MAX)

            try:
                trace_id, end, ctx = ek.start_http_trace(
                    f"{method} {path}",
                    method=method,
                    url=url,
                    headers=safe_headers,
                    upstream_trace_id=upstream_trace_id,
                    upstream_span_id=upstream_span_id,
                )
                token = _current_span.set(ctx)
            except Exception:
                await self.app(scope, receive, send)
                return

            status_code_holder: list[int] = [200]
            started: list[bool] = [False]
            response_headers_holder: list[dict] = [{}]

            async def send_wrapper(message: Any) -> None:
                try:
                    msg_type = message.get("type")
                    if msg_type == "http.response.start":
                        started[0] = True
                        status_code_holder[0] = message.get("status", 200)
                        raw_resp_headers = {
                            k.decode("utf-8", "replace"): v.decode("utf-8", "replace")
                            for k, v in message.get("headers", [])
                        }
                        response_headers_holder[0] = raw_resp_headers
                        response_tap.start(raw_resp_headers)
                        headers = list(message.get("headers", []))
                        headers.append((b"x-trace-id", trace_id.encode()))
                        headers.append((
                            b"traceparent",
                            f"00-{trace_id}-{ctx.span_id}-01".encode(),
                        ))
                        message = {**message, "headers": headers}
                    elif msg_type == "http.response.body":
                        response_tap.add(message.get("body", b""))
                except Exception:
                    pass
                await send(message)

            def body_attributes() -> dict:
                return {**request_tap.attributes(), **response_tap.attributes()}

            try:
                await self.app(scope, request_tap.receive, send_wrapper)
                sc = status_code_holder[0]
                try:
                    end(
                        "ERROR" if sc >= 400 else "OK",
                        status_code=sc,
                        response_body=response_tap.value(),
                        response_headers=response_headers_holder[0] or None,
                        request_body=request_tap.value(),
                        attributes=body_attributes() or None,
                    )
                except Exception:
                    pass
            except BaseException as exc:
                try:
                    end(
                        "ERROR",
                        status_code=_failed_status(exc, status_code_holder[0] if started[0] else None),
                        request_body=request_tap.value(),
                        attributes=body_attributes() or None,
                        exception=exc,
                    )
                except Exception:
                    pass
                raise
            finally:
                try:
                    _current_span.reset(token)
                except Exception:
                    pass
                release_ambient_trace(trace_id)

    return EvalKitLitestarMiddleware
