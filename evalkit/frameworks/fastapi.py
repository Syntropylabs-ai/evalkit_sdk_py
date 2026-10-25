from __future__ import annotations

from typing import Any, Callable

from ..core.client import EvalKitClient, release_ambient_trace
from ..core.context import SpanContext, _current_span, reset_identity, set_identity
from ..core.otel_provider import claim_framework_server_span, release_framework_server_span
from ..utils.headers import parse_traceparent, sanitize_headers
from ._bodies import RequestBodyTap, ResponseBodyTap
from ._identity import user_id_of
from ._lifespan import flush_before_shutdown_reply
from ._probe import answer_probe_untraced, is_ignored

_BODY_MAX = 64 * 1024


def _matched_template(routes: Any, scope: Any) -> "str | None":
    """Re-match the request against a router's routes, for a Starlette that does not record the match.

    Starlette before 1.0 keeps no route in the scope for its own `Route`s. The template is the root
    path the mounts in between added plus the matched route's path.
    """
    try:
        from starlette.routing import Match, Mount
    except Exception:
        return None
    for route in routes or ():
        try:
            match, child = route.matches(scope)
        except Exception:
            continue
        if match != Match.FULL:
            continue
        child_scope = {**scope, **child}
        if isinstance(route, Mount):
            inner = _matched_template(getattr(route, "routes", None), child_scope)
            if inner:
                return inner
        path = getattr(route, "path_format", None)
        if not isinstance(path, str) or not path.startswith("/"):
            return None
        prefix = scope.get("root_path") if isinstance(route, Mount) else child_scope.get("root_path")
        return (prefix or "").rstrip("/") + path
    return None


def route_template(
    scope: Any,
    entry_root_path: str = "",
    framework_span: Any = None,
    entry_path: "str | None" = None,
) -> "str | None":
    """The matched route's path template (`/orders/{order_id}`), read after the app ran, or None.

    In order: the `http.route` the framework's own request span recorded; the prefixed path FastAPI
    keeps for a route of an included router (whose `scope["route"]` carries the router's path
    without the include prefix); the route Starlette's router left in `scope["route"]`; the route a
    re-match of the router's routes finds. A mounted app adds its mount path to `root_path`; a Mount
    that is itself the match (static files) already carries its prefix in its own path.
    """
    try:
        recorded = (getattr(framework_span, "attributes", None) or {}).get("http.route")
        if isinstance(recorded, str) and recorded.startswith("/"):
            return recorded
    except Exception:
        pass
    try:
        prefix = (scope.get("root_path") or "").rstrip("/")
        effective = (scope.get("fastapi") or {}).get("effective_route_context")
        path = getattr(effective, "path_format", None)
        if isinstance(path, str) and path.startswith("/"):
            return prefix + path
        route = scope.get("route")
        path = getattr(route, "path_format", None) or getattr(route, "path", None)
        if isinstance(path, str) and path.startswith("/"):
            if type(route).__name__ == "Mount":
                prefix = entry_root_path.rstrip("/")
            return prefix + path
        router = scope.get("router")
        if router is None or entry_path is None:
            return None
        entry_scope = {
            "type": scope.get("type", "http"),
            "method": scope.get("method", "GET"),
            "path": entry_path,
            "root_path": entry_root_path,
        }
        return _matched_template(getattr(router, "routes", None), entry_scope)
    except Exception:
        return None


def _failed_status(exc: BaseException, sent: "int | None") -> "int | None":
    """The status a failed request is recorded with: what was sent, else the 500 the server answers
    with, except for a request cancelled because the client went away, which got no answer."""
    if sent is not None:
        return sent
    try:
        import asyncio

        if isinstance(exc, asyncio.CancelledError):
            return None
    except Exception:
        pass
    return 500


class EvalKitMiddleware:

    def __init__(self, app: Any, client: "EvalKitClient | None" = None) -> None:
        self._app = app
        self._client = client

    def _get_client(self) -> EvalKitClient:
        if self._client:
            return self._client
        import evalkit as _ek
        return _ek._get_instance()

    async def _lifespan(self, scope: Any, receive: Callable, send: Callable) -> None:
        # The server ends the worker soon after the app reports its shutdown complete, and uvicorn
        # then re-raises the signal that stopped it, which ends the process without atexit. The
        # queue, with what the app's own shutdown handlers recorded, goes out before that report.
        try:
            send = flush_before_shutdown_reply(send, lambda: [self._get_client()])
        except Exception:
            pass
        await self._app(scope, receive, send)

    async def _answer_untraced(self, scope: Any, receive: Callable, send: Callable) -> None:
        # A framework that traces requests itself opened the request's server span before this
        # middleware ran; claimed and never handed back, it is dropped at export with its inner spans.
        try:
            if _current_span.get() is None:
                claim_framework_server_span()
        except Exception:
            pass
        await answer_probe_untraced(self._app, scope, receive, send)

    def _route_before_routing(self, scope: Any) -> "str | None":
        # Starlette matches the route inside the app, after this middleware decides; a callable in
        # ignore_routes that reads the route has it matched here, once, the way the router will.
        router = getattr(scope.get("app"), "router", None) or getattr(self._app, "router", None)
        entry = {
            "type": "http",
            "method": scope.get("method", "GET"),
            "path": scope.get("path", "/"),
            "root_path": scope.get("root_path") or "",
        }
        return _matched_template(getattr(router, "routes", None), entry)

    async def __call__(self, scope: Any, receive: Callable, send: Callable) -> None:
        global _BODY_MAX
        if scope["type"] == "lifespan":
            await self._lifespan(scope, receive, send)
            return
        if scope["type"] not in ("http",):
            await self._app(scope, receive, send)
            return
        if is_ignored(scope.get("method"), scope.get("path"), resolve_route=lambda: self._route_before_routing(scope)):
            await self._answer_untraced(scope, receive, send)
            return

        try:
            client = self._get_client()
        except Exception:
            await self._app(scope, receive, send)
            return
        if not getattr(client, "enabled", True):
            await self._app(scope, receive, send)
            return
        _BODY_MAX = client._opts.max_body_bytes
        method = scope.get("method", "GET")
        path = scope.get("path", "/")
        entry_root_path = scope.get("root_path") or ""
        query = scope.get("query_string", b"").decode("utf-8", "replace")
        server = scope.get("server") or ("localhost", 80)
        url = f"{scope.get('scheme', 'http')}://{server[0]}{path}"
        if query:
            url = f"{url}?{query}"

        raw_headers = {k.decode("utf-8", "replace"): v.decode("utf-8", "replace") for k, v in scope.get("headers", [])}
        safe_headers = sanitize_headers(raw_headers)

        upstream_trace_id, upstream_span_id = None, None
        tp = raw_headers.get("traceparent", "")
        if tp:
            upstream_trace_id, upstream_span_id = parse_traceparent(tp)

        # The body is kept from the messages the app reads, as it reads them: reading it ahead would
        # hold an upload in memory before the app saw its first byte.
        request_tap = RequestBodyTap(receive, _BODY_MAX, raw_headers)
        response_tap = ResponseBodyTap(_BODY_MAX)

        identity_token = None
        request_user_id = user_id_of(scope.get("user"))
        if request_user_id:
            identity_token = set_identity(user_id=request_user_id)

        # A framework that traces requests itself (FastAPI's built-in telemetry) has already opened
        # this request's server span; taking its ids keeps one trace per request. That span read the
        # same traceparent, so its parent is the upstream span. A middleware on a mounted app, inside
        # one that already took the span, leaves it alone.
        claimed = claim_framework_server_span() if _current_span.get() is None else None
        claimed_span_id = claimed.span_id if claimed is not None else None
        framework_span = claimed.span if claimed is not None else None
        if claimed is not None:
            upstream_trace_id, upstream_span_id = claimed.trace_id, claimed.parent_span_id
        try:
            trace_id, end, ctx = client.start_http_trace(
                f"{method} {path}",
                method=method,
                url=url,
                headers=safe_headers,
                upstream_trace_id=upstream_trace_id,
                upstream_span_id=upstream_span_id,
                span_id=claimed_span_id,
            )
            token = _current_span.set(ctx)
        except Exception:
            release_framework_server_span(claimed)
            if identity_token is not None:
                reset_identity(identity_token)
            await self._app(scope, receive, send)
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
            await self._app(scope, request_tap.receive, send_wrapper)
            sc = status_code_holder[0]
            try:
                end(
                    "ERROR" if sc >= 400 else "OK",
                    status_code=sc,
                    response_body=response_tap.value(),
                    response_headers=response_headers_holder[0] or None,
                    route=route_template(scope, entry_root_path, framework_span, path),
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
                    route=route_template(scope, entry_root_path, framework_span, path),
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
            if identity_token is not None:
                reset_identity(identity_token)
            release_ambient_trace(trace_id)
