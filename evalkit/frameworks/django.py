from __future__ import annotations

import json
import sys
from typing import Any, Callable

from ..core.client import EvalKitClient, release_ambient_trace
from ..core.context import _current_span, reset_identity, set_identity
from ..core.http_status import is_server_error_status, raised_request_failed, server_span_status
from ..utils.headers import parse_traceparent
from ._bodies import is_binary, is_event_stream
from ._identity import user_id_of
from ._lifespan import answer_lifespan_when_refused, registered_clients
from ._probe import is_ignored, probe_untraced

_BODY_MAX = 64 * 1024

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
        for k, v in response.items():
            if str(k).lower() in _SENSITIVE_RESP:
                continue
            out[str(k)] = str(v)
    except Exception:
        pass
    return out


def _request_body(request: Any) -> Any:
    """The request body, when the request says it is small enough to keep and is not an upload.

    ``request.body`` reads the whole body into memory and keeps it there, so it is read only under the
    limit: an upload, or a body of unknown length, is left for the view to stream.
    """
    meta = getattr(request, "META", None) or {}
    try:
        length = int(meta.get("CONTENT_LENGTH") or 0)
    except (TypeError, ValueError):
        return None
    if length <= 0 or length > _BODY_MAX or is_binary(meta.get("CONTENT_TYPE")):
        return None
    raw = request.body[:_BODY_MAX]
    try:
        return json.loads(raw)
    except Exception:
        try:
            return raw.decode("utf-8", errors="replace") or None
        except Exception:
            return None


def django_route(request: Any) -> "str | None":
    """The resolved URL pattern (`/orders/<int:order_id>/`), or None when nothing resolved (a 404).

    Django's `ResolverMatch.route` joins the patterns of every included urlconf, without the
    leading slash the request path has.
    """
    try:
        route = getattr(getattr(request, "resolver_match", None), "route", None)
    except Exception:
        return None
    if not isinstance(route, str) or not route:
        return None
    return route if route.startswith("/") else "/" + route


def _route_before_resolving(request: Any) -> "str | None":
    """The URL pattern a request will resolve to, for a callable in ignore_routes that reads it: Django
    resolves the URL only after the middleware has let the request in."""
    from django.urls import resolve

    route = getattr(resolve(getattr(request, "path_info", None) or request.path), "route", None)
    if not isinstance(route, str) or not route:
        return None
    return route if route.startswith("/") else "/" + route


def _end_when_closed(response: Any, finish: Callable[[], None]) -> bool:
    """Run ``finish`` when the server closes a streamed response, after its last chunk, rather than when
    the view returned it. The WSGI server and Django's ASGI handler both close a response once it is sent."""
    if not getattr(response, "streaming", False):
        return False
    close = getattr(response, "close", None)
    if not callable(close):
        return False
    ended = [False]

    def close_and_finish() -> None:
        try:
            close()
        finally:
            if not ended[0]:
                ended[0] = True
                try:
                    finish()
                except Exception:
                    pass

    try:
        response.close = close_and_finish
    except Exception:
        return False
    return True


def _flush_on_asgi_shutdown() -> None:
    # Django's ASGI handler refuses the lifespan scope, so under uvicorn nothing of the application
    # runs once the server starts stopping, and the worker ends without atexit. The handler is
    # given a lifespan whose shutdown sends the queue. Under WSGI the ASGI handler is usually not
    # imported at all, and then nothing is changed.
    module = sys.modules.get("django.core.handlers.asgi")
    handler = getattr(module, "ASGIHandler", None) if module is not None else None
    if handler is None:
        return
    try:
        answer_lifespan_when_refused(handler, registered_clients)
    except Exception:
        pass


class EvalKitDjangoMiddleware:
    def __init__(self, get_response: Callable) -> None:
        self.get_response = get_response
        self._client: "EvalKitClient | None" = None
        _flush_on_asgi_shutdown()

    def _get_client(self) -> EvalKitClient:
        if self._client is None:
            import evalkit as _ek
            self._client = _ek._get_instance()
        return self._client

    def __call__(self, request: Any) -> Any:
        global _BODY_MAX
        if is_ignored(getattr(request, "method", None), getattr(request, "path", None),
                      resolve_route=lambda: _route_before_resolving(request)):
            with probe_untraced():
                return self.get_response(request)
        identity_token = None
        try:
            enabled = getattr(self._get_client(), "enabled", True)
        except Exception:
            enabled = True
        if not enabled:
            return self.get_response(request)
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
                    body_json = _request_body(request)
                except Exception:
                    body_json = None

            request_user_id = user_id_of(getattr(request, "user", None))
            if request_user_id:
                identity_token = set_identity(user_id=request_user_id)

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
            if identity_token is not None:
                reset_identity(identity_token)
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
                content_type = response.get("Content-Type") if hasattr(response, "get") else None
                if (not getattr(response, "streaming", False) and not is_event_stream(content_type)
                        and not is_binary(content_type)):
                    content = getattr(response, "content", b"") or b""
                    if content:
                        resp_body = content[:_BODY_MAX].decode("utf-8", errors="replace") or None
            except Exception:
                pass
            try:
                # A view's exception is turned into a response before it reaches here; process_exception
                # kept it. It failed the request only when that response is a server error: Http404,
                # PermissionDenied and SuspiciousOperation are answered with a 4xx.
                raised = getattr(request, "_evalkit_exception", None)
                if not isinstance(raised, BaseException) or not is_server_error_status(sc):
                    raised = None
                route = django_route(request)

                def finish() -> None:
                    end(server_span_status(sc), status_code=sc,
                        response_body=resp_body, response_headers=resp_headers,
                        route=route, exception=raised, exception_escaped=False)

                if not _end_when_closed(response, finish):
                    finish()
            except Exception:
                pass
            try:
                response["X-Trace-Id"] = trace_id
                response["traceparent"] = f"00-{trace_id}-{ctx.span_id}-01"
            except Exception:
                pass
            return response
        except BaseException as exc:
            try:
                failed = raised_request_failed(500)
                end("ERROR" if failed else "OK", status_code=500, route=django_route(request),
                    exception=exc if failed else None)
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
            try:
                request._evalkit_exception = None
            except Exception:
                pass

    def process_exception(self, request: Any, exception: Exception) -> None:
        # Kept for the request's span; returning None leaves the exception to Django's own handling.
        try:
            request._evalkit_exception = exception
        except Exception:
            pass
        return None
