"""Server spans of the framework integrations: one trace per request, named by the route template.

FastAPI's built-in telemetry opens a server span before any middleware runs, and EvalKit's middleware
opens its own; with nothing joining them every request exported as two traces. These tests run real
apps through the ASGI interface (no HTTP client, which the SDK would trace as well) with the SDK's
OpenTelemetry bridge on, as `evalkit.init` leaves it.
"""
import asyncio
import unittest

import evalkit
from tests.helpers import fresh, reset_context

UPSTREAM_TRACE = "a" * 32
UPSTREAM_SPAN = "b" * 16


def _call(app, method, path, *, body=b"", traceparent=None, root_path=""):
    headers = [(b"host", b"testserver"), (b"content-type", b"application/json")]
    if traceparent:
        headers.append((b"traceparent", traceparent.encode()))
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": method, "path": path, "raw_path": path.encode(), "root_path": root_path,
        "query_string": b"", "headers": headers, "scheme": "http",
        "server": ("testserver", 80), "client": ("127.0.0.1", 50000),
    }
    pending = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive():
        return pending.pop(0) if pending else {"type": "http.disconnect"}

    sent = []

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    return sent


class _ServerSpanCase(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self.client.flush()
        self.captured.clear()

    def exported(self):
        self.client.flush()
        return list(self.captured)

    def request_span(self, spans):
        servers = [e for e in spans if e.span_type == "http_call" and e.span_kind == "SERVER"]
        self.assertEqual(
            len(servers), 1,
            f"one server span per request; got {[(e.operation, (e.attributes or {}).get('evalkit.sdk_version')) for e in servers]}",
        )
        return servers[0]


class TestFastAPIOneTrace(_ServerSpanCase):
    def setUp(self):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            self.skipTest("fastapi not installed")
        super().setUp()

    def _app(self, **kwargs):
        from fastapi import APIRouter, FastAPI
        from opentelemetry import trace

        app = FastAPI(**kwargs)
        app.add_middleware(evalkit.EvalKitMiddleware, client=self.client)
        client = self.client

        @app.post("/orders/{order_id}/refunds")
        def refund(order_id: int):
            # A library with its own OpenTelemetry instrumentation (SQLAlchemy, botocore) opens its
            # span under whatever OpenTelemetry span is current; an SDK span opens under the request.
            with trace.get_tracer("app.db").start_as_current_span("SELECT orders"):
                pass
            end, _ = client.start_span("plan_refund")
            end("OK")
            return {"order_id": order_id}

        router = APIRouter(prefix="/orders")

        @router.get("/{order_id}")
        def get_order(order_id: int):
            return {"order_id": order_id}

        app.include_router(router, prefix="/api")
        return app

    def test_a_request_exports_as_one_trace(self):
        _call(self._app(), "POST", "/orders/4821/refunds")
        spans = self.exported()
        request = self.request_span(spans)
        self.assertEqual(
            {e.trace_id for e in spans}, {request.trace_id},
            "FastAPI's own server span and EvalKit's request span were two traces; every span of the request "
            f"belongs to one: {[(e.operation, e.trace_id[:8], e.parent_span_id) for e in spans]}",
        )
        self.assertIsNone(request.parent_span_id)
        self.assertFalse(
            [e.operation for e in spans if (e.operation or "").startswith("fastapi.")],
            "FastAPI's dependency, endpoint and serialization spans sit beside EvalKit's spans rather than "
            "around them and add spans to every request; they are left out of a request the middleware records",
        )
        by_name = {e.operation: e for e in spans}
        self.assertEqual(
            by_name["SELECT orders"].parent_span_id, request.span_id,
            "a span another OpenTelemetry library opens inside the endpoint hangs off the request span",
        )
        self.assertEqual(by_name["plan_refund"].parent_span_id, request.span_id)

    def test_the_request_span_is_named_by_the_route_template(self):
        _call(self._app(), "POST", "/orders/4821/refunds")
        request = self.request_span(self.exported())
        self.assertEqual(request.operation, "POST /orders/{order_id}/refunds", (
            "the raw path puts the order id in the name, so the Services table grew one row per order"
        ))
        self.assertEqual(request.attributes["http.route"], "/orders/{order_id}/refunds")
        self.assertTrue(request.attributes["http.url"].endswith("/orders/4821/refunds"), "the raw URL stays")

    def test_an_included_router_keeps_its_prefix(self):
        _call(self._app(), "GET", "/api/orders/4429")
        request = self.request_span(self.exported())
        self.assertEqual(request.operation, "GET /api/orders/{order_id}")

    def test_an_upstream_traceparent_is_still_the_parent(self):
        _call(self._app(), "POST", "/orders/1/refunds", traceparent=f"00-{UPSTREAM_TRACE}-{UPSTREAM_SPAN}-01")
        spans = self.exported()
        request = self.request_span(spans)
        self.assertEqual(request.trace_id, UPSTREAM_TRACE)
        self.assertEqual(request.parent_span_id, UPSTREAM_SPAN)
        self.assertEqual({e.trace_id for e in spans}, {UPSTREAM_TRACE})

    def test_the_response_traceparent_names_the_exported_request_span(self):
        sent = _call(self._app(), "POST", "/orders/7/refunds")
        request = self.request_span(self.exported())
        start = next(m for m in sent if m["type"] == "http.response.start")
        headers = dict(start["headers"])
        self.assertEqual(headers[b"traceparent"].decode(), f"00-{request.trace_id}-{request.span_id}-01")

    def test_with_fastapi_tracing_off_the_route_is_still_templated(self):
        _call(self._app(telemetry={"tracing": False}), "GET", "/api/orders/4429")
        spans = self.exported()
        request = self.request_span(spans)
        self.assertEqual(request.operation, "GET /api/orders/{order_id}")
        self.assertEqual({e.trace_id for e in spans}, {request.trace_id})

    def test_an_unmatched_path_keeps_the_raw_path(self):
        _call(self._app(), "GET", "/nowhere/1")
        request = self.request_span(self.exported())
        self.assertEqual(request.operation, "GET /nowhere/1")
        self.assertNotIn("http.route", request.attributes)


class TestStarletteRoutes(_ServerSpanCase):
    def setUp(self):
        try:
            import starlette  # noqa: F401
        except ImportError:
            self.skipTest("starlette not installed")
        super().setUp()

    def test_a_mounted_app_route_carries_the_mount_path(self):
        from starlette.applications import Starlette
        from starlette.middleware import Middleware
        from starlette.responses import JSONResponse
        from starlette.routing import Mount, Route

        async def item(request):
            return JSONResponse({"id": request.path_params["item_id"]})

        inner = Starlette(routes=[Route("/items/{item_id:int}", item)])
        app = Starlette(
            routes=[Mount("/shop", app=inner)],
            middleware=[Middleware(evalkit.EvalKitStarletteMiddleware, client=self.client)],
        )
        _call(app, "GET", "/shop/items/42")
        request = self.request_span(self.exported())
        self.assertEqual(request.operation, "GET /shop/items/{item_id}")

    def test_starlettes_own_opentelemetry_server_span_is_one_trace_with_evalkits(self):
        try:
            from starlette.middleware.opentelemetry import OpenTelemetryMiddleware
        except ImportError:
            self.skipTest("this Starlette has no OpenTelemetryMiddleware")
        from starlette.applications import Starlette
        from starlette.middleware import Middleware
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route

        async def order(request):
            return PlainTextResponse("ok")

        app = Starlette(
            routes=[Route("/orders/{order_id}", order)],
            middleware=[
                Middleware(OpenTelemetryMiddleware),
                Middleware(evalkit.EvalKitStarletteMiddleware, client=self.client),
            ],
        )
        _call(app, "GET", "/orders/4429")
        spans = self.exported()
        request = self.request_span(spans)
        self.assertEqual({e.trace_id for e in spans}, {request.trace_id})
        self.assertEqual(request.operation, "GET /orders/{order_id}")


class TestFlaskRoutes(_ServerSpanCase):
    def setUp(self):
        try:
            import flask  # noqa: F401
        except ImportError:
            self.skipTest("flask not installed")
        super().setUp()

    def _app(self):
        from flask import Flask

        app = Flask("orders")
        evalkit.instrument_flask(app)

        @app.get("/orders/<int:order_id>")
        def get_order(order_id):
            return {"order_id": order_id}

        @app.get("/boom/<int:order_id>")
        def boom(order_id):
            raise RuntimeError("down")

        return app

    def test_the_request_span_is_named_by_the_url_rule(self):
        self._app().test_client().get("/orders/4429")
        request = self.request_span(self.exported())
        self.assertEqual(request.operation, "GET /orders/<int:order_id>")
        self.assertEqual(request.attributes["http.route"], "/orders/<int:order_id>")

    def test_a_failed_request_is_named_by_the_url_rule_too(self):
        app = self._app()
        app.config["PROPAGATE_EXCEPTIONS"] = False
        app.test_client().get("/boom/1")
        request = self.request_span(self.exported())
        self.assertEqual(request.operation, "GET /boom/<int:order_id>")
        self.assertEqual(request.status, "ERROR")


class TestDjangoRoutes(_ServerSpanCase):
    def setUp(self):
        try:
            import django
            from django.conf import settings
        except ImportError:
            self.skipTest("django not installed")
        if not settings.configured:
            settings.configure(
                DEBUG=False, ALLOWED_HOSTS=["*"], ROOT_URLCONF=__name__, SECRET_KEY="test-only",
                MIDDLEWARE=["evalkit.frameworks.django.EvalKitDjangoMiddleware"],
            )
            django.setup()
        super().setUp()

    def test_the_request_span_is_named_by_the_resolved_pattern(self):
        from django.test import Client

        Client().get("/api/orders/4429/")
        request = self.request_span(self.exported())
        self.assertEqual(request.operation, "GET /api/orders/<int:order_id>/")
        self.assertEqual(request.attributes["http.route"], "/api/orders/<int:order_id>/")


def _django_urls():
    try:
        from django.http import JsonResponse
        from django.urls import include, path
    except Exception:
        return []

    def order(request, order_id):
        return JsonResponse({"order_id": order_id})

    return [path("api/", include([path("orders/<int:order_id>/", order)]))]


urlpatterns = _django_urls()
