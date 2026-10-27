"""Which answers make a request's span an error.

By default every 4xx and 5xx answer does: the SERVER span is ERROR, and an exception the server answers
with is recorded. ``server_error_status_codes=["500-599"]`` follows OpenTelemetry's HTTP conventions
instead: a 4xx (a scanner probing /.env, an expired token, a body that fails validation) is the caller's
mistake, so the span keeps its status code, stays OK and records no exception. The middleware checks run
with that setting, so each one shows that the answer decides the span; the default is checked on its own.
Outgoing calls keep a 4xx as an error either way.
"""
import asyncio
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import evalkit
from evalkit.core import http_status
from tests.helpers import fresh, reset_context
from tests.test_framework_spans import _call
from tests.test_init_safety import _run, _warnings


class OrderFailure(Exception):
    pass


def _exception_events(span):
    return [e for e in (span.events or []) if e.name == "exception"]


class _Case(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self.client.flush()
        self.captured.clear()
        # The setting that leaves a 4xx to the caller; DefaultIsEvery4xxAnd5xx checks the default.
        http_status.configure(["500-599"])
        self.addCleanup(http_status.reset)

    def exported(self):
        self.client.flush()
        return list(self.captured)

    def server_spans(self):
        return [e for e in self.exported() if e.span_type == "http_call" and e.span_kind == "SERVER"]

    def server_span(self):
        servers = self.server_spans()
        self.assertEqual(len(servers), 1, [e.operation for e in servers])
        self.captured.clear()
        return servers[0]

    def assert_client_error(self, span, code):
        self.assertEqual(span.status, "OK", f"a {code} is the caller's mistake, not the server's failure")
        self.assertEqual(span.attributes["http.status_code"], code, "the status code is still recorded")
        self.assertEqual(_exception_events(span), [], "no exception is recorded for a client error")
        self.assertNotIn("error.type", span.attributes)
        self.assertNotIn("evalkit.exception.frames", span.attributes)

    def assert_server_error(self, span, code, message=None, escaped=None):
        self.assertEqual(span.status, "ERROR")
        self.assertEqual(span.attributes["http.status_code"], code)
        events = _exception_events(span)
        if message is None:
            return
        self.assertEqual(len(events), 1, "the exception is recorded once")
        self.assertEqual(events[0].attributes["exception.message"], message)
        self.assertIn("evalkit.exception.frames", span.attributes)
        if escaped is not None:
            self.assertIs(events[0].attributes["exception.escaped"], escaped)


class TheRule(unittest.TestCase):
    def setUp(self):
        self.addCleanup(http_status.reset)

    def test_by_default_every_4xx_and_5xx_is_a_server_error(self):
        self.assertEqual(http_status.DEFAULT_SERVER_ERROR_STATUS_CODES, ("400-599",))
        for code in (400, 401, 403, 404, 405, 422, 429, 499, 500, 502, 503, 599):
            self.assertTrue(http_status.is_server_error_status(code), code)
            self.assertEqual(http_status.server_span_status(code), "ERROR")
        for code in (200, 204, 301, 399, None, "500", True):
            self.assertFalse(http_status.is_server_error_status(code), code)
            self.assertEqual(http_status.server_span_status(code), "OK")

    def test_500_599_leaves_a_4xx_to_the_caller(self):
        http_status.configure(["500-599"])
        for code in (400, 404, 429, 499):
            self.assertFalse(http_status.is_server_error_status(code), code)
        self.assertTrue(http_status.is_server_error_status(500))

    def test_codes_and_ranges_replace_the_default(self):
        http_status.configure([429, "500-599"])
        self.assertTrue(http_status.is_server_error_status(429))
        self.assertTrue(http_status.is_server_error_status(503))
        self.assertFalse(http_status.is_server_error_status(404))
        http_status.configure(["500-502", " 504 ", "520 - 530"])
        self.assertFalse(http_status.is_server_error_status(503), "a 503 left out of the list is not an error")
        self.assertTrue(http_status.is_server_error_status(504))
        self.assertTrue(http_status.is_server_error_status(525))
        http_status.reset()
        self.assertTrue(http_status.is_server_error_status(503))

    def test_a_value_that_cannot_be_read_keeps_the_default_with_a_warning(self):
        for value in ("500-599", 503, [600], [99], ["5xx"], ["599-500"], [True], [503.0], [None], {"a": 1}):
            with self.subTest(value=value):
                http_status.configure(["500-502"])
                with self.assertLogs("evalkit", "WARNING") as logs:
                    http_status.configure(value)
                self.assertIn("server_error_status_codes", logs.output[0])
                self.assertTrue(http_status.is_server_error_status(503))
                self.assertTrue(http_status.is_server_error_status(404), "the 400-599 default is back")
        self.assertIsNone(http_status.parse_server_error_status_codes([503, "500-599"])[1])

    def test_a_raised_request_fails_unless_it_was_answered_outside_the_codes(self):
        self.assertTrue(http_status.raised_request_failed(None), "a request that got no answer")
        self.assertTrue(http_status.raised_request_failed(200), "a response that broke off part way")
        self.assertTrue(http_status.raised_request_failed(500))
        self.assertTrue(http_status.raised_request_failed(404), "by default a 4xx answer is a failure too")
        http_status.configure(["500-599"])
        self.assertFalse(http_status.raised_request_failed(404))

    def test_http_answers_are_told_by_class_name_without_importing_the_framework(self):
        def answer(module, qualname, **attrs):
            base = type(qualname, (Exception,), {"__module__": module, "__qualname__": qualname})
            sub = type("Custom", (base,), {"__module__": "app.errors"})
            exc = sub("x")
            for name, value in attrs.items():
                setattr(exc, name, value)
            return exc

        self.assertEqual(http_status.http_answer_status(answer("starlette.exceptions", "HTTPException", status_code=401)), 401)
        self.assertEqual(http_status.http_answer_status(answer("werkzeug.exceptions", "HTTPException", code=404)), 404)
        self.assertEqual(http_status.http_answer_status(
            answer("litestar.exceptions.http_exceptions", "HTTPException", status_code=403)), 403)
        self.assertEqual(http_status.http_answer_status(answer("django.http.response", "Http404")), 404)
        self.assertEqual(http_status.http_answer_status(answer("django.core.exceptions", "PermissionDenied")), 403)
        self.assertEqual(http_status.http_answer_status(answer("django.core.exceptions", "SuspiciousOperation")), 400)
        self.assertEqual(http_status.http_answer_status(answer("django.core.exceptions", "BadRequest")), 400)
        self.assertIsNone(http_status.http_answer_status(answer("werkzeug.exceptions", "HTTPException", code=None)))
        self.assertIsNone(http_status.http_answer_status(answer("myapp", "HTTPException", status_code=404)))
        self.assertIsNone(http_status.http_answer_status(ValueError("x")))


class TheInitOption(unittest.TestCase):
    _AFTER = ("from evalkit.core.http_status import is_server_error_status as e\n"
              "EXTRA = [e(429), e(500), e(503)]")

    def test_init_passes_the_codes_on(self):
        result, stderr = _run(
            'dict(subscription_key="tk_test", base_url="http://127.0.0.1:1", function_tracing=False, '
            'server_error_status_codes=[429, "500-502"])', after=self._AFTER)
        self.assertTrue(result["enabled"])
        self.assertEqual(result["extra"], [True, True, False])
        self.assertEqual(_warnings(stderr), [])

    def test_an_invalid_value_keeps_the_sdk_on_and_the_default(self):
        result, stderr = _run(
            'dict(subscription_key="tk_test", base_url="http://127.0.0.1:1", function_tracing=False, '
            'server_error_status_codes="500-599")', after=self._AFTER)
        self.assertTrue(result["enabled"], "an unreadable status list does not turn the SDK off")
        self.assertEqual(result["extra"], [True, True, True], "the 400-599 default stays")
        warnings = _warnings(stderr)
        self.assertEqual(len(warnings), 1, warnings)
        self.assertIn("server_error_status_codes", warnings[0])

    def test_the_default_is_documented_on_init(self):
        import inspect

        self.assertIsNone(inspect.signature(evalkit.init).parameters["server_error_status_codes"].default)
        self.assertEqual(http_status.DEFAULT_SERVER_ERROR_STATUS_CODES, ("400-599",))


class TestASGIMiddlewareStatus(_Case):
    """The FastAPI / Starlette middleware on a bare ASGI app, so no framework is needed."""

    def _answer(self, status):
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": status, "headers": []})
            await send({"type": "http.response.body", "body": b"x"})

        _call(evalkit.EvalKitMiddleware(app, self.client), "GET", "/.env")
        return self.server_span()

    def test_client_errors_are_not_server_errors(self):
        for code in (400, 401, 403, 404, 405, 422, 429):
            with self.subTest(code=code):
                self.assert_client_error(self._answer(code), code)

    def test_a_5xx_is_an_error(self):
        for code in (500, 502, 503):
            with self.subTest(code=code):
                self.assert_server_error(self._answer(code), code)

    def test_the_configured_codes_decide(self):
        http_status.configure([429, "500-599"])
        self.assert_server_error(self._answer(429), 429)
        http_status.configure(["500-502"])
        self.assert_client_error(self._answer(503), 503)

    def test_a_raised_exception_is_a_500_with_the_exception(self):
        async def app(scope, receive, send):
            raise OrderFailure("ledger down")

        with self.assertRaises(OrderFailure):
            _call(evalkit.EvalKitMiddleware(app, self.client), "GET", "/orders/1")
        self.assert_server_error(self.server_span(), 500, "ledger down", escaped=True)

    def test_a_response_that_breaks_off_is_an_error(self):
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"part", "more_body": True})
            raise OrderFailure("stream broke")

        with self.assertRaises(OrderFailure):
            _call(evalkit.EvalKitMiddleware(app, self.client), "GET", "/export")
        self.assert_server_error(self.server_span(), 200, "stream broke", escaped=True)

    def test_an_exception_after_a_4xx_was_sent_is_not_recorded(self):
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 404, "headers": []})
            raise OrderFailure("after the answer")

        with self.assertRaises(OrderFailure):
            _call(evalkit.EvalKitMiddleware(app, self.client), "GET", "/missing")
        self.assert_client_error(self.server_span(), 404)

    def test_a_cancelled_request_stays_an_error(self):
        async def app(scope, receive, send):
            raise asyncio.CancelledError()

        with self.assertRaises(asyncio.CancelledError):
            _call(evalkit.EvalKitMiddleware(app, self.client), "GET", "/slow")
        span = self.server_span()
        self.assertEqual(span.status, "ERROR")
        self.assertNotIn("http.status_code", span.attributes)
        self.assertIs(span.attributes.get("evalkit.cancelled"), True)


class DefaultIsEvery4xxAnd5xx(_Case):
    """With no setting, a 4xx answer fails the server span, and a raised 4xx answer records its exception."""

    def setUp(self):
        super().setUp()
        http_status.reset()

    def _answer(self, status):
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": status, "headers": []})
            await send({"type": "http.response.body", "body": b"x"})

        _call(evalkit.EvalKitMiddleware(app, self.client), "GET", "/orders/7")
        return self.server_span()

    def test_a_4xx_answer_is_an_error(self):
        for code in (400, 401, 404, 429):
            with self.subTest(code=code):
                self.assert_server_error(self._answer(code), code)

    def test_a_2xx_and_3xx_answer_is_not(self):
        for code in (200, 302):
            with self.subTest(code=code):
                self.assertEqual(self._answer(code).status, "OK")


class TestFastAPIStatus(_Case):
    def setUp(self):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            self.skipTest("fastapi not installed")
        super().setUp()

    def _app(self):
        from fastapi import Depends, FastAPI, HTTPException
        from fastapi.responses import JSONResponse

        app = FastAPI()
        app.add_middleware(evalkit.EvalKitMiddleware, client=self.client)

        def signed_in():
            raise HTTPException(status_code=401, detail="expired token")

        @app.get("/orders/{order_id}")
        def order(order_id: int):
            if order_id == 0:
                raise HTTPException(status_code=400, detail="no such order id")
            return {"order_id": order_id}

        @app.get("/me")
        def me(user=Depends(signed_in)):
            return {}

        @app.get("/boom")
        def boom():
            raise OrderFailure("in the endpoint")

        @app.get("/busy")
        def busy():
            return JSONResponse({"busy": True}, status_code=503)

        return app

    def test_client_errors(self):
        app = self._app()
        _call(app, "GET", "/wp-config.php")
        span = self.server_span()
        self.assert_client_error(span, 404)
        self.assertEqual(span.operation, "GET /wp-config.php", "an unmatched path keeps the raw path")
        _call(app, "GET", "/me")
        self.assert_client_error(self.server_span(), 401)
        _call(app, "GET", "/orders/0")
        span = self.server_span()
        self.assert_client_error(span, 400)
        self.assertEqual(span.operation, "GET /orders/{order_id}")
        _call(app, "GET", "/orders/abc")
        self.assert_client_error(self.server_span(), 422)

    def test_server_errors(self):
        app = self._app()
        with self.assertRaises(OrderFailure):
            _call(app, "GET", "/boom")
        span = self.server_span()
        self.assert_server_error(span, 500, "in the endpoint", escaped=True)
        self.assertEqual(span.operation, "GET /boom")
        _call(app, "GET", "/busy")
        self.assert_server_error(self.server_span(), 503)


class TestStarletteStatus(_Case):
    def setUp(self):
        try:
            import starlette  # noqa: F401
        except ImportError:
            self.skipTest("starlette not installed")
        super().setUp()

    def test_client_and_server_errors(self):
        from starlette.applications import Starlette
        from starlette.exceptions import HTTPException
        from starlette.middleware import Middleware
        from starlette.routing import Route

        async def private(request):
            raise HTTPException(status_code=401)

        async def boom(request):
            raise OrderFailure("in the starlette endpoint")

        app = Starlette(
            routes=[Route("/private/{item_id}", private), Route("/boom", boom)],
            middleware=[Middleware(evalkit.EvalKitStarletteMiddleware, client=self.client)],
        )
        _call(app, "GET", "/.env")
        self.assert_client_error(self.server_span(), 404)
        _call(app, "GET", "/private/7")
        span = self.server_span()
        self.assert_client_error(span, 401)
        self.assertEqual(span.operation, "GET /private/{item_id}")
        with self.assertRaises(OrderFailure):
            _call(app, "GET", "/boom")
        self.assert_server_error(self.server_span(), 500, "in the starlette endpoint", escaped=True)


class TestFlaskStatus(_Case):
    def setUp(self):
        try:
            import flask  # noqa: F401
        except ImportError:
            self.skipTest("flask not installed")
        super().setUp()

    def _app(self):
        from flask import Flask, Response, abort

        app = Flask("status")
        app.config["PROPAGATE_EXCEPTIONS"] = False
        evalkit.instrument_flask(app)

        @app.get("/orders/<int:order_id>")
        def order(order_id):
            if order_id == 0:
                abort(400)
            return {"order_id": order_id}

        @app.get("/private")
        def private():
            abort(401)

        @app.get("/denied")
        def denied():
            return "no", 403

        @app.get("/boom")
        def boom():
            raise OrderFailure("in the flask view")

        @app.get("/busy")
        def busy():
            return "busy", 503

        @app.get("/stream")
        def stream():
            def chunks():
                for i in range(3):
                    time.sleep(0.05)
                    yield f"chunk {i}\n"

            return Response(chunks(), mimetype="text/plain")

        return app

    def test_client_errors(self):
        client = self._app().test_client()
        self.assertEqual(client.get("/.env").status_code, 404)
        span = self.server_span()
        self.assert_client_error(span, 404)
        self.assertEqual(span.operation, "GET /.env")
        client.get("/private")
        self.assert_client_error(self.server_span(), 401)
        client.get("/denied")
        self.assert_client_error(self.server_span(), 403)
        client.get("/orders/0")
        span = self.server_span()
        self.assert_client_error(span, 400)
        self.assertEqual(span.operation, "GET /orders/<int:order_id>")

    def test_server_errors(self):
        app = self._app()
        self.assertEqual(app.test_client().get("/boom").status_code, 500)
        self.assert_server_error(self.server_span(), 500, "in the flask view", escaped=False)
        app.test_client().get("/busy")
        self.assert_server_error(self.server_span(), 503)

    def test_an_exception_flask_propagates_has_escaped(self):
        app = self._app()
        app.config["PROPAGATE_EXCEPTIONS"] = True
        with self.assertRaises(OrderFailure):
            app.test_client().get("/boom")
        self.assert_server_error(self.server_span(), 500, "in the flask view", escaped=True)

    def test_the_configured_codes_decide(self):
        http_status.configure(["500-502"])
        self._app().test_client().get("/busy")
        self.assert_client_error(self.server_span(), 503)

    def test_a_streamed_response_ends_when_its_last_chunk_is_sent(self):
        from werkzeug.test import EnvironBuilder

        app = self._app()
        environ = EnvironBuilder(path="/stream").get_environ()
        body = app.wsgi_app(environ, lambda status, headers, exc_info=None: None)
        self.assertEqual(self.server_spans(), [], "the span is open while the response streams")
        try:
            chunks = list(body)
        finally:
            body.close()
        self.assertEqual(len(chunks), 3)
        span = self.server_span()
        self.assertEqual(span.status, "OK")
        self.assertGreaterEqual(span.latency_ms, 140, "the latency covers the stream")
        self.assertEqual(span.operation, "GET /stream")


class TestDjangoStatus(_Case):
    def setUp(self):
        try:
            import django
            from django.conf import settings
        except ImportError:
            self.skipTest("django not installed")
        if not settings.configured:
            settings.configure(
                DEBUG=False, ALLOWED_HOSTS=["*"], ROOT_URLCONF="tests.test_framework_spans", SECRET_KEY="test-only",
                MIDDLEWARE=["evalkit.frameworks.django.EvalKitDjangoMiddleware"],
            )
            django.setup()
        super().setUp()

    def _get(self, path):
        import logging

        from django.test import Client, override_settings

        # Django logs the 4xx it answers a SuspiciousOperation or a missing page with; the test reads spans.
        previous = logging.getLogger("django").level
        logging.getLogger("django").setLevel(logging.CRITICAL)
        try:
            with override_settings(ROOT_URLCONF=__name__):
                return Client(raise_request_exception=False).get(path)
        finally:
            logging.getLogger("django").setLevel(previous)

    def test_exceptions_django_answers_with_a_client_error_are_not_recorded(self):
        for path, code in (("/orders/0/", 404), ("/denied/", 403), ("/suspicious/", 400), ("/bad/", 400),
                           ("/private/", 401), ("/.env", 404)):
            with self.subTest(path=path):
                self.assertEqual(self._get(path).status_code, code)
                self.assert_client_error(self.server_span(), code)

    def test_the_route_is_kept_for_a_client_error(self):
        self._get("/orders/0/")
        self.assertEqual(self.server_span().operation, "GET /orders/<int:order_id>/")

    def test_a_view_that_raises_is_a_server_error_with_the_exception(self):
        self.assertEqual(self._get("/boom/").status_code, 500)
        span = self.server_span()
        self.assert_server_error(span, 500, "in the django view", escaped=False)
        self.assertEqual(span.operation, "GET /boom/")

    def test_a_streamed_response_ends_when_its_last_chunk_is_sent(self):
        response = self._get("/stream/")
        self.assertEqual(self.server_spans(), [], "the span is open while the response streams")
        # The test client closes a streamed response after its last chunk, as a WSGI server does.
        self.assertEqual(len(list(response.streaming_content)), 3)
        span = self.server_span()
        self.assertEqual(span.status, "OK")
        self.assertGreaterEqual(span.latency_ms, 140, "the latency covers the stream")


def _django_urls():
    try:
        from django.core.exceptions import BadRequest, PermissionDenied, SuspiciousOperation
        from django.http import Http404, HttpResponse, JsonResponse, StreamingHttpResponse
        from django.urls import path
    except Exception:
        return []

    def order(request, order_id):
        if order_id == 0:
            raise Http404("no such order")
        return JsonResponse({"order_id": order_id})

    def denied(request):
        raise PermissionDenied("not yours")

    def suspicious(request):
        raise SuspiciousOperation("odd header")

    def bad(request):
        raise BadRequest("malformed")

    def private(request):
        return HttpResponse("sign in", status=401)

    def boom(request):
        raise OrderFailure("in the django view")

    def stream(request):
        def chunks():
            for i in range(3):
                time.sleep(0.05)
                yield f"chunk {i}\n"

        return StreamingHttpResponse(chunks())

    return [
        path("orders/<int:order_id>/", order), path("denied/", denied), path("suspicious/", suspicious),
        path("bad/", bad), path("private/", private), path("boom/", boom), path("stream/", stream),
    ]


urlpatterns = _django_urls()


class TestLitestarStatus(_Case):
    def setUp(self):
        try:
            import litestar  # noqa: F401
        except ImportError:
            self.skipTest("litestar not installed")
        super().setUp()

    def _app(self):
        from litestar import Litestar, get
        from litestar.exceptions import HTTPException, NotAuthorizedException
        from litestar.response import Response

        @get("/orders/{order_id:int}", sync_to_thread=False)
        def order(order_id: int) -> dict:
            if order_id == 0:
                raise HTTPException(status_code=400, detail="no such order id")
            return {"order_id": order_id}

        @get("/private", sync_to_thread=False)
        def private() -> dict:
            raise NotAuthorizedException("expired token")

        @get("/boom", sync_to_thread=False)
        def boom() -> dict:
            raise OrderFailure("in the litestar handler")

        @get("/busy", sync_to_thread=False)
        def busy() -> Response:
            return Response({"busy": True}, status_code=503)

        return Litestar(route_handlers=[order, private, boom, busy],
                        middleware=[evalkit.create_litestar_middleware(self.client)])

    def test_client_errors(self):
        app = self._app()
        _call(app, "GET", "/private")
        self.assert_client_error(self.server_span(), 401)
        _call(app, "GET", "/orders/0")
        span = self.server_span()
        self.assert_client_error(span, 400)
        _call(app, "GET", "/.env")
        self.assertEqual(self.server_spans(), [], "Litestar answers an unmatched path before any middleware")

    def test_the_request_span_is_named_by_the_route_template(self):
        _call(self._app(), "GET", "/orders/4429")
        span = self.server_span()
        self.assertEqual(span.status, "OK")
        self.assertEqual(span.operation, "GET /orders/{order_id}")
        self.assertEqual(span.attributes["http.route"], "/orders/{order_id}")

    def test_server_errors(self):
        app = self._app()
        sent = _call(app, "GET", "/boom")
        self.assertEqual(next(m for m in sent if m["type"] == "http.response.start")["status"], 500)
        span = self.server_span()
        self.assert_server_error(span, 500, "in the litestar handler", escaped=False)
        self.assertEqual(span.operation, "GET /boom")
        _call(app, "GET", "/busy")
        span = self.server_span()
        self.assert_server_error(span, 503)
        self.assertEqual(_exception_events(span), [])


class TestFunctionSpans(_Case):
    def _span(self, name):
        spans = [e for e in self.exported() if e.operation == name]
        self.assertEqual(len(spans), 1, [e.operation for e in self.captured])
        self.captured.clear()
        return spans[0]

    def _http_exception(self, code):
        try:
            from starlette.exceptions import HTTPException
        except ImportError:
            HTTPException = type("HTTPException", (Exception,), {
                "__module__": "starlette.exceptions", "__qualname__": "HTTPException",
                "__init__": lambda self, status_code: setattr(self, "status_code", status_code),
            })
        return HTTPException(status_code=code)

    def test_a_client_error_answer_ends_the_span_ok(self):
        @self.client.trace_function("require_user")
        def require_user():
            raise self._http_exception(404)

        with self.assertRaises(Exception):
            require_user()
        span = self._span("require_user")
        self.assertEqual(span.status, "OK")
        self.assertEqual(_exception_events(span), [])
        self.assertEqual(span.attributes["http.response.status_code"], 404)
        self.assertNotIn("error.type", span.attributes)

    def test_an_async_dependency_raising_a_401_ends_ok(self):
        @self.client.trace_tool("check_token")
        async def check_token():
            raise self._http_exception(401)

        with self.assertRaises(Exception):
            asyncio.run(check_token())
        span = self._span("check_token")
        self.assertEqual(span.status, "OK")
        self.assertEqual(span.attributes["http.response.status_code"], 401)

    def test_a_server_error_answer_and_any_other_exception_stay_errors(self):
        @self.client.trace_function("upstream_down")
        def upstream_down():
            raise self._http_exception(503)

        @self.client.trace_function("parse")
        def parse():
            raise ValueError("bad input")

        with self.assertRaises(Exception):
            upstream_down()
        span = self._span("upstream_down")
        self.assertEqual(span.status, "ERROR")
        self.assertEqual(len(_exception_events(span)), 1)
        self.assertNotIn("http.response.status_code", span.attributes)
        with self.assertRaises(ValueError):
            parse()
        span = self._span("parse")
        self.assertEqual(span.status, "ERROR")
        self.assertEqual(_exception_events(span)[0].attributes["exception.message"], "bad input")

    def test_the_configured_codes_decide(self):
        http_status.configure([429, "500-599"])

        @self.client.trace_function("throttle")
        def throttle():
            raise self._http_exception(429)

        with self.assertRaises(Exception):
            throttle()
        self.assertEqual(self._span("throttle").status, "ERROR")

    def test_other_frameworks_answers(self):
        cases = []
        try:
            from werkzeug.exceptions import NotFound

            cases.append((NotFound(), 404))
        except ImportError:
            pass
        try:
            from django.core.exceptions import PermissionDenied
            from django.http import Http404

            cases += [(Http404(), 404), (PermissionDenied(), 403)]
        except ImportError:
            pass
        try:
            from litestar.exceptions import NotAuthorizedException

            cases.append((NotAuthorizedException(), 401))
        except ImportError:
            pass
        for exc, code in cases:
            with self.subTest(exc=type(exc).__name__):
                @self.client.trace_function("view")
                def view():
                    raise exc

                with self.assertRaises(type(exc)):
                    view()
                span = self._span("view")
                self.assertEqual(span.status, "OK")
                self.assertEqual(span.attributes["http.response.status_code"], code)


class TestOutgoingCallsAreUnchanged(_Case):
    def test_a_4xx_an_outgoing_call_gets_back_stays_an_error(self):
        try:
            import httpx
        except ImportError:
            self.skipTest("httpx not installed")

        class NotFoundHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *a):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), NotFoundHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            self.assertEqual(httpx.get(f"http://127.0.0.1:{server.server_address[1]}/missing").status_code, 404)
        finally:
            server.shutdown()
            server.server_close()
        calls = [e for e in self.exported() if e.span_type == "http_call" and e.span_kind == "CLIENT"]
        self.assertEqual(len(calls), 1, [e.operation for e in self.captured])
        self.assertEqual(calls[0].status, "ERROR", "for the app, a call that got a 404 failed")


if __name__ == "__main__":
    unittest.main()
