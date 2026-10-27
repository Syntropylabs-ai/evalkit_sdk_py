"""Health and readiness probes reach the app untraced, through every framework integration.

A probe is told by whole path segment, the rule outgoing calls are skipped by and the one the
TypeScript SDK uses, so /healthcare and /api/health-records are the app's own routes and are recorded.
Nothing the probe handler does is recorded, while work it starts that runs on after the response is.
The probe still gets its answer, unchanged.
"""
import logging
import threading
import unittest

import evalkit
from tests.helpers import fresh, reset_context
from tests.test_framework_spans import _call

# The paths the TypeScript SDK's health-probes test decides, then more absolute URLs and edge cases.
PROBES = [
    "/health", "/healthz", "/health/", "/health/live", "/health/ready", "/readyz", "/livez", "/startupz",
    "/actuator/health", "/actuator/health/liveness", "/v1/healthcheck?verbose=1", "/HEALTHZ", "/health-check",
    "/health_check", "/health?probe=k8s",
    "http://svc/health", "https://svc:8443/HEALTHZ", "/actuator/health/readiness", "/health/alive",
    "/api/v1/healthz#top", "/express/health/started", "/api/v2/health", "/services/billing/healthz",
]
APP_PATHS = [
    "/healthcare", "/api/health-records", "/healthcare/claims", "/patients/7/health/records",
    "/v1/orders?next=/health", "/wealth", "/health/records",
    "http://svc/healthcare", "/cb?next=http://h/health", "/healthz/details", "/", "",
    "/datasets/6650a1b2c3d4e5f601234567/health", "/datasets/:datasetId/health", "/users/42/healthz",
    "/runs/0b6f6a52-6d0e-4b0c-9a51-3f1f3d4b2a10/health/ready", "/projects/{projectId}/health",
]


class TestTheProbeRule(unittest.TestCase):
    def test_probes_are_told_by_whole_path_segment(self):
        from evalkit.frameworks._probe import is_probe

        for path in PROBES:
            with self.subTest(path):
                self.assertTrue(is_probe(path))
        for path in APP_PATHS:
            with self.subTest(path):
                self.assertFalse(is_probe(path), "the app's own route is recorded")
        self.assertFalse(is_probe(None))
        self.assertFalse(is_probe(b"/health"), "an ASGI path is text; anything else is not read")


class _ProbeCase(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self.client.flush()
        self.captured.clear()
        self.threads = []
        self.gates = []

    def exported(self):
        # The thread a probe started goes on once the probe has been answered.
        for gate in self.gates:
            gate.set()
        for thread in self.threads:
            thread.join(5)
        self.client.flush()
        return list(self.captured)

    def probe_work(self):
        """What a probe handler does: a span, a traced function, another library's span, a log line,
        and a thread that runs on after the response."""
        from opentelemetry import trace

        end, _ = self.client.start_span("probe.ping_database")
        end("OK")
        self.client.trace_function("probe.check_cache")(lambda: None)()
        with trace.get_tracer("app.db").start_as_current_span("SELECT 1"):
            pass
        logging.getLogger("probe.test").warning("probe log line")
        gate = threading.Event()

        def later():
            gate.wait(5)
            end_later, _ = self.client.start_span("background.after_probe")
            end_later("OK")

        thread = threading.Thread(target=later)
        thread.start()
        self.threads.append(thread)
        self.gates.append(gate)
        return gate

    def assert_only_background_work(self, spans):
        names = sorted(e.operation for e in spans)
        self.assertEqual(names, ["background.after_probe"], (
            "a probe exports nothing of its own, not even the framework's server span; the thread it "
            f"started and that ran on after the response is traced: {names}"
        ))

    def assert_app_route_traced(self, spans, path):
        servers = [e for e in spans if e.span_type == "http_call" and e.span_kind == "SERVER"]
        self.assertEqual(len(servers), 1, [e.operation for e in spans])
        self.assertTrue(servers[0].attributes["http.url"].endswith(path))
        self.assertIn("inside.app_route", {e.operation for e in spans})


class TestASGIMiddleware(_ProbeCase):
    def setUp(self):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            self.skipTest("fastapi not installed")
        super().setUp()

    def _app(self):
        from fastapi import BackgroundTasks, FastAPI

        app = FastAPI()
        app.add_middleware(evalkit.EvalKitMiddleware, client=self.client)
        case = self

        @app.get("/health")
        def health(tasks: BackgroundTasks):
            gate = case.probe_work()
            tasks.add_task(gate.set)
            return {"status": "ok"}

        @app.get("/healthcare")
        def healthcare():
            end, _ = case.client.start_span("inside.app_route")
            end("OK")
            return {"claims": 3}

        return app

    def test_a_probe_is_answered_untraced(self):
        sent = _call(self._app(), "GET", "/health")
        start = next(m for m in sent if m["type"] == "http.response.start")
        self.assertEqual(start["status"], 200)
        self.assertNotIn(b"x-trace-id", [k for k, _ in start.get("headers", [])])
        self.assertEqual(b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body"),
                         b'{"status":"ok"}')
        self.assert_only_background_work(self.exported())

    def test_the_apps_own_routes_and_the_next_request_are_traced(self):
        app = self._app()
        _call(app, "GET", "/health")
        self.exported()
        self.captured.clear()
        _call(app, "GET", "/healthcare")
        self.assert_app_route_traced(self.exported(), "/healthcare")


class TestFlask(_ProbeCase):
    def setUp(self):
        try:
            import flask  # noqa: F401
        except ImportError:
            self.skipTest("flask not installed")
        super().setUp()

    def _app(self):
        from flask import Flask

        app = Flask(__name__)
        from evalkit.frameworks.flask import instrument_flask

        instrument_flask(app, self.client)
        case = self

        @app.get("/health")
        def health():
            case.probe_work()
            return {"status": "ok"}

        @app.get("/healthcare")
        def healthcare():
            end, _ = case.client.start_span("inside.app_route")
            end("OK")
            return {"claims": 3}

        return app

    def test_a_probe_is_answered_untraced_and_the_next_request_is_traced(self):
        client = self._app().test_client()
        response = client.get("/health")
        self.assertEqual(response.get_json(), {"status": "ok"})
        self.assertNotIn("X-Trace-Id", response.headers)
        self.assert_only_background_work(self.exported())
        self.captured.clear()
        client.get("/healthcare")
        self.assert_app_route_traced(self.exported(), "/healthcare")


class TestDjango(_ProbeCase):
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

    def test_a_probe_is_answered_untraced(self):
        from django.http import JsonResponse
        from django.test import RequestFactory

        from evalkit.frameworks.django import EvalKitDjangoMiddleware

        def view(request):
            if request.path == "/health":
                self.probe_work()
            else:
                end, _ = self.client.start_span("inside.app_route")
                end("OK")
            return JsonResponse({"ok": True})

        middleware = EvalKitDjangoMiddleware(view)
        middleware._client = self.client
        response = middleware(RequestFactory().get("/health"))
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.has_header("X-Trace-Id"))
        self.assert_only_background_work(self.exported())
        self.captured.clear()
        middleware(RequestFactory().get("/healthcare"))
        self.assert_app_route_traced(self.exported(), "/healthcare")


class TestLitestar(_ProbeCase):
    def setUp(self):
        try:
            import litestar  # noqa: F401
        except ImportError:
            self.skipTest("litestar not installed")
        super().setUp()

    def test_a_probe_is_answered_untraced(self):
        from litestar import Litestar, get

        case = self

        @get("/health", sync_to_thread=False)
        def health() -> dict:
            case.probe_work()
            return {"status": "ok"}

        @get("/healthcare", sync_to_thread=False)
        def healthcare() -> dict:
            end, _ = case.client.start_span("inside.app_route")
            end("OK")
            return {"claims": 3}

        app = Litestar(route_handlers=[health, healthcare],
                       middleware=[evalkit.create_litestar_middleware(self.client)])
        sent = _call(app, "GET", "/health")
        start = next(m for m in sent if m["type"] == "http.response.start")
        self.assertEqual(start["status"], 200)
        self.assertNotIn(b"x-trace-id", [k for k, _ in start.get("headers", [])])
        self.assert_only_background_work(self.exported())
        self.captured.clear()
        _call(app, "GET", "/healthcare")
        self.assert_app_route_traced(self.exported(), "/healthcare")


class TestSuppressionEnds(unittest.TestCase):
    def test_a_failing_handler_still_ends_the_suppression(self):
        from evalkit.core.context import tracing_suppressed
        from evalkit.frameworks._probe import probe_untraced

        with self.assertRaises(ZeroDivisionError):
            with probe_untraced():
                self.assertTrue(tracing_suppressed())
                1 / 0
        self.assertFalse(tracing_suppressed())


if __name__ == "__main__":
    unittest.main()
