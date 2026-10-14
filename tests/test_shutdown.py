"""Spans queued when a process or worker stops still go out, and the flush never holds the exit up for long."""
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tests.helpers import fresh, reset_context

SDK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Receiver(BaseHTTPRequestHandler):
    spans = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        import gzip

        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        if self.headers.get("content-encoding") == "gzip":
            body = gzip.decompress(body)
        type(self).spans.extend(e.get("operation") for e in json.loads(body).get("events", []))
        self.send_response(200)
        self.send_header("content-length", "2")
        self.end_headers()
        self.wfile.write(b"{}")


def _silent_listener():
    """A socket that accepts connections and never answers: a receiver that is unreachable."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(64)
    return sock


class TestBoundedExitFlush(unittest.TestCase):
    def test_an_unreachable_receiver_holds_the_exit_flush_for_the_budget_only(self):
        from evalkit.core.exporter import _Exporter
        from evalkit.core.types import EvalKitOptions

        sock = _silent_listener()
        try:
            reset_context()
            client = fresh([])
            real = _Exporter(EvalKitOptions(subscription_key="tk_test", base_url=f"http://127.0.0.1:{sock.getsockname()[1]}",
                                            export_gzip=False))
            saved = client._exporter
            client._exporter = real
            try:
                end, _ = client.start_span("queued-at-exit")
                end("OK")
                started = time.monotonic()
                client.flush_before_exit(budget_s=1.0)
                took = time.monotonic() - started
            finally:
                client._exporter = saved
        finally:
            sock.close()
        self.assertLess(took, 2.0, f"the exit flush took {took:.1f}s; a dead receiver must not hold up a deploy")
        self.assertIsNone(real.exit_deadline, "an ordinary export afterwards keeps its full retry budget")


_PRELUDE = """import os, signal, sys, time
import evalkit
evalkit.init(subscription_key="tk_test_0123456789abcdef", base_url=os.environ["RECEIVER"],
             service_name="shutdown", function_tracing=False)
"""


def _script(body):
    return _PRELUDE + textwrap.dedent(body)


class TestSignals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _Receiver.spans = []
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Receiver)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.receiver = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def _run(self, body):
        env = {**os.environ, "PYTHONPATH": SDK, "RECEIVER": self.receiver, "LITELLM_LOCAL_MODEL_COST_MAP": "True"}
        return subprocess.run([sys.executable, "-c", _script(body)], env=env,
                              capture_output=True, text=True, timeout=60)

    def test_sigterm_with_the_default_action_sends_the_queue_and_still_ends_the_process(self):
        result = self._run("""
            end, _ = evalkit.start_span("before-sigterm"); end("OK")
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)
            print("still running")
        """)
        self.assertEqual(result.returncode, -signal.SIGTERM, "the signal keeps its default action")
        self.assertNotIn("still running", result.stdout)
        self.assertIn("before-sigterm", _Receiver.spans)

    def test_the_apps_own_handler_is_called_and_the_signal_is_not_swallowed(self):
        result = self._run("""
            seen = []
            def mine(signum, frame):
                seen.append(signum)
            signal.signal(signal.SIGTERM, mine)
            import importlib
            from evalkit.core.client import flush_on_termination_signals
            flush_on_termination_signals(evalkit._get_instance())
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.2)
            print("handled", seen == [signal.SIGTERM])
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("handled True", result.stdout)

    def test_ctrl_c_still_raises_keyboard_interrupt(self):
        result = self._run("""
            try:
                os.kill(os.getpid(), signal.SIGINT)
                time.sleep(2)
            except KeyboardInterrupt:
                print("KeyboardInterrupt")
        """)
        self.assertIn("KeyboardInterrupt", result.stdout)


class TestLifespanFlush(unittest.TestCase):
    def test_the_middleware_flushes_before_reporting_shutdown_complete(self):
        import evalkit

        reset_context()
        client = fresh([])
        order = []
        original = client.flush_before_exit
        client.flush_before_exit = lambda *a, **k: order.append("flush")

        async def app(scope, receive, send):
            message = await receive()
            assert message["type"] == "lifespan.shutdown"
            await send({"type": "lifespan.shutdown.complete"})

        async def receive():
            return {"type": "lifespan.shutdown"}

        async def send(message):
            order.append(message["type"])

        try:
            asyncio.run(evalkit.EvalKitMiddleware(app, client)({"type": "lifespan"}, receive, send))
        finally:
            client.flush_before_exit = original
        self.assertEqual(order, ["flush", "lifespan.shutdown.complete"],
                         "uvicorn ends the worker right after shutdown.complete, without atexit")



def _drive_lifespan(app, flushed_by=None, order=None):
    """Run an ASGI app through startup and shutdown; returns the replies, in order, with "flush"
    where the client's exit flush ran."""
    order = [] if order is None else order
    incoming = [{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}]

    async def receive():
        return incoming.pop(0)

    async def send(message):
        order.append(message["type"])

    reset_context()
    client = fresh([])
    original = client.flush_before_exit
    client.flush_before_exit = flushed_by or (lambda *a, **k: order.append("flush"))
    try:
        asyncio.run(app({"type": "lifespan", "asgi": {"version": "3.0"}}, receive, send))
    finally:
        client.flush_before_exit = original
    return order


class TestFrameworkClassLifespanFlush(unittest.TestCase):
    """Frameworks whose middleware never sees the lifespan scope flush from the application class."""

    def test_an_app_class_flushes_before_its_shutdown_reply(self):
        import inspect

        from evalkit.frameworks._lifespan import flush_on_lifespan_shutdown, registered_clients

        seen = []

        class App:
            async def __call__(self, scope, receive, send):
                seen.append(scope["type"])
                if scope["type"] == "lifespan":
                    await receive()
                    await send({"type": "lifespan.startup.complete"})
                    await receive()
                    seen.append("own shutdown hooks")
                    await send({"type": "lifespan.shutdown.complete"})

        self.assertTrue(flush_on_lifespan_shutdown(App, registered_clients))
        wrapped = App.__call__
        self.assertTrue(flush_on_lifespan_shutdown(App, registered_clients))
        self.assertIs(App.__call__, wrapped, "patching twice wraps once")
        self.assertTrue(inspect.iscoroutinefunction(App.__call__), "servers detect ASGI 3 by a coroutine __call__")
        order = _drive_lifespan(App())
        self.assertEqual(order, ["lifespan.startup.complete", "flush", "lifespan.shutdown.complete"])
        self.assertEqual(seen, ["lifespan", "own shutdown hooks"])

        sent = []

        async def send(message):
            sent.append(message)

        asyncio.run(App()({"type": "http"}, None, send))
        self.assertEqual(seen[-1], "http")
        self.assertEqual(sent, [], "other scopes pass through as they came")

    def test_a_failing_flush_still_reports_shutdown_complete(self):
        from evalkit.frameworks._lifespan import flush_on_lifespan_shutdown, registered_clients

        class App:
            async def __call__(self, scope, receive, send):
                await receive()
                await send({"type": "lifespan.startup.complete"})
                await receive()
                await send({"type": "lifespan.shutdown.complete"})

        flush_on_lifespan_shutdown(App, registered_clients)

        def broken(*a, **k):
            raise RuntimeError("receiver down")

        order = _drive_lifespan(App(), flushed_by=broken)
        self.assertEqual(order, ["lifespan.startup.complete", "lifespan.shutdown.complete"])

    def test_an_app_that_refuses_lifespan_is_given_one_with_the_flush(self):
        from evalkit.frameworks._lifespan import answer_lifespan_when_refused, registered_clients

        class Handler:
            async def __call__(self, scope, receive, send):
                if scope["type"] != "http":
                    raise ValueError("can only handle ASGI/HTTP connections, not %s." % scope["type"])

        answer_lifespan_when_refused(Handler, registered_clients)
        order = _drive_lifespan(Handler())
        self.assertEqual(order, ["lifespan.startup.complete", "flush", "lifespan.shutdown.complete"])

    def test_an_app_with_its_own_lifespan_keeps_it(self):
        from evalkit.frameworks._lifespan import answer_lifespan_when_refused, registered_clients

        class Handler:
            async def __call__(self, scope, receive, send):
                await receive()
                await send({"type": "lifespan.startup.complete"})
                await receive()
                await send({"type": "lifespan.shutdown.complete"})

        class Broken:
            async def __call__(self, scope, receive, send):
                await receive()
                raise ValueError("its own startup failed")

        answer_lifespan_when_refused(Handler, registered_clients)
        answer_lifespan_when_refused(Broken, registered_clients)
        self.assertEqual(_drive_lifespan(Handler()),
                         ["lifespan.startup.complete", "flush", "lifespan.shutdown.complete"])
        with self.assertRaises(ValueError, msg="an error of the app's own lifespan is not swallowed"):
            _drive_lifespan(Broken())

    def test_litestar_flushes_after_its_shutdown_hooks(self):
        try:
            from litestar import Litestar, get
        except ImportError:
            self.skipTest("litestar not installed")
        import evalkit

        order = []

        @get("/ping", sync_to_thread=False)
        def ping() -> dict:
            return {"ok": True}

        app = Litestar(route_handlers=[ping], middleware=[evalkit.create_litestar_middleware()],
                       on_shutdown=[lambda: order.append("on_shutdown")])
        _drive_lifespan(app, order=order)
        self.assertEqual(order, ["lifespan.startup.complete", "on_shutdown", "flush", "lifespan.shutdown.complete"],
                         "uvicorn ends the worker right after shutdown.complete, without atexit")

    def test_django_asgi_is_given_a_lifespan_with_the_flush(self):
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
        from django.core.asgi import get_asgi_application

        order = _drive_lifespan(get_asgi_application())
        self.assertEqual(order, ["lifespan.startup.complete", "flush", "lifespan.shutdown.complete"])



class TestModuleFlush(unittest.TestCase):
    """evalkit.flush() is called from shutdown hooks and test teardown whether or not init() ran."""

    def test_flush_before_init_returns_false_and_never_raises(self):
        import evalkit

        saved = evalkit._instance
        evalkit._instance = None
        try:
            self.assertIs(evalkit.flush(), False)
        finally:
            evalkit._instance = saved

    def test_flush_reports_whether_the_queue_went_out(self):
        import evalkit

        reset_context()
        client = fresh([])
        end, _ = client.start_span("queued")
        end("OK")
        self.assertIs(evalkit.flush(), True)
        original = client.flush

        def broken():
            raise RuntimeError("receiver down")

        client.flush = broken
        try:
            self.assertIs(evalkit.flush(), False)
        finally:
            client.flush = original


if __name__ == "__main__":
    unittest.main()
