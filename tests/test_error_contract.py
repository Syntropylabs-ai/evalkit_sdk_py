"""The exception contract a failed span carries, shared with the TypeScript SDK and the receiver.

Status ERROR described as "<type>: <message>"; error.type the fully qualified class name; one
`exception` event with exception.type, .message (2 KiB), .stacktrace (16 KiB, start and end kept) and
.escaped; evalkit.exception.frames (innermost first), .source, .causes, and .fingerprint when chosen.
"""
import asyncio
import inspect
import json
import logging
import os
import sys
import threading
import time
import unittest

import evalkit
from evalkit.core import errors
from evalkit.core.errors import ExceptionRecord, is_in_app, type_name
from evalkit.core.limits import utf8_len
from tests.helpers import OStreamChunk, fresh, reset_context

HERE = os.path.abspath(__file__)


class PaymentError(Exception):
    pass


def _event(span):
    events = [e for e in (span.events or []) if e.name == "exception"]
    assert len(events) == 1, f"one exception event per exception, got {[e.name for e in span.events or []]}"
    return events[0].attributes


class _Case(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def spans(self, operation=None):
        self.client.flush()
        return [e for e in self.captured if operation is None or e.operation == operation]

    def span(self, operation):
        found = self.spans(operation)
        self.assertEqual(len(found), 1, [e.operation for e in self.captured])
        return found[0]


class ContractTest(_Case):
    def test_a_failed_function_records_the_whole_contract(self):
        @self.client.trace_function("charge")
        def charge(amount):
            local_secret = "hunter2-local-value"  # noqa: F841 - never read by the SDK
            raise PaymentError(f"card declined for {amount}")

        with self.assertRaises(PaymentError):
            charge(42)
        span = self.span("charge")
        qualified = f"{PaymentError.__module__}.PaymentError"
        self.assertEqual(type_name(PaymentError), qualified)
        self.assertEqual(span.status, "ERROR")
        self.assertEqual(span.status_message, f"{qualified}: card declined for 42")
        self.assertEqual(span.attributes["error.type"], qualified)

        event = _event(span)
        self.assertEqual(event["exception.type"], type_name(PaymentError))
        self.assertEqual(event["exception.message"], "card declined for 42")
        self.assertTrue(event["exception.stacktrace"].startswith("Traceback"))
        self.assertIn("PaymentError: card declined for 42", event["exception.stacktrace"])
        self.assertIs(event["exception.escaped"], True)

        frames = json.loads(span.attributes["evalkit.exception.frames"])
        self.assertEqual(frames[0]["function"], "charge", "innermost first")
        self.assertEqual(frames[0]["file"], HERE)
        self.assertTrue(frames[0]["in_app"])
        self.assertIsInstance(frames[0]["line"], int)
        self.assertEqual(set(frames[0]) - {"col", "module"}, {"file", "line", "function", "in_app"})
        self.assertEqual(span.attributes["evalkit.exception.source"], f"{HERE}:{frames[0]['line']}")
        self.assertNotIn("evalkit.exception.fingerprint", span.attributes)

        for key in ("exception.type", "exception.message", "exception.stacktrace", "error.source"):
            self.assertIn(key, span.attributes, "the earlier attributes stay for this release")
        everything = json.dumps({**span.attributes, **event}, default=str)
        self.assertNotIn("hunter2-local-value", everything, "no frame's locals are ever read")

    def test_the_type_is_fully_qualified_without_private_modules(self):
        rate_limit = type("RateLimitError", (Exception,), {"__module__": "openai._exceptions"})
        self.assertEqual(type_name(rate_limit("slow down")), "openai.RateLimitError")
        self.assertEqual(type_name(ValueError()), "ValueError")
        self.assertEqual(type_name(asyncio.CancelledError()), "asyncio.exceptions.CancelledError")
        self.assertEqual(ExceptionRecord(ValueError()).status_message, "ValueError")

    def test_the_stacktrace_is_the_argument_not_the_exception_being_handled(self):
        try:
            raise KeyError("being handled")
        except KeyError:
            try:
                raise PaymentError("the one recorded")
            except PaymentError as recorded:
                exc = recorded
            record = ExceptionRecord(exc)
            # Recorded while another exception is being handled, as an except block does.
            self.assertIn("PaymentError: the one recorded", record.stacktrace)
            from evalkit.core.client import _exc_attrs

            self.assertIn("PaymentError", _exc_attrs(exc)["exception.stacktrace"])

    def test_causes_are_recorded_innermost_frames_first(self):
        def low():
            raise ConnectionError("socket closed")

        @self.client.trace_function("fetch")
        def fetch():
            try:
                low()
            except ConnectionError as exc:
                raise RuntimeError("fetch failed") from exc

        with self.assertRaises(RuntimeError):
            fetch()
        span = self.span("fetch")
        causes = json.loads(span.attributes["evalkit.exception.causes"])
        self.assertEqual(causes[0]["type"], "ConnectionError")
        self.assertEqual(causes[0]["message"], "socket closed")
        self.assertEqual(causes[0]["frames"][0]["function"], "low")
        self.assertIn("The above exception was the direct cause", _event(span)["exception.stacktrace"])

    def test_at_most_five_causes(self):
        exc = None
        for i in range(10):
            try:
                if exc is None:
                    raise ValueError("0")
                raise ValueError(str(i)) from exc
            except ValueError as caught:
                exc = caught
        record = ExceptionRecord(exc)
        self.assertEqual(len(record.causes), 5)

    def test_message_and_stacktrace_are_bounded(self):
        def recurse(n):
            return recurse(n + 1)

        try:
            recurse(0)
        except RecursionError as exc:
            record = ExceptionRecord(exc)
        self.assertLessEqual(utf8_len(record.stacktrace), 16 * 1024)
        self.assertTrue(record.stacktrace.startswith("Traceback"))
        self.assertIn("RecursionError", record.stacktrace[-300:], "the end, with the exception, is kept")
        self.assertLessEqual(len(record.frames), 64)
        self.assertLessEqual(utf8_len(ExceptionRecord(ValueError("m" * 10_000)).message), 2048)

    def test_in_app_frames(self):
        self.assertTrue(is_in_app(HERE, "tests.test_error_contract"))
        self.assertFalse(is_in_app(json.__file__, "json"), "the standard library is not the app")
        self.assertFalse(is_in_app("<frozen importlib._bootstrap>", None))
        self.assertFalse(is_in_app(errors.__file__, "evalkit.core.errors"), "nor is EvalKit")
        self.assertFalse(is_in_app("/srv/venv/lib/python3.12/site-packages/openai/_base.py", "openai._base"))
        errors.configure(in_app_include=["openai"], in_app_exclude=["tests.vendored"])
        try:
            self.assertTrue(is_in_app("/srv/venv/lib/python3.12/site-packages/openai/_base.py", "openai._base"))
            self.assertFalse(is_in_app(HERE, "tests.vendored.client"))
        finally:
            errors.configure()
        errors.configure(in_app_include=["mycompany"], in_app_exclude=["mycompany.vendored"])
        try:
            self.assertTrue(is_in_app("/srv/app/mycompany/orders.py", "mycompany.orders"))
            self.assertFalse(is_in_app("/srv/app/mycompany/vendored/x.py", "mycompany.vendored.x"),
                             "an exclusion inside an included package wins")
        finally:
            errors.configure()

    def test_a_fingerprint_set_by_the_app_is_recorded(self):
        @self.client.trace_function("grouped", error_fingerprint="payments-declined")
        def grouped():
            raise PaymentError("x")

        @self.client.trace_function("scoped")
        def scoped():
            raise PaymentError("y")

        with self.assertRaises(PaymentError):
            grouped()
        token = evalkit.set_error_fingerprint(["checkout", "timeout"])
        try:
            with self.assertRaises(PaymentError):
                scoped()
        finally:
            evalkit.clear_error_fingerprint(token)
        self.assertEqual(self.span("grouped").attributes["evalkit.exception.fingerprint"], "payments-declined")
        self.assertEqual(self.span("scoped").attributes["evalkit.exception.fingerprint"], "checkout|timeout")

    def test_secrets_in_the_exception_are_masked_in_the_event_too(self):
        @self.client.trace_function("leaky")
        def leaky():
            raise RuntimeError("GET https://api.example.com/v1?api_key=abc123&page=2 with sk-ABCDEFGHIJKLMNOPQRSTUV")

        with self.assertRaises(RuntimeError):
            leaky()
        span = self.span("leaky")
        event = _event(span)
        for text in (event["exception.message"], event["exception.stacktrace"], span.status_message,
                     span.attributes["exception.message"]):
            self.assertNotIn("abc123", text)
            self.assertNotIn("ABCDEFGHIJKLMNOPQRSTUV", text)
            self.assertIn("page=2", text)

    def test_the_exception_event_reaches_the_ingest_envelope(self):
        @self.client.trace_function("sent")
        def sent():
            raise ValueError("v")

        with self.assertRaises(ValueError):
            sent()
        d = self.span("sent").to_dict()
        [event] = [e for e in d["events"] if e["name"] == "exception"]
        self.assertEqual(set(event), {"name", "timestamp", "attributes"})
        self.assertEqual(event["attributes"]["exception.type"], "ValueError")
        self.assertNotIn("exception", d, "the exception object itself is never sent")
        self.assertIsNone(self.span("sent").exception, "nor kept once the span is queued")


class BaseExceptionTest(_Case):
    def test_a_cancelled_task_ends_its_span_as_cancelled_and_still_raises(self):
        @self.client.trace_function("slow")
        async def slow():
            await asyncio.sleep(10)

        async def main():
            task = asyncio.ensure_future(slow())
            await asyncio.sleep(0.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(main())
        span = self.span("slow")
        self.assertEqual(span.status, "ERROR")
        self.assertIs(span.attributes["evalkit.cancelled"], True)
        self.assertEqual(_event(span)["exception.type"], "asyncio.exceptions.CancelledError")

    def test_keyboard_interrupt_and_system_exit_end_the_span_and_pass_through(self):
        @self.client.trace_function("interrupted")
        def interrupted():
            raise KeyboardInterrupt()

        @self.client.trace_function("exits_clean")
        def exits_clean():
            sys.exit(0)

        @self.client.trace_function("exits_failed")
        def exits_failed():
            sys.exit(3)

        with self.assertRaises(KeyboardInterrupt):
            interrupted()
        with self.assertRaises(SystemExit):
            exits_clean()
        with self.assertRaises(SystemExit):
            exits_failed()
        self.assertEqual(self.span("interrupted").status, "ERROR")
        self.assertEqual(self.span("exits_clean").status, "OK")
        self.assertEqual(self.span("exits_failed").status, "ERROR")


class GeneratorTest(_Case):
    def test_a_generator_span_covers_its_iteration(self):
        @self.client.trace_function("rows")
        def rows(n):
            for i in range(n):
                time.sleep(0.02)
                yield i
            return "done"

        self.assertTrue(inspect.isgeneratorfunction(rows), "frameworks that inspect for generators still see one")
        gen = rows(3)
        self.assertEqual(self.spans("rows"), [], "no span before the first item is asked for")
        self.assertEqual(list(gen), [0, 1, 2])
        span = self.span("rows")
        self.assertEqual(span.status, "OK")
        self.assertGreaterEqual(span.latency_ms, 50)

    def test_an_error_during_iteration_is_recorded(self):
        @self.client.trace_function("broken_rows")
        def broken_rows():
            yield 1
            raise PaymentError("row 2 failed")

        with self.assertRaises(PaymentError):
            list(broken_rows())
        span = self.span("broken_rows")
        self.assertEqual(span.status, "ERROR")
        self.assertEqual(_event(span)["exception.message"], "row 2 failed")

    def test_send_throw_close_and_the_return_value_pass_through(self):
        seen = []

        @self.client.trace_function("echo")
        def echo():
            try:
                while True:
                    got = yield len(seen)
                    seen.append(got)
            except ValueError:
                yield "handled"
            return "returned"

        def outer():
            result = yield from echo()
            seen.append(result)

        gen = echo()
        self.assertEqual(next(gen), 0)
        self.assertEqual(gen.send("a"), 1)
        self.assertEqual(gen.throw(ValueError("x")), "handled")
        gen.close()
        self.assertEqual(seen, ["a"])
        self.assertEqual(self.span("echo").status, "OK", "closing a generator early is not an error")

        driver = outer()
        next(driver)
        with self.assertRaises(StopIteration):
            driver.throw(ValueError("y"))
            next(driver)
        self.assertIn("returned", seen)

    def test_the_span_is_current_inside_the_generator_only(self):
        from evalkit.core.context import _current_span

        inside = []

        @self.client.trace_function("scoped_gen")
        def scoped_gen():
            inside.append(_current_span.get())
            yield 1
            inside.append(_current_span.get())
            yield 2

        outside = []
        for _ in scoped_gen():
            outside.append(_current_span.get())
        span = self.span("scoped_gen")
        self.assertTrue(all(ctx is not None and ctx.span_id == span.span_id for ctx in inside))
        self.assertTrue(all(ctx is None for ctx in outside), "the consumer's code is not inside the span")

    def test_an_async_generator_span_covers_its_iteration_and_errors(self):
        @self.client.trace_function("stream_tokens")
        async def stream_tokens():
            for token in ("a", "b"):
                await asyncio.sleep(0.01)
                yield token
            raise PaymentError("stream broke")

        self.assertTrue(inspect.isasyncgenfunction(stream_tokens))

        async def main():
            got = []
            with self.assertRaises(PaymentError):
                async for token in stream_tokens():
                    got.append(token)
            return got

        self.assertEqual(asyncio.run(main()), ["a", "b"])
        span = self.span("stream_tokens")
        self.assertEqual(span.status, "ERROR")
        self.assertEqual(_event(span)["exception.message"], "stream broke")

    def test_an_async_generator_closed_early_ends_ok(self):
        @self.client.trace_function("endless")
        async def endless():
            while True:
                yield 1

        async def main():
            agen = endless()
            await agen.__anext__()
            await agen.aclose()

        asyncio.run(main())
        self.assertEqual(self.span("endless").status, "OK")


class MidStreamErrorTest(_Case):
    def test_a_stream_that_fails_part_way_records_the_exception(self):
        from tests.test_stream_failures import _install_openai_stream

        completions = _install_openai_stream(self.client, [OStreamChunk("Hel")], ConnectionResetError("peer reset"))
        with self.assertRaises(ConnectionResetError):
            list(completions.create(model="gpt-4o-mini", messages=[], stream=True))
        [span] = [e for e in self.spans() if e.span_type == "llm_call"]
        self.assertEqual(span.status_message, "ConnectionResetError: peer reset")
        event = _event(span)
        self.assertEqual(event["exception.type"], "ConnectionResetError")
        self.assertIs(event["exception.escaped"], True)
        self.assertIn("evalkit.exception.frames", span.attributes)


class LogRecordTest(_Case):
    def _handler(self):
        from evalkit.instrumentation.runtime.logging import EvalKitLoggingHandler

        logger = logging.getLogger("app.errors.contract")
        logger.handlers = [EvalKitLoggingHandler(self.client)]
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        return logger

    def test_a_logged_exception_is_recorded_as_not_escaped(self):
        logger = self._handler()
        try:
            raise PaymentError("logged")
        except PaymentError:
            logger.exception("charge failed")
            logger.warning("retrying", exc_info=True)
        spans = [e for e in self.spans() if e.span_type == "log"]
        error_span = next(e for e in spans if e.attributes["log.message"] == "charge failed")
        warning_span = next(e for e in spans if e.attributes["log.message"] == "retrying")
        self.assertEqual(error_span.status, "ERROR")
        self.assertIs(_event(error_span)["exception.escaped"], False)
        self.assertEqual(warning_span.status, "OK", "a warning with an exception stays a warning")
        self.assertEqual(_event(warning_span)["exception.message"], "logged")


class UnhandledTest(_Case):
    def setUp(self):
        super().setUp()
        import asyncio.base_events as base_events

        from evalkit.core import unhandled

        self.saved = (sys.excepthook, threading.excepthook, base_events.BaseEventLoop.call_exception_handler,
                      unhandled._installed, unhandled._client)
        self.chained = []
        sys.excepthook = lambda *a: self.chained.append(("sys", a[1]))
        threading.excepthook = lambda args: self.chained.append(("thread", args.exc_value))
        unhandled._installed = False
        unhandled.install(self.client)

    def tearDown(self):
        import asyncio.base_events as base_events

        from evalkit.core import unhandled

        (sys.excepthook, threading.excepthook, base_events.BaseEventLoop.call_exception_handler,
         unhandled._installed, unhandled._client) = self.saved

    def _unhandled(self):
        return [e for e in self.spans() if (e.attributes or {}).get("evalkit.exception.unhandled")]

    def test_an_uncaught_exception_is_recorded_and_the_previous_hook_still_runs(self):
        try:
            raise PaymentError("top level")
        except PaymentError as exc:
            sys.excepthook(type(exc), exc, exc.__traceback__)
        [span] = self._unhandled()
        self.assertEqual(span.status, "ERROR")
        self.assertEqual(span.attributes["evalkit.exception.handler"], "sys.excepthook")
        self.assertEqual([kind for kind, _ in self.chained], ["sys"])

    def test_a_thread_that_dies_is_recorded(self):
        def work():
            raise PaymentError("in a thread")

        thread = threading.Thread(target=work)
        thread.start()
        thread.join()
        [span] = self._unhandled()
        self.assertEqual(_event(span)["exception.message"], "in a thread")
        self.assertEqual([kind for kind, _ in self.chained], ["thread"])

    def test_an_asyncio_exception_reaching_the_loop_is_recorded(self):
        handled = []

        async def main():
            loop = asyncio.get_running_loop()
            loop.set_exception_handler(lambda loop, context: handled.append(context))
            loop.call_exception_handler({"message": "boom", "exception": PaymentError("in a callback")})

        asyncio.run(main())
        [span] = self._unhandled()
        self.assertEqual(span.attributes["evalkit.exception.handler"], "asyncio")
        self.assertEqual(len(handled), 1, "the loop's own handler still runs")

    def test_it_is_off_unless_asked_for(self):
        import inspect as _inspect

        self.assertIs(_inspect.signature(evalkit.init).parameters["capture_unhandled"].default, False)


class FrameworkExceptionTest(_Case):
    def test_fastapi_middleware_passes_the_exception_through(self):
        from tests.test_framework_spans import _call

        async def app(scope, receive, send):
            raise PaymentError("in the endpoint")

        with self.assertRaises(PaymentError):
            _call(evalkit.EvalKitMiddleware(app, self.client), "POST", "/pay")
        [span] = [e for e in self.spans() if e.span_type == "http_call"]
        self.assertEqual(span.status, "ERROR")
        self.assertEqual(span.attributes["http.status_code"], 500)
        self.assertEqual(_event(span)["exception.message"], "in the endpoint")
        self.assertIs(_event(span)["exception.escaped"], True)

    def test_flask_passes_the_exception_through(self):
        try:
            from flask import Flask
        except ImportError:
            self.skipTest("flask not installed")
        app = Flask("contract")
        app.config["PROPAGATE_EXCEPTIONS"] = False

        @app.route("/pay")
        def pay():
            raise PaymentError("in the view")

        evalkit.instrument_flask(app)
        self.assertEqual(app.test_client().get("/pay").status_code, 500)
        [span] = [e for e in self.spans() if e.span_type == "http_call"]
        self.assertEqual(span.status, "ERROR")
        self.assertEqual(span.attributes["http.status_code"], 500)
        self.assertEqual(_event(span)["exception.message"], "in the view")

    def test_django_passes_the_exception_through(self):
        try:
            import django
            from django.conf import settings
            from django.test import Client, override_settings
        except ImportError:
            self.skipTest("django not installed")
        if not settings.configured:
            settings.configure(
                DEBUG=False, ALLOWED_HOSTS=["*"], ROOT_URLCONF="tests.test_framework_spans", SECRET_KEY="test-only",
                MIDDLEWARE=["evalkit.frameworks.django.EvalKitDjangoMiddleware"],
            )
            django.setup()
        with override_settings(ROOT_URLCONF=__name__):
            response = Client(raise_request_exception=False).get("/pay/")
        self.assertEqual(response.status_code, 500)
        [span] = [e for e in self.spans() if e.span_type == "http_call"]
        self.assertEqual(span.status, "ERROR")
        self.assertEqual(_event(span)["exception.message"], "in the django view")


def _django_urls():
    try:
        from django.urls import path
    except Exception:
        return []

    def pay(request):
        raise PaymentError("in the django view")

    return [path("pay/", pay)]


urlpatterns = _django_urls()


if __name__ == "__main__":
    unittest.main(verbosity=2)
