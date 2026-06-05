import asyncio
import unittest

import evalkit
from tests.helpers import fresh, reset_context


class TestManualSpans(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_start_span_creates_function_call(self):
        end, ctx = self.client.start_span("my-operation")
        end("OK")
        self.client.flush()
        spans = [e for e in self.captured if e.operation == "my-operation"]
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].span_type, "function_call")
        self.assertEqual(spans[0].status, "OK")

    def test_start_span_nests_under_active_trace(self):
        tid, end_root, ctx = self.client.start_trace("root")
        with evalkit.with_trace(ctx):
            end, child_ctx = self.client.start_span("child-work")
            end("OK")
        end_root("OK")
        self.client.flush()
        spans = [e for e in self.captured if e.operation == "child-work"]
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].trace_id, tid)
        self.assertEqual(spans[0].parent_span_id, ctx.span_id)

    def test_start_trace_returns_32char_id(self):
        tid, end, ctx = self.client.start_trace("my-trace")
        self.assertIsInstance(tid, str)
        self.assertEqual(len(tid), 32)
        self.assertEqual(ctx.trace_id, tid)
        end("OK")
        self.client.flush()

    def test_trace_function_sync_decorator(self):
        @self.client.trace_function("decorated-fn")
        def do_work(x):
            return x * 2

        result = do_work(21)
        self.assertEqual(result, 42)
        self.client.flush()
        spans = [e for e in self.captured if e.operation == "decorated-fn"]
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status, "OK")
        self.assertEqual(spans[0].span_type, "function_call")

    def test_trace_function_captures_io(self):
        @self.client.trace_function("io-fn", capture_io=True)
        def compute(a, b):
            return a + b

        compute(3, 4)
        self.client.flush()
        spans = [e for e in self.captured if e.operation == "io-fn"]
        attrs = spans[0].attributes or {}
        self.assertIn("function.args", attrs)
        self.assertIn("function.result", attrs)

    def test_trace_function_async_decorator(self):
        @self.client.trace_function("async-fn")
        async def async_work(n):
            return n ** 2

        result = asyncio.run(async_work(5))
        self.assertEqual(result, 25)
        self.client.flush()
        spans = [e for e in self.captured if e.operation == "async-fn"]
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status, "OK")

    def test_trace_function_error_path(self):
        @self.client.trace_function("failing-fn")
        def boom():
            raise ValueError("test error")

        with self.assertRaises(ValueError):
            boom()
        self.client.flush()
        spans = [e for e in self.captured if e.operation == "failing-fn"]
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status, "ERROR")

    def test_start_http_trace_attributes(self):
        tid, end, ctx = self.client.start_http_trace(
            "GET /api/data",
            method="GET",
            url="http://example.com/api/data",
            attributes={"custom.key": "custom.value"},
        )
        end("OK", status_code=200)
        self.client.flush()
        http = [e for e in self.captured if e.span_type == "http_call"]
        self.assertEqual(len(http), 1)
        self.assertEqual(http[0].trace_id, tid)
        attrs = http[0].attributes or {}
        self.assertEqual(attrs.get("http.method"), "GET")
        self.assertEqual(attrs.get("http.status_code"), 200)
        self.assertEqual(attrs.get("custom.key"), "custom.value")


if __name__ == "__main__":
    unittest.main(verbosity=2)
