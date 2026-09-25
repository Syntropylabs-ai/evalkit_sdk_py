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

    def test_consecutive_root_traces_get_their_own_ids(self):
        # One trace per turn: a loop of turns must not fold into a single trace.
        ids = []
        for _ in range(3):
            tid, end, _ctx = self.client.start_trace("turn")
            ids.append(tid)
            end("OK")
        self.assertEqual(len(set(ids)), 3)

    def test_bare_call_after_a_trace_does_not_join_it(self):
        tid, end, _ctx = self.client.start_trace("turn")
        end("OK")
        end_bare, bare_ctx = self.client.start_span("after")
        end_bare("OK")
        self.assertNotEqual(bare_ctx.trace_id, tid)

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

    def test_trace_tool_emits_tool_call_span(self):
        @self.client.trace_tool("search_web")
        def search_web(query):
            return {"results": [query]}

        out = search_web("weather")
        self.assertEqual(out, {"results": ["weather"]})
        self.client.flush()
        tools = [e for e in self.captured if e.span_type == "tool_call"]
        self.assertEqual(len(tools), 1)
        attrs = tools[0].attributes or {}
        self.assertEqual(attrs.get("tool.name"), "search_web")
        self.assertIn("tool.arguments", attrs)
        self.assertIn("tool.result", attrs)
        self.assertIn("weather", attrs["tool.result"])

    def test_traced_class_decorator(self):
        @evalkit.traced(prefix="svc")
        class Service:
            def place(self, n):
                return n + 1

            def cancel(self, n):
                return n - 1

        s = Service()
        self.assertEqual(s.place(1), 2)
        self.assertEqual(s.cancel(5), 4)
        self.client.flush()
        ops = {e.operation for e in self.captured if e.span_type == "function_call"}
        self.assertIn("svc.place", ops)
        self.assertIn("svc.cancel", ops)

    def test_trace_module_wraps_only_own_functions(self):
        import types as _types
        import json as _json

        mod = _types.ModuleType("ek_fake_svc")

        def add(a, b):
            return a + b

        def mul(a, b):
            return a * b

        add.__module__ = "ek_fake_svc"
        mul.__module__ = "ek_fake_svc"
        mod.add = add
        mod.mul = mul
        mod.dumps = _json.dumps

        count = evalkit.trace_module(mod)
        self.assertEqual(count, 2, "should wrap only the 2 functions defined in this module")

        self.assertEqual(mod.add(2, 3), 5)
        self.assertEqual(mod.mul(2, 3), 6)
        self.client.flush()

        fns = [e for e in self.captured if e.span_type == "function_call"]
        ops = {e.operation for e in fns}
        self.assertIn("ek_fake_svc.add", ops)
        self.assertIn("ek_fake_svc.mul", ops)
        self.assertNotIn("ek_fake_svc.dumps", ops)
        add_span = next(e for e in fns if e.operation == "ek_fake_svc.add")
        self.assertIn("function.args", add_span.attributes or {})
        self.assertIn("function.result", add_span.attributes or {})

    def test_trace_package_recurses_real_submodules(self):
        import os
        import sys
        import shutil
        import tempfile

        tmp = tempfile.mkdtemp()
        pkgdir = os.path.join(tmp, "ek_fake_pkg")
        os.makedirs(pkgdir)
        with open(os.path.join(pkgdir, "__init__.py"), "w") as f:
            f.write("")
        with open(os.path.join(pkgdir, "services.py"), "w") as f:
            f.write(
                "def place(order):\n"
                "    return {'placed': order}\n\n"
                "def cancel(order_id):\n"
                "    return order_id\n"
            )
        sys.path.insert(0, tmp)
        try:
            import ek_fake_pkg
            n = evalkit.trace_package(ek_fake_pkg)
            self.assertGreaterEqual(n, 2, "should wrap functions across the package's submodules")

            import ek_fake_pkg.services as svc
            self.assertEqual(svc.place("o-1"), {"placed": "o-1"})
            self.assertEqual(svc.cancel("c-1"), "c-1")
            self.client.flush()

            ops = {e.operation for e in self.captured if e.span_type == "function_call"}
            self.assertIn("services.place", ops)
            self.assertIn("services.cancel", ops)
        finally:
            sys.path.remove(tmp)
            sys.modules.pop("ek_fake_pkg", None)
            sys.modules.pop("ek_fake_pkg.services", None)
            shutil.rmtree(tmp, ignore_errors=True)

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
