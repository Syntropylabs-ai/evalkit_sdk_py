import asyncio
import importlib
import os
import sys
import tempfile
import textwrap
import unittest

from evalkit.instrumentation import functrace
from tests.helpers import fresh, reset_context


def _make_pkg(root: str, pkg: str, body: str) -> None:
    d = os.path.join(root, pkg)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "__init__.py"), "w").close()
    with open(os.path.join(d, "mod.py"), "w") as f:
        f.write(textwrap.dedent(body))


class TestFunctionTracer(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        functrace._installed = False

    def tearDown(self):
        sys.meta_path[:] = [
            f for f in sys.meta_path if type(f).__name__ != "_EvalKitTracingFinder"
        ]
        functrace._installed = False

    def test_in_scope_module_is_traced(self):
        tmp = tempfile.mkdtemp()
        sys.path.insert(0, tmp)
        _make_pkg(tmp, "appx", """
            def add(a, b):
                return a + b

            def _hidden(x):
                return x
        """)
        functrace.install_function_tracer(self.client, tmp)
        mod = importlib.import_module("appx.mod")

        self.assertEqual(mod.add(2, 3), 5)
        self.assertTrue(hasattr(mod.add, "__wrapped__"))
        self.assertFalse(hasattr(mod._hidden, "__wrapped__"))

        self.client.flush()
        ops = [getattr(e, "operation", "") for e in self.captured]
        self.assertTrue(any("add" in o for o in ops), f"no function_call span for add: {ops}")

    def test_out_of_scope_module_not_traced(self):
        scope_dir = tempfile.mkdtemp()
        other_dir = tempfile.mkdtemp()
        sys.path.insert(0, other_dir)
        _make_pkg(other_dir, "appy", "def mul(a, b):\n    return a * b\n")

        functrace.install_function_tracer(self.client, scope_dir)
        mod = importlib.import_module("appy.mod")

        self.assertEqual(mod.mul(2, 3), 6)
        self.assertFalse(hasattr(mod.mul, "__wrapped__"), "out-of-scope module must not be wrapped")

    def test_decorated_functions_are_not_traced_a_second_time(self):
        tmp = tempfile.mkdtemp()
        sys.path.insert(0, tmp)
        _make_pkg(tmp, "appdeco", """
            import asyncio
            import functools

            import evalkit


            @evalkit.trace_function(capture_io=False)
            def lookup(card_number):
                return "ok"


            @evalkit.trace_tool(capture_io=False)
            def refund(order_id):
                return "refunded"


            @evalkit.trace_function(capture_io=False)
            async def fetch(card_number):
                return "fetched"


            def retry(fn):
                @functools.wraps(fn)
                def wrapper(*args, **kwargs):
                    return fn(*args, **kwargs)
                return wrapper


            @retry
            @evalkit.trace_function(capture_io=False)
            def charge(card_number):
                return "charged"


            def plain(x):
                return x
        """)
        functrace.install_function_tracer(self.client, tmp)
        mod = importlib.import_module("appdeco.mod")

        self.assertEqual(mod.lookup("4242 4242 4242 4242"), "ok")
        self.assertEqual(mod.refund("1001"), "refunded")
        self.assertEqual(asyncio.run(mod.fetch("4242 4242 4242 4242")), "fetched")
        self.assertEqual(mod.charge("4242 4242 4242 4242"), "charged")
        self.assertEqual(mod.plain(7), 7)
        for name in ("lookup", "refund", "fetch", "charge"):
            self.assertTrue(getattr(getattr(mod, name), "_evalkit_traced", False), name)

        self.client.flush()
        ops = [e.operation for e in self.captured if e.span_type in ("function_call", "tool_call")]
        for name in ("lookup", "refund", "fetch", "charge"):
            matching = [o for o in ops if o.rsplit(".", 1)[-1] == name]
            self.assertEqual(
                len(matching), 1,
                f"{name} ran once but recorded {matching}: automatic tracing wrapped a function its "
                "decorator already traces",
            )
        self.assertEqual(len([o for o in ops if o.endswith("plain")]), 1)
        flat = repr([e.attributes for e in self.captured])
        self.assertNotIn("4242", flat, "capture_io=False must hold: no outer span may record the arguments")
        self.assertNotIn("1001", flat)

    def test_install_is_idempotent(self):
        tmp = tempfile.mkdtemp()
        functrace.install_function_tracer(self.client, tmp)
        functrace.install_function_tracer(self.client, tmp)
        finders = [f for f in sys.meta_path if type(f).__name__ == "_EvalKitTracingFinder"]
        self.assertEqual(len(finders), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
