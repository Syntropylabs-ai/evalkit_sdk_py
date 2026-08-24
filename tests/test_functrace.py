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

    def test_install_is_idempotent(self):
        tmp = tempfile.mkdtemp()
        functrace.install_function_tracer(self.client, tmp)
        functrace.install_function_tracer(self.client, tmp)
        finders = [f for f in sys.meta_path if type(f).__name__ == "_EvalKitTracingFinder"]
        self.assertEqual(len(finders), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
