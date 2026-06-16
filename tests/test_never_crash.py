import unittest

from tests.helpers import fresh, reset_context


class TestNeverCrash(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_host_runs_when_tracing_setup_fails(self):
        @self.client.trace_function("add")
        def add(a, b):
            return a + b

        orig = self.client.start_span
        self.client.start_span = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            self.assertEqual(add(2, 3), 5)  # tracing blew up; host still returns
        finally:
            self.client.start_span = orig

    def test_none_result_not_captured(self):
        @self.client.trace_function("void")
        def void():
            return None

        void()
        self.client.flush()
        spans = [e for e in self.captured if e.operation == "void"]
        self.assertTrue(spans, "expected a function_call span")
        self.assertNotIn("function.result", spans[0].attributes or {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
