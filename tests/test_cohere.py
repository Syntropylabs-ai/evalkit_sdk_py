import sys
import types
import unittest

from tests.helpers import fresh, reset_context


class TestCohere(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self._injected = []

    def tearDown(self):
        for n in self._injected:
            sys.modules.pop(n, None)

    def test_chat_span(self):
        cohere_mod = types.ModuleType("cohere")

        class _CTokens:
            input_tokens = 6
            output_tokens = 11

        class _CUsage:
            tokens = _CTokens()

        class _CResult:
            text = "Cohere response"
            usage = _CUsage()

        class FakeCohereClient:
            def chat(self, *a, **k):
                return _CResult()

        cohere_mod.Client = FakeCohereClient
        sys.modules["cohere"] = cohere_mod
        self._injected.append("cohere")

        from evalkit.instrumentation.auto import _patch_cohere_class
        _patch_cohere_class(self.client, FakeCohereClient)

        FakeCohereClient().chat(model="command-r-plus", message="Tell me something")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].provider, "cohere")
        self.assertEqual(llm[0].completion, "Cohere response")
        self.assertEqual(llm[0].tokens_in, 6)
        self.assertEqual(llm[0].tokens_out, 11)

    def test_error_emits_error_span(self):
        cohere_mod = types.ModuleType("cohere")

        class FailingCohere:
            def chat(self, *a, **k):
                raise ValueError("cohere down")

        cohere_mod.Client = FailingCohere
        sys.modules["cohere"] = cohere_mod
        self._injected.append("cohere")

        from evalkit.instrumentation.auto import _patch_cohere_class
        _patch_cohere_class(self.client, FailingCohere)

        with self.assertRaises(ValueError):
            FailingCohere().chat(model="command-r-plus", message="hi")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].status, "ERROR")


if __name__ == "__main__":
    unittest.main(verbosity=2)
