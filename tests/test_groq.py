import unittest

from tests.helpers import OResp, fresh, reset_context


class TestGroq(unittest.TestCase):
    """Groq uses the same OpenAI-compatible patch with provider='groq'."""

    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_groq_provider_tag(self):
        from evalkit.instrumentation.auto import _patch_openai_class_sync

        class FakeGroqCompletions:
            def create(self, *a, **k):
                return OResp("Groq fast response", pin=3, pout=6)

        _patch_openai_class_sync(self.client, FakeGroqCompletions, provider="groq")
        FakeGroqCompletions().create(model="llama3-8b-8192", messages=[{"role": "user", "content": "hi"}])
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].provider, "groq")
        self.assertEqual(llm[0].completion, "Groq fast response")

    def test_groq_tool_call_span(self):
        from evalkit.instrumentation.auto import _patch_openai_class_sync

        class FakeGroqCompletions:
            def create(self, *a, **k):
                return OResp(tool=("web_search", '{"query":"LLM"}'))

        _patch_openai_class_sync(self.client, FakeGroqCompletions, provider="groq")
        FakeGroqCompletions().create(model="llama3-70b-8192", messages=[])
        self.client.flush()
        tools = [e for e in self.captured if e.span_type == "tool_call"]
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].operation, "web_search")


if __name__ == "__main__":
    unittest.main(verbosity=2)
