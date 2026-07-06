import sys
import types
import unittest

import evalkit
from tests.helpers import GenAIResp, fresh, reset_context, install_fake_genai


class TestGoogleGenAI(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_six_calls_single_trace(self):
        m = install_fake_genai(self.client)
        for i in range(6):
            m.generate_content(model="gemini-2.0-flash", contents=f"q{i}")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 6)
        self.assertEqual(len({e.trace_id for e in llm}), 1)
        self.assertEqual(llm[0].provider, "google-genai")
        self.assertEqual(llm[0].tokens_in, 3)
        self.assertEqual(llm[0].tokens_out, 5)

    def test_joins_active_trace(self):
        m = install_fake_genai(self.client)
        trace_id, end, ctx = self.client.start_trace("root")
        with evalkit.with_trace(ctx):
            m.generate_content(model="g", contents="hi")
        end("OK")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(llm[0].trace_id, trace_id)
        self.assertEqual(llm[0].parent_span_id, ctx.span_id)

    def test_tool_events_captured(self):
        m = install_fake_genai(self.client, response=GenAIResp(tool=("get_weather", {"city": "london"})))
        m.generate_content(model="g", contents="weather?")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertTrue(llm.events)
        self.assertEqual(llm.events[0].attributes["gen_ai.tool.name"], "get_weather")

    def test_client_side_tool_no_child_span(self):
        # Client-side function tool -> event only, no empty zero-duration child span.
        m = install_fake_genai(self.client, response=GenAIResp(tool=("get_weather", {"city": "london"})))
        m.generate_content(model="g", contents="weather?")
        self.client.flush()
        tools = [e for e in self.captured if e.span_type == "tool_call"]
        self.assertEqual(len(tools), 0)

    def test_streaming(self):
        chunks = [GenAIResp(text="Hel"), GenAIResp(text="lo!", pin=0, pout=7)]
        m = install_fake_genai(self.client, stream_chunks=chunks)
        out = list(m.generate_content_stream(model="g", contents="hi"))
        self.assertEqual(len(out), 2)
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(llm.completion, "Hello!")
        self.assertTrue(llm.attributes.get("gen_ai.streaming"))

    def test_streaming_captures_request_and_response_body(self):
        # Regression test: the streaming path used to never attach body attrs
        # (only the non-streaming generate_content path did).
        chunks = [GenAIResp(text="Hel"), GenAIResp(text="lo!", pin=0, pout=7)]
        m = install_fake_genai(self.client, stream_chunks=chunks)
        list(m.generate_content_stream(model="g", contents="hi"))
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertIn("gen_ai.request.body", llm.attributes)
        self.assertIn("gen_ai.response.body", llm.attributes)

    def test_thinking_captured_non_streaming(self):
        m = install_fake_genai(self.client, response=GenAIResp(text="42", thinking="let me think..."))
        m.generate_content(model="g", contents="what is 6*7?")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(llm.completion, "42")
        self.assertEqual(llm.attributes.get("gen_ai.response.thinking"), "let me think...")

    def test_reasoning_tokens_captured_non_streaming(self):
        m = install_fake_genai(self.client, response=GenAIResp(text="42", thoughts_tokens=15))
        m.generate_content(model="g", contents="what is 6*7?")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(llm.attributes.get("gen_ai.usage.reasoning_tokens"), 15)

    def test_thinking_and_reasoning_tokens_captured_streaming(self):
        chunks = [
            GenAIResp(thinking="step one...", pin=0, pout=0),
            GenAIResp(text="42", pin=5, pout=8, thoughts_tokens=12),
        ]
        m = install_fake_genai(self.client, stream_chunks=chunks)
        list(m.generate_content_stream(model="g", contents="what is 6*7?"))
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(llm.completion, "42")
        self.assertEqual(llm.attributes.get("gen_ai.response.thinking"), "step one...")
        self.assertEqual(llm.attributes.get("gen_ai.usage.reasoning_tokens"), 12)


class TestGoogleAILegacy(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self._injected = []

    def tearDown(self):
        for n in self._injected:
            sys.modules.pop(n, None)

    def test_generate_content_span(self):
        genai_mod = types.ModuleType("google.generativeai")

        class _LegacyResult:
            text = "Legacy Gemini response"

        class FakeGenerativeModel:
            model_name = "gemini-1.5-pro"

            def generate_content(self, *a, **k):
                return _LegacyResult()

        genai_mod.GenerativeModel = FakeGenerativeModel
        sys.modules["google"] = types.ModuleType("google")
        sys.modules["google.generativeai"] = genai_mod
        self._injected += ["google", "google.generativeai"]

        from evalkit.instrumentation.auto import _patch_google_genai_legacy_class
        _patch_google_genai_legacy_class(self.client, FakeGenerativeModel)

        FakeGenerativeModel().generate_content("What is AI?")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].provider, "google-ai")
        self.assertEqual(llm[0].model, "gemini-1.5-pro")
        self.assertEqual(llm[0].completion, "Legacy Gemini response")


if __name__ == "__main__":
    unittest.main(verbosity=2)
