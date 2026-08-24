import types
import unittest

import evalkit
from tests.helpers import (
    AResp, GenAIResp,
    fresh, reset_context,
    install_fake_anthropic, install_fake_genai,
)


class TestAnthropicClaude(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_llm_span_with_tool(self):
        m = install_fake_anthropic(self.client, AResp(text="hi", tool=("get_weather", {"city": "london"})))
        m.create(model="claude-3-5-sonnet", system="be helpful",
                 messages=[{"role": "user", "content": "weather?"}])
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].provider, "anthropic")
        self.assertEqual(llm[0].completion, "hi")
        self.assertEqual(llm[0].tokens_in, 3)
        self.assertEqual(llm[0].tokens_out, 5)
        self.assertTrue(llm[0].events)
        self.assertEqual(llm[0].events[0].attributes["gen_ai.tool.name"], "get_weather")
        tools = [e for e in self.captured if e.span_type == "tool_call"]
        self.assertEqual(len(tools), 0)

    def test_simulate_user_tool_correctness(self):
        m = install_fake_anthropic(self.client, AResp(text="resolved", tool=("issue_refund", {"id": "1"})))

        def agent(ctx):
            m.create(model="claude-3-5-sonnet", messages=[{"role": "user", "content": ctx.message}])
            return "ok"

        report = evalkit.simulate_user(
            agent,
            [{"name": "refund", "starting_prompt": "charged twice", "expected_tools": ["issue_refund"]}],
        )
        self.assertEqual(report["results"][0]["status"], "OK")
        self.assertEqual(report["results"][0]["scores"].get("tool_correctness"), 1.0)

    def test_text_only_no_tool_span(self):
        m = install_fake_anthropic(self.client, AResp(text="Just text, no tools."))
        m.create(model="claude-3-5-sonnet", messages=[{"role": "user", "content": "summary?"}])
        self.client.flush()
        tools = [e for e in self.captured if e.span_type == "tool_call"]
        self.assertEqual(len(tools), 0)

    def test_thinking_captured_on_response(self):
        m = install_fake_anthropic(self.client, AResp(text="42", thinking="Let me work through this..."))
        m.create(model="claude-3-7-sonnet", messages=[{"role": "user", "content": "what is 6*7?"}])
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].completion, "42")
        self.assertEqual(llm[0].attributes.get("gen_ai.response.thinking"), "Let me work through this...")
        self.assertNotIn("gen_ai.response.has_redacted_thinking", llm[0].attributes)

    def test_redacted_thinking_flagged(self):
        m = install_fake_anthropic(self.client, AResp(text="42", redacted_thinking=True))
        m.create(model="claude-3-7-sonnet", messages=[{"role": "user", "content": "what is 6*7?"}])
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertTrue(llm[0].attributes.get("gen_ai.response.has_redacted_thinking"))
        self.assertNotIn("gen_ai.response.thinking", llm[0].attributes)

    def test_streaming_captures_time_to_first_token(self):
        from evalkit.instrumentation.auto import _patch_anthropic_class_sync

        events = [
            types.SimpleNamespace(type="message_start", message=types.SimpleNamespace(usage=types.SimpleNamespace(input_tokens=3))),
            types.SimpleNamespace(type="content_block_delta", delta=types.SimpleNamespace(type="text_delta", text="hi")),
            types.SimpleNamespace(type="message_delta", usage=types.SimpleNamespace(output_tokens=5)),
        ]

        class FakeMessages:
            def create(self, *a, **k):
                return iter(events)

        _patch_anthropic_class_sync(self.client, FakeMessages)
        list(FakeMessages().create(model="claude-3-5-sonnet", messages=[], stream=True))
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        ttft = llm.attributes.get("gen_ai.server.time_to_first_token_ms")
        self.assertIsNotNone(ttft)
        self.assertIsInstance(ttft, int)
        self.assertGreaterEqual(ttft, 0)

    def test_error_emits_error_span(self):
        from evalkit.instrumentation.auto import _patch_anthropic_class_sync

        class FailingMessages:
            def create(self, *a, **k):
                raise RuntimeError("anthropic down")

        _patch_anthropic_class_sync(self.client, FailingMessages)
        with self.assertRaises(RuntimeError):
            FailingMessages().create(model="claude-3-5-sonnet", messages=[])
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].status, "ERROR")


if __name__ == "__main__":
    unittest.main(verbosity=2)
