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
        tools = [e for e in self.captured if e.span_type == "tool_call"]
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].operation, "get_weather")
        self.assertEqual(tools[0].parent_span_id, llm[0].span_id)

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
