import asyncio
import sys
import types
import unittest

from tests.helpers import (
    SDKAssistantMessage, SDKResultMessage,
    fresh, reset_context,
)


def _install_sdk(client, messages, *, injected, model="claude-sonnet-4-6"):
    sdk_mod = types.ModuleType("claude_agent_sdk")

    async def _gen():
        for m in messages:
            yield m

    async def fake_query(*, prompt="", options=None, **kwargs):
        async for m in _gen():
            yield m

    sdk_mod.query = fake_query

    class FakeSDKClient:
        _options = types.SimpleNamespace(model=model)
        _last_message = "user query"

        async def receive_response(self):
            async for m in _gen():
                yield m

    sdk_mod.ClaudeSDKClient = FakeSDKClient
    sys.modules["claude_agent_sdk"] = sdk_mod
    injected.append("claude_agent_sdk")

    from evalkit.instrumentation.llm.claude_agent_sdk import patch_claude_agent_sdk
    patch_claude_agent_sdk(client)
    return sdk_mod


class TestClaudeAgentSDK(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self._injected = []

    def tearDown(self):
        for n in self._injected:
            sys.modules.pop(n, None)

    def test_tokens_from_result_message(self):
        sdk = _install_sdk(self.client, [
            SDKAssistantMessage("The answer is 42"),
            SDKResultMessage(inp=10, out=20),
        ], injected=self._injected)

        asyncio.run(_consume(sdk.query(prompt="What is the answer?")))
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].tokens_in, 10)
        self.assertEqual(llm[0].tokens_out, 20)
        self.assertEqual(llm[0].provider, "anthropic")
        self.assertEqual(llm[0].operation, "claude_agent_sdk.query")

    def test_cost_and_stop_reason(self):
        sdk = _install_sdk(self.client, [
            SDKAssistantMessage("Done"),
            SDKResultMessage(cost=0.0009, stop="end_turn", turns=2),
        ], injected=self._injected)

        asyncio.run(_consume(sdk.query(prompt="Do something")))
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        attrs = llm.attributes or {}
        self.assertIn("llm.cost_usd", attrs)
        self.assertEqual(attrs.get("gen_ai.finish_reason"), "end_turn")
        self.assertEqual(attrs.get("llm.num_turns"), "2")

    def test_tool_events_no_child_span(self):
        sdk = _install_sdk(self.client, [
            SDKAssistantMessage(text="", tool=("lookup_order", {"order_id": "ORD-1001"})),
            SDKResultMessage(),
        ], injected=self._injected)

        asyncio.run(_consume(sdk.query(prompt="Where is my order?")))
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertTrue(llm.events)
        self.assertEqual(llm.events[0].attributes["gen_ai.tool.name"], "lookup_order")
        tools = [e for e in self.captured if e.span_type == "tool_call"]
        self.assertEqual(len(tools), 0)

    def test_model_from_assistant_message(self):
        sdk = _install_sdk(self.client, [
            SDKAssistantMessage("Hi", model="claude-opus-4-8"),
            SDKResultMessage(),
        ], injected=self._injected)

        asyncio.run(_consume(sdk.query(prompt="Which model?")))
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(llm.model, "claude-opus-4-8")

    def test_error_emits_error_span(self):
        sdk_mod = types.ModuleType("claude_agent_sdk")

        async def failing_query(*, prompt="", **kwargs):
            yield SDKAssistantMessage("starting")
            raise RuntimeError("SDK crashed")

        sdk_mod.query = failing_query
        sdk_mod.ClaudeSDKClient = None
        sys.modules["claude_agent_sdk"] = sdk_mod
        self._injected.append("claude_agent_sdk")

        from evalkit.instrumentation.llm.claude_agent_sdk import patch_claude_agent_sdk
        patch_claude_agent_sdk(self.client)

        async def run():
            with self.assertRaises(RuntimeError):
                async for _ in sdk_mod.query(prompt="crash please"):
                    pass

        asyncio.run(run())
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].status, "ERROR")


async def _consume(gen):
    async for _ in gen:
        pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
