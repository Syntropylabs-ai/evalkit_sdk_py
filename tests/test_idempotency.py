import asyncio
import sys
import types
import unittest

import evalkit
from tests.helpers import (
    OResp, AResp, SDKResultMessage,
    fresh, reset_context,
)


class TestIdempotency(unittest.TestCase):

    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_openai_not_double_patched(self):
        from evalkit.instrumentation.auto import _patch_openai_class_sync

        class FreshCompletions:
            def create(self, *a, **k):
                return OResp("once")

        _patch_openai_class_sync(self.client, FreshCompletions)
        _patch_openai_class_sync(self.client, FreshCompletions)
        FreshCompletions().create(model="gpt-4o-mini", messages=[])
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1, "second patch must not double-wrap")

    def test_anthropic_not_double_patched(self):
        from evalkit.instrumentation import auto as _auto

        class FreshMessages:
            def create(self, *a, **k):
                return AResp("once")

        _auto._patch_anthropic_class_sync(self.client, FreshMessages)
        _auto._patch_anthropic_class_sync(self.client, FreshMessages)
        FreshMessages().create(model="claude-3-5-sonnet", messages=[{"role": "user", "content": "hi"}])
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1, "second anthropic patch must not double-wrap")

    def test_claude_agent_sdk_not_double_patched(self):
        sdk_mod = types.ModuleType("claude_agent_sdk")

        async def fake_query(*, prompt="", **kwargs):
            yield SDKResultMessage()

        sdk_mod.query = fake_query
        sdk_mod.ClaudeSDKClient = None
        sys.modules["claude_agent_sdk"] = sdk_mod

        try:
            from evalkit.instrumentation.llm.claude_agent_sdk import patch_claude_agent_sdk
            patch_claude_agent_sdk(self.client)
            patch_claude_agent_sdk(self.client)

            async def run():
                async for _ in sdk_mod.query(prompt="hi"):
                    pass

            asyncio.run(run())
            self.client.flush()
            llm = [e for e in self.captured if e.span_type == "llm_call"]
            self.assertEqual(len(llm), 1, "double claude_agent_sdk patch must not double-emit")
        finally:
            sys.modules.pop("claude_agent_sdk", None)

    def test_litellm_not_double_patched(self):
        mod = types.ModuleType("litellm")
        from tests.helpers import LLMResult

        def fake_completion(model="gpt-4o", messages=None, **k):
            return LLMResult()

        mod.completion = fake_completion
        sys.modules["litellm"] = mod

        try:
            from evalkit.instrumentation.llm.litellm import patch_litellm
            patch_litellm(self.client)
            patch_litellm(self.client)

            mod.completion(model="gpt-4o-mini", messages=[])
            self.client.flush()
            llm = [e for e in self.captured if e.span_type == "llm_call"]
            self.assertEqual(len(llm), 1)
        finally:
            sys.modules.pop("litellm", None)

    def test_auto_instrumentation_idempotent(self):
        from evalkit.instrumentation.auto import setup_auto_instrumentation
        setup_auto_instrumentation(self.client)
        setup_auto_instrumentation(self.client)

    def test_no_crash_noop_patches(self):
        evalkit.patch_langchain()
        evalkit.patch_litellm()


if __name__ == "__main__":
    unittest.main(verbosity=2)
