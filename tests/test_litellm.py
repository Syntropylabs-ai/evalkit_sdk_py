import asyncio
import sys
import types
import unittest

from tests.helpers import LLMResult, fresh, reset_context


def _make_fake_litellm():
    mod = types.ModuleType("litellm")

    def _fake_completion(model="gpt-4o", messages=None, **kwargs):
        if kwargs.get("stream"):
            final = types.SimpleNamespace(
                choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=""))],
                usage=types.SimpleNamespace(prompt_tokens=7, completion_tokens=12),
            )
            return iter([
                types.SimpleNamespace(
                    choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content="Hi "))],
                    usage=None,
                ),
                types.SimpleNamespace(
                    choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content="there!"))],
                    usage=None,
                ),
                final,
            ])
        return LLMResult()

    async def _fake_acompletion(model="gpt-4o", messages=None, **kwargs):
        return LLMResult()

    mod.completion = _fake_completion
    mod.acompletion = _fake_acompletion
    return mod


class TestLiteLLM(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self._mod = _make_fake_litellm()
        sys.modules["litellm"] = self._mod
        from evalkit.instrumentation.llm.litellm import patch_litellm
        patch_litellm(self.client)

    def tearDown(self):
        sys.modules.pop("litellm", None)

    def test_prefixed_model_preserved(self):
        self._mod.completion(model="gemini/gemini-2.0-flash", messages=[{"role": "user", "content": "hello"}])
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].model, "gemini/gemini-2.0-flash")
        self.assertEqual(llm[0].provider, "litellm")

    def test_async_prefixed_model_preserved(self):
        asyncio.run(
            self._mod.acompletion(model="anthropic/claude-sonnet-4-6", messages=[{"role": "user", "content": "hi"}])
        )
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(llm[0].model, "anthropic/claude-sonnet-4-6")
        self.assertEqual(llm[0].provider, "litellm")

    def test_no_prefix_model_unchanged(self):
        self._mod.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "no prefix"}])
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(llm[0].model, "gpt-4o-mini")
        self.assertEqual(llm[0].provider, "litellm")

    def test_no_cost_attribute_sync(self):
        self._mod.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "cost"}])
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertNotIn("llm.cost_usd", llm.attributes or {})

    def test_no_cost_attribute_async(self):
        asyncio.run(self._mod.acompletion(model="gpt-4o", messages=[{"role": "user", "content": "async cost"}]))
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertNotIn("llm.cost_usd", llm.attributes or {})

    def test_operation_sync(self):
        self._mod.completion(model="gpt-4o-mini", messages=[])
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(llm.operation, "litellm.completion")

    def test_operation_async(self):
        asyncio.run(self._mod.acompletion(model="gpt-4o", messages=[]))
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(llm.operation, "litellm.acompletion")

    def test_tokens_captured_sync(self):
        self._mod.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "tokens"}])
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(llm.tokens_in, 7)
        self.assertEqual(llm.tokens_out, 12)

    def test_streaming_assembles_completion(self):
        stream = self._mod.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "stream"}], stream=True)
        list(stream)
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].completion, "Hi there!")
        self.assertEqual(llm[0].tokens_in, 7)
        self.assertEqual(llm[0].tokens_out, 12)
        self.assertTrue(llm[0].attributes.get("gen_ai.streaming"))
        self.assertEqual(llm[0].operation, "litellm.completion")

    def test_idempotent_no_double_span(self):
        from evalkit.instrumentation.llm.litellm import patch_litellm
        patch_litellm(self.client)
        self._mod.completion(model="gpt-4o-mini", messages=[])
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1, "double patch must not emit double spans")


if __name__ == "__main__":
    unittest.main(verbosity=2)
