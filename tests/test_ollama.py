import asyncio
import unittest

import ollama

from tests.helpers import fresh, reset_context


def _resp(content="42", thinking=None, tool=None, prompt_eval=5, eval_count=8, done_reason="stop"):
    message = ollama.Message(role="assistant", content=content, thinking=thinking)
    if tool:
        message.tool_calls = [ollama.Message.ToolCall(function=ollama.Message.ToolCall.Function(name=tool[0], arguments=tool[1]))]
    return ollama.ChatResponse(
        model="llama3.2", message=message, done=True, done_reason=done_reason,
        prompt_eval_count=prompt_eval, eval_count=eval_count,
    )


def _chunk(content="", thinking=None, done=False, prompt_eval=None, eval_count=None):
    message = ollama.Message(role="assistant", content=content, thinking=thinking)
    return ollama.ChatResponse(
        model="llama3.2", message=message, done=done,
        prompt_eval_count=prompt_eval, eval_count=eval_count,
    )


class TestOllama(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_sync_chat_emits_llm_call_span(self):
        from evalkit.instrumentation.llm.ollama import _patch_sync

        class FakeClient:
            def chat(self, model="", messages=None, *, stream=False, **kwargs):
                return _resp("42")

        _patch_sync(self.client, FakeClient)
        FakeClient().chat(model="llama3.2", messages=[{"role": "user", "content": "what is 6*7?"}])
        self.client.flush()

        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].provider, "ollama")
        self.assertEqual(llm[0].model, "llama3.2")
        self.assertEqual(llm[0].completion, "42")
        self.assertEqual(llm[0].tokens_in, 5)
        self.assertEqual(llm[0].tokens_out, 8)
        self.assertIn("gen_ai.request.body", llm[0].attributes)
        self.assertIn("gen_ai.response.body", llm[0].attributes)

    def test_thinking_captured(self):
        from evalkit.instrumentation.llm.ollama import _patch_sync

        class FakeClient:
            def chat(self, model="", messages=None, *, stream=False, **kwargs):
                return _resp("42", thinking="let me think...")

        _patch_sync(self.client, FakeClient)
        FakeClient().chat(model="llama3.2", messages=[{"role": "user", "content": "q"}])
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(llm.attributes.get("gen_ai.response.thinking"), "let me think...")

    def test_tool_call_event(self):
        from evalkit.instrumentation.llm.ollama import _patch_sync

        class FakeClient:
            def chat(self, model="", messages=None, *, stream=False, **kwargs):
                return _resp("", tool=("get_weather", {"city": "sf"}))

        _patch_sync(self.client, FakeClient)
        FakeClient().chat(model="llama3.2", messages=[{"role": "user", "content": "weather?"}])
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertTrue(llm.events)
        self.assertEqual(llm.events[0].attributes["gen_ai.tool.name"], "get_weather")

    def test_error_emits_error_span(self):
        from evalkit.instrumentation.llm.ollama import _patch_sync

        class FakeClient:
            def chat(self, model="", messages=None, *, stream=False, **kwargs):
                raise RuntimeError("model not found")

        _patch_sync(self.client, FakeClient)
        with self.assertRaises(RuntimeError):
            FakeClient().chat(model="llama3.2", messages=[])
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].status, "ERROR")

    def test_streaming_assembles_completion_and_ttft(self):
        from evalkit.instrumentation.llm.ollama import _patch_sync

        chunks = [
            _chunk("Hel"),
            _chunk("lo!", done=True, prompt_eval=5, eval_count=8),
        ]

        class FakeClient:
            def chat(self, model="", messages=None, *, stream=False, **kwargs):
                return iter(chunks)

        _patch_sync(self.client, FakeClient)
        list(FakeClient().chat(model="llama3.2", messages=[], stream=True))
        self.client.flush()

        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(llm.completion, "Hello!")
        self.assertEqual(llm.tokens_in, 5)
        self.assertEqual(llm.tokens_out, 8)
        self.assertTrue(llm.attributes.get("gen_ai.streaming"))
        ttft = llm.attributes.get("gen_ai.server.time_to_first_token_ms")
        self.assertIsInstance(ttft, int)
        self.assertGreaterEqual(ttft, 0)

    def test_async_chat_emits_llm_call_span(self):
        from evalkit.instrumentation.llm.ollama import _patch_async

        class FakeAsyncClient:
            async def chat(self, model="", messages=None, *, stream=False, **kwargs):
                return _resp("async 42")

        _patch_async(self.client, FakeAsyncClient)

        async def run():
            await FakeAsyncClient().chat(model="llama3.2", messages=[{"role": "user", "content": "q"}])

        asyncio.run(run())
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(llm.completion, "async 42")


if __name__ == "__main__":
    unittest.main(verbosity=2)
