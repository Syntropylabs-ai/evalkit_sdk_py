import asyncio
import types
import unittest

import evalkit
from tests.helpers import (
    OResp, OStreamChunk, ORespResult,
    fresh, reset_context,
    install_fake_openai_sync, install_fake_openai_async,
    install_fake_openai_responses_sync,
)


class TestOpenAI(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_sync_basic_llm_span(self):
        fc = install_fake_openai_sync(self.client, OResp("hello", pin=5, pout=8))
        fc.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].provider, "openai")
        self.assertEqual(llm[0].model, "gpt-4o-mini")
        self.assertEqual(llm[0].completion, "hello")
        self.assertEqual(llm[0].tokens_in, 5)
        self.assertEqual(llm[0].tokens_out, 8)
        self.assertEqual(llm[0].status, "OK")

    def test_sync_tool_call_event(self):
        # Client-side function tools surface as a gen_ai.tool.call event on the
        # llm_call span (no result -> no child tool_call span; wrap execution for that).
        fc = install_fake_openai_sync(self.client, OResp(tool=("search", '{"q":"weather"}')))
        fc.create(model="gpt-4o", messages=[{"role": "user", "content": "weather?"}])
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertTrue(llm[0].events)
        self.assertEqual(llm[0].events[0].attributes["gen_ai.tool.name"], "search")
        tools = [e for e in self.captured if e.span_type == "tool_call"]
        self.assertEqual(len(tools), 0, "client-side tool must not emit an empty child span")

    def test_sync_error_emits_error_span(self):
        from evalkit.instrumentation.auto import _patch_openai_class_sync

        class FakeCompletionsError:
            def create(self, *a, **k):
                raise ValueError("rate limit")

        _patch_openai_class_sync(self.client, FakeCompletionsError)
        with self.assertRaises(ValueError):
            FakeCompletionsError().create(model="gpt-4o-mini", messages=[])
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].status, "ERROR")
        self.assertIn("rate limit", llm[0].status_message or "")

    def test_async_llm_span(self):
        async def run():
            fc = install_fake_openai_async(self.client, OResp("async reply", pin=4, pout=6))
            await fc.create(model="gpt-4o", messages=[{"role": "user", "content": "async test"}])

        asyncio.run(run())
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].completion, "async reply")
        self.assertEqual(llm[0].tokens_in, 4)

    def test_sync_streaming_assembles_completion(self):
        final_usage = types.SimpleNamespace(prompt_tokens=5, completion_tokens=10)
        chunks = [
            OStreamChunk("Hello"),
            OStreamChunk(" world"),
            OStreamChunk("", usage=final_usage),
        ]
        fc = install_fake_openai_sync(self.client, chunks)
        stream = fc.create(model="gpt-4o-mini", messages=[], stream=True)
        list(stream)
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].completion, "Hello world")
        self.assertEqual(llm[0].tokens_in, 5)
        self.assertEqual(llm[0].tokens_out, 10)
        self.assertTrue(llm[0].attributes.get("gen_ai.streaming"))

    def test_joins_active_trace(self):
        fc = install_fake_openai_sync(self.client, OResp("nested"))
        trace_id, end, ctx = self.client.start_trace("request")
        with evalkit.with_trace(ctx):
            fc.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
        end("OK")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(llm[0].trace_id, trace_id)
        self.assertEqual(llm[0].parent_span_id, ctx.span_id)


class TestOpenAIResponsesAPI(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_basic_responses_span(self):
        fc = install_fake_openai_responses_sync(self.client, ORespResult("Agents reply", pin=6, pout=9))
        fc.create(model="gpt-4o-mini", input="Hello from agents SDK")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].operation, "responses.create")
        self.assertEqual(llm[0].provider, "openai")
        self.assertEqual(llm[0].completion, "Agents reply")
        self.assertEqual(llm[0].tokens_in, 6)
        self.assertEqual(llm[0].tokens_out, 9)

    def test_responses_tool_call_event(self):
        # A function_call output item is a client-side tool (no result) -> event only.
        resp = ORespResult(tool=("get_weather", '{"city":"Paris"}'))
        fc = install_fake_openai_responses_sync(self.client, resp)
        fc.create(model="gpt-4o", input="Weather in Paris?")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertTrue(llm.events)
        self.assertEqual(llm.events[0].attributes["gen_ai.tool.name"], "get_weather")
        tools = [e for e in self.captured if e.span_type == "tool_call"]
        self.assertEqual(len(tools), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
