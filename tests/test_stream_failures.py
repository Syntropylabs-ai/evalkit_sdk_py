import asyncio
import types
import unittest

from tests.helpers import OStreamChunk, fresh, reset_context

TTFT = "gen_ai.server.time_to_first_token_ms"


def _install_openai_stream(client, chunks, error=None):
    from evalkit.instrumentation.auto import _patch_openai_class_sync

    def gen():
        for c in chunks:
            yield c
        if error is not None:
            raise error

    class FakeCompletions:
        def create(self, *a, **k):
            return gen()

    _patch_openai_class_sync(client, FakeCompletions)
    return FakeCompletions()


def _install_openai_async_stream(client, chunks, error=None):
    from evalkit.instrumentation.auto import _patch_openai_class_async

    async def agen():
        for c in chunks:
            yield c
        if error is not None:
            raise error

    class FakeAsyncCompletions:
        async def create(self, *a, **k):
            return agen()

    _patch_openai_class_async(client, FakeAsyncCompletions)
    return FakeAsyncCompletions()


def _install_anthropic_stream(client, events, error=None):
    from evalkit.instrumentation import auto as _auto

    def gen():
        for e in events:
            yield e
        if error is not None:
            raise error

    class FakeMessages:
        def create(self, *a, **k):
            return gen()

    _auto._patch_anthropic_class_sync(client, FakeMessages)
    return FakeMessages()


class TestStreamFailures(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def _llm(self):
        self.client.flush()
        spans = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(spans), 1, "a stream must emit exactly one llm_call span")
        return spans[0]

    def test_failed_stream_reports_error(self):
        fc = _install_openai_stream(self.client, [OStreamChunk("Hel")], ValueError("connection reset"))
        with self.assertRaises(ValueError):
            list(fc.create(model="gpt-4o-mini", messages=[], stream=True))
        span = self._llm()
        self.assertEqual(span.status, "ERROR",
                         "a stream that died mid-flight was recorded as a success, so failed streaming calls "
                         "never show up in the dashboard's error rate")
        self.assertIn("connection reset", span.status_message or "",
                      "without the provider's message there is nothing to debug the failure with")

    def test_failed_stream_keeps_the_partial_completion(self):
        fc = _install_openai_stream(
            self.client,
            [OStreamChunk("Hel"), OStreamChunk("lo", usage=types.SimpleNamespace(prompt_tokens=5, completion_tokens=2))],
            RuntimeError("upstream closed"),
        )
        with self.assertRaises(RuntimeError):
            list(fc.create(model="gpt-4o-mini", messages=[], stream=True))
        span = self._llm()
        self.assertEqual(span.completion, "Hello",
                         "the text received before the failure is the only evidence of what the model was doing; "
                         "dropping it makes a failed stream indistinguishable from one that never started")
        self.assertEqual(span.tokens_in, 5,
                         "the provider bills for tokens it already streamed, so a failed stream must still carry usage")

    def test_failed_stream_has_no_time_to_first_token(self):
        fc = _install_openai_stream(self.client, [OStreamChunk("Hel")], ValueError("boom"))
        with self.assertRaises(ValueError):
            list(fc.create(model="gpt-4o-mini", messages=[], stream=True))
        span = self._llm()
        self.assertNotIn(TTFT, span.attributes,
                         "a truncated stream must stay out of the TTFT percentiles, otherwise a provider outage "
                         "that fails fast reads as a latency improvement")

    def test_failed_async_stream_reports_error(self):
        async def run():
            fc = _install_openai_async_stream(self.client, [OStreamChunk("hi")], ValueError("async reset"))
            stream = await fc.create(model="gpt-4o-mini", messages=[], stream=True)
            async for _ in stream:
                pass

        with self.assertRaises(ValueError):
            asyncio.run(run())
        span = self._llm()
        self.assertEqual(span.status, "ERROR", "the async tracer must report failures like the sync one")
        self.assertIn("async reset", span.status_message or "")
        self.assertNotIn(TTFT, span.attributes)

    def test_failed_anthropic_stream_reports_error(self):
        events = [
            types.SimpleNamespace(type="message_start", message=types.SimpleNamespace(
                usage=types.SimpleNamespace(input_tokens=3))),
            types.SimpleNamespace(type="content_block_delta", delta=types.SimpleNamespace(
                type="text_delta", text="Hel")),
        ]
        fm = _install_anthropic_stream(self.client, events, ValueError("overloaded"))
        with self.assertRaises(ValueError):
            list(fm.create(model="claude-sonnet-4-5", messages=[], stream=True))
        span = self._llm()
        self.assertEqual(span.status, "ERROR",
                         "the anthropic branch of the shared tracer must report failures too, or Claude streams "
                         "are the one provider whose errors are invisible")
        self.assertIn("overloaded", span.status_message or "")
        self.assertNotIn(TTFT, span.attributes)

    def test_abandoned_stream_is_not_a_failure(self):
        fc = _install_openai_stream(self.client, [OStreamChunk("Hel"), OStreamChunk("lo")])
        stream = fc.create(model="gpt-4o-mini", messages=[], stream=True)
        it = iter(stream)
        next(it)
        del it
        del stream
        span = self._llm()
        self.assertEqual(span.status, "OK",
                         "a caller that stops reading (a cancelled generation, a `break`) has not hit a provider "
                         "error; GeneratorExit must not be recorded as one")
        self.assertIn(TTFT, span.attributes,
                      "the first token did arrive, so abandoning the rest of the stream must not discard its TTFT")

    def test_successful_stream_still_reports_ok_with_ttft(self):
        fc = _install_openai_stream(self.client, [OStreamChunk("Hel"), OStreamChunk("lo")])
        list(fc.create(model="gpt-4o-mini", messages=[], stream=True))
        span = self._llm()
        self.assertEqual(span.status, "OK")
        self.assertEqual(span.completion, "Hello")
        self.assertIsInstance(span.attributes.get(TTFT), int,
                              "TTFT must serialize as a number; trace-service2 reads it with JSONExtractFloat "
                              "and a string comes back as 0")

    def test_a_broken_enqueue_does_not_mask_the_provider_error(self):
        original = self.client.enqueue

        def exploding_enqueue(*a, **k):
            raise RuntimeError("mask hook blew up")

        fc = _install_openai_stream(self.client, [OStreamChunk("Hel")], ValueError("real provider error"))
        self.client.enqueue = exploding_enqueue
        try:
            with self.assertRaises(ValueError) as ctx:
                list(fc.create(model="gpt-4o-mini", messages=[], stream=True))
        finally:
            self.client.enqueue = original
        self.assertIn("real provider error", str(ctx.exception),
                      "the span is emitted from a finally block, so an exporter or mask failure there would "
                      "replace the caller's exception with ours and send them debugging the wrong thing")


if __name__ == "__main__":
    unittest.main()
