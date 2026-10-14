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



class ProviderRefused(Exception):
    """What a provider client raises when the request fails before any chunk arrives."""


def _raising(exc, is_async=False):
    if is_async:
        async def create(*a, **k):
            raise exc
    else:
        def create(*a, **k):
            raise exc
    return create


class TestStreamOpenFailures(unittest.TestCase):
    """A streamed call that fails before its stream opens is recorded like the same call without stream=True."""

    KWARGS = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Where is order 1001?"}],
              "stream": True, "temperature": 0.2}

    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def _span(self):
        self.client.flush()
        spans = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(spans), 1, "a stream that never opened must still leave exactly one model-call span")
        return spans[0]

    def _check(self, span, exc, model="gpt-4o-mini"):
        self.assertEqual(span.status, "ERROR")
        self.assertEqual(span.model, model)
        self.assertIn("refused", span.status_message or "")
        self.assertIn("Where is order 1001?", span.prompt or "")
        self.assertIs(span.attributes.get("gen_ai.streaming"), True)
        self.assertEqual(span.attributes.get("gen_ai.request.model"), model)
        self.assertEqual(span.attributes.get("exception.type"), type(exc).__qualname__)
        self.assertNotIn(TTFT, span.attributes)

    def _call(self, create, is_async=False, **over):
        exc = ProviderRefused("connection refused")
        kwargs = {**self.KWARGS, **over}
        if is_async:
            async def run():
                await create(**kwargs)
            with self.assertRaises(ProviderRefused) as raised:
                asyncio.run(run())
        else:
            with self.assertRaises(ProviderRefused) as raised:
                create(**kwargs)
        return raised.exception

    def _patched_class(self, patcher, exc, is_async=False):
        from evalkit.instrumentation import auto as _auto

        class FakeCompletions:
            pass

        FakeCompletions.create = _raising(exc, is_async)
        getattr(_auto, patcher)(self.client, FakeCompletions)
        return FakeCompletions()

    def test_openai_sync(self):
        exc = ProviderRefused("connection refused")
        fc = self._patched_class("_patch_openai_class_sync", exc)
        with self.assertRaises(ProviderRefused) as raised:
            fc.create(**self.KWARGS)
        self.assertIs(raised.exception, exc, "the provider's own exception object must reach the caller")
        span = self._span()
        self._check(span, exc)
        self.assertEqual(span.attributes.get("gen_ai.request.temperature"), 0.2)

    def test_openai_async(self):
        exc = ProviderRefused("connection refused")
        fc = self._patched_class("_patch_openai_class_async", exc, is_async=True)

        async def run():
            await fc.create(**self.KWARGS)
        with self.assertRaises(ProviderRefused) as raised:
            asyncio.run(run())
        self.assertIs(raised.exception, exc)
        self._check(self._span(), exc)

    def test_anthropic_sync_and_async(self):
        for patcher, is_async in (("_patch_anthropic_class_sync", False), ("_patch_anthropic_class_async", True)):
            with self.subTest(patcher=patcher):
                self.captured.clear()
                exc = ProviderRefused("connection refused")
                fm = self._patched_class(patcher, exc, is_async)
                kwargs = {**self.KWARGS, "model": "claude-sonnet-4-5", "max_tokens": 64}
                if is_async:
                    async def run():
                        await fm.create(**kwargs)
                    with self.assertRaises(ProviderRefused):
                        asyncio.run(run())
                else:
                    with self.assertRaises(ProviderRefused):
                        fm.create(**kwargs)
                span = self._span()
                self._check(span, exc, model="claude-sonnet-4-5")
                self.assertEqual(span.attributes.get("gen_ai.request.max_tokens"), 64)

    def test_explicit_client_patchers(self):
        from evalkit.instrumentation.llm.anthropic import patch_anthropic
        from evalkit.instrumentation.llm.anthropic_async import patch_async_anthropic
        from evalkit.instrumentation.llm.openai import patch_openai
        from evalkit.instrumentation.llm.openai_async import patch_async_openai
        from evalkit.instrumentation.llm.vertex_ai import patch_anthropic_vertex

        cases = (
            ("openai", patch_openai, False, "chat"),
            ("openai_async", patch_async_openai, True, "chat"),
            ("anthropic", patch_anthropic, False, "messages"),
            ("anthropic_async", patch_async_anthropic, True, "messages"),
            ("anthropic_vertex", patch_anthropic_vertex, False, "messages"),
        )
        for name, patcher, is_async, shape in cases:
            with self.subTest(name):
                self.captured.clear()
                exc = ProviderRefused("connection refused")
                api = types.SimpleNamespace(create=_raising(exc, is_async), stream=None)
                fake = (types.SimpleNamespace(chat=types.SimpleNamespace(completions=api)) if shape == "chat"
                        else types.SimpleNamespace(messages=api))
                patcher(self.client, fake)
                raised = self._call(api.create, is_async)
                self.assertIsInstance(raised, ProviderRefused)
                span = self._span()
                self.assertEqual(span.status, "ERROR")
                self.assertEqual(span.model, "gpt-4o-mini")
                self.assertIs(span.attributes.get("gen_ai.streaming"), True)
                self.assertIn("gen_ai.request.body", span.attributes, "the request is recorded, as on the non-stream path")
                self.assertEqual(span.attributes.get("exception.type"), "ProviderRefused")

    def test_litellm_sync_and_async(self):
        from evalkit.instrumentation.llm import litellm as ek_litellm

        for is_async in (False, True):
            with self.subTest(is_async=is_async):
                self.captured.clear()
                exc = ProviderRefused("connection refused")
                fake = types.SimpleNamespace(
                    completion=_raising(exc), acompletion=_raising(exc, is_async=True),
                )
                ek_litellm._patch_sync(self.client, fake)
                ek_litellm._patch_async(self.client, fake)
                raised = self._call(fake.acompletion if is_async else fake.completion, is_async)
                self.assertIs(raised, exc)
                span = self._span()
                self.assertEqual(span.status, "ERROR")
                self.assertEqual(span.operation, "litellm.acompletion" if is_async else "litellm.completion")
                self.assertEqual(span.attributes.get("exception.type"), "ProviderRefused")

    def test_real_clients_with_a_refused_connection(self):
        """The provider SDKs themselves, pointed at a closed local port: no request leaves the machine."""
        import anthropic
        import openai

        oai = openai.OpenAI(base_url="http://127.0.0.1:1/v1", api_key="sk-test-not-a-real-key-000000", max_retries=0)
        ant = anthropic.Anthropic(base_url="http://127.0.0.1:1", api_key="sk-ant-test", max_retries=0)
        with self.assertRaises(openai.APIConnectionError):
            oai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], stream=True)
        with self.assertRaises(anthropic.APIConnectionError):
            ant.messages.create(model="claude-sonnet-4-5", max_tokens=8, messages=[{"role": "user", "content": "hi"}],
                                stream=True)
        self.client.flush()
        spans = sorted((e.operation, e.model, e.status) for e in self.captured if e.span_type == "llm_call")
        self.assertEqual(spans, [
            ("chat.completions.create", "gpt-4o-mini", "ERROR"),
            ("messages.create", "claude-sonnet-4-5", "ERROR"),
        ])


if __name__ == "__main__":
    unittest.main()
