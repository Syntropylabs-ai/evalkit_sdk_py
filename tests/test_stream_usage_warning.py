import asyncio
import types
import unittest

from evalkit.instrumentation import _streaming
from tests.helpers import OStreamChunk, fresh, reset_context

USAGE = types.SimpleNamespace(prompt_tokens=11, completion_tokens=2)


def _install_sync(client, chunks, error=None, provider="openai"):
    from evalkit.instrumentation.auto import _patch_openai_class_sync

    def gen():
        for c in chunks:
            yield c
        if error is not None:
            raise error

    class FakeCompletions:
        def create(self, *a, **k):
            return gen()

    _patch_openai_class_sync(client, FakeCompletions, provider)
    return FakeCompletions()


def _install_async(client, chunks, error=None, provider="openai"):
    from evalkit.instrumentation.auto import _patch_openai_class_async

    async def agen():
        for c in chunks:
            yield c
        if error is not None:
            raise error

    class FakeAsyncCompletions:
        async def create(self, *a, **k):
            return agen()

    _patch_openai_class_async(client, FakeAsyncCompletions, provider)
    return FakeAsyncCompletions()


class TestStreamUsageWarning(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        _streaming._stream_usage_warned = False

    def _llm(self):
        self.client.flush()
        spans = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(spans), 1, "a stream must emit exactly one llm_call span")
        return spans[0]

    def test_missing_stream_options_warns_once_with_the_exact_remedy(self):
        fc = _install_sync(self.client, [OStreamChunk("Par"), OStreamChunk("is")])
        with self.assertLogs("evalkit", level="WARNING") as logs:
            list(fc.create(model="gpt-4o-mini", messages=[], stream=True))
        span = self._llm()
        self.assertEqual(span.completion, "Paris",
                         "the fake chunk shape does not match what the tracer reads, so this test proves nothing")
        self.assertEqual(span.tokens_in, 0,
                         "this test only means something if a stream with no usage chunk really does report 0 "
                         "input tokens")
        self.assertEqual(span.tokens_out, 0,
                         "this test only means something if a stream with no usage chunk really does report 0 "
                         "output tokens")
        self.assertEqual(len(logs.records), 1,
                         "a streamed OpenAI call with no stream_options reports 0/0 tokens and therefore $0 cost for "
                         "every such call, and silently under-reporting spend is worse than one log line, so exactly "
                         "one warning must fire; got %d" % len(logs.records))
        message = logs.records[0].getMessage()
        self.assertIn("stream_options", message,
                      "a warning that does not name stream_options leaves the caller with a symptom and no fix")
        self.assertIn("include_usage", message,
                      "the warning must name the exact field to add, not just the option object")

    def test_warning_is_once_per_process(self):
        fc = _install_sync(self.client, [OStreamChunk("Par")])
        with self.assertLogs("evalkit", level="WARNING") as logs:
            list(fc.create(model="gpt-4o-mini", messages=[], stream=True))
            list(fc.create(model="gpt-4o-mini", messages=[], stream=True))
            list(fc.create(model="gpt-4o-mini", messages=[], stream=True))
        self.assertEqual(len(logs.records), 1,
                         "a chat loop streams thousands of times; a per-call warning would flood stderr and get the "
                         "whole SDK muted, so the warning must fire once per process; got %d" % len(logs.records))

    def test_include_usage_true_is_silent(self):
        fc = _install_sync(self.client, [OStreamChunk("Par"), OStreamChunk("is", usage=USAGE)])
        with self.assertNoLogs("evalkit", level="WARNING"):
            list(fc.create(model="gpt-4o-mini", messages=[], stream=True,
                           stream_options={"include_usage": True}))
        span = self._llm()
        self.assertEqual(span.tokens_in, 11,
                         "a stream that DID send a usage chunk must report its real prompt tokens, otherwise the "
                         "warning being suppressed here would have been suppressed for the wrong reason")

    def test_truthy_non_bool_include_usage_is_silent(self):
        fc = _install_sync(self.client, [OStreamChunk("Par")])
        with self.assertNoLogs("evalkit", level="WARNING"):
            list(fc.create(model="gpt-4o-mini", messages=[], stream=True,
                           stream_options={"include_usage": 1}))
        self.assertEqual(self._llm().tokens_in, 0,
                         "this case is only load-bearing while there is no usage chunk: with real tokens the warning "
                         "is suppressed by the token check and the include_usage gate is never consulted")

    def test_stream_options_as_object_is_silent(self):
        fc = _install_sync(self.client, [OStreamChunk("Par")])
        options = types.SimpleNamespace(include_usage=True)
        with self.assertNoLogs("evalkit", level="WARNING"):
            list(fc.create(model="gpt-4o-mini", messages=[], stream=True, stream_options=options))
        self.assertEqual(self._llm().tokens_in, 0,
                         "openai-python accepts stream_options as a typed object as well as a dict, and a caller who "
                         "already asked for usage must not be told to ask for it again; with no usage chunk present "
                         "the getattr path is the only thing keeping this silent")

    def test_failed_stream_does_not_warn(self):
        fc = _install_sync(self.client, [OStreamChunk("Par")], ValueError("connection reset"))
        with self.assertNoLogs("evalkit", level="WARNING"):
            with self.assertRaises(ValueError):
                list(fc.create(model="gpt-4o-mini", messages=[], stream=True))
        self.assertEqual(self._llm().status, "ERROR",
                         "a stream that died mid-flight must not be recorded as a success")

    def test_non_openai_provider_does_not_warn(self):
        fc = _install_sync(self.client, [OStreamChunk("Par")], provider="groq")
        with self.assertNoLogs("evalkit", level="WARNING"):
            list(fc.create(model="llama-3.3-70b", messages=[], stream=True))
        self.assertEqual(self._llm().provider, "groq",
                         "this test is only meaningful if the non-openai provider was actually recorded as such")

    def test_async_stream_warns(self):
        fc = _install_async(self.client, [OStreamChunk("Par")])

        async def run():
            stream = await fc.create(model="gpt-4o-mini", messages=[], stream=True)
            async for _ in stream:
                pass

        with self.assertLogs("evalkit", level="WARNING") as logs:
            asyncio.run(run())
        self.assertEqual(len(logs.records), 1,
                         "async streaming loses tokens exactly like sync streaming does, so leaving the async path "
                         "unwarned would silently under-report the majority of production traffic")

    def test_explicit_patcher_warns(self):
        from evalkit.instrumentation.llm.openai import patch_openai

        def gen():
            yield OStreamChunk("Par")

        class FakeCompletions:
            def create(self, *a, **k):
                return gen()

        fake = types.SimpleNamespace(chat=types.SimpleNamespace(completions=FakeCompletions()))
        patch_openai(self.client, fake)
        with self.assertLogs("evalkit", level="WARNING") as logs:
            list(fake.chat.completions.create(model="gpt-4o-mini", messages=[], stream=True))
        self.assertEqual(len(logs.records), 1,
                         "patch_openai_client() is the documented explicit entrypoint, so a caller who opts out of "
                         "autotrace would lose both the token counts and the warning that explains why; got %d"
                         % len(logs.records))

    def test_explicit_patcher_respects_include_usage(self):
        from evalkit.instrumentation.llm.openai import patch_openai

        def gen():
            yield OStreamChunk("Par", usage=USAGE)

        class FakeCompletions:
            def create(self, *a, **k):
                return gen()

        fake = types.SimpleNamespace(chat=types.SimpleNamespace(completions=FakeCompletions()))
        patch_openai(self.client, fake)
        with self.assertNoLogs("evalkit", level="WARNING"):
            list(fake.chat.completions.create(model="gpt-4o-mini", messages=[], stream=True,
                                             stream_options={"include_usage": True}))

    def test_explicit_async_patcher_warns(self):
        from evalkit.instrumentation.llm.openai_async import patch_async_openai

        async def agen():
            yield OStreamChunk("Par")

        class FakeAsyncCompletions:
            async def create(self, *a, **k):
                return agen()

        fake = types.SimpleNamespace(chat=types.SimpleNamespace(completions=FakeAsyncCompletions()))
        patch_async_openai(self.client, fake)

        async def run():
            stream = await fake.chat.completions.create(model="gpt-4o-mini", messages=[], stream=True)
            async for _ in stream:
                pass

        with self.assertLogs("evalkit", level="WARNING") as logs:
            asyncio.run(run())
        self.assertEqual(len(logs.records), 1,
                         "the explicit async patcher is the entrypoint for AsyncOpenAI, which is what production "
                         "services actually use; got %d" % len(logs.records))

    def test_a_raising_warning_never_costs_us_the_span(self):
        original = _streaming._warn_missing_stream_usage

        def boom():
            raise RuntimeError("logging misconfigured")

        _streaming._warn_missing_stream_usage = boom
        try:
            fc = _install_sync(self.client, [OStreamChunk("Par"), OStreamChunk("is")])
            list(fc.create(model="gpt-4o-mini", messages=[], stream=True))
        finally:
            _streaming._warn_missing_stream_usage = original

        span = self._llm()
        self.assertEqual(span.completion, "Paris",
                         "_finish swallows every exception from on_done, so an unguarded warn that raises would "
                         "silently drop the whole span - trading an under-priced call for a call that never "
                         "happened at all")
