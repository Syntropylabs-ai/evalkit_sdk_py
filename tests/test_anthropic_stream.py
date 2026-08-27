import asyncio
import unittest

from evalkit.core.context import current_span_ctx
from tests.helpers import (
    ACacheUsage, AResp,
    FakeAsyncMessageStream, FakeAsyncStreamManager, FakeMessageStream, FakeStreamManager,
    fresh, reset_context,
    install_fake_anthropic_class_stream, install_fake_anthropic_streaming,
)

MODEL = "claude-sonnet-4-5"
MESSAGES = [{"role": "user", "content": "why is the sky blue?"}]


class TestAnthropicMessagesStream(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def _llm(self):
        self.client.flush()
        return [e for e in self.captured if e.span_type == "llm_call"]

    def _stream(self, snapshot=None, enter_error=None, async_=False):
        snapshot = AResp() if snapshot is None else snapshot
        stream_cls = FakeAsyncMessageStream if async_ else FakeMessageStream
        mgr_cls = FakeAsyncStreamManager if async_ else FakeStreamManager
        stream = stream_cls(snapshot)
        return stream, mgr_cls(stream, enter_error=enter_error)

    def test_explicit_patch_traces_messages_stream(self):
        stream, mgr = self._stream(AResp(text="because of Rayleigh scattering"))
        messages = install_fake_anthropic_streaming(self.client, mgr)

        with messages.stream(model=MODEL, system="be brief", messages=MESSAGES) as s:
            self.assertIs(s, stream, "the patched stream must hand the caller the real MessageStream, not the wrapper")

        llm = self._llm()
        self.assertEqual(
            len(llm), 1,
            "patch_anthropic only ever patched messages.create, so an app using Anthropic's recommended "
            "with client.messages.stream(...) form produced no span at all and every one of those calls was "
            "invisible in the dashboard and free in the cost rollups",
        )
        self.assertEqual(llm[0].operation, "messages.stream")
        self.assertEqual(llm[0].provider, "anthropic")
        self.assertEqual(llm[0].model, MODEL)
        self.assertEqual(llm[0].completion, "because of Rayleigh scattering")
        self.assertEqual(llm[0].status, "OK")
        self.assertTrue(llm[0].attributes["gen_ai.streaming"])

    def test_stream_span_folds_cache_tokens_into_tokens_in(self):
        snapshot = AResp(text="cached")
        snapshot.usage = ACacheUsage(100, 20, cache_read=800, cache_creation=200)
        _, mgr = self._stream(snapshot)
        messages = install_fake_anthropic_streaming(self.client, mgr)

        with messages.stream(model=MODEL, messages=MESSAGES):
            pass

        span = self._llm()[0]
        self.assertEqual(
            span.tokens_in, 1100,
            "Anthropic reports input_tokens EXCLUSIVE of the cache buckets while the dashboard computes full-price "
            "input as max(0, tokensIn - cacheRead - cacheCreation); the class-level stream patcher read only "
            "input_tokens, so a heavily cached streaming call was billed on 100 tokens instead of 1100",
        )
        self.assertEqual(
            span.cache_read_tokens, 800,
            "folding the cache into tokens_in must not consume the bucket - the dashboard reprices these tokens at "
            "the cache-read rate, which is roughly a tenth of the input rate",
        )
        self.assertEqual(
            span.cache_creation_tokens, 200,
            "cache writes are billed ABOVE the input rate, so dropping this bucket undercharges a cache-priming call",
        )
        self.assertEqual(span.tokens_out, 20)

    def test_async_explicit_patch_traces_messages_stream(self):
        stream, mgr = self._stream(AResp(text="async answer"), async_=True)
        messages = install_fake_anthropic_streaming(self.client, mgr, flavor="async")

        async def run():
            async with messages.stream(model=MODEL, messages=MESSAGES) as s:
                self.assertIs(s, stream)

        asyncio.run(run())

        llm = self._llm()
        self.assertEqual(
            len(llm), 1,
            "patch_async_anthropic has the same hole as the sync patcher, so async apps streaming the recommended "
            "way reported nothing",
        )
        self.assertEqual(llm[0].operation, "messages.stream")
        self.assertEqual(llm[0].completion, "async answer")

    def test_vertex_explicit_patch_names_its_own_provider(self):
        _, mgr = self._stream()
        messages = install_fake_anthropic_streaming(self.client, mgr, flavor="vertex")

        with messages.stream(model=MODEL, messages=MESSAGES):
            pass

        self.assertEqual(
            self._llm()[0].provider, "anthropic-vertex",
            "Claude on Vertex is priced and quota'd separately from Claude on Anthropic's own API, so a Vertex "
            "stream reported as plain anthropic lands in the wrong provider rollup",
        )

    def test_stream_span_reports_the_error_that_killed_it(self):
        _, mgr = self._stream()
        messages = install_fake_anthropic_streaming(self.client, mgr)

        with self.assertRaises(RuntimeError):
            with messages.stream(model=MODEL, messages=MESSAGES):
                raise RuntimeError("overloaded_error: upstream capacity")

        span = self._llm()[0]
        self.assertEqual(span.status, "ERROR")
        self.assertIn(
            "overloaded_error", span.status_message or "",
            "the class-level stream patcher set status ERROR but never set status_message, so the dashboard showed a "
            "red span with no reason and debugging a failed stream meant guessing",
        )

    def test_stream_span_exists_when_the_connection_never_opens(self):
        _, mgr = self._stream(enter_error=ValueError("401 invalid x-api-key"))
        messages = install_fake_anthropic_streaming(self.client, mgr)

        with self.assertRaises(ValueError):
            with messages.stream(model=MODEL, messages=MESSAGES):
                pass

        llm = self._llm()
        self.assertEqual(
            len(llm), 1,
            "__exit__ is never called when __enter__ raises, so a stream that dies on connect emitted no span at all "
            "- the single most important failure to see is the one that leaves no trace",
        )
        self.assertEqual(llm[0].status, "ERROR")
        self.assertIn("401", llm[0].status_message or "")

    def test_instrumentation_never_forces_the_stream_to_drain(self):
        snapshot = AResp(text="partial")
        snapshot.usage = ACacheUsage(11, 7)
        stream, mgr = self._stream(snapshot)
        messages = install_fake_anthropic_streaming(self.client, mgr)

        with messages.stream(model=MODEL, messages=MESSAGES):
            pass

        self.assertFalse(
            stream.forced_drain,
            "get_final_message() calls until_done(), which reads the rest of the HTTP response; calling it from "
            "__exit__ makes instrumentation pull tokens the uninstrumented app would never have generated, so "
            "tracing an early-abandoning caller changes their bill",
        )
        span = self._llm()[0]
        self.assertEqual(
            span.tokens_out, 7,
            "reading the non-blocking current_message_snapshot instead still has to yield whatever usage the "
            "provider had already reported, or declining to drain would cost us the numbers entirely",
        )

    def test_stream_span_survives_an_untouched_stream(self):
        stream, mgr = self._stream()
        stream._snapshot_error = AssertionError("snapshot accessed before any event")
        messages = install_fake_anthropic_streaming(self.client, mgr)

        with messages.stream(model=MODEL, messages=MESSAGES):
            pass

        llm = self._llm()
        self.assertEqual(
            len(llm), 1,
            "current_message_snapshot is a property that asserts non-None, so a caller who opens the stream and "
            "never reads a single event would otherwise take an AssertionError out of our own __exit__",
        )
        self.assertEqual(llm[0].tokens_in, 0)
        self.assertEqual(llm[0].completion, "")

    def test_stream_span_carries_the_request_it_was_asked_to_make(self):
        _, mgr = self._stream()
        messages = install_fake_anthropic_streaming(self.client, mgr)

        with messages.stream(model=MODEL, messages=MESSAGES, temperature=0.2, max_tokens=512):
            pass

        attrs = self._llm()[0].attributes
        self.assertIn(
            "gen_ai.request.body", attrs,
            "messages.create records the request body and messages.stream did not, so the same prompt was "
            "reproducible from a non-streaming span and not from a streaming one",
        )
        self.assertEqual(attrs["gen_ai.request.temperature"], 0.2)
        self.assertEqual(attrs["gen_ai.request.max_tokens"], 512)

    def test_stream_prompt_flattens_block_content(self):
        _, mgr = self._stream()
        messages = install_fake_anthropic_streaming(self.client, mgr)

        with messages.stream(model=MODEL, messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image", "source": {"type": "base64", "data": "AAAA"}},
            ],
        }]):
            pass

        prompt = self._llm()[0].prompt
        self.assertIn(
            "describe this", prompt,
            "the class-level stream patcher had its own inline prompt builder that interpolated the content list "
            "with str(), so a multimodal prompt was stored as a Python repr with the whole base64 image inlined "
            "instead of the text the user actually typed",
        )
        self.assertNotIn("base64", prompt)

    def test_stream_span_captures_thinking_and_tool_calls(self):
        snapshot = AResp(text="42", tool=("lookup_order", {"id": "ORD-1"}), thinking="let me check")
        _, mgr = self._stream(snapshot)
        messages = install_fake_anthropic_streaming(self.client, mgr)

        with messages.stream(model=MODEL, messages=MESSAGES):
            pass

        span = self._llm()[0]
        self.assertEqual(span.attributes["gen_ai.response.thinking"], "let me check")
        self.assertEqual(
            span.completion, "42",
            "thinking text must stay out of the completion, or every tool-metric and response-match evaluator "
            "scores the model's scratchpad instead of its answer",
        )
        self.assertTrue(span.events)
        self.assertEqual(span.events[0].attributes["gen_ai.tool.name"], "lookup_order")

    def test_class_level_patch_carries_the_same_columns(self):
        snapshot = AResp(text="ok")
        snapshot.usage = ACacheUsage(50, 5, cache_read=450)
        stream, mgr = self._stream(snapshot)
        messages = install_fake_anthropic_class_stream(self.client, mgr)

        with messages.stream(model=MODEL, messages=MESSAGES):
            pass

        span = self._llm()[0]
        self.assertEqual(
            span.tokens_in, 500,
            "the autotrace path and the explicit patcher have to agree on the token maths, or the same app reports "
            "different costs depending on whether init() found anthropic before the user built their client",
        )
        self.assertEqual(span.cache_read_tokens, 450)
        self.assertFalse(
            stream.forced_drain,
            "the autotrace path is the one that used to force the drain, so it is the one that has to stop",
        )

    def test_the_with_body_is_not_nested_under_the_stream_span(self):
        _, mgr = self._stream()
        messages = install_fake_anthropic_streaming(self.client, mgr)
        before = current_span_ctx()

        with messages.stream(model=MODEL, messages=MESSAGES):
            self.assertEqual(
                current_span_ctx(), before,
                "the context token is reset at the end of __enter__ on purpose - the body of a `with stream` block "
                "is the caller's own code, not work done inside the LLM call, and re-parenting it would silently "
                "change every existing autotraced waterfall",
            )

    def test_patching_twice_does_not_double_count_the_call(self):
        _, mgr = self._stream()
        messages = install_fake_anthropic_streaming(self.client, mgr)
        from evalkit.instrumentation.llm._anthropic_stream import patch_messages_stream

        patch_messages_stream(self.client, messages, "anthropic")

        with messages.stream(model=MODEL, messages=MESSAGES):
            pass

        self.assertEqual(
            len(self._llm()), 1,
            "init() patches the Messages class and an explicit patch_anthropic() patches the instance; without the "
            "_evalkit guard an app that does both writes the same span twice, and evalkit.traces is a plain "
            "MergeTree with no dedupe so the double charge is permanent",
        )

    def test_instrumentation_failure_still_returns_a_working_stream(self):
        stream, mgr = self._stream()
        messages = install_fake_anthropic_streaming(self.client, mgr)

        class Exploding(list):
            def __iter__(self):
                raise TypeError("no")

        with messages.stream(model=MODEL, messages=Exploding()) as s:
            self.assertIs(
                s, stream,
                "building the span must never be able to break the call it is observing: a prompt shape we cannot "
                "read has to degrade to an untraced stream, not to an exception in the customer's request path",
            )

        self.assertEqual(len(self._llm()), 0)


if __name__ == "__main__":
    unittest.main()
