import sys
import types
import unittest

import evalkit
from tests.helpers import GenAIResp, fresh, reset_context, install_fake_genai

TTFT = "gen_ai.server.time_to_first_token_ms"


class TestGoogleGenAI(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_six_calls_single_trace(self):
        m = install_fake_genai(self.client)
        for i in range(6):
            m.generate_content(model="gemini-2.0-flash", contents=f"q{i}")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 6)
        self.assertEqual(len({e.trace_id for e in llm}), 1)
        self.assertEqual(llm[0].provider, "google-genai")
        self.assertEqual(llm[0].tokens_in, 3)
        self.assertEqual(llm[0].tokens_out, 5)

    def test_joins_active_trace(self):
        m = install_fake_genai(self.client)
        trace_id, end, ctx = self.client.start_trace("root")
        with evalkit.with_trace(ctx):
            m.generate_content(model="g", contents="hi")
        end("OK")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(llm[0].trace_id, trace_id)
        self.assertEqual(llm[0].parent_span_id, ctx.span_id)

    def test_tool_events_captured(self):
        m = install_fake_genai(self.client, response=GenAIResp(tool=("get_weather", {"city": "london"})))
        m.generate_content(model="g", contents="weather?")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertTrue(llm.events)
        self.assertEqual(llm.events[0].attributes["gen_ai.tool.name"], "get_weather")

    def test_client_side_tool_no_child_span(self):
        m = install_fake_genai(self.client, response=GenAIResp(tool=("get_weather", {"city": "london"})))
        m.generate_content(model="g", contents="weather?")
        self.client.flush()
        tools = [e for e in self.captured if e.span_type == "tool_call"]
        self.assertEqual(len(tools), 0)

    def test_streaming(self):
        chunks = [GenAIResp(text="Hel"), GenAIResp(text="lo!", pin=0, pout=7)]
        m = install_fake_genai(self.client, stream_chunks=chunks)
        out = list(m.generate_content_stream(model="g", contents="hi"))
        self.assertEqual(len(out), 2)
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(llm.completion, "Hello!")
        self.assertTrue(llm.attributes.get("gen_ai.streaming"))

    def test_streaming_captures_request_and_response_body(self):
        chunks = [GenAIResp(text="Hel"), GenAIResp(text="lo!", pin=0, pout=7)]
        m = install_fake_genai(self.client, stream_chunks=chunks)
        list(m.generate_content_stream(model="g", contents="hi"))
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertIn("gen_ai.request.body", llm.attributes)
        self.assertIn("gen_ai.response.body", llm.attributes)

    def test_thinking_captured_non_streaming(self):
        m = install_fake_genai(self.client, response=GenAIResp(text="42", thinking="let me think..."))
        m.generate_content(model="g", contents="what is 6*7?")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(llm.completion, "42")
        self.assertEqual(llm.attributes.get("gen_ai.response.thinking"), "let me think...")

    def test_reasoning_tokens_captured_non_streaming(self):
        m = install_fake_genai(self.client, response=GenAIResp(text="42", thoughts_tokens=15))
        m.generate_content(model="g", contents="what is 6*7?")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(llm.attributes.get("gen_ai.usage.reasoning_tokens"), 15)

    def test_thinking_and_reasoning_tokens_captured_streaming(self):
        chunks = [
            GenAIResp(thinking="step one...", pin=0, pout=0),
            GenAIResp(text="42", pin=5, pout=8, thoughts_tokens=12),
        ]
        m = install_fake_genai(self.client, stream_chunks=chunks)
        list(m.generate_content_stream(model="g", contents="what is 6*7?"))
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(llm.completion, "42")
        self.assertEqual(llm.attributes.get("gen_ai.response.thinking"), "step one...")
        self.assertEqual(llm.attributes.get("gen_ai.usage.reasoning_tokens"), 12)

    def test_thoughts_and_tool_use_tokens_counted_as_billable_non_streaming(self):
        m = install_fake_genai(self.client, response=GenAIResp(
            text="42", pin=2000, pout=40, thoughts_tokens=900,
            cached_tokens=1800, tool_use_tokens=150, total_tokens=3090))
        m.generate_content(model="g", contents="what is 6*7?")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(llm.tokens_in, 2150)
        self.assertEqual(llm.tokens_out, 940)
        self.assertEqual(llm.cache_read_tokens, 1800)
        self.assertEqual(llm.attributes.get("gen_ai.usage.total_tokens"), 3090)
        self.assertEqual(llm.attributes.get("gen_ai.usage.tool_use_prompt_tokens"), 150)
        self.assertEqual(llm.attributes.get("gen_ai.usage.reasoning_tokens"), 900)

    def test_thoughts_and_tool_use_tokens_counted_as_billable_streaming(self):
        chunks = [
            GenAIResp(text="Hel", pin=0, pout=0),
            GenAIResp(text="lo!", pin=2000, pout=40, thoughts_tokens=900,
                      cached_tokens=1800, tool_use_tokens=150, total_tokens=3090),
        ]
        m = install_fake_genai(self.client, stream_chunks=chunks)
        list(m.generate_content_stream(model="g", contents="hi"))
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(llm.tokens_in, 2150)
        self.assertEqual(llm.tokens_out, 940)
        self.assertEqual(llm.cache_read_tokens, 1800)
        self.assertEqual(llm.attributes.get("gen_ai.usage.total_tokens"), 3090)
        self.assertEqual(llm.attributes.get("gen_ai.usage.tool_use_prompt_tokens"), 150)

    def test_cached_tokens_captured_non_streaming(self):
        m = install_fake_genai(self.client, response=GenAIResp(text="42", pin=394505, cached_tokens=392425))
        m.generate_content(model="g", contents="what is 6*7?")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(llm.tokens_in, 394505)
        self.assertEqual(llm.attributes.get("gen_ai.usage.cached_tokens"), 392425)

    def test_cached_tokens_captured_streaming(self):
        chunks = [
            GenAIResp(text="Hel", pin=0, pout=0),
            GenAIResp(text="lo!", pin=394505, pout=7, cached_tokens=392425),
        ]
        m = install_fake_genai(self.client, stream_chunks=chunks)
        list(m.generate_content_stream(model="g", contents="hi"))
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(llm.tokens_in, 394505)
        self.assertEqual(llm.attributes.get("gen_ai.usage.cached_tokens"), 392425)

    def test_streaming_captures_time_to_first_token(self):
        chunks = [GenAIResp(text="Hel", pin=0, pout=0), GenAIResp(text="lo!", pin=3, pout=7)]
        m = install_fake_genai(self.client, stream_chunks=chunks)
        list(m.generate_content_stream(model="g", contents="hi"))
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        ttft = llm.attributes.get(TTFT)
        self.assertIsInstance(ttft, int,
                              "Gemini streams carried no TTFT while openai/anthropic/bedrock did, which reads as a "
                              "broken dashboard column rather than a missing SDK feature")
        self.assertGreaterEqual(ttft, 0)
        self.assertLessEqual(ttft, llm.latency_ms,
                             "TTFT can never exceed the span's own latency")

    def test_thinking_only_chunk_marks_time_to_first_token(self):
        chunks = [GenAIResp(thinking="let me think", pin=0, pout=0), GenAIResp(text="42", pin=3, pout=7)]
        m = install_fake_genai(self.client, stream_chunks=chunks)
        list(m.generate_content_stream(model="g", contents="what is 6*7?"))
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertIsInstance(llm.attributes.get(TTFT), int,
                              "a thinking model has produced its first token even before any visible text, so "
                              "ignoring thought parts makes reasoning models look far slower than they are")

    def test_failed_stream_has_no_time_to_first_token(self):
        from evalkit.instrumentation.llm import google_genai as gg

        def gen():
            yield GenAIResp(text="Hel", pin=0, pout=0)
            raise ValueError("stream reset")

        class FakeModels:
            def generate_content(self, *, model, contents, config=None, **k):
                return GenAIResp(text="hello")

            def generate_content_stream(self, *, model, contents, config=None, **k):
                return gen()

        gg._patch_sync_models(self.client, types.SimpleNamespace(Models=FakeModels, AsyncModels=None))
        with self.assertRaises(ValueError):
            list(FakeModels().generate_content_stream(model="g", contents="hi"))
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(llm.status, "ERROR")
        self.assertNotIn(TTFT, llm.attributes,
                         "a truncated stream must stay out of the TTFT percentiles, or a provider outage that "
                         "fails fast reads as a latency improvement")

    def test_async_streaming_captures_time_to_first_token(self):
        import asyncio

        from evalkit.instrumentation.llm import google_genai as gg

        chunks = [GenAIResp(text="Hel", pin=0, pout=0), GenAIResp(text="lo!", pin=3, pout=7)]

        async def agen():
            for c in chunks:
                yield c

        class FakeAsyncModels:
            async def generate_content(self, *, model, contents, config=None, **k):
                return GenAIResp(text="hello")

            async def generate_content_stream(self, *, model, contents, config=None, **k):
                return agen()

        gg._patch_async_models(self.client, types.SimpleNamespace(AsyncModels=FakeAsyncModels))

        async def run():
            stream = await FakeAsyncModels().generate_content_stream(model="g", contents="hi")
            async for _ in stream:
                pass

        asyncio.run(run())
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertIsInstance(llm.attributes.get(TTFT), int,
                              "the async path is the one most apps use; instrumenting only the sync one leaves the "
                              "column empty for exactly the traffic that matters")


class TestGoogleAILegacy(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self._injected = []

    def tearDown(self):
        for n in self._injected:
            sys.modules.pop(n, None)

    def test_generate_content_span(self):
        genai_mod = types.ModuleType("google.generativeai")

        class _LegacyResult:
            text = "Legacy Gemini response"

        class FakeGenerativeModel:
            model_name = "gemini-1.5-pro"

            def generate_content(self, *a, **k):
                return _LegacyResult()

        genai_mod.GenerativeModel = FakeGenerativeModel
        sys.modules["google"] = types.ModuleType("google")
        sys.modules["google.generativeai"] = genai_mod
        self._injected += ["google", "google.generativeai"]

        from evalkit.instrumentation.auto import _patch_google_genai_legacy_class
        _patch_google_genai_legacy_class(self.client, FakeGenerativeModel)

        FakeGenerativeModel().generate_content("What is AI?")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].provider, "google-ai")
        self.assertEqual(llm[0].model, "gemini-1.5-pro")
        self.assertEqual(llm[0].completion, "Legacy Gemini response")

    def _llm(self):
        self.client.flush()
        spans = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(spans), 1, "a generate_content call must emit exactly one llm_call span")
        return spans[0]

    def _install_legacy(self, *, result=None, chunks=None, error=None):
        from evalkit.instrumentation.llm.google_ai import patch_google_ai

        class LegacyModel:
            model_name = "gemini-2.5-pro"

            def generate_content(self, contents, **kwargs):
                if kwargs.get("stream"):
                    return _LegacyStream(chunks or [], error)
                return result

        m = LegacyModel()
        patch_google_ai(self.client, m)
        return m

    def test_legacy_unary_keeps_thought_parts_out_of_the_completion(self):
        m = self._install_legacy(result=GenAIResp(
            text="the answer", thinking="internal reasoning", pin=4, pout=9, thoughts_tokens=6))
        m.generate_content("hi")
        span = self._llm()
        self.assertEqual(span.completion, "the answer",
                         "google marks reasoning parts with `thought=True` but folds them into `.text`; using "
                         "`.text` as the completion leaked the model's private reasoning into every eval's input")
        self.assertEqual(span.attributes.get("gen_ai.response.thinking"), "internal reasoning")
        self.assertEqual(span.attributes.get("gen_ai.usage.reasoning_tokens"), 6)

    def test_legacy_unary_emits_tool_events(self):
        m = self._install_legacy(result=GenAIResp(text="", tool=("get_weather", {"city": "london"})))
        m.generate_content("weather?")
        span = self._llm()
        events = [e for e in (span.events or []) if e.name == "gen_ai.tool.call"]
        self.assertEqual(len(events), 1,
                         "the legacy google path dropped function calls entirely, so a Gemini agent's tool use "
                         "was invisible in the trace")
        self.assertEqual(events[0].attributes["gen_ai.tool.name"], "get_weather")

    def test_legacy_stream_records_text_usage_and_ttft(self):
        m = self._install_legacy(chunks=[
            GenAIResp(text="Hel", pin=0, pout=0, total_tokens=0),
            GenAIResp(text="lo", pin=4, pout=7, total_tokens=11),
        ])
        out = list(m.generate_content("hi", stream=True))
        self.assertEqual(len(out), 2, "the tracer must pass every chunk through to the caller")
        span = self._llm()
        self.assertEqual(span.status, "OK")
        self.assertEqual(span.completion, "Hello",
                         "reading the aggregate response instead of the chunks produced an empty completion for "
                         "every streamed Gemini call")
        self.assertEqual((span.tokens_in, span.tokens_out), (4, 7))
        self.assertTrue(span.attributes.get("gen_ai.streaming"))
        self.assertIsInstance(span.attributes.get(TTFT), int,
                              "TTFT must serialize as a number; trace-service2 reads it with JSONExtractFloat "
                              "and a string comes back as 0")

    def test_legacy_stream_never_touches_the_un_iterated_response(self):
        m = self._install_legacy(chunks=[GenAIResp(text="ok", pin=1, pout=1, total_tokens=2)])
        stream = m.generate_content("hi", stream=True)
        list(stream)
        span = self._llm()
        self.assertEqual(span.completion, "ok",
                         "google's streaming response raises IncompleteIterationError from `.candidates` and "
                         "`.text` until it has been consumed; a tracer that reads them inside a bare except "
                         "records a silently empty span instead of failing loudly")

    def test_legacy_stream_separates_thinking_from_the_completion(self):
        m = self._install_legacy(chunks=[
            GenAIResp(text="", thinking="plan", pin=0, pout=0, total_tokens=0),
            GenAIResp(text="answer", pin=2, pout=5, total_tokens=7),
        ])
        list(m.generate_content("hi", stream=True))
        span = self._llm()
        self.assertEqual(span.completion, "answer")
        self.assertEqual(span.attributes.get("gen_ai.response.thinking"), "plan")

    def test_legacy_tool_only_stream_has_no_time_to_first_token(self):
        m = self._install_legacy(chunks=[
            GenAIResp(text="", tool=("get_weather", {"city": "SF"}), pin=3, pout=4, total_tokens=7),
        ])
        list(m.generate_content("weather?", stream=True))
        span = self._llm()
        events = [e for e in (span.events or []) if e.name == "gen_ai.tool.call"]
        self.assertEqual(len(events), 1)
        self.assertNotIn(TTFT, span.attributes,
                         "a function call is not user-visible text; counting it as the first token makes "
                         "tool-calling Gemini look faster than it is")

    def test_legacy_stream_records_finish_reason_by_name(self):
        chunk = GenAIResp(text="done", pin=1, pout=1, total_tokens=2)
        chunk.candidates[0].finish_reason = types.SimpleNamespace(name="STOP")
        m = self._install_legacy(chunks=[chunk])
        list(m.generate_content("hi", stream=True))
        span = self._llm()
        self.assertEqual(span.attributes.get("gen_ai.response.finish_reasons"), ["STOP"],
                         "google's finish_reason is a proto enum whose str() is the integer; storing that makes "
                         "the dashboard's stop-reason breakdown unreadable")

    def test_legacy_stream_drops_unspecified_finish_reason(self):
        chunk = GenAIResp(text="done", pin=1, pout=1, total_tokens=2)
        chunk.candidates[0].finish_reason = types.SimpleNamespace(name="FINISH_REASON_UNSPECIFIED")
        m = self._install_legacy(chunks=[chunk])
        list(m.generate_content("hi", stream=True))
        span = self._llm()
        self.assertNotIn("gen_ai.response.finish_reasons", span.attributes,
                         "every non-final chunk carries an unspecified finish reason; recording it would report "
                         "'FINISH_REASON_UNSPECIFIED' as the outcome of a perfectly normal call")

    def test_legacy_failed_stream_keeps_partials_and_drops_ttft(self):
        m = self._install_legacy(
            chunks=[GenAIResp(text="Hel", pin=4, pout=1, total_tokens=5)],
            error=RuntimeError("upstream closed"),
        )
        with self.assertRaises(RuntimeError):
            list(m.generate_content("hi", stream=True))
        span = self._llm()
        self.assertEqual(span.status, "ERROR")
        self.assertIn("upstream closed", span.status_message or "")
        self.assertEqual(span.completion, "Hel")
        self.assertEqual(span.tokens_in, 4, "the provider bills for tokens it already streamed")
        self.assertNotIn(TTFT, span.attributes, "a truncated stream must stay out of the TTFT percentiles")

    def test_legacy_abandoned_stream_stays_ok_and_keeps_ttft(self):
        m = self._install_legacy(chunks=[
            GenAIResp(text="Hel", pin=0, pout=0, total_tokens=0),
            GenAIResp(text="lo", pin=4, pout=2, total_tokens=6),
        ])
        stream = m.generate_content("hi", stream=True)
        it = iter(stream)
        next(it)
        del it
        del stream
        span = self._llm()
        self.assertEqual(span.status, "OK",
                         "a caller that stops reading has not hit a provider error")
        self.assertIn(TTFT, span.attributes)

    def test_legacy_auto_class_patch_traces_streaming(self):
        from evalkit.instrumentation.auto import _patch_google_genai_legacy_class

        chunks = [GenAIResp(text="via ", pin=0, pout=0, total_tokens=0),
                  GenAIResp(text="init", pin=2, pout=3, total_tokens=5)]

        class AutoModel:
            model_name = "gemini-2.5-flash"

            def generate_content(self, contents, **kwargs):
                if kwargs.get("stream"):
                    return _LegacyStream(chunks, None)
                return GenAIResp(text="unary")

        _patch_google_genai_legacy_class(self.client, AutoModel)
        list(AutoModel().generate_content("hi", stream=True))
        span = self._llm()
        self.assertEqual(span.completion, "via init",
                         "init() patches GenerativeModel at the class level; without a streaming seam there the "
                         "auto-instrumented path silently recorded nothing for streamed calls")
        self.assertEqual((span.tokens_in, span.tokens_out), (2, 3))


class _LegacyStream:
    def __init__(self, chunks, error):
        self._chunks = chunks
        self._error = error

    def __iter__(self):
        for c in self._chunks:
            yield c
        if self._error is not None:
            raise self._error

    @property
    def candidates(self):
        raise RuntimeError("IncompleteIterationError")

    @property
    def text(self):
        raise RuntimeError("IncompleteIterationError")


if __name__ == "__main__":
    unittest.main(verbosity=2)
