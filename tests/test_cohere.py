import asyncio
import sys
import types
import unittest

from tests.helpers import fresh, reset_context

TTFT = "gen_ai.server.time_to_first_token_ms"
THINKING = "gen_ai.response.thinking"
FINISH = "gen_ai.response.finish_reasons"


def _ns(**kw):
    return types.SimpleNamespace(**kw)


def _v2_units(pin, pout, cached=0):
    return _ns(billed_units=_ns(input_tokens=pin, output_tokens=pout), cached_tokens=cached)


def _v2_content_delta(text=None, thinking=None):
    content = _ns()
    if text is not None:
        content.text = text
    if thinking is not None:
        content.thinking = thinking
    return _ns(type="content-delta", delta=_ns(message=_ns(content=content)))


def _v2_tool_event(etype, index, *, call_id=None, name=None, arguments=None):
    function = _ns()
    if name is not None:
        function.name = name
    if arguments is not None:
        function.arguments = arguments
    call = _ns(function=function)
    if call_id is not None:
        call.id = call_id
    return _ns(type=etype, index=index, delta=_ns(message=_ns(tool_calls=call)))


def _v1_stream_end(pin, pout, finish="COMPLETE"):
    return _ns(
        event_type="stream-end",
        finish_reason=finish,
        response=_ns(
            response_id="r1",
            meta=_ns(billed_units=_ns(input_tokens=pin, output_tokens=pout)),
        ),
    )


def _install_cohere_stream(client, events, error=None, is_async=False):
    from evalkit.instrumentation.llm.cohere import patch_cohere

    if is_async:
        class FakeAsyncCohere:
            async def chat(self, **k):
                return _ns(text="unused")

            async def chat_stream(self, **k):
                for e in events:
                    yield e
                if error is not None:
                    raise error

        co = FakeAsyncCohere()
    else:
        class FakeCohere:
            def chat(self, **k):
                return _ns(text="unused")

            def chat_stream(self, **k):
                for e in events:
                    yield e
                if error is not None:
                    raise error

        co = FakeCohere()

    patch_cohere(client, co)
    return co


class TestCohere(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self._injected = []

    def tearDown(self):
        for n in self._injected:
            sys.modules.pop(n, None)

    def _llm(self):
        self.client.flush()
        spans = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(spans), 1, "a cohere call must emit exactly one llm_call span")
        return spans[0]

    def test_chat_span(self):
        cohere_mod = types.ModuleType("cohere")

        class _CTokens:
            input_tokens = 6
            output_tokens = 11

        class _CUsage:
            tokens = _CTokens()

        class _CResult:
            text = "Cohere response"
            usage = _CUsage()

        class FakeCohereClient:
            def chat(self, *a, **k):
                return _CResult()

        cohere_mod.Client = FakeCohereClient
        sys.modules["cohere"] = cohere_mod
        self._injected.append("cohere")

        from evalkit.instrumentation.auto import _patch_cohere_class
        _patch_cohere_class(self.client, FakeCohereClient)

        FakeCohereClient().chat(model="command-r-plus", message="Tell me something")
        span = self._llm()
        self.assertEqual(span.provider, "cohere")
        self.assertEqual(span.completion, "Cohere response")
        self.assertEqual(span.tokens_in, 6)
        self.assertEqual(span.tokens_out, 11)

    def test_error_emits_error_span(self):
        cohere_mod = types.ModuleType("cohere")

        class FailingCohere:
            def chat(self, *a, **k):
                raise ValueError("cohere down")

        cohere_mod.Client = FailingCohere
        sys.modules["cohere"] = cohere_mod
        self._injected.append("cohere")

        from evalkit.instrumentation.auto import _patch_cohere_class
        _patch_cohere_class(self.client, FailingCohere)

        with self.assertRaises(ValueError):
            FailingCohere().chat(model="command-r-plus", message="hi")
        span = self._llm()
        self.assertEqual(span.status, "ERROR")

    def test_v1_unary_usage_comes_from_meta(self):
        from evalkit.instrumentation.llm.cohere import patch_cohere

        result = _ns(
            text="v1 answer",
            response_id="r-v1",
            finish_reason="COMPLETE",
            meta=_ns(billed_units=_ns(input_tokens=4, output_tokens=9)),
        )

        class FakeV1:
            def chat(self, **k):
                return result

        co = FakeV1()
        patch_cohere(self.client, co)
        co.chat(model="command-r", message="hi")
        span = self._llm()
        self.assertEqual(span.completion, "v1 answer")
        self.assertEqual((span.tokens_in, span.tokens_out), (4, 9),
                         "cohere v1 reports usage under `meta`, not `usage`; reading only `usage` left every v1 "
                         "chat span at zero tokens, which zeroes its cost in the dashboard")
        self.assertEqual(span.attributes.get("gen_ai.response.id"), "r-v1")

    def test_v2_unary_keeps_thinking_out_of_the_completion(self):
        from evalkit.instrumentation.llm.cohere import patch_cohere

        result = _ns(
            id="c-v2",
            finish_reason="COMPLETE",
            message=_ns(
                content=[
                    _ns(type="thinking", thinking="let me think"),
                    _ns(type="text", text="the answer"),
                ],
                tool_plan=None,
                tool_calls=None,
            ),
            usage=_v2_units(5, 7),
        )

        class FakeV2:
            def chat(self, **k):
                return result

        co = FakeV2()
        patch_cohere(self.client, co)
        co.chat(model="command-a", messages=[{"role": "user", "content": "hi"}])
        span = self._llm()
        self.assertEqual(span.completion, "the answer",
                         "reasoning text is not the answer; folding it into the completion corrupts every "
                         "eval that scores the model's output")
        self.assertEqual(span.attributes.get(THINKING), "let me think")

    def test_v2_stream_records_text_usage_and_ttft(self):
        co = _install_cohere_stream(self.client, [
            _ns(type="message-start", id="c-1"),
            _v2_content_delta(text="Hel"),
            _v2_content_delta(text="lo"),
            _ns(type="message-end", delta=_ns(finish_reason="COMPLETE", usage=_v2_units(4, 2, cached=3))),
        ])
        list(co.chat_stream(model="command-a", messages=[]))
        span = self._llm()
        self.assertEqual(span.status, "OK")
        self.assertEqual(span.completion, "Hello")
        self.assertEqual((span.tokens_in, span.tokens_out), (4, 2),
                         "a cohere stream that reports no usage bills as free; the message-end delta is the only "
                         "place the counts arrive")
        self.assertEqual(span.cache_read_tokens, 3)
        self.assertEqual(span.attributes.get(FINISH), ["COMPLETE"])
        self.assertIsInstance(span.attributes.get(TTFT), int,
                             "TTFT must serialize as a number; trace-service2 reads it with JSONExtractFloat "
                             "and a string comes back as 0")
        self.assertTrue(span.attributes.get("gen_ai.streaming"))
        self.assertEqual(span.attributes.get("gen_ai.response.id"), "c-1")

    def test_v2_stream_separates_thinking_from_the_completion(self):
        co = _install_cohere_stream(self.client, [
            _v2_content_delta(thinking="weighing options"),
            _v2_content_delta(text="answer"),
            _ns(type="message-end", delta=_ns(finish_reason="COMPLETE", usage=_v2_units(1, 1))),
        ])
        list(co.chat_stream(model="command-a", messages=[]))
        span = self._llm()
        self.assertEqual(span.completion, "answer")
        self.assertEqual(span.attributes.get(THINKING), "weighing options")

    def test_v2_stream_emits_tool_events_with_accumulated_arguments(self):
        co = _install_cohere_stream(self.client, [
            _v2_tool_event("tool-call-start", 0, call_id="tc_1", name="get_weather", arguments=""),
            _v2_tool_event("tool-call-delta", 0, arguments='{"city":'),
            _v2_tool_event("tool-call-delta", 0, arguments='"SF"}'),
            _ns(type="message-end", delta=_ns(finish_reason="TOOL_CALL", usage=_v2_units(2, 2))),
        ])
        list(co.chat_stream(model="command-a", messages=[]))
        span = self._llm()
        events = [e for e in (span.events or []) if e.name == "gen_ai.tool.call"]
        self.assertEqual(len(events), 1,
                         "a streamed tool call arrives as start + N argument deltas; emitting one event per delta "
                         "or none at all both make agent traces unreadable")
        self.assertEqual(events[0].attributes["gen_ai.tool.name"], "get_weather")
        self.assertEqual(events[0].attributes["gen_ai.tool.call.arguments"], '{"city":"SF"}')
        self.assertEqual(events[0].attributes["gen_ai.tool.call.id"], "tc_1")

    def test_tool_only_v2_stream_has_no_time_to_first_token(self):
        co = _install_cohere_stream(self.client, [
            _v2_tool_event("tool-call-start", 0, call_id="tc_1", name="get_weather", arguments=""),
            _v2_tool_event("tool-call-delta", 0, arguments='{"city":"SF"}'),
            _ns(type="message-end", delta=_ns(finish_reason="TOOL_CALL", usage=_v2_units(2, 2))),
        ])
        list(co.chat_stream(model="command-a", messages=[]))
        span = self._llm()
        self.assertNotIn(TTFT, span.attributes,
                         "tool-argument deltas are not user-visible tokens; counting them as the first token makes "
                         "tool-calling models look faster than they are")

    def test_v2_tool_plan_is_recorded_as_thinking_and_marks_the_first_token(self):
        co = _install_cohere_stream(self.client, [
            _ns(type="tool-plan-delta", delta=_ns(message=_ns(tool_plan="I will call "))),
            _ns(type="tool-plan-delta", delta=_ns(message=_ns(tool_plan="the weather tool"))),
            _v2_tool_event("tool-call-start", 0, call_id="tc_1", name="get_weather", arguments="{}"),
            _ns(type="message-end", delta=_ns(finish_reason="TOOL_CALL", usage=_v2_units(2, 5))),
        ])
        list(co.chat_stream(model="command-a", messages=[]))
        span = self._llm()
        self.assertEqual(span.completion, "",
                         "the tool plan is not the assistant's answer, so it must not become the completion")
        self.assertEqual(span.attributes.get(THINKING), "I will call the weather tool")
        self.assertIn(TTFT, span.attributes,
                      "the tool plan is generated, billed output text, so the moment it arrives is a real "
                      "time-to-first-token")

    def test_v1_stream_records_text_and_usage(self):
        co = _install_cohere_stream(self.client, [
            _ns(event_type="stream-start", generation_id="g1"),
            _ns(event_type="text-generation", text="Hel"),
            _ns(event_type="text-generation", text="lo"),
            _v1_stream_end(4, 9),
        ])
        list(co.chat_stream(model="command-r", message="hi"))
        span = self._llm()
        self.assertEqual(span.completion, "Hello",
                         "v1 and v2 are different event unions on the same method; one dispatcher has to serve "
                         "both or half of cohere's users get empty spans")
        self.assertEqual((span.tokens_in, span.tokens_out), (4, 9))
        self.assertEqual(span.attributes.get("gen_ai.response.id"), "r1")
        self.assertEqual(span.attributes.get(FINISH), ["COMPLETE"])
        self.assertIn(TTFT, span.attributes)

    def test_v1_stream_prompt_uses_the_positional_message(self):
        co = _install_cohere_stream(self.client, [
            _ns(event_type="text-generation", text="ok"),
            _v1_stream_end(1, 1),
        ])
        list(co.chat_stream(model="command-r", message="what is the weather"))
        span = self._llm()
        self.assertIn("what is the weather", span.prompt,
                      "cohere v1 passes the prompt as `message=`, not `messages=`; missing it left the prompt "
                      "column empty for every v1 call")

    def test_v1_stream_tool_calls_are_not_duplicated_by_the_final_generation(self):
        co = _install_cohere_stream(self.client, [
            _ns(event_type="tool-calls-chunk", tool_call_delta=_ns(
                index=0, name="get_weather", parameters='{"city":', text="I will ")),
            _ns(event_type="tool-calls-chunk", tool_call_delta=_ns(
                index=0, name=None, parameters='"SF"}', text="check")),
            _ns(event_type="tool-calls-generation",
                tool_calls=[_ns(name="get_weather", parameters={"city": "SF"})],
                text="I will check"),
            _v1_stream_end(3, 6, finish="COMPLETE"),
        ])
        list(co.chat_stream(model="command-r", message="weather?"))
        span = self._llm()
        events = [e for e in (span.events or []) if e.name == "gen_ai.tool.call"]
        self.assertEqual(len(events), 1,
                         "v1 streams the tool call incrementally and then repeats it in tool-calls-generation; "
                         "keeping both would double every tool call in the trace")
        self.assertEqual(events[0].attributes["gen_ai.tool.name"], "get_weather")
        self.assertEqual(span.attributes.get(THINKING), "I will check",
                         "the plan accumulated from the deltas is the whole plan, so the final event's copy of it "
                         "must not be appended again")

    def test_failed_stream_keeps_partials_and_drops_ttft(self):
        co = _install_cohere_stream(
            self.client,
            [_v2_content_delta(text="Hel"),
             _ns(type="message-end", delta=_ns(finish_reason="COMPLETE", usage=_v2_units(4, 1)))],
            error=RuntimeError("upstream closed"),
        )
        with self.assertRaises(RuntimeError):
            list(co.chat_stream(model="command-a", messages=[]))
        span = self._llm()
        self.assertEqual(span.status, "ERROR")
        self.assertIn("upstream closed", span.status_message or "")
        self.assertEqual(span.completion, "Hel",
                         "the text received before the failure is the only evidence of what the model was doing")
        self.assertEqual(span.tokens_in, 4,
                         "the provider bills for tokens it already streamed")
        self.assertNotIn(TTFT, span.attributes,
                         "a truncated stream must stay out of the TTFT percentiles")

    def test_stream_call_that_raises_immediately_emits_an_error_span(self):
        from evalkit.instrumentation.llm.cohere import patch_cohere

        class Failing:
            def chat_stream(self, **k):
                raise ValueError("bad api key")

        co = Failing()
        patch_cohere(self.client, co)
        with self.assertRaises(ValueError):
            co.chat_stream(model="command-a", messages=[])
        span = self._llm()
        self.assertEqual(span.status, "ERROR")
        self.assertIn("bad api key", span.status_message or "",
                      "a stream that never opens is the most common failure of all; it has to be traced even "
                      "though no chunk ever arrives")

    def test_abandoned_stream_stays_ok_and_keeps_ttft(self):
        co = _install_cohere_stream(self.client, [
            _v2_content_delta(text="Hel"),
            _v2_content_delta(text="lo"),
            _ns(type="message-end", delta=_ns(finish_reason="COMPLETE", usage=_v2_units(4, 2))),
        ])
        stream = co.chat_stream(model="command-a", messages=[])
        it = iter(stream)
        next(it)
        del it
        del stream
        span = self._llm()
        self.assertEqual(span.status, "OK",
                         "a caller that stops reading has not hit a provider error")
        self.assertIn(TTFT, span.attributes)

    def test_async_stream_is_traced(self):
        async def run():
            co = _install_cohere_stream(self.client, [
                _v2_content_delta(text="async "),
                _v2_content_delta(text="hello"),
                _ns(type="message-end", delta=_ns(finish_reason="COMPLETE", usage=_v2_units(2, 3))),
            ], is_async=True)
            async for _ in co.chat_stream(model="command-a", messages=[]):
                pass

        asyncio.run(run())
        span = self._llm()
        self.assertEqual(span.status, "OK")
        self.assertEqual(span.completion, "async hello",
                         "cohere's async chat_stream is an async generator function, so the wrapper must not "
                         "await it before handing it to the async tracer")
        self.assertEqual((span.tokens_in, span.tokens_out), (2, 3))

    def test_async_unary_chat_is_traced(self):
        from evalkit.instrumentation.llm.cohere import patch_cohere

        class FakeAsyncV2:
            async def chat(self, **k):
                return _ns(id="c-a", finish_reason="COMPLETE",
                           message=_ns(content=[_ns(type="text", text="async answer")],
                                       tool_plan=None, tool_calls=None),
                           usage=_v2_units(3, 4))

        co = FakeAsyncV2()
        patch_cohere(self.client, co)
        asyncio.run(co.chat(model="command-a", messages=[]))
        span = self._llm()
        self.assertEqual(span.completion, "async answer",
                         "AsyncClient.chat is a coroutine; wrapping it with the sync recorder recorded the "
                         "coroutine object instead of the response")
        self.assertEqual((span.tokens_in, span.tokens_out), (3, 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
