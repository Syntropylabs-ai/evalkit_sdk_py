import asyncio
import json
import unittest

from tests.helpers import (
    AStreamBlockStart, AStreamJsonDelta, AStreamMessageDelta, AStreamMessageStart, AStreamTextDelta,
    OStreamToolCall, OStreamToolChunk,
    fresh, reset_context, stream_usage,
)

TOOL_NAME = "gen_ai.tool.name"
TOOL_ARGS = "gen_ai.tool.call.arguments"
TOOL_ID = "gen_ai.tool.call.id"
FINISH = "gen_ai.response.finish_reasons"


def _openai_sync(client, chunks):
    from evalkit.instrumentation.auto import _patch_openai_class_sync

    class FakeCompletions:
        def create(self, *a, **k):
            return iter(chunks)

    _patch_openai_class_sync(client, FakeCompletions)
    return FakeCompletions()


def _openai_async(client, chunks):
    from evalkit.instrumentation.auto import _patch_openai_class_async

    async def agen():
        for c in chunks:
            yield c

    class FakeAsyncCompletions:
        async def create(self, *a, **k):
            return agen()

    _patch_openai_class_async(client, FakeAsyncCompletions)
    return FakeAsyncCompletions()


def _anthropic_sync(client, events):
    from evalkit.instrumentation import auto as _auto

    class FakeMessages:
        def create(self, *a, **k):
            return iter(events)

    _auto._patch_anthropic_class_sync(client, FakeMessages)
    return FakeMessages()


def _explicit_openai(client, chunks):
    import types as _types

    from evalkit.instrumentation.llm.openai import patch_openai

    class FakeCompletions:
        def create(self, *a, **k):
            return iter(chunks)

    completions = FakeCompletions()
    fake = _types.SimpleNamespace(chat=_types.SimpleNamespace(completions=completions))
    patch_openai(client, fake)
    return completions


class StreamToolCallBase(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def _span(self):
        self.client.flush()
        spans = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(spans), 1, "a stream must emit exactly one llm_call span")
        return spans[0]

    def _tools(self, span):
        return [e for e in (span.events or []) if e.name == "gen_ai.tool.call"]


class TestOpenAIStreamingToolCalls(StreamToolCallBase):
    def _weather_stream(self):
        return [
            OStreamToolChunk(tool_calls=[OStreamToolCall(0, id="call_abc", name="get_weather", arguments="")]),
            OStreamToolChunk(tool_calls=[OStreamToolCall(0, arguments='{"city"')]),
            OStreamToolChunk(tool_calls=[OStreamToolCall(0, arguments=': "Paris"}')]),
            OStreamToolChunk(finish_reason="tool_calls"),
            OStreamToolChunk(usage=stream_usage(prompt_tokens=42, completion_tokens=17)),
        ]

    def test_streamed_tool_call_is_reassembled_from_its_deltas(self):
        fc = _openai_sync(self.client, self._weather_stream())
        list(fc.create(model="gpt-4o-mini", messages=[], stream=True))

        span = self._span()
        tools = self._tools(span)
        self.assertEqual(
            len(tools), 1,
            "process_openai_chunk never looked at delta.tool_calls, so a streamed tool call produced a span with no "
            "tool events at all; tool-metric.ts:362 then sees claimsAction && !ranTool and reports fabricated_action "
            "at critical severity against a model that did exactly what it was asked",
        )
        self.assertEqual(tools[0].attributes[TOOL_NAME], "get_weather")
        self.assertEqual(
            json.loads(tools[0].attributes[TOOL_ARGS]), {"city": "Paris"},
            "OpenAI streams function arguments as a sequence of partial JSON strings across chunks, so the accumulator "
            "has to concatenate them before the JSON is parseable",
        )
        self.assertEqual(
            tools[0].attributes[TOOL_ID], "call_abc",
            "the tool id arrives only on the first delta of each call and every later delta omits it, so the "
            "accumulator must key on index and not overwrite the id with an empty one",
        )

    def test_streamed_tool_call_records_the_finish_reason(self):
        fc = _openai_sync(self.client, self._weather_stream())
        list(fc.create(model="gpt-4o-mini", messages=[], stream=True))

        span = self._span()
        self.assertEqual(
            span.attributes.get(FINISH), ["tool_calls"],
            "the non-streaming branch records finish_reasons and the streaming branch did not, so you could not tell "
            "a stream that stopped to call a tool from one that hit the token limit",
        )

    def test_two_parallel_tool_calls_stay_separate(self):
        chunks = [
            OStreamToolChunk(tool_calls=[
                OStreamToolCall(0, id="call_1", name="get_weather", arguments=""),
                OStreamToolCall(1, id="call_2", name="get_time", arguments=""),
            ]),
            OStreamToolChunk(tool_calls=[OStreamToolCall(0, arguments='{"city":"Oslo"}')]),
            OStreamToolChunk(tool_calls=[OStreamToolCall(1, arguments='{"tz":"UTC"}')]),
            OStreamToolChunk(finish_reason="tool_calls"),
        ]
        fc = _openai_sync(self.client, chunks)
        list(fc.create(model="gpt-4o-mini", messages=[], stream=True))

        tools = self._tools(self._span())
        self.assertEqual(
            [t.attributes[TOOL_NAME] for t in tools], ["get_weather", "get_time"],
            "parallel tool calls interleave their deltas, so keying the accumulator on delta index is what keeps one "
            "call's arguments out of the other's",
        )
        self.assertEqual(json.loads(tools[0].attributes[TOOL_ARGS]), {"city": "Oslo"})
        self.assertEqual(json.loads(tools[1].attributes[TOOL_ARGS]), {"tz": "UTC"})

    def test_text_and_tool_calls_in_one_stream_do_not_contaminate_each_other(self):
        chunks = [
            OStreamToolChunk(content="Let me check. "),
            OStreamToolChunk(tool_calls=[OStreamToolCall(0, id="call_x", name="lookup", arguments='{"q":1}')]),
            OStreamToolChunk(content="Done."),
            OStreamToolChunk(finish_reason="tool_calls"),
        ]
        fc = _openai_sync(self.client, chunks)
        list(fc.create(model="gpt-4o-mini", messages=[], stream=True))

        span = self._span()
        self.assertEqual(
            span.completion, "Let me check. Done.",
            "tool arguments must never land in the completion text - evaluators score the completion, and a JSON blob "
            "spliced into the prose fails every response-match check",
        )
        self.assertEqual(len(self._tools(span)), 1)

    def test_deprecated_function_call_streaming_is_still_captured(self):
        import types as _t

        chunks = [
            OStreamToolChunk(function_call=_t.SimpleNamespace(name="legacy_fn", arguments="")),
            OStreamToolChunk(function_call=_t.SimpleNamespace(name=None, arguments='{"a":1}')),
            OStreamToolChunk(finish_reason="function_call"),
        ]
        fc = _openai_sync(self.client, chunks)
        list(fc.create(model="gpt-4o-mini", messages=[], stream=True))

        tools = self._tools(self._span())
        self.assertEqual(
            len(tools), 1,
            "delta.function_call is the deprecated single-function form, but apps built before tools= still stream it "
            "and their spans were just as blank",
        )
        self.assertEqual(tools[0].attributes[TOOL_NAME], "legacy_fn")
        self.assertEqual(json.loads(tools[0].attributes[TOOL_ARGS]), {"a": 1})

    def test_stream_records_the_response_id_and_model_it_actually_hit(self):
        fc = _openai_sync(self.client, [
            OStreamToolChunk(content="hi", id="chatcmpl-real", model="gpt-4o-mini-2024-07-18"),
        ])
        list(fc.create(model="gpt-4o-mini", messages=[], stream=True))

        attrs = self._span().attributes
        self.assertEqual(
            attrs.get("gen_ai.response.model"), "gpt-4o-mini-2024-07-18",
            "an alias like gpt-4o-mini resolves to a dated snapshot server-side; without the resolved id a regression "
            "after a silent snapshot rollover is invisible",
        )
        self.assertEqual(attrs.get("gen_ai.response.id"), "chatcmpl-real")

    def test_streamed_refusal_is_not_silently_dropped(self):
        fc = _openai_sync(self.client, [
            OStreamToolChunk(refusal="I can't help with that."),
            OStreamToolChunk(finish_reason="stop"),
        ])
        list(fc.create(model="gpt-4o-mini", messages=[], stream=True))

        span = self._span()
        self.assertEqual(
            span.attributes.get("gen_ai.response.refusal"), "I can't help with that.",
            "on a refusal delta.content is None and the text lives in delta.refusal, so the span recorded an empty "
            "completion and the trace looked like the model returned nothing",
        )

    def test_explicit_patch_captures_streamed_tool_calls_too(self):
        completions = _explicit_openai(self.client, self._weather_stream())
        list(completions.create(model="gpt-4o-mini", messages=[], stream=True))

        tools = self._tools(self._span())
        self.assertEqual(
            len(tools), 1,
            "patch_openai and the class-level autotrace patcher are separate code paths and both were blind; fixing "
            "only one means the same app reports different tool activity depending on how it was instrumented",
        )
        self.assertEqual(tools[0].attributes[TOOL_NAME], "get_weather")

    def test_async_streamed_tool_calls_are_reassembled(self):
        fc = _openai_async(self.client, self._weather_stream())

        async def run():
            stream = await fc.create(model="gpt-4o-mini", messages=[], stream=True)
            return [c async for c in stream]

        asyncio.run(run())

        tools = self._tools(self._span())
        self.assertEqual(len(tools), 1)
        self.assertEqual(json.loads(tools[0].attributes[TOOL_ARGS]), {"city": "Paris"})

    def test_streaming_span_reports_total_tokens_like_the_non_streaming_one(self):
        fc = _openai_sync(self.client, self._weather_stream())
        list(fc.create(model="gpt-4o-mini", messages=[], stream=True))

        span = self._span()
        self.assertEqual(
            span.attributes.get("gen_ai.usage.total_tokens"), 59,
            "the non-streaming branch sets total_tokens and the streaming branch did not, so any dashboard panel "
            "reading that attribute silently treated every streamed call as zero",
        )


class TestAnthropicStreamingToolCalls(StreamToolCallBase):
    def _order_stream(self):
        return [
            AStreamMessageStart(input_tokens=30, cache_read=0, cache_creation=0),
            AStreamBlockStart(0, type="text"),
            AStreamTextDelta("Checking your order."),
            AStreamBlockStart(1, type="tool_use", name="lookup_order", id="toolu_01"),
            AStreamJsonDelta(1, '{"order_id"'),
            AStreamJsonDelta(1, ': "ORD-42"}'),
            AStreamMessageDelta(output_tokens=25, stop_reason="tool_use"),
        ]

    def test_streamed_tool_use_block_is_reassembled(self):
        fm = _anthropic_sync(self.client, self._order_stream())
        list(fm.create(model="claude-sonnet-4-5", messages=[], stream=True))

        span = self._span()
        tools = self._tools(span)
        self.assertEqual(
            len(tools), 1,
            "process_anthropic_chunk handled text_delta and thinking_delta but ignored tool_use blocks entirely, so "
            "every streamed Claude tool call was invisible while the same call made non-streaming was recorded",
        )
        self.assertEqual(tools[0].attributes[TOOL_NAME], "lookup_order")
        self.assertEqual(
            json.loads(tools[0].attributes[TOOL_ARGS]), {"order_id": "ORD-42"},
            "Anthropic sends tool input as input_json_delta.partial_json fragments that only form valid JSON once "
            "concatenated in arrival order",
        )
        self.assertEqual(tools[0].attributes[TOOL_ID], "toolu_01")

    def test_tool_json_does_not_leak_into_the_completion(self):
        fm = _anthropic_sync(self.client, self._order_stream())
        list(fm.create(model="claude-sonnet-4-5", messages=[], stream=True))

        self.assertEqual(
            self._span().completion, "Checking your order.",
            "input_json_delta arrives on the same content_block_delta event type as text_delta, so routing it by "
            "delta.type is what keeps the tool payload out of the answer text",
        )

    def test_streamed_stop_reason_is_recorded(self):
        fm = _anthropic_sync(self.client, self._order_stream())
        list(fm.create(model="claude-sonnet-4-5", messages=[], stream=True))

        self.assertEqual(
            self._span().attributes.get(FINISH), ["tool_use"],
            "stop_reason lives on message_delta.delta and the processor only ever read message_delta.usage, so a "
            "streamed Claude call never reported why it stopped",
        )

    def test_server_side_tool_use_is_recorded_as_a_tool_call(self):
        events = [
            AStreamMessageStart(input_tokens=10),
            AStreamBlockStart(0, type="server_tool_use", name="web_search", id="srvtoolu_1"),
            AStreamJsonDelta(0, '{"query":"weather"}'),
            AStreamMessageDelta(output_tokens=5, stop_reason="end_turn"),
        ]
        fm = _anthropic_sync(self.client, events)
        list(fm.create(model="claude-sonnet-4-5", messages=[], stream=True))

        span = self._span()
        tools = self._tools(span)
        self.assertEqual(
            len(tools), 1,
            "server_tool_use is how Anthropic reports a tool it ran on its own side, such as web_search; it is real "
            "billed work and dropping it made the model look like it answered from nothing",
        )
        self.assertEqual(tools[0].attributes[TOOL_NAME], "web_search")
        self.assertTrue(
            span.attributes.get("gen_ai.response.has_server_side_tool_use"),
            "a server-side tool has no matching client-side execution span, so the span has to say so or a duplicate "
            "detector will read the missing pair as a fabricated action",
        )

    def test_message_delta_without_input_usage_does_not_erase_the_cache_buckets(self):
        events = [
            AStreamMessageStart(input_tokens=100, cache_read=800, cache_creation=200),
            AStreamTextDelta("cached answer"),
            AStreamMessageDelta(output_tokens=12, stop_reason="end_turn"),
        ]
        fm = _anthropic_sync(self.client, events)
        list(fm.create(model="claude-sonnet-4-5", messages=[], stream=True))

        span = self._span()
        self.assertEqual(
            span.tokens_in, 1100,
            "Anthropic reports input_tokens EXCLUSIVE of the cache buckets while the dashboard subtracts them back "
            "out, so message_start has to fold them in",
        )
        self.assertEqual(
            span.cache_read_tokens, 800,
            "MessageDeltaUsage declares cache_read_input_tokens as Optional and Anthropic usually omits it, so "
            "re-reading usage on message_delta must not overwrite a known bucket with a missing one",
        )
        self.assertEqual(span.cache_creation_tokens, 200)
        self.assertEqual(span.tokens_out, 12)

    def test_message_delta_that_does_carry_cumulative_input_usage_is_believed(self):
        events = [
            AStreamMessageStart(input_tokens=100, cache_read=0, cache_creation=0),
            AStreamTextDelta("hi"),
            AStreamMessageDelta(output_tokens=9, stop_reason="end_turn",
                                input_tokens=150, cache_read=50, cache_creation=0),
        ]
        fm = _anthropic_sync(self.client, events)
        list(fm.create(model="claude-sonnet-4-5", messages=[], stream=True))

        span = self._span()
        self.assertEqual(
            span.tokens_in, 200,
            "MessageDeltaUsage is documented as cumulative, so when the final event does report input usage it is the "
            "authoritative number and still has to have its cache buckets folded in",
        )
        self.assertEqual(span.cache_read_tokens, 50)

    def test_stream_records_the_message_id_and_resolved_model(self):
        fm = _anthropic_sync(self.client, [
            AStreamMessageStart(input_tokens=1, id="msg_01XYZ", model="claude-sonnet-4-5-20250929"),
            AStreamTextDelta("ok"),
            AStreamMessageDelta(output_tokens=1, stop_reason="end_turn"),
        ])
        list(fm.create(model="claude-sonnet-4-5", messages=[], stream=True))

        attrs = self._span().attributes
        self.assertEqual(attrs.get("gen_ai.response.id"), "msg_01XYZ")
        self.assertEqual(attrs.get("gen_ai.response.model"), "claude-sonnet-4-5-20250929")

    def test_two_tool_blocks_keyed_by_block_index_stay_separate(self):
        events = [
            AStreamMessageStart(input_tokens=5),
            AStreamBlockStart(0, type="tool_use", name="first", id="t0"),
            AStreamJsonDelta(0, '{"a":1}'),
            AStreamBlockStart(1, type="tool_use", name="second", id="t1"),
            AStreamJsonDelta(1, '{"b":2}'),
            AStreamMessageDelta(output_tokens=3, stop_reason="tool_use"),
        ]
        fm = _anthropic_sync(self.client, events)
        list(fm.create(model="claude-sonnet-4-5", messages=[], stream=True))

        tools = self._tools(self._span())
        self.assertEqual([t.attributes[TOOL_NAME] for t in tools], ["first", "second"])
        self.assertEqual(json.loads(tools[0].attributes[TOOL_ARGS]), {"a": 1})
        self.assertEqual(json.loads(tools[1].attributes[TOOL_ARGS]), {"b": 2})


class TestAbandonedAndCancelledStreams(StreamToolCallBase):
    def test_a_stream_abandoned_early_still_emits_what_it_saw(self):
        chunks = [
            OStreamToolChunk(tool_calls=[OStreamToolCall(0, id="c1", name="partial_tool", arguments='{"x"')]),
            OStreamToolChunk(content="never read"),
        ]
        fc = _openai_sync(self.client, chunks)

        for _ in fc.create(model="gpt-4o-mini", messages=[], stream=True):
            break

        span = self._span()
        self.assertEqual(
            span.status, "OK",
            "breaking out of the loop closes the generator and raises GeneratorExit at the yield; abandoning a stream "
            "early is normal client behaviour, so it must not be reported as a failed call",
        )
        self.assertEqual(len(self._tools(span)), 1)

    def test_a_cancelled_async_stream_is_reported_as_an_error(self):
        fc = _openai_async(self.client, [OStreamToolChunk(content="a"), OStreamToolChunk(content="b")])

        async def run():
            stream = await fc.create(model="gpt-4o-mini", messages=[], stream=True)
            async for _ in stream:
                raise asyncio.CancelledError("client went away")

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(run())

        span = self._span()
        self.assertEqual(
            span.status, "ERROR",
            "asyncio.CancelledError inherits from BaseException, not Exception, so an except Exception handler let a "
            "cancelled stream fall through and record status OK - a timed-out or client-aborted request looked "
            "successful in the dashboard",
        )


if __name__ == "__main__":
    unittest.main()
