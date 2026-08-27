import asyncio
import types
import unittest

from tests.helpers import fresh, reset_context

TTFT = "gen_ai.server.time_to_first_token_ms"
THINKING = "gen_ai.response.thinking"
FINISH = "gen_ai.response.finish_reasons"


def _ns(**kw):
    return types.SimpleNamespace(**kw)


def _delta(content=None, tool_calls=None):
    return _ns(content=content, tool_calls=tool_calls)


def _chunk(delta=None, finish=None, usage=None, cid="cmpl-1", model="mistral-large-latest"):
    choices = [_ns(index=0, delta=delta if delta is not None else _delta(), finish_reason=finish)]
    return _ns(data=_ns(id=cid, model=model, choices=choices, usage=usage))


def _usage(pin, pout):
    return _ns(prompt_tokens=pin, completion_tokens=pout, total_tokens=pin + pout)


def _tool_call(index=0, call_id="null", name=None, arguments=None):
    function = _ns()
    if name is not None:
        function.name = name
    if arguments is not None:
        function.arguments = arguments
    return _ns(index=index, id=call_id, type="function", function=function)


def _thinking_part(text):
    return _ns(type="thinking", thinking=[_ns(type="text", text=text)])


class _FakeChat:
    def __init__(self, *, result=None, chunks=None, error=None, raise_on_open=None):
        self._result = result
        self._chunks = chunks or []
        self._error = error
        self._raise_on_open = raise_on_open

    def complete(self, **k):
        if self._raise_on_open is not None:
            raise self._raise_on_open
        return self._result

    async def complete_async(self, **k):
        if self._raise_on_open is not None:
            raise self._raise_on_open
        return self._result

    def stream(self, **k):
        if self._raise_on_open is not None:
            raise self._raise_on_open
        return self._gen()

    async def stream_async(self, **k):
        if self._raise_on_open is not None:
            raise self._raise_on_open
        return self._agen()

    def _gen(self):
        for c in self._chunks:
            yield c
        if self._error is not None:
            raise self._error

    async def _agen(self):
        for c in self._chunks:
            yield c
        if self._error is not None:
            raise self._error


class TestMistral(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def _install(self, **kw):
        from evalkit.instrumentation.llm.mistral import patch_mistral_chat

        chat = _FakeChat(**kw)
        patch_mistral_chat(self.client, chat)
        return chat

    def _llm(self):
        self.client.flush()
        spans = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(spans), 1, "a mistral call must emit exactly one llm_call span")
        return spans[0]

    def test_unary_complete_span(self):
        chat = self._install(result=_ns(
            id="cmpl-x", model="mistral-small-latest",
            choices=[_ns(index=0, message=_ns(content="the answer", tool_calls=None), finish_reason="stop")],
            usage=_usage(6, 3),
        ))
        chat.complete(model="mistral-small-latest", messages=[{"role": "user", "content": "hi"}])
        span = self._llm()
        self.assertEqual(span.provider, "mistral")
        self.assertEqual(span.completion, "the answer")
        self.assertEqual((span.tokens_in, span.tokens_out), (6, 3))
        self.assertEqual(span.attributes.get(FINISH), ["stop"])

    def test_unary_thinking_content_is_not_a_list_in_the_completion(self):
        chat = self._install(result=_ns(
            id="cmpl-x", model="magistral-medium-latest",
            choices=[_ns(index=0, message=_ns(
                content=[_thinking_part("step one"), _ns(type="text", text="final answer")],
                tool_calls=None), finish_reason="stop")],
            usage=_usage(4, 9),
        ))
        chat.complete(model="magistral-medium-latest", messages=[])
        span = self._llm()
        self.assertEqual(span.completion, "final answer",
                         "mistral's thinking models return content as a list of chunks; assigning the list itself "
                         "to the completion produced a Python repr in the trace instead of the model's text")
        self.assertEqual(span.attributes.get(THINKING), "step one")

    def test_unary_complete_async_records_finish_reason(self):
        chat = self._install(result=_ns(
            id="cmpl-a", model="mistral-large-latest",
            choices=[_ns(index=0, message=_ns(content="async answer", tool_calls=None), finish_reason="stop")],
            usage=_usage(2, 5),
        ))
        asyncio.run(chat.complete_async(model="mistral-large-latest", messages=[]))
        span = self._llm()
        self.assertEqual(span.completion, "async answer")
        self.assertEqual(span.attributes.get(FINISH), ["stop"],
                         "the async unary path was missing the finish-reason block the sync one had, so async "
                         "callers could not tell a truncated response from a complete one")

    def test_stream_records_text_usage_and_ttft(self):
        chat = self._install(chunks=[
            _chunk(_delta(content="Hel")),
            _chunk(_delta(content="lo")),
            _chunk(_delta(), finish="stop", usage=_usage(7, 2)),
        ])
        list(chat.stream(model="mistral-large-latest", messages=[]))
        span = self._llm()
        self.assertEqual(span.status, "OK")
        self.assertEqual(span.completion, "Hello",
                         "mistral yields CompletionEvent objects wrapping the chunk under `.data`; reading the "
                         "event as if it were the chunk left every streamed completion empty")
        self.assertEqual((span.tokens_in, span.tokens_out), (7, 2))
        self.assertEqual(span.attributes.get(FINISH), ["stop"])
        self.assertEqual(span.attributes.get("gen_ai.response.id"), "cmpl-1")
        self.assertEqual(span.attributes.get("gen_ai.response.model"), "mistral-large-latest")
        self.assertTrue(span.attributes.get("gen_ai.streaming"))
        self.assertIsInstance(span.attributes.get(TTFT), int,
                             "TTFT must serialize as a number; trace-service2 reads it with JSONExtractFloat "
                             "and a string comes back as 0")

    def test_stream_separates_thinking_from_the_completion(self):
        chat = self._install(chunks=[
            _chunk(_delta(content=[_thinking_part("reasoning")])),
            _chunk(_delta(content="answer")),
            _chunk(_delta(), finish="stop", usage=_usage(3, 8)),
        ])
        list(chat.stream(model="magistral-medium-latest", messages=[]))
        span = self._llm()
        self.assertEqual(span.completion, "answer")
        self.assertEqual(span.attributes.get(THINKING), "reasoning")

    def test_stream_emits_one_tool_event_with_accumulated_arguments(self):
        chat = self._install(chunks=[
            _chunk(_delta(tool_calls=[_tool_call(0, "tc_1", name="get_weather", arguments="")])),
            _chunk(_delta(tool_calls=[_tool_call(0, "null", arguments='{"city":')])),
            _chunk(_delta(tool_calls=[_tool_call(0, "null", arguments='"SF"}')])),
            _chunk(_delta(), finish="tool_calls", usage=_usage(5, 4)),
        ])
        list(chat.stream(model="mistral-large-latest", messages=[]))
        span = self._llm()
        events = [e for e in (span.events or []) if e.name == "gen_ai.tool.call"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].attributes["gen_ai.tool.name"], "get_weather")
        self.assertEqual(events[0].attributes["gen_ai.tool.call.arguments"], '{"city":"SF"}')
        self.assertEqual(events[0].attributes["gen_ai.tool.call.id"], "tc_1",
                         "mistral defaults a streamed tool call's id to the literal string 'null'; storing that "
                         "over the real id makes tool results impossible to correlate")

    def test_tool_only_stream_has_no_time_to_first_token(self):
        chat = self._install(chunks=[
            _chunk(_delta(tool_calls=[_tool_call(0, "tc_1", name="get_weather", arguments='{"city":"SF"}')])),
            _chunk(_delta(), finish="tool_calls", usage=_usage(5, 4)),
        ])
        list(chat.stream(model="mistral-large-latest", messages=[]))
        span = self._llm()
        self.assertNotIn(TTFT, span.attributes,
                         "tool-argument deltas are not user-visible tokens; counting them as the first token makes "
                         "tool-calling models look faster than they are")

    def test_zero_usage_in_a_later_chunk_does_not_erase_reported_counts(self):
        chat = self._install(chunks=[
            _chunk(_delta(content="hi"), usage=_usage(9, 1)),
            _chunk(_delta(), finish="stop", usage=_usage(0, 0)),
        ])
        list(chat.stream(model="mistral-large-latest", messages=[]))
        span = self._llm()
        self.assertEqual((span.tokens_in, span.tokens_out), (9, 1),
                         "mistral's UsageInfo fields default to 0, so a trailing chunk carrying an empty usage "
                         "object would zero out real counts and make the call look free")

    def test_failed_stream_keeps_partials_and_drops_ttft(self):
        chat = self._install(
            chunks=[_chunk(_delta(content="Hel"), usage=_usage(4, 1))],
            error=RuntimeError("upstream closed"),
        )
        with self.assertRaises(RuntimeError):
            list(chat.stream(model="mistral-large-latest", messages=[]))
        span = self._llm()
        self.assertEqual(span.status, "ERROR")
        self.assertIn("upstream closed", span.status_message or "")
        self.assertEqual(span.completion, "Hel")
        self.assertEqual(span.tokens_in, 4,
                         "the provider bills for tokens it already streamed")
        self.assertNotIn(TTFT, span.attributes,
                         "a truncated stream must stay out of the TTFT percentiles")

    def test_abandoned_stream_stays_ok_and_keeps_ttft(self):
        chat = self._install(chunks=[
            _chunk(_delta(content="Hel")),
            _chunk(_delta(content="lo")),
            _chunk(_delta(), finish="stop", usage=_usage(4, 2)),
        ])
        stream = chat.stream(model="mistral-large-latest", messages=[])
        it = iter(stream)
        next(it)
        del it
        del stream
        span = self._llm()
        self.assertEqual(span.status, "OK",
                         "a caller that stops reading has not hit a provider error")
        self.assertIn(TTFT, span.attributes)

    def test_stream_call_that_raises_immediately_emits_an_error_span(self):
        chat = self._install(raise_on_open=ValueError("bad api key"))
        with self.assertRaises(ValueError):
            chat.stream(model="mistral-large-latest", messages=[])
        span = self._llm()
        self.assertEqual(span.status, "ERROR")
        self.assertIn("bad api key", span.status_message or "")

    def test_stream_async_is_awaited_before_tracing(self):
        chat = self._install(chunks=[
            _chunk(_delta(content="async ")),
            _chunk(_delta(content="hello")),
            _chunk(_delta(), finish="stop", usage=_usage(2, 3)),
        ])

        async def run():
            stream = await chat.stream_async(model="mistral-large-latest", messages=[])
            async for _ in stream:
                pass

        asyncio.run(run())
        span = self._llm()
        self.assertEqual(span.status, "OK")
        self.assertEqual(span.completion, "async hello",
                         "mistral's stream_async is a coroutine that returns the event stream, so the wrapper has "
                         "to await it; treating it like an async generator broke the caller's `await`")
        self.assertEqual((span.tokens_in, span.tokens_out), (2, 3))
        self.assertEqual(span.operation, "chat.stream_async")

    def test_failed_async_stream_reports_error(self):
        chat = self._install(
            chunks=[_chunk(_delta(content="par"), usage=_usage(3, 1))],
            error=ValueError("async reset"),
        )

        async def run():
            stream = await chat.stream_async(model="mistral-large-latest", messages=[])
            async for _ in stream:
                pass

        with self.assertRaises(ValueError):
            asyncio.run(run())
        span = self._llm()
        self.assertEqual(span.status, "ERROR")
        self.assertIn("async reset", span.status_message or "")
        self.assertEqual(span.completion, "par")
        self.assertNotIn(TTFT, span.attributes)

    def test_auto_class_patch_upgrades_the_chat_instance(self):
        from evalkit.instrumentation.auto import _patch_mistral_chat_class

        class _StreamChat(_FakeChat):
            pass

        chunks = [
            _chunk(_delta(content="via ")),
            _chunk(_delta(content="init")),
            _chunk(_delta(), finish="stop", usage=_usage(1, 2)),
        ]
        _patch_mistral_chat_class(self.client, _StreamChat)
        chat = _StreamChat(chunks=chunks)
        list(chat.stream(model="mistral-large-latest", messages=[]))
        span = self._llm()
        self.assertEqual(span.completion, "via init",
                         "init() patches the Chat class, not a client instance; without the class-level seam the "
                         "auto-instrumented path gets no mistral streaming at all")
        self.assertEqual(span.operation, "chat.stream")

    def test_auto_class_patch_traces_unary_complete(self):
        from evalkit.instrumentation.auto import _patch_mistral_chat_class

        class _UnaryChat(_FakeChat):
            pass

        result = _ns(
            id="cmpl-auto", model="mistral-small-latest",
            choices=[_ns(index=0, message=_ns(content="auto answer", tool_calls=None), finish_reason="stop")],
            usage=_usage(3, 4),
        )
        _patch_mistral_chat_class(self.client, _UnaryChat)
        chat = _UnaryChat(result=result)
        chat.complete(model="mistral-small-latest", messages=[])
        span = self._llm()
        self.assertEqual(span.completion, "auto answer")
        self.assertEqual((span.tokens_in, span.tokens_out), (3, 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
