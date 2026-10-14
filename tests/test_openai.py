import asyncio
import json
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import evalkit
from tests.helpers import (
    OResp, OStreamChunk, ORespResult,
    fresh, reset_context, responses_sse,
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

    def test_streaming_captures_time_to_first_token(self):
        chunks = [
            OStreamChunk("Hel"),
            OStreamChunk("lo!", usage=types.SimpleNamespace(prompt_tokens=5, completion_tokens=10)),
        ]
        fc = install_fake_openai_sync(self.client, chunks)
        list(fc.create(model="gpt-4o-mini", messages=[], stream=True))
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        ttft = llm.attributes.get("gen_ai.server.time_to_first_token_ms")
        self.assertIsNotNone(ttft)
        self.assertIsInstance(ttft, int)
        self.assertGreaterEqual(ttft, 0)

    def test_non_streaming_has_no_ttft(self):
        fc = install_fake_openai_sync(self.client, OResp("hello"))
        fc.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertNotIn("gen_ai.server.time_to_first_token_ms", llm.attributes)

    def test_reasoning_tokens_captured(self):
        fc = install_fake_openai_sync(self.client, OResp("42", pin=5, pout=8, reasoning_tokens=120))
        fc.create(model="o3-mini", messages=[{"role": "user", "content": "what is 6*7?"}])
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(llm[0].attributes.get("gen_ai.usage.reasoning_tokens"), 120)

    def test_no_reasoning_tokens_key_when_absent(self):
        fc = install_fake_openai_sync(self.client, OResp("hello", pin=5, pout=8))
        fc.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertNotIn("gen_ai.usage.reasoning_tokens", llm[0].attributes)

    def test_reasoning_tokens_captured_streaming(self):
        final_usage = types.SimpleNamespace(
            prompt_tokens=5, completion_tokens=10,
            completion_tokens_details=types.SimpleNamespace(reasoning_tokens=64),
        )
        chunks = [OStreamChunk("42"), OStreamChunk("", usage=final_usage)]
        fc = install_fake_openai_sync(self.client, chunks)
        list(fc.create(model="o3-mini", messages=[], stream=True))
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(llm[0].attributes.get("gen_ai.usage.reasoning_tokens"), 64)

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

    def test_responses_reasoning_summary_and_tokens(self):
        resp = ORespResult("42", pin=6, pout=9)
        resp.output.insert(0, types.SimpleNamespace(
            type="reasoning", summary=[types.SimpleNamespace(type="summary_text", text="Six times seven.")]))
        resp.usage.output_tokens_details = types.SimpleNamespace(reasoning_tokens=5)
        resp.usage.input_tokens_details = types.SimpleNamespace(cached_tokens=2)
        install_fake_openai_responses_sync(self.client, resp).create(model="o4-mini", input="6*7?")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(llm.completion, "42")
        self.assertEqual(llm.attributes["gen_ai.response.thinking"], "Six times seven.")
        self.assertEqual(llm.attributes["gen_ai.usage.reasoning_tokens"], 5)
        self.assertEqual(llm.cache_read_tokens, 2)
        self.assertNotIn("gen_ai.streaming", llm.attributes)

    def test_reading_the_request_never_fails_the_call(self):
        from evalkit.instrumentation import auto as _auto

        fc = install_fake_openai_responses_sync(self.client, ORespResult("fine"))
        reply = fc.create(model="gpt-4o", input=[{"role": "user", "content": [{"type": "input_text", "text": None}]}])
        self.assertEqual(reply.output_text, "fine")
        original = _auto._responses_prompt
        _auto._responses_prompt = lambda kwargs: 1 / 0
        try:
            self.assertEqual(fc.create(model="gpt-4o", input="hi").output_text, "fine")
        finally:
            _auto._responses_prompt = original
        self.client.flush()
        self.assertEqual(len([e for e in self.captured if e.span_type == "llm_call"]), 1,
                         "the unreadable part is empty text; a failing span start sends the call untraced")

    def test_responses_tool_call_event(self):
        resp = ORespResult(tool=("get_weather", '{"city":"Paris"}'))
        fc = install_fake_openai_responses_sync(self.client, resp)
        fc.create(model="gpt-4o", input="Weather in Paris?")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertTrue(llm.events)
        self.assertEqual(llm.events[0].attributes["gen_ai.tool.name"], "get_weather")
        tools = [e for e in self.captured if e.span_type == "tool_call"]
        self.assertEqual(len(tools), 0)


class _ResponsesServer(BaseHTTPRequestHandler):
    """POST /v1/responses as the API streams it; the model name picks how the stream goes."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, status, data, content_type, length=None):
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(data) if length is None else length))
        self.send_header("x-request-id", "req_local")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        model = body.get("model")
        if model == "fail-401":
            self._send(401, json.dumps({"error": {"message": "invalid api key", "type": "invalid_request_error"}})
                       .encode(), "application/json")
            return
        if model == "cut":
            whole = responses_sse().encode()
            # The length of the whole stream, then the connection closes after its first delta.
            self._send(200, responses_sse(cut_after=5).encode(), "text/event-stream", length=len(whole))
            self.close_connection = True
            return
        if not body.get("stream"):
            self._send(400, b'{"error": {"message": "stream only"}}', "application/json")
            return
        tool = ("lookup_order", '{"order_id": "1001"}') if model == "tool" else None
        reasoning = "The user asks about order 1001." if model == "reasoning" else None
        end = {"failed": "response.failed", "error-event": "error"}.get(model, "response.completed")
        self._send(200, responses_sse(tool=tool, end=end, reasoning=reasoning).encode(), "text/event-stream")


class TestOpenAIResponsesStreams(unittest.TestCase):
    """A streamed Responses call, on the sync and the async client, is one llm_call span."""

    INPUT = [{"role": "user", "content": "Where is order 1001?"}]

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _ResponsesServer)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self.async_clients = []

    def openai(self, async_=False):
        import openai

        cls = openai.AsyncOpenAI if async_ else openai.OpenAI
        client = cls(base_url=self.base, api_key="sk-test-not-a-real-key-000000", max_retries=0)
        if async_:
            self.async_clients.append(client)
        return client

    def arun(self, make):
        """Run a coroutine function, closing the async clients it made on its own loop."""
        async def main():
            try:
                return await make()
            finally:
                while self.async_clients:
                    await self.async_clients.pop().close()

        return asyncio.run(main())

    def _llm(self):
        self.client.flush()
        return [e for e in self.captured if e.span_type == "llm_call"]

    def _assert_streamed_reply(self, span):
        self.assertEqual(span.operation, "responses.create")
        self.assertEqual(span.provider, "openai")
        self.assertEqual(span.status, "OK")
        self.assertEqual(span.model, "gpt-4o-mini")
        self.assertEqual(span.prompt, "system: Be brief.\nuser: Where is order 1001?")
        self.assertEqual(span.completion, "Order 1001 shipped.")
        self.assertEqual((span.tokens_in, span.tokens_out, span.cache_read_tokens), (21, 7, 4))
        attrs = span.attributes
        self.assertTrue(attrs["gen_ai.streaming"])
        self.assertIsInstance(attrs["gen_ai.server.time_to_first_token_ms"], int)
        self.assertGreaterEqual(attrs["gen_ai.server.time_to_first_token_ms"], 0)
        self.assertEqual(attrs["gen_ai.usage.total_tokens"], 28)
        self.assertEqual(attrs["gen_ai.usage.reasoning_tokens"], 2)
        self.assertEqual(attrs["gen_ai.response.id"], "resp_local")
        self.assertEqual(attrs["gen_ai.response.model"], "gpt-4o-mini-2024-07-18")
        self.assertEqual(attrs["gen_ai.request.max_tokens"], 64)

    def _create(self, client, model="gpt-4o-mini", **extra):
        return client.responses.create(model=model, instructions="Be brief.", input=self.INPUT,
                                       max_output_tokens=64, stream=True, **extra)

    def test_sync_create_stream(self):
        import openai

        stream = self._create(self.openai())
        self.assertIsInstance(stream, openai.Stream)
        self.assertEqual(stream.response.headers["x-request-id"], "req_local")
        self.assertEqual(next(stream).type, "response.created")
        types_seen = [event.type for event in stream]
        self.assertEqual(types_seen[-1], "response.completed")
        spans = self._llm()
        self.assertEqual(len(spans), 1)
        self._assert_streamed_reply(spans[0])

    def test_async_create_stream(self):
        import openai

        async def run():
            stream = await self._create(self.openai(async_=True))
            self.assertIsInstance(stream, openai.AsyncStream)
            self.assertEqual(stream.response.headers["x-request-id"], "req_local")
            first = await anext(stream)
            rest = [event.type async for event in stream]
            return first.type, rest

        first, rest = self.arun(run)
        self.assertEqual((first, rest[-1]), ("response.created", "response.completed"))
        spans = self._llm()
        self.assertEqual(len(spans), 1)
        self._assert_streamed_reply(spans[0])

    def test_async_stream_helper(self):
        async def run():
            client = self.openai(async_=True)
            async with client.responses.stream(model="gpt-4o-mini", instructions="Be brief.", input=self.INPUT,
                                               max_output_tokens=64) as stream:
                deltas = [event.delta async for event in stream if event.type == "response.output_text.delta"]
                final = await stream.get_final_response()
            async with client.responses.stream(model="gpt-4o-mini", instructions="Be brief.", input=self.INPUT,
                                               max_output_tokens=64) as stream:
                await stream.until_done()
            return deltas, final

        deltas, final = self.arun(run)
        self.assertEqual("".join(deltas), "Order 1001 shipped.")
        self.assertEqual(final.output_text, "Order 1001 shipped.")
        spans = self._llm()
        self.assertEqual(len(spans), 2)
        for span in spans:
            self._assert_streamed_reply(span)

    def test_sync_stream_helper(self):
        with self.openai().responses.stream(model="gpt-4o-mini", instructions="Be brief.", input=self.INPUT,
                                            max_output_tokens=64) as stream:
            final = stream.get_final_response()
        self.assertEqual(final.output_text, "Order 1001 shipped.")
        spans = self._llm()
        self.assertEqual(len(spans), 1)
        self._assert_streamed_reply(spans[0])

    def test_tool_calls_are_assembled(self):
        async def run_async():
            return [event async for event in await self._create(self.openai(async_=True), model="tool")]

        list(self._create(self.openai(), model="tool"))
        self.arun(run_async)
        spans = self._llm()
        self.assertEqual(len(spans), 2)
        for span in spans:
            calls = [e.attributes for e in span.events or [] if e.name == "gen_ai.tool.call"]
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["gen_ai.tool.name"], "lookup_order")
            self.assertEqual(json.loads(calls[0]["gen_ai.tool.call.arguments"]), {"order_id": "1001"})
            self.assertEqual(calls[0]["gen_ai.tool.call.id"], "call_local")

    def test_reasoning_is_recorded_beside_the_completion(self):
        async def run_async():
            async with self.openai(async_=True).responses.stream(model="reasoning", input=self.INPUT) as stream:
                return await stream.get_final_response()

        list(self._create(self.openai(), model="reasoning"))
        final = self.arun(run_async)
        self.assertEqual(final.output_text, "Order 1001 shipped.")
        spans = self._llm()
        self.assertEqual(len(spans), 2)
        for span in spans:
            self.assertEqual(span.completion, "Order 1001 shipped.")
            self.assertEqual(span.attributes["gen_ai.response.thinking"], "The user asks about order 1001.")
            self.assertIsInstance(span.attributes["gen_ai.server.time_to_first_token_ms"], int)

    def test_a_stream_that_fails_to_open_is_an_error_span(self):
        import openai

        async def run_async():
            client = self.openai(async_=True)
            for call in (lambda: self._create(client, model="fail-401"),
                         lambda: client.responses.stream(model="fail-401", input=self.INPUT).__aenter__()):
                try:
                    await call()
                    self.fail("the provider's error was swallowed")
                except openai.AuthenticationError as exc:
                    self.assertIs(type(exc), openai.AuthenticationError)

        with self.assertRaises(openai.AuthenticationError):
            self._create(self.openai(), model="fail-401")
        self.arun(run_async)
        spans = self._llm()
        self.assertEqual(len(spans), 3)
        for span in spans:
            self.assertEqual(span.status, "ERROR")
            self.assertEqual(span.model, "fail-401")
            self.assertIn("user: Where is order 1001?", span.prompt)
            self.assertIn("invalid api key", span.status_message)
            self.assertEqual(span.attributes["exception.type"], "AuthenticationError")
            self.assertIn("exception.stacktrace", span.attributes)

    def test_a_stream_cut_off_keeps_the_partial_text(self):
        async def run_async():
            stream = await self._create(self.openai(async_=True), model="cut")
            seen = []
            try:
                async for event in stream:
                    seen.append(event.type)
            except Exception as exc:
                return seen, exc
            return seen, None

        stream = self._create(self.openai(), model="cut")
        seen, raised = [], None
        try:
            for event in stream:
                seen.append(event.type)
        except Exception as exc:
            raised = exc
        async_seen, async_raised = self.arun(run_async)
        self.assertIsNotNone(raised, "the cut connection reaches the caller")
        self.assertIsNotNone(async_raised)
        self.assertEqual(seen[-1], "response.output_text.delta")
        self.assertEqual(async_seen, seen)
        spans = self._llm()
        self.assertEqual(len(spans), 2)
        for span in spans:
            self.assertEqual(span.status, "ERROR")
            self.assertEqual(span.completion, "Order 1001 ")
            self.assertTrue(span.status_message)

    def test_a_response_that_fails_inside_the_stream_is_an_error(self):
        async def run_async():
            stream = await self._create(self.openai(async_=True), model="error-event")
            try:
                return [event.type async for event in stream]
            except Exception as exc:
                return exc

        events = [event.type for event in self._create(self.openai(), model="failed")]
        self.assertEqual(events[-1], "response.failed")
        self.arun(run_async)
        spans = self._llm()
        self.assertEqual(len(spans), 2)
        self.assertEqual([s.status for s in spans], ["ERROR", "ERROR"])
        self.assertEqual(spans[0].status_message, "the model crashed")
        self.assertEqual(spans[0].completion, "Order 1001 shipped.")
        self.assertTrue(spans[1].status_message)

    def test_raw_responses_are_handed_back_untouched(self):
        import openai

        async def run():
            raw = await self.openai(async_=True).responses.with_raw_response.create(
                model="gpt-4o-mini", input=self.INPUT, stream=True)
            parsed = raw.parse()
            self.assertIsInstance(parsed, openai.AsyncStream)
            return raw.headers["x-request-id"], [event.type async for event in parsed]

        request_id, events = self.arun(run)
        self.assertEqual(request_id, "req_local")
        self.assertEqual(events[-1], "response.completed")

    def test_a_failing_event_reader_never_breaks_the_stream(self):
        from evalkit.instrumentation import _streaming

        original = _streaming._StreamState.process_openai_responses_event

        def broken(*a, **k):
            raise RuntimeError("evalkit bug")

        _streaming._StreamState.process_openai_responses_event = broken
        try:
            async def run():
                return [event.type async for event in await self._create(self.openai(async_=True))]

            sync_events = [event.type for event in self._create(self.openai())]
            async_events = self.arun(run)
        finally:
            _streaming._StreamState.process_openai_responses_event = original
        self.assertEqual(sync_events[-1], "response.completed")
        self.assertEqual(async_events, sync_events)
        self.assertEqual(len(self._llm()), 2, "the span still ends, with what could be read")


class TestResponsesStreamState(unittest.TestCase):
    def test_tool_calls_come_from_the_deltas_when_the_final_response_never_arrived(self):
        from evalkit.instrumentation._streaming import _StreamState, stream_tool_events

        ns = types.SimpleNamespace
        state = _StreamState(0.0)
        for event in (
            ns(type="response.output_item.added", output_index=1,
               item=ns(type="function_call", call_id="call_9", id="fc_9", name="refund", arguments="")),
            ns(type="response.function_call_arguments.delta", output_index=1, delta='{"order'),
            ns(type="response.function_call_arguments.delta", output_index=1, delta='": "1001"}'),
        ):
            state.process_openai_responses_event(event)
        self.assertIsNone(state.ttft_ms, "tool arguments are not generated text")
        events = stream_tool_events(state, "2026-01-01T00:00:00Z")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].attributes["gen_ai.tool.name"], "refund")
        self.assertEqual(events[0].attributes["gen_ai.tool.call.arguments"], '{"order": "1001"}')
        self.assertEqual(events[0].attributes["gen_ai.tool.call.id"], "call_9")

    def test_text_refusal_and_reasoning_end_the_wait_for_a_first_token(self):
        from evalkit.instrumentation._streaming import _StreamState

        ns = types.SimpleNamespace
        for event in (ns(type="response.output_text.delta", delta="Hi"),
                      ns(type="response.refusal.delta", delta="No."),
                      ns(type="response.reasoning_summary_text.delta", delta="Thinking")):
            state = _StreamState(0.0)
            state.process_openai_responses_event(event)
            self.assertIsNotNone(state.ttft_ms, event.type)
        self.assertEqual(state.chunks, [], "reasoning is not part of the completion")
        self.assertEqual(state.thinking_chunks, ["Thinking"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
