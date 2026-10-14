"""Instrumentation never changes what the instrumented call does.

1. A failure inside EvalKit's own hooks never reaches the caller; the call runs and returns as usual.
2. A provider's exception reaches the caller unchanged: the same object, type and traceback.
3. Stream wrappers are transparent: the provider's methods, attributes, context managers and
   isinstance checks all keep working, exercised here on the real OpenAI and Anthropic SDKs
   against a local server.
4. Chunks pass through as they arrive; early break, close(), garbage collection and cancellation
   end the span once; what the span keeps of a stream is bounded.
5. The caller's request (kwargs, messages, headers) is never changed or consumed.
6. Patching is idempotent and tolerates a client class that lacks what a newer version has.
7. Recording a span never waits on the network.
"""
import asyncio
import gc
import json
import sys
import threading
import time
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tests.helpers import OStreamChunk, fresh, reset_context

MESSAGES = [{"role": "user", "content": "Where is order 1001?"}]


def _openai_chunk(delta, finish=None, usage=None):
    body = {"id": "chatcmpl-local", "object": "chat.completion.chunk", "created": 0, "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        body = {**body, "choices": [], "usage": usage}
    return "data: " + json.dumps(body) + "\n\n"


def _anthropic_events():
    events = [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_local", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5",
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": "Order 1001 "}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": "shipped."}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                           "usage": {"output_tokens": 6}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


class _Provider(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        type(self).requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
        if body.get("model") == "fail-401":
            payload = json.dumps({"error": {"message": "invalid api key", "type": "authentication_error"}}).encode()
            self.send_response(401)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path.endswith("/messages"):
            text = _anthropic_events()
        else:
            text = (_openai_chunk({"role": "assistant", "content": ""}) + _openai_chunk({"content": "Hello "})
                    + _openai_chunk({"content": "there."}) + _openai_chunk({}, finish="stop"))
            if (body.get("stream_options") or {}).get("include_usage"):
                text += _openai_chunk({}, usage={"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14})
            text += "data: [DONE]\n\n"
        data = text.encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(data)))
        self.send_header("x-request-id", "req_local")
        self.end_headers()
        self.wfile.write(data)


class _LocalProviderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Provider)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        _Provider.requests = []

    def _llm(self):
        self.client.flush()
        return [e for e in self.captured if e.span_type == "llm_call"]

    def openai(self, async_=False):
        import openai

        cls = openai.AsyncOpenAI if async_ else openai.OpenAI
        return cls(base_url=self.base + "/v1", api_key="sk-test-not-a-real-key-000000", max_retries=0)

    def anthropic(self, async_=False):
        import anthropic

        cls = anthropic.AsyncAnthropic if async_ else anthropic.Anthropic
        return cls(base_url=self.base, api_key="sk-ant-test", max_retries=0)


class TestOpenAIStreamIsTheProvidersStream(_LocalProviderTest):
    def test_sync_stream_keeps_the_provider_surface(self):
        import openai

        stream = self.openai().chat.completions.create(model="gpt-4o-mini", messages=MESSAGES, stream=True)
        self.assertIsInstance(stream, openai.Stream)
        self.assertEqual(stream.response.headers["x-request-id"], "req_local")
        first = next(stream)
        self.assertEqual(first.choices[0].delta.role, "assistant")
        rest = [c.choices[0].delta.content for c in stream if c.choices]
        self.assertEqual("".join(r for r in rest if r), "Hello there.")
        spans = self._llm()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].completion, "Hello there.")

    def test_sync_stream_context_manager_and_close(self):
        with self.openai().chat.completions.create(model="gpt-4o-mini", messages=MESSAGES, stream=True) as stream:
            next(iter(stream))
        self.assertTrue(stream.response.is_closed)
        other = self.openai().chat.completions.create(model="gpt-4o-mini", messages=MESSAGES, stream=True)
        other.close()
        self.assertTrue(other.response.is_closed)
        self.assertEqual(len(self._llm()), 2, "exit and close() each end their span once")

    def test_the_stream_helper_reads_through_the_tracer(self):
        with self.openai().chat.completions.stream(model="gpt-4o-mini", messages=MESSAGES) as stream:
            final = stream.get_final_completion()
        self.assertEqual(final.choices[0].message.content, "Hello there.")
        self.assertEqual(len(self._llm()), 1)

    def test_raw_and_streaming_responses_are_handed_back_untouched(self):
        import openai

        completions = self.openai().chat.completions
        raw = completions.with_raw_response.create(model="gpt-4o-mini", messages=MESSAGES, stream=True)
        self.assertEqual(raw.headers["x-request-id"], "req_local")
        parsed = raw.parse()
        self.assertIsInstance(parsed, openai.Stream)
        self.assertEqual(len([c for c in parsed]), 4)
        with completions.with_streaming_response.create(model="gpt-4o-mini", messages=MESSAGES, stream=True) as resp:
            lines = [line for line in resp.iter_lines() if line]
        self.assertTrue(lines[-1].endswith("[DONE]"))

    def test_async_stream_keeps_the_provider_surface(self):
        import openai

        async def run():
            client = self.openai(async_=True)
            stream = await client.chat.completions.create(model="gpt-4o-mini", messages=MESSAGES, stream=True)
            self.assertIsInstance(stream, openai.AsyncStream)
            first = await anext(stream)
            texts = [c.choices[0].delta.content async for c in stream if c.choices]
            async with await client.chat.completions.create(
                    model="gpt-4o-mini", messages=MESSAGES, stream=True) as second:
                await anext(second)
            third = await client.chat.completions.create(model="gpt-4o-mini", messages=MESSAGES, stream=True)
            await third.close()
            return first, texts

        first, texts = asyncio.run(run())
        self.assertEqual(first.choices[0].delta.role, "assistant")
        self.assertEqual("".join(t for t in texts if t), "Hello there.")
        self.assertEqual(len(self._llm()), 3)

    def test_message_objects_from_an_earlier_reply_can_be_sent_back(self):
        client = self.openai()
        first = client.chat.completions.create(model="gpt-4o-mini", messages=MESSAGES, stream=True)
        list(first)
        reply = types.SimpleNamespace(role="assistant", content="Hello there.")
        from openai.types.chat import ChatCompletionMessage

        message = ChatCompletionMessage(role="assistant", content="Hello there.")
        stream = client.chat.completions.create(
            model="gpt-4o-mini", messages=[*MESSAGES, message, {"role": "user", "content": "thanks"}], stream=True)
        list(stream)
        self.assertEqual(_Provider.requests[-1]["body"]["messages"][1]["content"], "Hello there.")
        self.assertIn("assistant: Hello there.", self._llm()[-1].prompt)
        del reply


class TestAnthropicStreamIsTheProvidersStream(_LocalProviderTest):
    def test_create_stream_keeps_the_provider_surface(self):
        import anthropic

        stream = self.anthropic().messages.create(model="claude-sonnet-4-5", max_tokens=64, messages=MESSAGES,
                                                  stream=True)
        # anthropic.Stream's metaclass accepts only the exact class in isinstance(), which no wrapper
        # (nor any subclass) can be; the class the stream reports is still the provider's.
        self.assertIs(stream.__class__, anthropic.Stream)
        self.assertEqual(stream.response.headers["x-request-id"], "req_local")
        types_seen = [event.type for event in stream]
        self.assertEqual(types_seen[0], "message_start")
        span = self._llm()[0]
        self.assertEqual(span.completion, "Order 1001 shipped.")
        self.assertEqual((span.tokens_in, span.tokens_out), (12, 6))

    def test_messages_stream_helper_keeps_every_reader(self):
        import anthropic

        client = self.anthropic()
        with client.messages.stream(model="claude-sonnet-4-5", max_tokens=64, messages=MESSAGES) as stream:
            self.assertIsInstance(stream, anthropic.MessageStream)
            self.assertEqual("".join(stream.text_stream), "Order 1001 shipped.")
            self.assertEqual(stream.get_final_message().content[0].text, "Order 1001 shipped.")
            self.assertEqual(stream.get_final_text(), "Order 1001 shipped.")
            self.assertEqual(stream.current_message_snapshot.usage.output_tokens, 6)
        with client.messages.stream(model="claude-sonnet-4-5", max_tokens=64, messages=MESSAGES) as stream:
            stream.until_done()
        with client.messages.stream(model="claude-sonnet-4-5", max_tokens=64, messages=MESSAGES) as stream:
            self.assertIn("message_stop", [event.type for event in stream])
        manager = client.messages.stream(model="claude-sonnet-4-5", max_tokens=64, messages=MESSAGES)
        self.assertIsInstance(manager, anthropic.lib.streaming.MessageStreamManager)
        with manager as stream:
            stream.until_done()
        spans = self._llm()
        self.assertEqual(len(spans), 4)
        self.assertTrue(all(s.completion == "Order 1001 shipped." for s in spans))

    def test_async_messages_stream_helper(self):
        async def run():
            client = self.anthropic(async_=True)
            async with client.messages.stream(model="claude-sonnet-4-5", max_tokens=64, messages=MESSAGES) as stream:
                text = "".join([t async for t in stream.text_stream])
                final = await stream.get_final_message()
            created = await client.messages.create(model="claude-sonnet-4-5", max_tokens=64, messages=MESSAGES,
                                                   stream=True)
            events = [event.type async for event in created]
            return text, final, events

        text, final, events = asyncio.run(run())
        self.assertEqual(text, "Order 1001 shipped.")
        self.assertEqual(final.usage.output_tokens, 6)
        self.assertEqual(events[-1], "message_stop")
        self.assertEqual(len(self._llm()), 2)


class TestProviderErrorsReachTheCallerUnchanged(_LocalProviderTest):
    def test_a_401_is_the_providers_exception_with_its_traceback(self):
        import anthropic
        import openai

        for call, error in (
            (lambda: self.openai().chat.completions.create(model="fail-401", messages=MESSAGES, stream=True),
             openai.AuthenticationError),
            (lambda: self.openai().chat.completions.create(model="fail-401", messages=MESSAGES), openai.AuthenticationError),
            (lambda: self.anthropic().messages.create(model="fail-401", max_tokens=8, messages=MESSAGES, stream=True),
             anthropic.AuthenticationError),
        ):
            with self.subTest(error=error.__name__):
                try:
                    call()
                    self.fail("the provider's error was swallowed")
                except error as exc:
                    caught = exc
                self.assertIs(type(caught), error)
                frames = []
                tb = caught.__traceback__
                while tb is not None:
                    frames.append(tb.tb_frame.f_code.co_filename)
                    tb = tb.tb_next
                self.assertTrue(any(f.endswith("_base_client.py") for f in frames),
                                "the traceback still runs down into the provider's own code")
        self.assertEqual([s.status for s in self._llm()], ["ERROR", "ERROR", "ERROR"])

    def test_the_same_exception_object_is_reraised(self):
        from evalkit.instrumentation import auto as _auto

        error = ConnectionResetError("peer reset")

        class Completions:
            def create(self, *a, **k):
                raise error

        _auto._patch_openai_class_sync(self.client, Completions)
        for stream in (False, True):
            with self.subTest(stream=stream):
                with self.assertRaises(ConnectionResetError) as raised:
                    Completions().create(model="gpt-4o-mini", messages=MESSAGES, stream=stream)
                self.assertIs(raised.exception, error)


class TestOurFailuresNeverReachTheCaller(_LocalProviderTest):
    def _break(self, target, name):
        original = getattr(target, name)

        def broken(*a, **k):
            raise RuntimeError("evalkit bug")

        setattr(target, name, broken)
        self.addCleanup(setattr, target, name, original)

    def test_a_failing_span_start_falls_back_to_the_plain_call(self):
        from evalkit.instrumentation import auto as _auto

        self._break(_auto, "_chat_prompt")
        reply = self.openai().chat.completions.create(model="gpt-4o-mini", messages=MESSAGES, stream=True)
        self.assertEqual(len(list(reply)), 4)
        events = list(self.anthropic().messages.create(model="claude-sonnet-4-5", max_tokens=8, messages=MESSAGES,
                                                       stream=True))
        self.assertEqual(events[-1].type, "message_stop")

    def test_a_failing_recorder_leaves_the_result_alone(self):
        from evalkit.instrumentation import auto as _auto

        self._break(self.client, "enqueue")
        self._break(_auto, "_request_attrs")
        stream = self.openai().chat.completions.create(model="gpt-4o-mini", messages=MESSAGES, stream=True)
        self.assertEqual(len(list(stream)), 4)
        with self.anthropic().messages.stream(model="claude-sonnet-4-5", max_tokens=8, messages=MESSAGES) as s:
            self.assertEqual(s.get_final_text(), "Order 1001 shipped.")

    def test_a_failing_chunk_reader_never_breaks_the_stream(self):
        from evalkit.instrumentation import _streaming

        self._break(_streaming._StreamState, "process_openai_chunk")
        stream = self.openai().chat.completions.create(model="gpt-4o-mini", messages=MESSAGES, stream=True)
        self.assertEqual(len(list(stream)), 4)
        self.assertEqual(len(self._llm()), 1, "the span still ends, with what could be read")

    def test_a_failing_recorder_never_breaks_a_traced_function(self):
        traced = self.client.trace_function("charge")(lambda amount: amount * 2)
        tool = self.client.trace_tool("refund")(lambda order: f"refunded {order}")
        self._break(self.client, "enqueue")
        self.assertEqual(traced(21), 42)
        self.assertEqual(tool("1001"), "refunded 1001")
        self._break(self.client, "start_span")
        self.assertEqual(traced(4), 8)

    def test_a_failing_http_recorder_leaves_requests_alone(self):
        import requests

        from evalkit.instrumentation.http import requests as ek_requests

        self._break(ek_requests, "sanitize_headers")
        resp = requests.post(self.base + "/v1/chat/completions", json={"model": "gpt-4o-mini"})
        self.assertEqual(resp.status_code, 200)


class TestStreamsAreHandledCorrectly(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def _tracer(self, chunks, pulled, is_async=False):
        from evalkit.instrumentation._streaming import _AsyncStreamTracer, _SyncStreamTracer

        def gen():
            for c in chunks:
                pulled.append(c)
                yield c

        async def agen():
            for c in chunks:
                pulled.append(c)
                yield c

        done = []
        tracer = (_AsyncStreamTracer if is_async else _SyncStreamTracer)(
            agen() if is_async else gen(), lambda *a: done.append(a), start_ms=time.time() * 1000)
        return tracer, done

    def test_chunks_pass_through_one_at_a_time(self):
        pulled = []
        tracer, done = self._tracer([OStreamChunk("a"), OStreamChunk("b"), OStreamChunk("c")], pulled)
        it = iter(tracer)
        next(it)
        self.assertEqual(len(pulled), 1, "the tracer must never read ahead of the caller")
        next(it)
        self.assertEqual(len(pulled), 2)
        self.assertEqual(done, [])

    def test_every_way_of_stopping_ends_the_span_once(self):
        def stop_by_break(t):
            for _ in t:
                break

        def stop_by_next_then_close(t):
            next(t)
            t.close()

        def stop_by_exit(t):
            with t:
                next(t)

        def stop_by_dropping(t):
            next(t)

        for stop in (stop_by_break, stop_by_next_then_close, stop_by_exit, stop_by_dropping):
            with self.subTest(stop.__name__):
                pulled = []
                tracer, done = self._tracer([OStreamChunk("a"), OStreamChunk("b")], pulled)
                stop(tracer)
                del tracer
                gc.collect()
                self.assertEqual(len(done), 1)
                self.assertIsNone(done[0][-1], "stopping early is not an error")

    def test_cancellation_ends_the_async_span_once(self):
        pulled = []
        tracer, done = self._tracer([OStreamChunk("a"), OStreamChunk("b")], pulled, is_async=True)

        async def run():
            async for _ in tracer:
                raise asyncio.CancelledError()

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(run())
        gc.collect()
        self.assertEqual(len(done), 1)

    def test_what_a_span_keeps_of_a_stream_is_bounded(self):
        from evalkit.instrumentation import _streaming

        state = _streaming._StreamState(0.0)
        piece = "x" * 65536
        for _ in range(64):
            state.process_openai_chunk(OStreamChunk(piece))
        self.assertLessEqual(sum(map(len, state.chunks)), _streaming._TEXT_KEEP)
        delta = types.SimpleNamespace(tool_calls=[types.SimpleNamespace(
            index=0, id="c", function=types.SimpleNamespace(name="write_file", arguments=piece))])
        for _ in range(64):
            state._add_openai_tool_deltas(delta)
        self.assertLessEqual(len(state.tool_calls[0]["arguments"]), _streaming._ARGS_KEEP)


class TestTheRequestIsNeverChanged(_LocalProviderTest):
    def test_kwargs_messages_and_headers_go_out_as_given(self):
        import copy

        import requests

        kwargs = {"model": "gpt-4o-mini", "messages": [dict(m) for m in MESSAGES], "stream": True,
                  "temperature": 0.2, "extra_headers": {"x-team": "billing"}}
        before = copy.deepcopy(kwargs)
        list(self.openai().chat.completions.create(**kwargs))
        self.assertEqual(kwargs, before)
        sent = _Provider.requests[-1]
        self.assertNotIn("stream_options", sent["body"], "the SDK must never add stream_options to a request")
        self.assertEqual(sent["body"]["messages"], MESSAGES)
        self.assertEqual(sent["headers"].get("x-team"), "billing")

        headers = {"x-a": "1"}
        requests.post(self.base + "/v1/chat/completions", json={"model": "gpt-4o-mini"}, headers=headers)
        self.assertEqual(headers, {"x-a": "1"}, "the caller's headers dict is left as it was")
        self.assertIn("traceparent", {k.lower() for k in _Provider.requests[-1]["headers"]})

    def test_a_generator_of_messages_is_not_consumed(self):
        list(self.openai().chat.completions.create(
            model="gpt-4o-mini", messages=(m for m in MESSAGES), stream=True))
        self.assertEqual(_Provider.requests[-1]["body"]["messages"], MESSAGES)


class TestPatchingIsSafe(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_patching_twice_records_once(self):
        from evalkit.instrumentation import auto as _auto

        class Completions:
            def create(self, *a, **k):
                return iter([OStreamChunk("hi")])

        _auto._patch_openai_class_sync(self.client, Completions)
        first = Completions.create
        _auto._patch_openai_class_sync(self.client, Completions)
        self.assertIs(Completions.create, first)
        list(Completions().create(model="m", messages=[], stream=True))
        self.client.flush()
        self.assertEqual(len([e for e in self.captured if e.span_type == "llm_call"]), 1)

    def test_a_missing_library_or_attribute_is_skipped(self):
        from evalkit.instrumentation import auto as _auto
        from evalkit.instrumentation.db.psycopg2 import patch_psycopg2

        saved = {name: sys.modules.get(name) for name in ("psycopg2", "openai.resources.chat.completions.completions")}
        sys.modules["psycopg2"] = None
        try:
            self.assertFalse(patch_psycopg2(self.client))
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod
        _auto._try_patch_anthropic(self.client)
        _auto.setup_auto_instrumentation(self.client)


class TestNoBlockingOnTheRequestPath(unittest.TestCase):
    def test_recording_does_not_wait_for_a_slow_receiver(self):
        reset_context()
        captured = []
        client = fresh(captured)
        slow = threading.Event()
        client._exporter.export = lambda batch: (slow.wait(3), captured.extend(batch), True)[2]
        try:
            started = time.perf_counter()
            for _ in range(50):
                end, _ = client.start_span("fast")
                end("OK")
            self.assertLess(time.perf_counter() - started, 0.5, "spans are queued; export happens in the background")
        finally:
            slow.set()
            client.flush()


if __name__ == "__main__":
    unittest.main()
