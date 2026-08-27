import asyncio
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import evalkit
from tests.helpers import fresh, reset_context

SSE_CHUNKS = [
    b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
    b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
    b"data: [DONE]\n\n",
]
SSE_BODY = b"".join(SSE_CHUNKS).decode()


def _start_server(chunks=None, delay=0.0, status=200):
    payload = SSE_CHUNKS if chunks is None else chunks

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for part in payload:
                self.wfile.write(b"%x\r\n" % len(part) + part + b"\r\n")
                self.wfile.flush()
                if delay:
                    time.sleep(delay)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        do_POST = do_GET

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class StreamBodyBase(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self.srv = _start_server(*self._server_args())
        self.port = self.srv.server_address[1]

    def _server_args(self):
        return ()

    def tearDown(self):
        self.srv.shutdown()

    def url(self, path="/stream"):
        return f"http://127.0.0.1:{self.port}{path}"

    def _spans(self):
        self.client.flush()
        return [
            e for e in self.captured
            if e.span_type == "http_call" and "127.0.0.1" in (e.attributes or {}).get("http.url", "")
        ]

    def _span(self):
        spans = self._spans()
        self.assertEqual(
            len(spans), 1,
            "a streamed response must produce exactly one span; the ClickHouse traces table is a plain MergeTree with "
            "no dedupe or update path, so a second row for the same call is a permanent double-count",
        )
        return spans[0]


class TestRequestsStreaming(StreamBodyBase):
    def test_streamed_body_is_captured_chunk_by_chunk(self):
        import requests

        tid, end, ctx = self.client.start_trace("req-stream")
        with evalkit.with_trace(ctx):
            resp = requests.get(self.url(), stream=True)
            seen = [c for c in resp.iter_content(chunk_size=None) if c]
        end("OK")

        self.assertTrue(seen, "the caller must still receive every chunk it would have without instrumentation")
        span = self._span()
        attrs = span.attributes
        self.assertEqual(span.trace_id, tid)
        self.assertEqual(
            attrs.get("http.response.body"), SSE_BODY,
            "with stream=True requests never materialises .text, so the old code skipped the body entirely and an SSE "
            "response - which is what every streaming LLM API returns - was recorded with no payload at all; the "
            "capture must be every chunk in order, not just the first",
        )
        self.assertTrue(
            attrs.get("http.response.streamed"),
            "the span has to say the body arrived incrementally, otherwise a truncated capture is indistinguishable "
            "from a genuinely short response",
        )
        self.assertTrue(
            attrs.get("http.response.stream.complete"),
            "the iterator ran to exhaustion, so the capture is the whole body and not a partial read",
        )

    def test_body_reaches_the_caller_unchanged(self):
        import requests

        with evalkit.with_trace(self.client.start_trace("req-passthrough")[2]):
            resp = requests.get(self.url(), stream=True)
            body = b"".join(resp.iter_content(chunk_size=None))

        self.assertEqual(
            body, b"".join(SSE_CHUNKS),
            "the tee must yield each chunk straight through; consuming or reordering the stream would break every "
            "SSE client the SDK is installed next to",
        )

    def test_json_on_a_stream_response_still_captures_the_body(self):
        import requests

        srv = _start_server(chunks=[b'{"ok":true}'])
        port = srv.server_address[1]
        try:
            with evalkit.with_trace(self.client.start_trace("req-json")[2]):
                resp = requests.get(f"http://127.0.0.1:{port}/j", stream=True)
                self.assertEqual(resp.json(), {"ok": True})
        finally:
            srv.shutdown()

        self.client.flush()
        spans = [e for e in self.captured
                 if e.span_type == "http_call" and str(port) in (e.attributes or {}).get("http.url", "")]
        self.assertEqual(len(spans), 1)
        self.assertEqual(
            spans[0].attributes.get("http.response.body"), '{"ok":true}',
            "requests routes .json()/.text/.content through iter_content, so teeing that one method is what makes "
            "stream=True responses readable no matter which accessor the caller reaches for",
        )

    def test_a_stream_that_is_closed_without_being_read_still_emits(self):
        import requests

        with evalkit.with_trace(self.client.start_trace("req-abandon")[2]):
            resp = requests.get(self.url(), stream=True)
            resp.close()

        from evalkit.instrumentation.http import _stream_body as sb

        self.assertEqual(
            sb.pending_count(), 0,
            "close() is the emit hook for a response nobody reads; if it only got picked up by the next flush, the "
            "span would arrive minutes late or not at all in a short-lived process",
        )
        span = self._span()
        self.assertFalse(
            span.attributes.get("http.response.stream.complete"),
            "emission is deferred until the body finishes, so a response closed without being read must still produce "
            "a span - and it must admit the body is incomplete rather than imply an empty response",
        )
        self.assertEqual(span.attributes["http.status_code"], 200)

    def test_a_partially_read_stream_records_what_was_pulled(self):
        import requests

        from evalkit.instrumentation.http import _stream_body as sb

        with evalkit.with_trace(self.client.start_trace("req-partial")[2]):
            resp = requests.get(self.url(), stream=True)
            for _ in resp.iter_content(chunk_size=None):
                break
            resp.close()

        self.assertEqual(
            sb.pending_count(), 0,
            "abandoning the iterator drops its last reference, and the resulting GeneratorExit is what closes the "
            "capture; without handling it the span would sit in the pending map waiting for a flush",
        )
        span = self._span()
        self.assertFalse(span.attributes.get("http.response.stream.complete"))
        self.assertGreaterEqual(
            span.attributes.get("http.response.stream.chunks", 0), 1,
            "breaking out of the loop is normal SSE client behaviour once the answer is complete, and the chunks the "
            "caller did pull are exactly the ones worth keeping",
        )
        self.assertIn(
            "http.response.body", span.attributes,
            "a partial capture is still the most useful thing available for debugging a stream the client gave up on",
        )

    def test_time_to_headers_is_kept_separately_from_total_duration(self):
        import requests

        srv = _start_server(delay=0.05)
        port = srv.server_address[1]
        try:
            with evalkit.with_trace(self.client.start_trace("req-ttfb")[2]):
                resp = requests.get(f"http://127.0.0.1:{port}/slow", stream=True)
                list(resp.iter_content(chunk_size=None))
        finally:
            srv.shutdown()

        self.client.flush()
        span = [e for e in self.captured
                if e.span_type == "http_call" and str(port) in (e.attributes or {}).get("http.url", "")][0]
        attrs = span.attributes
        self.assertIn(
            "http.server.time_to_headers_ms", attrs,
            "latency_ms now spans the whole stream, so the old meaning - time until the server responded - needs its "
            "own attribute or the responsiveness signal is lost",
        )
        self.assertGreaterEqual(
            span.latency_ms, attrs["http.server.time_to_headers_ms"],
            "a streamed call cannot finish before its own headers arrived",
        )
        self.assertGreater(
            span.latency_ms, 50,
            "the server sleeps between chunks, so full-stream duration must be visibly longer than time-to-headers",
        )
        self.assertIn(
            "http.server.time_to_first_byte_ms", attrs,
            "time to first body byte is the streaming equivalent of time-to-first-token and is the number a user "
            "actually feels; it is not the same as time-to-headers",
        )
        self.assertLess(
            attrs["http.server.time_to_first_byte_ms"], span.latency_ms - 40,
            "the first-byte stamp must be taken on the first chunk and never overwritten, otherwise it silently "
            "becomes time-to-last-byte and duplicates latency_ms",
        )

    def test_the_opt_out_restores_immediate_emission(self):
        import requests

        from evalkit.instrumentation.http import requests as mod

        saved = mod._STREAM_BODIES
        mod._STREAM_BODIES = False
        try:
            with evalkit.with_trace(self.client.start_trace("req-optout")[2]):
                resp = requests.get(self.url(), stream=True)
                self.client.flush()
                emitted_before_reading = len(self._spans())
                list(resp.iter_content(chunk_size=None))
        finally:
            mod._STREAM_BODIES = saved

        self.assertEqual(
            emitted_before_reading, 1,
            "capture_stream_bodies=False must fall back to emitting as soon as the headers land; anyone who turns the "
            "tee off is opting out of deferred emission too, not just out of the body",
        )
        attrs = self._span().attributes
        self.assertNotIn(
            "http.response.body", attrs,
            "with the tee disabled there is no safe way to read a stream=True body, so the span carries none rather "
            "than consuming the caller's stream to get one",
        )
        self.assertNotIn("http.response.streamed", attrs)

    def test_capture_is_capped_and_says_so(self):
        import requests

        from evalkit.instrumentation.http import requests as mod

        srv = _start_server(chunks=[b"x" * 200, b"y" * 200])
        port = srv.server_address[1]
        saved = mod._BODY_MAX
        mod._BODY_MAX = 64
        try:
            with evalkit.with_trace(self.client.start_trace("req-cap")[2]):
                resp = requests.get(f"http://127.0.0.1:{port}/big", stream=True)
                list(resp.iter_content(chunk_size=None))
        finally:
            mod._BODY_MAX = saved
            srv.shutdown()

        self.client.flush()
        attrs = [e for e in self.captured
                 if e.span_type == "http_call" and str(port) in (e.attributes or {}).get("http.url", "")][0].attributes
        self.assertEqual(
            len(attrs["http.response.body"]), 64,
            "an unbounded tee on a long-lived stream is a memory leak, so the buffer stops at max_body_bytes",
        )
        self.assertTrue(
            attrs.get("http.response.body.truncated"),
            "a body clipped at the cap must be flagged, or it reads as a complete short response",
        )
        self.assertEqual(
            attrs.get("http.response.body.bytes"), 400,
            "the full byte count is still worth recording even when the payload itself is clipped",
        )

    def test_a_single_oversized_chunk_is_clipped_and_flagged(self):
        import requests

        from evalkit.instrumentation.http import requests as mod

        srv = _start_server(chunks=[b"z" * 300])
        port = srv.server_address[1]
        saved = mod._BODY_MAX
        mod._BODY_MAX = 64
        try:
            with evalkit.with_trace(self.client.start_trace("req-one-big")[2]):
                resp = requests.get(f"http://127.0.0.1:{port}/one", stream=True)
                list(resp.iter_content(chunk_size=None))
        finally:
            mod._BODY_MAX = saved
            srv.shutdown()

        self.client.flush()
        attrs = [e for e in self.captured
                 if e.span_type == "http_call" and str(port) in (e.attributes or {}).get("http.url", "")][0].attributes
        self.assertEqual(len(attrs["http.response.body"]), 64)
        self.assertTrue(
            attrs.get("http.response.body.truncated"),
            "one chunk larger than the whole budget is the common case for a big JSON response, and it is the only "
            "case where the partial-room branch runs on its own; that branch has to raise the flag itself",
        )
        self.assertEqual(attrs.get("http.response.body.bytes"), 300)

    def test_a_chunk_arriving_after_the_cap_is_already_full_is_flagged(self):
        import requests

        from evalkit.instrumentation.http import requests as mod

        srv = _start_server(chunks=[b"x" * 200, b"y" * 200])
        port = srv.server_address[1]
        saved = mod._BODY_MAX
        mod._BODY_MAX = 200
        try:
            with evalkit.with_trace(self.client.start_trace("req-cap-exact")[2]):
                resp = requests.get(f"http://127.0.0.1:{port}/exact", stream=True)
                list(resp.iter_content(chunk_size=None))
        finally:
            mod._BODY_MAX = saved
            srv.shutdown()

        self.client.flush()
        attrs = [e for e in self.captured
                 if e.span_type == "http_call" and str(port) in (e.attributes or {}).get("http.url", "")][0].attributes
        self.assertEqual(len(attrs["http.response.body"]), 200)
        self.assertTrue(
            attrs.get("http.response.body.truncated"),
            "the first chunk fills the buffer exactly, so truncation is only discoverable on the chunk that finds no "
            "room left at all; that branch has to set the flag too or a long stream looks complete",
        )
        self.assertEqual(attrs.get("http.response.body.bytes"), 400)
        self.assertEqual(
            attrs.get("http.response.stream.chunks"), 2,
            "chunks past the cap are still counted - the count is about the stream's shape, not the buffer's",
        )


class TestHttpxStreaming(StreamBodyBase):
    def test_sync_stream_body_is_captured(self):
        import httpx

        with evalkit.with_trace(self.client.start_trace("httpx-stream")[2]):
            with httpx.Client() as c:
                with c.stream("GET", self.url()) as resp:
                    chunks = [t for t in resp.iter_text()]

        self.assertTrue(chunks)
        attrs = self._span().attributes
        self.assertEqual(
            attrs.get("http.response.body"), SSE_BODY,
            "httpx is the transport under the openai and anthropic clients, so a streamed chat completion went out "
            "over this path and came back with no recorded payload",
        )
        self.assertTrue(attrs.get("http.response.stream.complete"))
        self.assertIn(
            "http.server.time_to_headers_ms", attrs,
            "the deferred path makes latency_ms mean full-stream duration, so httpx spans need the headers timing "
            "carried over too or the bump silently redefines what the dashboard is plotting",
        )
        self.assertIn("http.server.time_to_first_byte_ms", attrs)

    def test_async_stream_body_is_captured(self):
        import httpx

        async def run():
            with evalkit.with_trace(self.client.start_trace("httpx-astream")[2]):
                async with httpx.AsyncClient() as c:
                    async with c.stream("GET", self.url()) as resp:
                        return [chunk async for chunk in resp.aiter_bytes()]

        chunks = asyncio.run(run())
        self.assertEqual(
            b"".join(chunks), b"".join(SSE_CHUNKS),
            "aiter_bytes is what read()/aiter_text()/aiter_lines() all funnel through, so teeing it must not disturb "
            "what the caller sees",
        )
        attrs = self._span().attributes
        self.assertEqual(attrs.get("http.response.body"), SSE_BODY)
        self.assertTrue(attrs.get("http.response.streamed"))
        self.assertTrue(
            attrs.get("http.response.stream.complete"),
            "httpx's aiter_raw awaits aclose() at natural end-of-stream from inside the generator, so a close hook "
            "that emits eagerly would report every fully-consumed async stream as incomplete",
        )

    def test_a_close_from_inside_the_iterator_does_not_preempt_completion(self):
        from evalkit.instrumentation.http import _stream_body as sb

        seen = []
        capture = sb.new_capture(1024, time.time() * 1000, lambda cap, err: seen.append(cap))

        def source():
            yield b"a"
            yield b"b"
            capture.finish(deferrable=True)

        self.assertEqual(b"".join(capture.tee_sync(source())), b"ab")
        self.assertEqual(len(seen), 1)
        self.assertTrue(
            seen[0].complete,
            "httpx closes the response from inside iter_raw once the body is exhausted, so close() arriving while the "
            "tee is still unwinding must not win the race and stamp the span incomplete",
        )

    def test_a_close_with_no_active_iterator_still_emits_immediately(self):
        from evalkit.instrumentation.http import _stream_body as sb

        seen = []
        capture = sb.new_capture(1024, time.time() * 1000, lambda cap, err: seen.append(cap))
        capture.finish(deferrable=True)
        self.assertEqual(
            len(seen), 1,
            "deferral is only correct while a tee is mid-flight; a response closed without ever being iterated has "
            "nothing left to wait for and must emit at once",
        )
        self.assertFalse(seen[0].complete)

    def test_non_streaming_httpx_still_reads_the_body_eagerly(self):
        import httpx

        with evalkit.with_trace(self.client.start_trace("httpx-plain")[2]):
            with httpx.Client() as c:
                c.get(self.url())

        attrs = self._span().attributes
        self.assertEqual(
            attrs.get("http.response.body"), SSE_BODY,
            "httpx materialises _content for non-streaming sends, so that path must keep its immediate read and not "
            "be pushed onto the deferred tee",
        )
        self.assertFalse(
            attrs.get("http.response.streamed"),
            "a response httpx already buffered was not consumed incrementally by the caller",
        )

    def test_async_stream_closed_without_reading_still_emits(self):
        import httpx

        async def run():
            with evalkit.with_trace(self.client.start_trace("httpx-abandon")[2]):
                async with httpx.AsyncClient() as c:
                    async with c.stream("GET", self.url()):
                        pass

        asyncio.run(run())
        span = self._span()
        self.assertEqual(span.attributes["http.status_code"], 200)
        self.assertFalse(span.attributes.get("http.response.stream.complete"))


class TestAiohttpStreaming(StreamBodyBase):
    def setUp(self):
        super().setUp()
        from evalkit.instrumentation.http.aiohttp import patch_aiohttp
        patch_aiohttp(self.client)

    def test_streamed_body_is_captured(self):
        import aiohttp

        async def run():
            with evalkit.with_trace(self.client.start_trace("aio-stream")[2]):
                async with aiohttp.ClientSession() as sess:
                    async with sess.get(self.url()) as resp:
                        return [c async for c in resp.content.iter_chunked(8)]

        chunks = asyncio.run(run())
        self.assertTrue(chunks, "the caller must still receive the stream")
        attrs = self._span().attributes
        self.assertEqual(
            attrs.get("http.response.body"), SSE_BODY,
            "the aiohttp tracer recorded no response body at all, streamed or not, because reading it in the tracer "
            "would have consumed the stream out from under the caller",
        )
        self.assertTrue(attrs.get("http.response.streamed"))

    def test_read_captures_the_body_without_consuming_it_twice(self):
        import aiohttp

        async def run():
            with evalkit.with_trace(self.client.start_trace("aio-read")[2]):
                async with aiohttp.ClientSession() as sess:
                    async with sess.get(self.url()) as resp:
                        return await resp.read()

        body = asyncio.run(run())
        self.assertEqual(
            body, b"".join(SSE_CHUNKS),
            "resp.read() must return the full payload; teeing at the StreamReader means the tracer sees the bytes on "
            "their way to the caller instead of racing for them",
        )
        attrs = self._span().attributes
        self.assertEqual(attrs.get("http.response.body"), SSE_BODY)
        self.assertFalse(
            attrs.get("http.response.streamed"),
            "a single whole-body read is not incremental consumption",
        )
        self.assertTrue(
            attrs.get("http.response.stream.complete"),
            "an unbounded read() returns the entire body, so the capture is complete by construction and must not be "
            "reported as a partial read",
        )

    def test_json_still_works_through_the_tee(self):
        import aiohttp

        srv = _start_server(chunks=[b'{"ok":true}'])
        port = srv.server_address[1]

        async def run():
            with evalkit.with_trace(self.client.start_trace("aio-json")[2]):
                async with aiohttp.ClientSession() as sess:
                    async with sess.get(f"http://127.0.0.1:{port}/j") as resp:
                        return await resp.json(content_type=None)

        try:
            self.assertEqual(asyncio.run(run()), {"ok": True})
        finally:
            srv.shutdown()

        self.client.flush()
        spans = [e for e in self.captured
                 if e.span_type == "http_call" and str(port) in (e.attributes or {}).get("http.url", "")]
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].attributes.get("http.response.body"), '{"ok":true}')

    def test_response_released_without_reading_still_emits(self):
        import aiohttp

        async def run():
            with evalkit.with_trace(self.client.start_trace("aio-release")[2]):
                async with aiohttp.ClientSession() as sess:
                    async with sess.get(self.url()) as resp:
                        resp.release()

        asyncio.run(run())
        span = self._span()
        self.assertEqual(
            span.attributes["http.status_code"], 200,
            "deferring emission means an unread aiohttp response must still be reported, or instrumenting aiohttp "
            "would lose spans that used to be emitted immediately",
        )


class TestPendingCaptureSafety(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_flush_emits_a_stream_that_was_never_finished(self):
        from evalkit.instrumentation.http import _stream_body as sb

        sb.flush_pending()
        seen = []
        capture = sb.new_capture(1024, time.time() * 1000, lambda cap, err: seen.append((cap, err)))
        capture.add(b"partial")
        self.assertEqual(
            sb.pending_count(), 1,
            "an in-flight capture must be registered, since the registry is the only handle flush and eviction have "
            "on a stream nobody is going to close",
        )

        self.client.flush()
        self.assertEqual(
            sb.pending_count(), 0,
            "a capture that has emitted must leave the registry, or the map grows forever and every later flush "
            "re-walks dead entries",
        )
        self.assertEqual(
            len(seen), 1,
            "a response that is never read, never closed and never garbage collected would otherwise be a span that "
            "silently never exists; flush is the backstop that makes deferred emission safe",
        )
        self.assertFalse(seen[0][0].complete)
        self.assertEqual(seen[0][0].body(), "partial")

    def test_time_to_first_byte_is_stamped_once_on_the_first_chunk(self):
        from evalkit.instrumentation.http import _stream_body as sb

        seen = []
        capture = sb.new_capture(1024, time.time() * 1000, lambda cap, err: seen.append(cap))
        capture.add(b"first")
        time.sleep(0.12)
        capture.add(b"last")
        capture.finish(complete=True)

        ttfb = seen[0].attrs()["http.server.time_to_first_byte_ms"]
        self.assertLess(
            ttfb, 100,
            "the first-byte stamp is the streaming analogue of time-to-first-token; if a later chunk overwrites it, "
            "the attribute silently becomes time-to-last-byte and stops being a responsiveness signal at all",
        )

    def test_a_capture_emits_at_most_once(self):
        from evalkit.instrumentation.http import _stream_body as sb

        seen = []
        capture = sb.new_capture(1024, time.time() * 1000, lambda cap, err: seen.append(cap))
        capture.finish(complete=True)
        capture.finish()
        sb.flush_pending()
        self.assertEqual(
            len(seen), 1,
            "exhaustion, close() and the garbage collector can all fire for the same response, so the once-guard is "
            "what stops a MergeTree table with no dedupe from double-counting the call",
        )

    def test_the_pending_registry_is_bounded(self):
        from evalkit.instrumentation.http import _stream_body as sb

        sb.flush_pending()
        seen = []
        saved = sb._MAX_PENDING
        sb._MAX_PENDING = 4
        try:
            for _ in range(10):
                sb.new_capture(16, time.time() * 1000, lambda cap, err: seen.append(cap))
            self.assertLessEqual(
                sb.pending_count(), 4,
                "the registry holds a strong reference so ids stay stable, which means an app that leaks responses "
                "would grow it without bound; the oldest capture is flushed out instead",
            )
        finally:
            sb._MAX_PENDING = saved
            sb.flush_pending()
        self.assertGreaterEqual(len(seen), 6, "evicted captures must be emitted, not dropped")


if __name__ == "__main__":
    unittest.main()
