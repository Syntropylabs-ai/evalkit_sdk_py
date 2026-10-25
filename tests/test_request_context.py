"""Server spans: the request's trace ends with the request, the traceparent is read by the W3C rules,
and bodies are kept from what the app reads, up to max_body_bytes, never by reading ahead of it."""
import asyncio
import json
import unittest

import evalkit
from evalkit.core.context import _ambient_trace_id
from evalkit.utils.headers import parse_traceparent
from tests.helpers import fresh, reset_context

TRACE = "4bf92f3577b34da6a3ce929d0e0e4736"
SPAN = "00f067aa0ba902b7"


class TraceparentTest(unittest.TestCase):
    def test_a_valid_header_is_read(self):
        self.assertEqual(parse_traceparent(f"00-{TRACE}-{SPAN}-01"), (TRACE, SPAN))
        self.assertEqual(parse_traceparent(f"  00-{TRACE}-{SPAN}-00\t"), (TRACE, SPAN))

    def test_invalid_headers_are_ignored(self):
        for header in (
            f"00-{'0' * 32}-{SPAN}-01",       # all-zero trace id
            f"00-{TRACE}-{'0' * 16}-01",      # all-zero span id
            f"00-{TRACE.upper()}-{SPAN}-01",  # uppercase hex
            f"00-{TRACE[:-1]}g-{SPAN}-01",    # not hex
            f"ff-{TRACE}-{SPAN}-01",          # version ff
            f"00-{TRACE}-{SPAN}-01-extra",    # version 00 has four fields
            f"00-{TRACE}-{SPAN}",             # no flags
            "", "garbage", None,
        ):
            with self.subTest(header=header):
                self.assertEqual(parse_traceparent(header), (None, None))

    def test_a_later_version_with_more_fields_is_joined(self):
        self.assertEqual(parse_traceparent(f"01-{TRACE}-{SPAN}-01-what-the-future-adds"), (TRACE, SPAN))
        self.assertEqual(parse_traceparent(f"cc-{TRACE}-{SPAN}-09"), (TRACE, SPAN))


def _scope(method="POST", path="/api", headers=()):
    return {
        "type": "http", "method": method, "path": path, "query_string": b"",
        "headers": [(b"host", b"testserver"), *headers], "scheme": "http", "server": ("testserver", 80),
    }


class _Case(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def server_span(self):
        self.client.flush()
        [span] = [e for e in self.captured if e.span_type == "http_call" and e.span_kind == "SERVER"]
        return span


class AsgiBodyTest(_Case):
    def _run(self, app, *, messages, headers=()):
        mw = evalkit.EvalKitMiddleware(app, self.client)
        pending = list(messages)
        pulled = []

        async def receive():
            message = pending.pop(0) if pending else {"type": "http.disconnect"}
            pulled.append(message)
            return message

        sent = []

        async def send(message):
            sent.append(message)

        async def main():
            await mw(_scope(headers=headers), receive, send)
            return _ambient_trace_id.get()

        ambient_after = asyncio.run(main())
        return pulled, sent, ambient_after

    @staticmethod
    def _reading_app(response_chunks=(b'{"ok":true}',), content_type=b"application/json"):
        async def app(scope, receive, send):
            while True:
                message = await receive()
                if not message.get("more_body"):
                    break
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", content_type)]})
            for i, chunk in enumerate(response_chunks):
                await send({"type": "http.response.body", "body": chunk, "more_body": i < len(response_chunks) - 1})
        return app

    def test_nothing_is_read_before_the_app_asks(self):
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 204, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        pulled, _sent, _ = self._run(app, messages=[{"type": "http.request", "body": b"x" * 100, "more_body": False}])
        self.assertEqual(pulled, [], "the middleware never pulls the body ahead of the app")
        self.assertNotIn("http.request.body", self.server_span().attributes)

    def test_the_body_the_app_reads_is_kept_up_to_the_cap(self):
        saved = self.client._opts.max_body_bytes
        self.client._opts.max_body_bytes = 100
        try:
            chunks = [{"type": "http.request", "body": b"a" * 80, "more_body": True},
                      {"type": "http.request", "body": b"b" * 80, "more_body": False}]
            self._run(self._reading_app(), messages=chunks, headers=[(b"content-type", b"text/plain")])
        finally:
            self.client._opts.max_body_bytes = saved
        attrs = self.server_span().attributes
        self.assertEqual(attrs["http.request.body"], "a" * 80 + "b" * 20)
        self.assertIs(attrs["http.request.body.truncated"], True)
        self.assertEqual(attrs["http.request.body.bytes"], 160)

    def test_uploads_are_not_kept(self):
        for content_type in (b"multipart/form-data; boundary=x", b"application/octet-stream"):
            with self.subTest(content_type=content_type):
                self.captured.clear()
                self._run(self._reading_app(), messages=[{"type": "http.request", "body": b"FILEDATA"}],
                          headers=[(b"content-type", content_type)])
                self.assertNotIn("http.request.body", self.server_span().attributes)

    def test_the_response_is_kept_from_its_chunks_up_to_the_cap(self):
        saved = self.client._opts.max_body_bytes
        self.client._opts.max_body_bytes = 10
        try:
            app = self._reading_app(response_chunks=(b"0123456", b"789abc", b"def"), content_type=b"text/plain")
            _p, sent, _ = self._run(app, messages=[{"type": "http.request", "body": b""}])
        finally:
            self.client._opts.max_body_bytes = saved
        body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
        self.assertEqual(body, b"0123456789abcdef", "the client gets every byte")
        attrs = self.server_span().attributes
        self.assertEqual(attrs["http.response.body"], "0123456789")
        self.assertEqual(attrs["http.response.body.bytes"], 16)

    def test_a_server_sent_event_stream_is_not_buffered(self):
        app = self._reading_app(response_chunks=(b"data: 1\n\n", b"data: 2\n\n"), content_type=b"text/event-stream")
        self._run(app, messages=[{"type": "http.request", "body": b""}])
        self.assertNotIn("http.response.body", self.server_span().attributes)

    def test_the_request_trace_is_not_left_as_the_ambient_one(self):
        _p, _s, ambient_after = self._run(self._reading_app(), messages=[{"type": "http.request", "body": b""}])
        self.assertIsNone(ambient_after)


class FlaskContextTest(_Case):
    def setUp(self):
        try:
            from flask import Flask, request
        except ImportError:
            self.skipTest("flask not installed")
        super().setUp()
        self.app = Flask("context")
        self.request = request

        @self.app.route("/echo", methods=["POST"])
        def echo():
            return {"len": len(self.request.get_data())}

        evalkit.instrument_flask(self.app)

    def test_a_reused_thread_does_not_keep_the_previous_request_trace(self):
        self.app.test_client().post("/echo", data=b"{}", content_type="application/json")
        request_trace = self.server_span().trace_id
        self.assertNotEqual(_ambient_trace_id.get(), request_trace)
        end, ctx = self.client.start_span("after the request")
        end()
        self.assertNotEqual(ctx.trace_id, request_trace, "a later span on the same thread starts its own trace")

    def test_a_body_over_the_cap_or_an_upload_is_not_read(self):
        saved = self.client._opts.max_body_bytes
        import evalkit.frameworks.flask as flask_mod

        flask_mod._BODY_MAX = 16
        try:
            self.app.test_client().post("/echo", data=b'{"a":"' + b"x" * 100 + b'"}', content_type="application/json")
            self.assertNotIn("http.request.body", self.server_span().attributes)
            self.captured.clear()
            self.app.test_client().post("/echo", data=b'{"a":1}', content_type="application/json")
            self.assertEqual(json.loads(self.server_span().attributes["http.request.body"]), {"a": 1})
            self.captured.clear()
            self.app.test_client().post("/echo", data=b"bytes", content_type="application/octet-stream")
            self.assertNotIn("http.request.body", self.server_span().attributes)
        finally:
            flask_mod._BODY_MAX = saved

    def test_a_streamed_response_is_not_consumed_to_record_it(self):
        from flask import Response

        produced = []

        @self.app.route("/stream")
        def stream():
            def gen():
                for i in range(3):
                    produced.append(i)
                    yield f"chunk {i}\n"
            return Response(gen(), mimetype="text/plain")

        response = self.app.test_client().get("/stream", buffered=False)
        # Werkzeug's test client pulls the first chunk itself; reading the body to record it would pull all.
        self.assertLessEqual(len(produced), 1, "the middleware must not drain a streamed body")
        self.assertEqual(b"".join(response.response), b"chunk 0\nchunk 1\nchunk 2\n")
        response.close()


class DjangoContextTest(_Case):
    def setUp(self):
        try:
            import django
            from django.conf import settings
        except ImportError:
            self.skipTest("django not installed")
        if not settings.configured:
            settings.configure(
                DEBUG=False, ALLOWED_HOSTS=["*"], ROOT_URLCONF="tests.test_framework_spans", SECRET_KEY="test-only",
                MIDDLEWARE=["evalkit.frameworks.django.EvalKitDjangoMiddleware"],
            )
            django.setup()
        super().setUp()

    def test_a_reused_thread_does_not_keep_the_previous_request_trace(self):
        from django.test import Client, override_settings

        with override_settings(ROOT_URLCONF=__name__):
            Client().post("/echo/", data=b'{"a":1}', content_type="application/json")
        span = self.server_span()
        self.assertEqual(json.loads(span.attributes["http.request.body"]), {"a": 1})
        self.assertNotEqual(_ambient_trace_id.get(), span.trace_id)

    def test_an_upload_is_not_read_to_record_it(self):
        from django.test import Client, override_settings

        with override_settings(ROOT_URLCONF=__name__):
            Client().post("/echo/", data=b"x" * 100, content_type="application/octet-stream")
        self.assertNotIn("http.request.body", self.server_span().attributes)


def _django_urls():
    try:
        from django.http import JsonResponse
        from django.urls import path
        from django.views.decorators.csrf import csrf_exempt
    except Exception:
        return []

    @csrf_exempt
    def echo(request):
        return JsonResponse({"len": len(request.body)})

    return [path("echo/", echo)]


urlpatterns = _django_urls()


if __name__ == "__main__":
    unittest.main(verbosity=2)
