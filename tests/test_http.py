import asyncio
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import evalkit
from tests.helpers import fresh, reset_context


def _start_server():
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class TestRequestsHTTP(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_requests_emits_http_call_span(self):
        try:
            import requests
        except ImportError:
            self.skipTest("requests not installed")

        srv = _start_server()
        port = srv.server_address[1]
        try:
            tid, end, ctx = self.client.start_trace("req")
            with evalkit.with_trace(ctx):
                requests.get(f"http://127.0.0.1:{port}/data")
            end("OK")
        finally:
            srv.shutdown()

        self.client.flush()
        http = [
            e for e in self.captured
            if e.span_type == "http_call" and "127.0.0.1" in (e.attributes or {}).get("http.url", "")
        ]
        self.assertTrue(http, "expected an http_call span")
        self.assertEqual(http[0].trace_id, tid)
        self.assertIn("traceparent", (http[0].attributes or {}).get("http.request.headers", ""))


def _start_recording_server(seen_traceparents):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            seen_traceparents.append(self.headers.get("traceparent"))
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _urllib3_instrumented():
    """OpenTelemetry's urllib3 instrumentation, on, as `setup_otel` turns it on when it is installed."""
    try:
        from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor
    except ImportError:
        return False
    instrumentor = URLLib3Instrumentor()
    if not instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.instrument()
    return True


class TestRequestsOverUrllib3(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self.client.flush()
        self.captured.clear()

    def _calls_to(self, port):
        return [
            e for e in self.captured
            if e.span_type == "http_call"
            and f"127.0.0.1:{port}" in str((e.attributes or {}).get("http.url") or (e.attributes or {}).get("url.full") or "")
        ]

    def test_a_requests_call_is_one_span_with_urllib3_instrumentation_installed(self):
        try:
            import requests
        except ImportError:
            self.skipTest("requests not installed")
        if not _urllib3_instrumented():
            self.skipTest("opentelemetry-instrumentation-urllib3 not installed")

        seen = []
        srv = _start_recording_server(seen)
        port = srv.server_address[1]
        try:
            tid, end, ctx = self.client.start_trace("req")
            with evalkit.with_trace(ctx):
                requests.get(f"http://127.0.0.1:{port}/charges/ch_demo_4821")
            end("OK")
        finally:
            srv.shutdown()
        self.client.flush()

        calls = self._calls_to(port)
        self.assertEqual(
            len(calls), 1,
            "requests sends through urllib3, so the urllib3 instrumentation recorded the same call as a second "
            f"POST/GET span next to the SDK's; got {[(e.operation, e.span_kind) for e in calls]}",
        )
        span = calls[0]
        self.assertEqual(span.trace_id, tid)
        self.assertEqual(seen, [f"00-{tid}-{span.span_id}-01"], (
            "the traceparent the server receives must name the SDK's span; the urllib3 instrumentation "
            "would replace it with the id of a span that is never exported"
        ))

    def test_urllib3_used_directly_is_still_traced(self):
        try:
            import urllib3
        except ImportError:
            self.skipTest("urllib3 not installed")
        if not _urllib3_instrumented():
            self.skipTest("opentelemetry-instrumentation-urllib3 not installed")

        srv = _start_recording_server([])
        port = srv.server_address[1]
        try:
            tid, end, ctx = self.client.start_trace("boto-like")
            with evalkit.with_trace(ctx):
                urllib3.PoolManager().request("GET", f"http://127.0.0.1:{port}/bucket/key")
            end("OK")
        finally:
            srv.shutdown()
        self.client.flush()

        calls = self._calls_to(port)
        self.assertEqual(
            len(calls), 1,
            "botocore and boto3 call urllib3 directly; only the call requests makes on the SDK's behalf may be "
            "left out, so a direct urllib3 call keeps its span",
        )
        self.assertEqual(calls[0].trace_id, tid)


class TestClientSpanNames(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def _only_call(self, port):
        self.client.flush()
        calls = [e for e in self.captured
                 if e.span_type == "http_call" and f"127.0.0.1:{port}" in (e.attributes or {}).get("http.url", "")]
        self.assertEqual(len(calls), 1, [(e.operation, e.span_kind) for e in calls])
        return calls[0]

    def test_requests_span_is_named_by_method_and_keeps_the_url(self):
        try:
            import requests
        except ImportError:
            self.skipTest("requests not installed")
        srv = _start_server()
        port = srv.server_address[1]
        url = f"http://127.0.0.1:{port}/orders/4429?expand=items"
        try:
            with evalkit.with_trace(self.client.start_trace("req")[2]):
                requests.get(url)
        finally:
            srv.shutdown()
        span = self._only_call(port)
        self.assertEqual(span.operation, "GET", (
            "a client span named by its path makes every order id a separate operation; OpenTelemetry names a "
            "client span with no known URL template by its method"
        ))
        self.assertEqual(span.span_kind, "CLIENT")
        self.assertEqual(span.attributes["http.url"], url, "the full URL, ids and query, stays on the span")

    def test_httpx_span_is_named_by_method(self):
        try:
            import httpx
        except ImportError:
            self.skipTest("httpx not installed")
        srv = _start_server()
        port = srv.server_address[1]
        try:
            with evalkit.with_trace(self.client.start_trace("httpx")[2]):
                httpx.get(f"http://127.0.0.1:{port}/orders/4429")
        finally:
            srv.shutdown()
        span = self._only_call(port)
        self.assertEqual(span.operation, "GET")
        self.assertEqual(span.span_kind, "CLIENT")
        self.assertTrue(span.attributes["http.url"].endswith("/orders/4429"))


class TestHttpxHTTP(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_httpx_async_emits_http_call_span(self):
        try:
            import httpx
        except ImportError:
            self.skipTest("httpx not installed")

        srv = _start_server()
        port = srv.server_address[1]
        try:
            tid, end, ctx = self.client.start_trace("httpx-req")
            with evalkit.with_trace(ctx):
                async def run():
                    async with httpx.AsyncClient() as c:
                        await c.get(f"http://127.0.0.1:{port}/test")
                asyncio.run(run())
            end("OK")
        finally:
            srv.shutdown()

        self.client.flush()
        http = [
            e for e in self.captured
            if e.span_type == "http_call" and "127.0.0.1" in (e.attributes or {}).get("http.url", "")
        ]
        self.assertTrue(http, "expected an http_call span from httpx")
        self.assertEqual(http[0].trace_id, tid)


class TestAiohttpHTTP(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_aiohttp_emits_http_call_span(self):
        try:
            import aiohttp
        except ImportError:
            self.skipTest("aiohttp not installed")

        from evalkit.instrumentation.http.aiohttp import patch_aiohttp
        patch_aiohttp(self.client)

        srv = _start_server()
        port = srv.server_address[1]

        async def run():
            tid, end, ctx = self.client.start_trace("aiohttp-req")
            with evalkit.with_trace(ctx):
                async with aiohttp.ClientSession() as sess:
                    async with sess.get(f"http://127.0.0.1:{port}/test") as resp:
                        await resp.read()
            end("OK")
            return tid

        try:
            tid = asyncio.run(run())
        finally:
            srv.shutdown()

        self.client.flush()
        http = [
            e for e in self.captured
            if e.span_type == "http_call" and "127.0.0.1" in (e.attributes or {}).get("http.url", "")
        ]
        self.assertTrue(http, "expected an http_call span from aiohttp")
        self.assertEqual(http[0].trace_id, tid)
        self.assertIn("http.method", http[0].attributes)
        self.assertEqual(http[0].attributes["http.status_code"], 200)

    def test_aiohttp_patch_idempotent(self):
        try:
            import aiohttp
        except ImportError:
            self.skipTest("aiohttp not installed")

        from evalkit.instrumentation.http.aiohttp import patch_aiohttp
        orig = aiohttp.ClientSession._request
        patch_aiohttp(self.client)
        patched_once = aiohttp.ClientSession._request
        patch_aiohttp(self.client)
        self.assertIs(aiohttp.ClientSession._request, patched_once)
        aiohttp.ClientSession._request = orig


if __name__ == "__main__":
    unittest.main(verbosity=2)
