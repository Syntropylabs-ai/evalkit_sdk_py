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


if __name__ == "__main__":
    unittest.main(verbosity=2)
