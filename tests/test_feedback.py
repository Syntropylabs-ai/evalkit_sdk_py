import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import evalkit
from tests.helpers import fresh, reset_context


def _server(status=200, payload=None):
    recv = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            ln = int(self.headers.get("Content-Length", 0))
            recv["path"] = self.path
            recv["key"] = self.headers.get("X-Subscription-Key")
            recv["body"] = json.loads(self.rfile.read(ln) or b"{}")
            body = json.dumps(payload if payload is not None else {"scoreId": "s1", "recorded": True}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, recv


class TestFeedback(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_thumbs_up_posts_normalised_payload(self):
        srv, recv = _server()
        try:
            url = f"http://127.0.0.1:{srv.server_address[1]}"
            out = evalkit.feedback("t" * 32, "thumbs", True, comment="great", user_id="u1", session_id="s1", api_url=url)
        finally:
            srv.shutdown()
        self.assertEqual(out["recorded"], True)
        self.assertEqual(recv["path"], "/v1/feedback")
        self.assertEqual(recv["key"], "tk_test")
        self.assertEqual(recv["body"], {"traceId": "t" * 32, "kind": "thumbs", "value": 1.0, "sessionId": "s1", "comment": "great", "userId": "u1"})
        self.client.flush()
        spans = [e for e in self.captured if (e.attributes or {}).get("http.url", "").endswith("/v1/feedback")]
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status, "OK")

    def test_reward_and_rating(self):
        srv, recv = _server()
        try:
            url = f"http://127.0.0.1:{srv.server_address[1]}"
            evalkit.reward("t" * 32, 0.8, span_id="sp1", name="task_reward", api_url=url)
            self.assertEqual(recv["body"], {"traceId": "t" * 32, "kind": "reward", "value": 0.8, "spanId": "sp1", "name": "task_reward"})
            evalkit.feedback("t" * 32, "rating", 4, api_url=url)
            self.assertEqual(recv["body"]["value"], 4.0)
            evalkit.feedback("t" * 32, "text", "too verbose", api_url=url)
            self.assertEqual(recv["body"]["value"], "too verbose")
        finally:
            srv.shutdown()

    def test_validation(self):
        with self.assertRaises(ValueError):
            evalkit.feedback("t" * 32, "stars", 5)
        with self.assertRaises(ValueError):
            evalkit.feedback("", "thumbs", True)
        with self.assertRaises(ValueError):
            evalkit.feedback("t" * 32, "rating", "high")
        with self.assertRaises(ValueError):
            evalkit.feedback("t" * 32, "text", "")

    def test_http_error_raises_and_marks_span(self):
        srv, _ = _server(status=404, payload={"message": "trace not found"})
        try:
            url = f"http://127.0.0.1:{srv.server_address[1]}"
            with self.assertRaises(Exception):
                evalkit.feedback("t" * 32, "thumbs", False, api_url=url)
        finally:
            srv.shutdown()
        self.client.flush()
        spans = [e for e in self.captured if (e.attributes or {}).get("http.url", "").endswith("/v1/feedback")]
        self.assertEqual(spans[-1].status, "ERROR")
        self.assertEqual(spans[-1].attributes["http.status_code"], 404)
