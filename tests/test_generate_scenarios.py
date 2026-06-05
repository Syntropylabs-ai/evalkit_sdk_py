import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import evalkit
from tests.helpers import fresh, reset_context


class TestGenerateScenarios(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def _start_scenarios_server(self):
        recv = {}

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                ln = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(ln)
                recv["path"] = self.path
                recv["key"] = self.headers.get("X-Subscription-Key")
                recv["body"] = json.loads(body or b"{}")
                payload = json.dumps({
                    "scenarios": [{"name": "s1", "starting_prompt": "hi", "expected_tools": ["t"]}]
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv, recv

    def test_request_shape_and_response(self):
        srv, recv = self._start_scenarios_server()
        port = srv.server_address[1]
        try:
            scenarios = evalkit.generate_scenarios(
                "You are a billing agent.",
                tools=["issue_refund"],
                count=2,
                api_url=f"http://127.0.0.1:{port}",
            )
        finally:
            srv.shutdown()

        self.assertEqual(len(scenarios), 1)
        self.assertEqual(recv["path"], "/scenarios/generate")
        self.assertEqual(recv["key"], "tk_test")
        self.assertEqual(recv["body"]["agentInstructions"], "You are a billing agent.")
        self.assertEqual(recv["body"]["tools"], ["issue_refund"])
        self.assertEqual(recv["body"]["count"], 2)

    def test_emits_http_span(self):
        srv, recv = self._start_scenarios_server()
        port = srv.server_address[1]
        try:
            evalkit.generate_scenarios(
                "billing agent",
                tools=["refund"],
                api_url=f"http://127.0.0.1:{port}",
            )
        finally:
            srv.shutdown()

        self.client.flush()
        spans = [
            e for e in self.captured
            if (e.operation or "").startswith("POST /scenarios/generate")
        ]
        self.assertTrue(spans, "generate_scenarios should emit an http_call span")


if __name__ == "__main__":
    unittest.main(verbosity=2)
