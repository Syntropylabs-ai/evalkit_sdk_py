import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import evalkit
from tests.helpers import fresh, reset_context


class TestEvaluateSimulation(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def _start_eval_server(self):
        recv = {}

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                ln = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(ln)
                recv["path"] = self.path
                recv["key"] = self.headers.get("X-Subscription-Key")
                recv["body"] = json.loads(body or b"{}")
                payload = json.dumps({
                    "simulationId": "sim_abc",
                    "runId": "run_xyz",
                    "scenarios": [
                        {
                            "scenarioId": "scn_1",
                            "name": "happy path",
                            "traceId": "trace_001",
                            "status": "OK",
                            "overallScore": 0.9,
                            "passed": True,
                            "metrics": [
                                {
                                    "ruleId": "r1", "ruleName": "Relevance",
                                    "category": "quality", "score": 0.9,
                                    "passed": True, "reason": "On topic",
                                }
                            ],
                        }
                    ],
                    "aggregate": {
                        "averageScore": 0.9, "passRate": 1.0,
                        "totalScenarios": 1, "evaluatedScenarios": 1,
                    },
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
        srv, recv = self._start_eval_server()
        port = srv.server_address[1]
        try:
            result = evalkit.evaluate_simulation(
                "sim_abc", "col_xyz",
                provider="openai", model="gpt-4o-mini", api_key="sk-test",
                run_id="run_xyz",
                api_url=f"http://127.0.0.1:{port}",
            )
        finally:
            srv.shutdown()

        self.assertEqual(recv["path"], "/simulations/evaluate")
        self.assertEqual(recv["key"], "tk_test")
        self.assertEqual(recv["body"]["simulationId"], "sim_abc")
        self.assertEqual(recv["body"]["collectionId"], "col_xyz")
        self.assertEqual(recv["body"]["runId"], "run_xyz")
        self.assertEqual(recv["body"]["provider"], "openai")
        self.assertEqual(recv["body"]["model"], "gpt-4o-mini")
        self.assertEqual(recv["body"]["apiKey"], "sk-test")
        self.assertNotIn("maxTokens", recv["body"])
        self.assertEqual(len(result["scenarios"]), 1)
        self.assertAlmostEqual(result["aggregate"]["averageScore"], 0.9)

    def test_optional_fields_omitted_when_not_provided(self):
        srv, recv = self._start_eval_server()
        port = srv.server_address[1]
        try:
            evalkit.evaluate_simulation(
                "sim_abc", "col_xyz",
                provider="openai", model="gpt-4o-mini", api_key="sk-test",
                api_url=f"http://127.0.0.1:{port}",
            )
        finally:
            srv.shutdown()

        self.assertNotIn("runId", recv["body"])
        self.assertNotIn("maxTokens", recv["body"])

    def test_max_tokens_included_when_provided(self):
        srv, recv = self._start_eval_server()
        port = srv.server_address[1]
        try:
            evalkit.evaluate_simulation(
                "sim_abc", "col_xyz",
                provider="openai", model="gpt-4o-mini", api_key="sk-test",
                max_tokens=512,
                api_url=f"http://127.0.0.1:{port}",
            )
        finally:
            srv.shutdown()

        self.assertEqual(recv["body"]["maxTokens"], 512)

    def test_emits_http_span_without_leaking_api_key(self):
        srv, recv = self._start_eval_server()
        port = srv.server_address[1]
        try:
            evalkit.evaluate_simulation(
                "sim_abc", "col_xyz",
                provider="openai", model="gpt-4o-mini", api_key="sk-secret",
                api_url=f"http://127.0.0.1:{port}",
            )
        finally:
            srv.shutdown()

        self.client.flush()
        spans = [
            e for e in self.captured
            if (e.operation or "").startswith("POST /simulations/evaluate")
        ]
        self.assertTrue(spans, "evaluate_simulation should emit an http_call span")
        for s in spans:
            attrs = getattr(s, "attributes", {}) or {}
            self.assertNotIn("sk-secret", json.dumps(attrs, default=str))


if __name__ == "__main__":
    unittest.main(verbosity=2)
