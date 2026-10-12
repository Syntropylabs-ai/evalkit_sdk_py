import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import evalkit
from evalkit.simulation import AgentTurnResult
from tests.helpers import fresh, reset_context


class _UserTurnServer:
    """Stands in for POST /simulations/user-turn: `answer(body)` returns (status, payload)."""

    def __init__(self, answer):
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                outer.requests.append({"path": self.path, "key": self.headers.get("X-Subscription-Key"), "body": body})
                status, payload = answer(body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def close(self):
        self.srv.shutdown()


USAGE = {"inputTokens": 400, "outputTokens": 12, "costUsd": 0.0001}


def refund_user(body):
    """Done once the agent confirms the refund; otherwise hands over the order number."""
    last = body["history"][-1]["content"]
    if "refunded" in last:
        return 200, {"done": True, "usage": USAGE}
    return 200, {"done": False, "message": "It is order 1042.", "usage": USAGE}


SCENARIO = {
    "name": "refund",
    "starting_prompt": "I want a refund.",
    "user_goal": "get the refund for order 1042",
    "persona": "IMPATIENT",
    "category": "tool_failure",
    "hidden_info": {"order": {"value": "order 1042", "reveal_trigger": "the order number"}},
    "tool_fixtures": [{"tool": "lookup_order", "input": {"order_id": "1042"}, "output": {"found": True}}],
    "expected_tools": ["lookup_order", "issue_refund"],
}


def refund_agent(ctx):
    if "1042" not in ctx.message:
        return "Which order is it?"
    return AgentTurnResult(
        text="Done, I refunded $59 for order 1042.",
        tool_calls=[
            {"name": "lookup_order", "args": {"order_id": "1042"}, "result": {"amount": 59}},
            {"name": "issue_refund", "args": {"order_id": "1042"}, "error": "payment provider timeout"},
        ],
    )


class TestLiveSimulatedUser(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def _run(self, answer, entrypoint, scenarios, **kw):
        server = _UserTurnServer(answer)
        try:
            report = evalkit.simulate_user(entrypoint, scenarios, api_url=server.url, **kw)
        finally:
            server.close()
        return report, server.requests

    def test_plays_the_user_until_it_is_done(self):
        seen = []

        def agent(ctx):
            seen.append(ctx.message)
            return refund_agent(ctx)

        report, requests = self._run(refund_user, agent, [SCENARIO], user_model="mock gpt-4o-mini", agent="refund-agent")
        result = report["results"][0]
        self.assertEqual(result["status"], "OK")
        self.assertEqual(seen, ["I want a refund.", "It is order 1042."])
        self.assertEqual(result["turns"], 2)
        self.assertEqual(result["ended_by"], "user")
        self.assertEqual(
            [m["content"] for m in result["transcript"]],
            ["I want a refund.", "Which order is it?", "It is order 1042.", "Done, I refunded $59 for order 1042."],
        )
        self.assertEqual(result["simulated_user_usage"], {"calls": 2, "input_tokens": 800, "output_tokens": 24, "cost_usd": 0.0002})
        self.assertEqual(result["scores"].get("tool_correctness"), 1.0)

        self.assertEqual([r["path"] for r in requests], ["/simulations/user-turn"] * 2)
        self.assertEqual(requests[0]["key"], "tk_test")
        first, second = (r["body"] for r in requests)
        self.assertEqual(first["turn"], 2)
        self.assertEqual(first["model"], "mock gpt-4o-mini")
        self.assertEqual(
            first["scenario"],
            {
                "maxTurns": 12,
                "persona": "IMPATIENT",
                "userGoal": "get the refund for order 1042",
                "name": "refund",
                "scenarioType": "tool_failure",
                "hiddenInfo": SCENARIO["hidden_info"],
                "fixtures": SCENARIO["tool_fixtures"],
            },
        )
        self.assertEqual(first["history"], [
            {"role": "user", "content": "I want a refund."},
            {"role": "assistant", "content": "Which order is it?"},
        ])
        self.assertEqual(second["history"][-1], {
            "role": "assistant",
            "content": "Done, I refunded $59 for order 1042.",
            "toolCalls": [
                {"name": "lookup_order", "arguments": {"order_id": "1042"}, "result": {"amount": 59}},
                {"name": "issue_refund", "arguments": {"order_id": "1042"}, "error": "payment provider timeout"},
            ],
        })

    def test_records_each_turn_as_the_simulations_user_never_the_agent(self):
        report, _ = self._run(refund_user, refund_agent, [SCENARIO], user_model="m", agent="refund-agent")
        roots = [e for e in self.captured if (e.operation or "").startswith("scenario:")]
        users = [e for e in self.captured if e.operation == "simulated_user"]
        self.assertEqual(len(roots), 1)
        self.assertEqual(len(users), 2)
        self.assertEqual(roots[0].agent_name, "refund-agent")
        for span in users:
            self.assertEqual(span.attributes["evalkit.simulation.role"], "user")
            self.assertEqual(span.span_type, "function_call")
            self.assertEqual(span.parent_span_id, roots[0].span_id)
            self.assertEqual(span.trace_id, roots[0].trace_id)
            self.assertEqual(span.session_id, roots[0].session_id)
            self.assertFalse(span.agent_name)
            self.assertNotIn("evalkit.agent_name", span.attributes)
            self.assertNotIn("gen_ai.agent.name", span.attributes)
            self.assertIsNone(span.tokens_in)
            self.assertTrue(span.is_simulation)
            self.assertEqual(span.simulation_id, report["simulation_id"])
        self.assertEqual([u.completion for u in users], ["It is order 1042.", None])
        self.assertEqual([u.attributes["evalkit.simulated_user.done"] for u in users], [False, True])
        self.assertEqual(users[0].attributes["evalkit.simulated_user.input_tokens"], 400)
        self.assertFalse([e for e in self.captured if e.span_type == "http_call"], "the user turn's request is the user span, not a second span")

    def test_max_turns_caps_a_user_who_is_never_satisfied(self):
        report, requests = self._run(
            lambda body: (200, {"done": False, "message": "That is not good enough.", "usage": USAGE}),
            lambda ctx: "Sorry.",
            [SCENARIO],
            max_turns=3,
            user_model="m",
        )
        result = report["results"][0]
        self.assertEqual(result["turns"], 3)
        self.assertEqual(result["ended_by"], "max_turns")
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["body"]["scenario"]["maxTurns"], 3)

    def test_turns_only_scenarios_replay_as_written(self):
        report, requests = self._run(
            refund_user,
            lambda ctx: "ok",
            [{"name": "s", "turns": ["one", "two"]}, {"name": "p", "starting_prompt": "hi", "user_goal": "g", "turns": ["hi"]}],
            user_model="m",
        )
        self.assertEqual([r["turns"] for r in report["results"]], [2, 1])
        self.assertNotIn("transcript", report["results"][0])
        self.assertEqual(requests, [])

    def test_live_user_true_continues_after_the_scripted_turns(self):
        seen = []

        def agent(ctx):
            seen.append(ctx.message)
            return refund_agent(ctx)

        report, requests = self._run(
            refund_user, agent, [{**SCENARIO, "turns": ["I want a refund.", "Hello?"]}], live_user=True, model="fallback"
        )
        self.assertEqual(seen, ["I want a refund.", "Hello?", "It is order 1042."])
        self.assertEqual(requests[0]["body"]["turn"], 3)
        self.assertEqual(requests[0]["body"]["model"], "fallback")
        self.assertEqual(report["results"][0]["ended_by"], "user")

    def test_turn_events_play_before_the_live_user(self):
        seen = []
        _, requests = self._run(
            refund_user,
            lambda ctx: (seen.append(ctx.message), refund_agent(ctx))[1],
            [{**SCENARIO, "turn_events": [{"at_user_turn": 2, "message": "Also, my order is 1042."}]}],
            user_model="m",
        )
        self.assertEqual(seen, ["I want a refund.", "Also, my order is 1042."])
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["body"]["turn"], 3)

    def test_async_entrypoints_meet_the_same_user(self):
        async def agent(ctx):
            return refund_agent(ctx)

        report, requests = self._run(refund_user, agent, [SCENARIO], user_model="m", agent="refund-agent")
        result = report["results"][0]
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["turns"], 2)
        self.assertEqual(len(requests), 2)
        users = [e for e in self.captured if e.operation == "simulated_user"]
        root = next(e for e in self.captured if (e.operation or "").startswith("scenario:"))
        self.assertEqual(len(users), 2)
        self.assertTrue(all(u.parent_span_id == root.span_id and not u.agent_name for u in users))

    def test_a_refused_turn_ends_the_scenario_with_the_platforms_reason(self):
        report, _ = self._run(
            lambda body: (400, {"statusCode": 400, "message": "No model is configured for the simulated user."}),
            lambda ctx: "Which order?",
            [SCENARIO],
        )
        result = report["results"][0]
        self.assertEqual(result["status"], "ERROR")
        self.assertIn("simulated user failed at turn 2: 400 No model is configured", result["error"])
        self.assertEqual(result["turns"], 1)
        failed = [e for e in self.captured if e.operation == "simulated_user"]
        self.assertEqual([f.status for f in failed], ["ERROR"])

    def test_generate_scenarios_asks_for_a_live_user_unless_scripted_turns_is_set(self):
        server = _UserTurnServer(lambda body: (200, {"scenarios": [{"name": "s"}]}))
        try:
            evalkit.generate_scenarios("agent", api_url=server.url)
            evalkit.generate_scenarios("agent", api_url=server.url, scripted_turns=True)
        finally:
            server.close()
        self.assertEqual([r["body"]["scriptedTurns"] for r in server.requests], [False, True])


if __name__ == "__main__":
    unittest.main()
