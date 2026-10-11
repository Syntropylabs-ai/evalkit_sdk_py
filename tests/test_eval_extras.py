import unittest

import evalkit
from tests.helpers import RunServer, fresh, reset_context

RULE = "665f0c1234567890abcdef12"


class TestEvalExtras(unittest.TestCase):
    def setUp(self):
        reset_context()
        fresh([])

    def test_hyperparameters_and_auto_baseline_sent_once_on_create(self):
        with RunServer() as server:
            result = evalkit.Eval(
                "hp",
                [{"input": str(i), "output": "b"} for i in range(250)],
                evaluators=[RULE],
                hyperparameters={"model": "gpt-5-mini", "temperature": 0.2},
                auto_baseline=True,
                api_url=server.url,
            )
        create = server.requests[0]["body"]
        self.assertEqual(create["hyperparameters"], {"model": "gpt-5-mini", "temperature": 0.2})
        self.assertIs(create["autoBaseline"], True)
        self.assertEqual(len(server.bodies("/rows")), 2)
        for body in server.bodies("/rows"):
            self.assertEqual(set(body), {"rows"})
        self.assertEqual(result.job_id, "job_run_1")
        self.assertEqual(result.total_rows, 250)

    def test_legacy_fallback_keeps_single_batch_only_fields(self):
        with RunServer(mode="no-rows-endpoint") as server:
            evalkit.Eval(
                "hp",
                [{"input": str(i), "output": "b"} for i in range(3)],
                evaluators=[RULE],
                hyperparameters={"model": "m"},
                auto_baseline=True,
                batch_size=2,
                api_url=server.url,
            )
        legacy = [b for b in server.bodies("/evaluation-runs") if "rows" in b]
        self.assertEqual(len(legacy), 2)
        self.assertTrue(all("autoBaseline" not in b for b in legacy))
        self.assertTrue(all(b["hyperparameters"] == {"model": "m"} for b in legacy))


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Find an order by id",
            "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
        },
    }
]

MESSAGES = [
    {"role": "user", "content": "Where is order 42?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup_order", "arguments": '{"order_id": "42"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "shipped"}'},
    {"role": "assistant", "content": "Order 42 has shipped."},
]


class TestEvalAgentRows(unittest.TestCase):
    def setUp(self):
        reset_context()
        fresh([])

    def _rows_sent(self, data):
        with RunServer() as server:
            evalkit.Eval("agent", data, evaluators=[RULE], api_url=server.url)
        return server.bodies("/rows")[0]["rows"]

    def test_agent_fields_travel_as_top_level_keys(self):
        rows = self._rows_sent([
            {"input": "q0", "output": "a0", "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736"},
            {"input": "q1", "output": "a1", "session_id": "sess_1", "metadata": {"tier": "pro"}},
            {
                "input": "Where is order 42?",
                "output": "Order 42 has shipped.",
                "messages": MESSAGES,
                "tools": TOOLS,
                "system_prompt": "You are a support agent.",
            },
        ])
        self.assertEqual(rows[0], {
            "rowIndex": 0, "input": "q0", "output": "a0", "traceId": "4bf92f3577b34da6a3ce929d0e0e4736",
        })
        self.assertEqual(rows[1], {
            "rowIndex": 1, "input": "q1", "output": "a1", "sessionId": "sess_1", "metadata": {"tier": "pro"},
        })
        self.assertEqual(rows[2]["messages"], MESSAGES)
        self.assertEqual(rows[2]["tools"], TOOLS)
        self.assertEqual(rows[2]["systemPrompt"], "You are a support agent.")
        self.assertNotIn("metadata", rows[2])

    def test_absent_or_empty_agent_fields_are_omitted(self):
        rows = self._rows_sent([
            {"input": "q0", "output": "a0"},
            {"input": "q1", "output": "a1", "messages": [], "tools": [], "system_prompt": "", "trace_id": None},
        ])
        wire = {"traceId", "sessionId", "messages", "tools", "systemPrompt"}
        for row in rows:
            self.assertFalse(wire & set(row), row)
            self.assertFalse({"trace_id", "session_id", "system_prompt"} & set(row), row)

    def test_bad_agent_field_types_raise_with_row_index(self):
        cases = [
            ({"trace_id": 42}, TypeError, "data[1]['trace_id'] must be a string"),
            ({"session_id": ""}, ValueError, "data[1]['session_id'] is empty"),
            ({"trace_id": "   "}, ValueError, "data[1]['trace_id'] is empty"),
            ({"messages": "hello"}, TypeError, "data[1]['messages'] must be a list"),
            ({"tools": {"type": "function"}}, TypeError, "data[1]['tools'] must be a list"),
            ({"system_prompt": ["a"]}, TypeError, "data[1]['system_prompt'] must be a string"),
        ]
        for extra, error, message in cases:
            with self.subTest(extra=extra):
                data = [{"input": "q0", "output": "a0"}, {"input": "q1", "output": "a1", **extra}]
                with self.assertRaises(error) as cm:
                    evalkit.Eval("x", data, evaluators=[RULE], api_url="http://127.0.0.1:1")
                self.assertIn(message, str(cm.exception))

    def test_bad_agent_field_raises_before_the_task_runs(self):
        calls = []
        with self.assertRaises(TypeError):
            evalkit.Eval(
                "x",
                [{"input": "q0", "messages": "hello"}],
                task=lambda row: calls.append(row) or "a",
                evaluators=[RULE],
                api_url="http://127.0.0.1:1",
            )
        self.assertEqual(calls, [])
