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
        with RunServer(mode="legacy") as server:
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
