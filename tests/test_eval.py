import contextlib
import io
import json
import os
import unittest
import urllib.error

import evalkit
from evalkit.eval import MAX_ROWS_PER_BATCH, EvalResult, _rows_endpoint_support
from tests.helpers import DATASET_HASH, RunServer, fresh, reset_context

RULE_A = "665f0c1234567890abcdef12"
RULE_B = "665f0c1234567890abcdef34"
BASELINE = "665f0c1234567890abcdefaa"


def _pattern_outcome(g):
    if g % 10 == 0:
        return dict(score=None, passed=None, status="error", error="boom")
    if g % 10 == 1:
        return dict(score=None, passed=None, status="no_verdict")
    if g % 10 in (2, 3, 4):
        return dict(score=0.4, passed=False)
    return dict(score=0.9, passed=True)


def _patterned_row(index, row, score, passed, status="ok", error=None):
    metric_status = "ok" if status == "ok" else ("judge_error" if status == "error" else "unparseable")
    result = {
        "rowIndex": index,
        "input": row["input"],
        "output": row["output"],
        "metrics": [
            {
                "ruleId": RULE_A, "metricName": "Relevance", "score": score,
                "passed": passed, "status": metric_status, "reason": "r",
            }
        ],
        "overallScore": score,
        "passed": passed,
        "status": status,
        "counts": {
            "ok": 1 if metric_status == "ok" else 0,
            "unparseable": 1 if metric_status == "unparseable" else 0,
            "judgeError": 1 if metric_status == "judge_error" else 0,
            "skipped": 0,
            "unsupported": 0,
        },
    }
    if error is not None:
        result["error"] = error
    return result


def _patterned(body, n):
    rows = body["rows"]
    results = [
        _patterned_row(i, r, **_pattern_outcome(int(r["input"][1:]))) for i, r in enumerate(rows)
    ]
    return 200, {
        "jobId": f"job_{n}",
        "name": body.get("name"),
        "datasetHash": f"sha256:batch{n}",
        "totalRows": len(rows),
        "scoringVersion": 2,
        "gateResult": {"passed": False, "checks": [], "evaluatedAt": "x"},
        "comparison": {"comparable": False},
        "results": results,
    }


def _bad_request(body, n):
    return 400, {"statusCode": 400, "message": "judge model not found"}


class TestEval(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def _run(self, server, **kwargs):
        kwargs.setdefault("api_url", server.url)
        return evalkit.Eval(**kwargs)

    def test_exported_from_package(self):
        self.assertIs(evalkit.Eval, __import__("evalkit.eval", fromlist=["Eval"]).Eval)
        self.assertIs(evalkit.EvalResult, EvalResult)
        self.assertIn("Eval", evalkit.__all__)
        self.assertIn("EvalResult", evalkit.__all__)

    def test_create_rows_complete_sequence(self):
        with RunServer() as server:
            result = self._run(
                server,
                name="nightly",
                data=[
                    {"input": "q1", "output": "a1", "target": "t1"},
                    {"input": "q2", "output": "a2", "context": ["c1", "c2"], "metadata": {"tier": "pro"}},
                ],
                evaluators=[RULE_A],
                judge_model="gpt-4o",
            )

        self.assertEqual(
            server.paths(),
            ["/evaluation-runs", "/evaluation-runs/run_1/rows", "/evaluation-runs/run_1/complete"],
        )

        create = server.requests[0]["body"]
        self.assertEqual(create["name"], "nightly")
        self.assertEqual(create["ruleIds"], [RULE_A])
        self.assertEqual(create["judgeModelName"], "gpt-4o")
        self.assertNotIn("rows", create)
        self.assertNotIn("gate", create)
        self.assertNotIn("judgeModelConfigId", create)

        self.assertEqual(server.requests[1]["body"]["rows"], [
            {"rowIndex": 0, "input": "q1", "output": "a1", "expected": "t1"},
            {"rowIndex": 1, "input": "q2", "output": "a2",
             "metadata": {"tier": "pro", "context": ["c1", "c2"]}},
        ])
        self.assertEqual(server.completes, [{}])

        self.assertIsInstance(result, EvalResult)
        self.assertEqual(result.run_id, "run_1")
        self.assertEqual(result.url, "https://app.test/o/o1/p/p1/evaluations/run_1")
        self.assertEqual(result.job_id, "job_run_1")
        self.assertEqual(result.job_ids, ["job_run_1"])
        self.assertEqual(result.name, "nightly")
        self.assertEqual(result.dataset_hash, DATASET_HASH)
        self.assertEqual(result.total_rows, 2)
        self.assertAlmostEqual(result.average_score, 0.9)
        self.assertAlmostEqual(result.pass_rate, 1.0)
        self.assertEqual(result.counts["ok"], 2)
        self.assertEqual(result.scoring_version, 2)
        self.assertEqual(result.overall_scores, {"Relevance": 0.9})
        self.assertIsNone(result.gate_result)
        self.assertIsNone(result.passed)
        self.assertEqual([row["rowIndex"] for row in result.results], [0, 1])
        self.assertEqual(result.local_scores, {})

    def test_rows_with_task(self):
        seen = []

        def task(row):
            seen.append(row["input"])
            return "answer:" + row["input"]

        with RunServer() as server:
            result = self._run(
                server,
                name="with-task",
                data=[{"input": "q1", "output": "stale"}, {"input": "q2"}],
                task=task,
                evaluators=[RULE_A],
                judge_model_config_id="665f0c1234567890abcdef99",
            )

        self.assertEqual(seen, ["q1", "q2"])
        self.assertEqual(
            [row["output"] for row in server.requests[1]["body"]["rows"]],
            ["answer:q1", "answer:q2"],
        )
        self.assertEqual(server.requests[0]["body"]["judgeModelConfigId"], "665f0c1234567890abcdef99")
        self.assertNotIn("judgeModelName", server.requests[0]["body"])
        self.assertEqual(result.total_rows, 2)

    def test_missing_output_without_task_raises(self):
        with self.assertRaises(ValueError) as cm:
            evalkit.Eval("x", [{"input": "q1"}], evaluators=[RULE_A], api_url="http://127.0.0.1:1")
        self.assertIn("data[0]", str(cm.exception))

    def test_empty_data_raises(self):
        with self.assertRaises(ValueError) as cm:
            evalkit.Eval("x", [], evaluators=[RULE_A], api_url="http://127.0.0.1:1")
        self.assertIn("data is empty", str(cm.exception))

    def test_local_scores_travel_with_rows(self):
        def exact_match(row, output):
            return 1.0 if output == row.get("target") else 0.0

        def length_ok(row, output):
            return None if not output else float(len(output) <= 3)

        def detailed(row, output):
            return {"score": 0.25, "passed": False, "reason": "too terse"}

        with RunServer() as server:
            result = self._run(
                server,
                name="mixed",
                data=[
                    {"input": "q1", "output": "a1", "target": "a1"},
                    {"input": "q2", "output": "", "target": "zz"},
                ],
                evaluators=[RULE_A, exact_match, RULE_B, length_ok, detailed, lambda row, output: None],
                gate={"minPassRate": 0.5},
            )

        self.assertEqual(server.requests[0]["body"]["ruleIds"], [RULE_A, RULE_B])
        rows = server.requests[1]["body"]["rows"]
        self.assertEqual(rows[0]["localScores"], [
            {"name": "exact_match", "score": 1.0},
            {"name": "length_ok", "score": 1.0},
            {"name": "detailed", "score": 0.25, "passed": False, "reason": "too terse"},
        ])
        self.assertEqual(rows[1]["localScores"], [
            {"name": "exact_match", "score": 0.0},
            {"name": "detailed", "score": 0.25, "passed": False, "reason": "too terse"},
        ])
        self.assertEqual(result.local_scores["exact_match"], [1.0, 0.0])
        self.assertEqual(result.local_scores["length_ok"], [1.0, None])
        self.assertEqual(result.local_scores["<lambda>"], [None, None])
        for scores in result.local_scores.values():
            self.assertEqual(len(scores), 2)

    def test_boolean_local_score_carries_passed(self):
        with RunServer() as server:
            self._run(
                server,
                name="bool",
                data=[{"input": "q", "output": "a"}],
                evaluators=[RULE_A, lambda row, output: True],
            )
        self.assertEqual(
            server.requests[1]["body"]["rows"][0]["localScores"],
            [{"name": "<lambda>", "score": 1.0, "passed": True}],
        )

    def test_local_score_of_unsupported_type_raises(self):
        with self.assertRaises(ValueError) as cm:
            evalkit.Eval(
                "x", [{"input": "q", "output": "a"}],
                evaluators=[RULE_A, lambda row, output: "great"],
                api_url="http://127.0.0.1:1",
            )
        self.assertIn("local evaluator", str(cm.exception))

    def test_invalid_evaluator_entry_raises(self):
        for bad in ("relevance", "665f0c1234567890abcdef1", 42, None):
            with self.assertRaises(ValueError) as cm:
                evalkit.Eval("x", [{"input": "q", "output": "a"}], evaluators=[RULE_A, bad], api_url="http://127.0.0.1:1")
            self.assertIn(repr(bad), str(cm.exception))
            self.assertIn("evaluators[1]", str(cm.exception))

    def test_local_only_evaluators_raise(self):
        with self.assertRaises(ValueError):
            evalkit.Eval("x", [{"input": "q", "output": "a"}], evaluators=[lambda r, o: 1.0], api_url="http://127.0.0.1:1")

    def test_batch_boundaries(self):
        for total, expected in ((199, [199]), (200, [200]), (201, [200, 1]), (450, [200, 200, 50])):
            reset_context()
            with RunServer() as server:
                result = self._run(
                    server,
                    name=f"n{total}",
                    data=[{"input": f"q{i}", "output": f"a{i}"} for i in range(total)],
                    evaluators=[RULE_A],
                )
            row_bodies = server.bodies("/rows")
            self.assertEqual([len(body["rows"]) for body in row_bodies], expected, total)
            self.assertEqual(len(server.bodies("/evaluation-runs")), 1, total)
            self.assertEqual(len(server.completes), 1, total)
            indexes = [row["rowIndex"] for body in row_bodies for row in body["rows"]]
            self.assertEqual(indexes, list(range(total)), total)
            self.assertEqual([row["rowIndex"] for row in server.stored_rows()], list(range(total)))
            self.assertEqual(result.total_rows, total)

    def test_batch_size_never_exceeds_server_cap(self):
        with RunServer() as server:
            self._run(
                server,
                name="clamped",
                data=[{"input": f"q{i}", "output": "a"} for i in range(300)],
                evaluators=[RULE_A],
                batch_size=1000,
            )
        self.assertEqual([len(b["rows"]) for b in server.bodies("/rows")], [MAX_ROWS_PER_BATCH, 100])

    def test_batch_size_below_one_raises(self):
        with self.assertRaises(ValueError):
            evalkit.Eval("x", [{"input": "q", "output": "a"}], evaluators=[RULE_A], batch_size=0,
                         api_url="http://127.0.0.1:1")

    def test_retried_batch_is_idempotent_on_row_index(self):
        with RunServer(fail_rows_times=1) as server:
            result = self._run(
                server,
                name="flaky",
                data=[{"input": f"q{i}", "output": "a"} for i in range(250)],
                evaluators=[RULE_A],
            )
        row_bodies = server.bodies("/rows")
        self.assertEqual(len(row_bodies), 3)
        self.assertEqual(row_bodies[0], row_bodies[1])
        self.assertEqual([len(b["rows"]) for b in row_bodies], [200, 200, 50])
        self.assertEqual([row["rowIndex"] for row in server.stored_rows()], list(range(250)))
        self.assertEqual(result.total_rows, 250)

    def test_gate_is_applied_at_complete(self):
        gate = {"minPassRate": 0.8, "maxRegressionsVsBaseline": 0}
        with RunServer() as server:
            result = self._run(
                server,
                name="gated",
                data=[{"input": "q1", "output": "a1"}],
                evaluators=[RULE_A],
                baseline_job_id=BASELINE,
                gate=gate,
                auto_baseline=True,
                hyperparameters={"model": "gpt-5-mini"},
            )
        create = server.requests[0]["body"]
        self.assertEqual(create["baselineJobId"], BASELINE)
        self.assertIs(create["autoBaseline"], True)
        self.assertEqual(create["hyperparameters"], {"model": "gpt-5-mini"})
        self.assertNotIn("gate", create)
        self.assertEqual(server.completes, [{"gate": gate}])
        self.assertIs(result.passed, True)
        self.assertEqual(result.gate_result["checks"][0]["threshold"], 0.8)
        self.assertTrue(result.comparison["comparable"])

    def test_url_printed_once(self):
        buffer = io.StringIO()
        with RunServer() as server:
            with contextlib.redirect_stdout(buffer):
                result = self._run(
                    server,
                    name="printed",
                    data=[{"input": f"q{i}", "output": "a"} for i in range(400)],
                    evaluators=[RULE_A],
                )
        printed = [line for line in buffer.getvalue().splitlines() if result.url in line]
        self.assertEqual(len(printed), 1, buffer.getvalue())

    def test_url_not_printed_when_quiet(self):
        buffer = io.StringIO()
        os.environ["EVALKIT_QUIET"] = "1"
        try:
            with RunServer() as server:
                with contextlib.redirect_stdout(buffer):
                    result = self._run(
                        server, name="quiet", data=[{"input": "q", "output": "a"}], evaluators=[RULE_A]
                    )
        finally:
            os.environ.pop("EVALKIT_QUIET", None)
        self.assertEqual(buffer.getvalue(), "")
        self.assertEqual(result.url, "https://app.test/o/o1/p/p1/evaluations/run_1")

    def test_rows_endpoint_404_falls_back_to_single_call(self):
        with RunServer(mode="no-rows-endpoint") as server:
            result = self._run(
                server,
                name="old-server",
                data=[{"input": "q1", "output": "a1", "target": "t1"}],
                evaluators=[RULE_A],
                gate={"minPassRate": 0.5},
            )
            self.assertEqual(
                server.paths(), ["/evaluation-runs", "/evaluation-runs/run_1/rows", "/evaluation-runs"]
            )
            legacy = server.requests[2]["body"]
            self.assertEqual(legacy["rows"], [{"input": "q1", "output": "a1", "target": "t1"}])
            self.assertEqual(legacy["gate"], {"minPassRate": 0.5})
            self.assertEqual(result.job_id, "job_1")
            self.assertIsNone(result.run_id)
            self.assertIsNone(result.url)
            self.assertIs(result.passed, True)

            server.requests.clear()
            self._run(server, name="second", data=[{"input": "q2", "output": "a2"}], evaluators=[RULE_A])

        self.assertEqual(server.paths(), ["/evaluation-runs"])
        self.assertIn("rows", server.requests[0]["body"])

    def test_rowless_create_rejected_falls_back_to_single_call(self):
        with RunServer(mode="legacy") as server:
            result = self._run(
                server, name="old", data=[{"input": "q1", "output": "a1"}], evaluators=[RULE_A]
            )
        self.assertEqual(server.paths(), ["/evaluation-runs", "/evaluation-runs"])
        self.assertEqual(result.job_id, "job_1")
        self.assertIsNone(result.url)

    def test_legacy_multi_batch_rolls_up_client_side_without_gate(self):
        rows = [{"input": f"q{g}", "output": f"a{g}"} for g in range(450)]
        with RunServer(mode="legacy", legacy_respond=_patterned) as server:
            result = self._run(
                server,
                name="big",
                data=rows,
                evaluators=[RULE_A, lambda row, output: 1.0],
                baseline_job_id=BASELINE,
                gate={"minPassRate": 0.9},
            )

        legacy_bodies = [b for b in server.bodies("/evaluation-runs") if "rows" in b]
        self.assertEqual([len(b["rows"]) for b in legacy_bodies], [200, 200, 50])
        for body in legacy_bodies:
            self.assertNotIn("gate", body)
            self.assertNotIn("baselineJobId", body)
            self.assertNotIn("localScores", json.dumps(body))

        self.assertEqual(result.job_ids, ["job_1", "job_2", "job_3"])
        self.assertEqual(result.total_rows, 450)
        self.assertEqual([r["rowIndex"] for r in result.results], list(range(450)))
        self.assertAlmostEqual(result.average_score, 0.7125)
        self.assertAlmostEqual(result.pass_rate, 0.625)
        self.assertEqual(result.errored_rows, 45)
        self.assertEqual(result.indeterminate_rows, 45)
        self.assertEqual(result.counts, {"ok": 360, "unparseable": 45, "judgeError": 45, "skipped": 0, "unsupported": 0})
        self.assertIsNone(result.gate_result)
        self.assertIsNone(result.comparison)
        self.assertIsNone(result.passed)
        self.assertEqual(len(result.local_scores["<lambda>"]), 450)

    def test_passed_property_mirrors_gate_result(self):
        base = dict(job_id="j", job_ids=["j"], name="n", dataset_hash=None, total_rows=0, average_score=None,
                    pass_rate=None, counts=None, indeterminate_rows=0, errored_rows=0, scoring_version=2,
                    overall_scores={}, results=[], comparison=None, local_scores={})
        self.assertIsNone(EvalResult(gate_result=None, **base).passed)
        self.assertIs(EvalResult(gate_result={"passed": False, "checks": []}, **base).passed, False)
        self.assertIsNone(EvalResult(gate_result={"passed": None, "checks": []}, **base).passed)

    def test_header_present_and_no_key_in_body(self):
        with RunServer() as server:
            self._run(server, name="keys", data=[{"input": "q1", "output": "a1"}], evaluators=[RULE_A])

        for request in server.requests:
            self.assertEqual(request["key"], "tk_test")
            self.assertNotIn("tk_test", request["raw"])
            for forbidden in ("apiKey", "api_key", "subscriptionKey", "subscription_key"):
                self.assertNotIn(forbidden, request["raw"])
        self.assertEqual(set(server.requests[0]["body"]), {"name", "ruleIds"})

        self.client.flush()
        spans = [e for e in self.captured if (e.operation or "").startswith("POST /evaluation-runs")]
        self.assertEqual(
            [e.operation for e in spans],
            ["POST /evaluation-runs", "POST /evaluation-runs/:runId/rows", "POST /evaluation-runs/:runId/complete"],
        )
        for span in spans:
            attrs = getattr(span, "attributes", {}) or {}
            self.assertNotIn("tk_test", json.dumps(attrs, default=str))
            self.assertEqual(span.span_type, "http_call")
            self.assertIn(attrs.get("http.status_code"), (200, 201, 202))

    def test_http_error_propagates(self):
        with RunServer(mode="legacy", legacy_respond=_bad_request) as server:
            with self.assertRaises(urllib.error.HTTPError) as cm:
                self._run(server, name="bad", data=[{"input": "q1", "output": "a1"}], evaluators=[RULE_A])
        self.assertEqual(cm.exception.code, 400)
        self.assertIs(_rows_endpoint_support.get(self.client), False)
        self.client.flush()
        spans = [e for e in self.captured if (e.operation or "").startswith("POST /evaluation-runs")]
        self.assertTrue(spans)
        self.assertEqual((getattr(spans[0], "attributes", {}) or {}).get("http.status_code"), 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
