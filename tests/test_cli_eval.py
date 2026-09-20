import io
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import evalkit
from evalkit_cli.evaluation import (
    EXIT_ERROR,
    EXIT_FAILED,
    EXIT_PASSED,
    EXIT_UNDECIDED,
    EvalCliError,
    RunReport,
    exit_code,
    load_json_spec,
    load_module_spec,
    normalise_spec,
    render_gate_table,
    resolve_spec,
    verdict,
)
from evalkit_cli.main import main
from tests.helpers import RunServer, fresh, reset_context

RULE = "665f0c1234567890abcdef12"
UNDECIDED_GATE = {
    "passed": None,
    "checks": [
        {"name": "minPassRate", "threshold": 0.9, "actual": 1.0, "passed": True},
        {"name": "maxRegressionsVsBaseline", "threshold": 0, "actual": None, "passed": None},
    ],
    "evaluatedAt": "2026-09-16T00:00:00.000Z",
}
FAILED_GATE = {
    "passed": False,
    "checks": [{"name": "minPassRate", "threshold": 0.9, "actual": 0.5, "passed": False}],
    "evaluatedAt": "2026-09-16T00:00:00.000Z",
}


class GateServer(RunServer):
    def __init__(self, gate="server", **kwargs):
        super().__init__(**kwargs)
        self.gate = gate
        self.runs_by_id = {}

    def complete_payload(self, run_id, body):
        payload = super().complete_payload(run_id, body)
        if self.gate != "server":
            payload["gateResult"] = self.gate
        self.runs_by_id[run_id] = dict(payload, runId=run_id, url=f"https://app.test/o/o1/p/p1/evaluations/{run_id}")
        return payload

    def route_get(self, path):
        parts = path.split("/")
        if len(parts) == 3 and parts[1] == "evaluation-runs" and parts[2] in self.runs_by_id:
            return 200, self.runs_by_id[parts[2]]
        return 404, {"statusCode": 404, "message": f"Cannot GET {path}"}


def _config(directory, **extra):
    spec = {"name": "ci", "data": [{"input": "q1", "output": "a1"}, {"input": "q2", "output": "a2"}], "evaluators": [RULE]}
    spec.update(extra)
    path = Path(directory) / "eval.json"
    path.write_text(json.dumps(spec))
    return str(path)


def _run(argv, home):
    out = io.StringIO()
    code = main(argv, home=Path(home), out=out)
    return code, out.getvalue()


class TestExitCodes(unittest.TestCase):
    def test_mapping_fails_closed_on_undecided(self):
        self.assertEqual(exit_code(True), EXIT_PASSED)
        self.assertEqual(exit_code(True, allow_undecided=True), EXIT_PASSED)
        self.assertEqual(exit_code(False), EXIT_FAILED)
        self.assertEqual(exit_code(False, allow_undecided=True), EXIT_FAILED)
        self.assertEqual(exit_code(None), EXIT_UNDECIDED)
        self.assertEqual(exit_code(None, allow_undecided=True), EXIT_PASSED)
        self.assertEqual((EXIT_PASSED, EXIT_FAILED, EXIT_UNDECIDED, EXIT_ERROR), (0, 1, 2, 3))

    def test_verdict_words(self):
        self.assertEqual([verdict(True), verdict(False), verdict(None)], ["passed", "failed", "undecided"])

    def test_report_passed_reads_only_a_boolean(self):
        base = dict(run_id="r", job_id="j", url=None, name="n", total_rows=1, errored_rows=0, indeterminate_rows=0,
                    pass_rate=1.0, average_score=0.9, overall_scores={})
        self.assertIs(RunReport(gate=None, **base).passed, None)
        self.assertIs(RunReport(gate={"passed": "true"}, **base).passed, None)
        self.assertIs(RunReport(gate={"passed": False}, **base).passed, False)
        self.assertEqual(RunReport(gate=UNDECIDED_GATE, **base).to_json()["verdict"], "undecided")


class TestGateTable(unittest.TestCase):
    def test_table_has_one_row_per_check_with_null_actuals_shown(self):
        lines = render_gate_table(UNDECIDED_GATE)
        self.assertEqual(lines[0], "gate:")
        self.assertIn("check", lines[1])
        self.assertRegex(lines[2], r"minPassRate\s+0\.900\s+1\.000\s+passed")
        self.assertRegex(lines[3], r"maxRegressionsVsBaseline\s+0\s+—\s+undecided")

    def test_no_gate_is_said_not_hidden(self):
        self.assertEqual(render_gate_table(None), ["gate: none (no gate was set on this run)"])


class TestSpecLoading(unittest.TestCase):
    def test_camel_case_keys_become_eval_arguments(self):
        spec = normalise_spec({"name": "n", "judgeModel": "gpt-4o", "baselineJobId": "b", "autoBaseline": True,
                               "gate": {"minPassRate": 0.9}, "timeoutMs": 5000, "batchSize": 50})
        self.assertEqual(spec, {"name": "n", "judge_model": "gpt-4o", "baseline_job_id": "b", "auto_baseline": True,
                                "gate": {"minPassRate": 0.9}, "timeout": 5.0, "batch_size": 50})

    def test_unknown_key_is_refused(self):
        with self.assertRaises(EvalCliError):
            normalise_spec({"name": "n", "judge": "x"})

    def test_json_config_with_jsonl_data_file_relative_to_the_config(self):
        with TemporaryDirectory() as tmp:
            Path(tmp, "rows.jsonl").write_text('{"input": "a", "output": "b"}\n\n{"input": "c", "output": "d"}\n')
            Path(tmp, "eval.json").write_text(json.dumps({"name": "n", "dataFile": "rows.jsonl", "evaluators": [RULE]}))
            spec = load_json_spec(Path(tmp, "eval.json"))
        self.assertEqual(spec["data"], [{"input": "a", "output": "b"}, {"input": "c", "output": "d"}])
        self.assertEqual(spec["evaluators"], [RULE])

    def test_json_config_refuses_data_and_data_file_together(self):
        with TemporaryDirectory() as tmp:
            Path(tmp, "eval.json").write_text(json.dumps({"name": "n", "data": [], "dataFile": "x.json"}))
            with self.assertRaises(EvalCliError):
                load_json_spec(Path(tmp, "eval.json"))

    def test_module_target_default_attr_callable_and_missing(self):
        with TemporaryDirectory() as tmp:
            Path(tmp, "checkout.py").write_text(
                "eval = {'name': 'm', 'data': [{'input': 'q', 'output': 'a'}], 'evaluators': ['" + RULE + "']}\n"
                "def build():\n    return {'name': 'built', 'data': [], 'evaluators': []}\n"
            )
            mapping = load_module_spec("checkout.py", Path(tmp))
            self.assertEqual(mapping["name"], "m")
            spec, result = resolve_spec(load_module_spec("checkout.py:build", Path(tmp)))
            self.assertEqual(spec["name"], "built")
            self.assertIsNone(result)
            with self.assertRaises(EvalCliError):
                load_module_spec("checkout.py:nope", Path(tmp))
            with self.assertRaises(EvalCliError):
                load_module_spec("missing.py", Path(tmp))

    def test_resolve_spec_rejects_other_shapes(self):
        with self.assertRaises(EvalCliError):
            resolve_spec(42)


class TestEvalRunCommand(unittest.TestCase):
    def setUp(self):
        reset_context()
        fresh([])
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name, "home")
        self.home.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_passed_gate_exits_zero_and_prints_url_rate_and_table(self):
        with GateServer() as server:
            cfg = _config(self.tmp.name, gate={"minPassRate": 0.9})
            code, out = _run(["eval", "run", cfg, "--api-url", server.url], self.home)
        self.assertEqual(code, EXIT_PASSED)
        self.assertIn("https://app.test/o/o1/p/p1/evaluations/run_1", out)
        self.assertIn("pass rate:     100.0%", out)
        self.assertRegex(out, r"minPassRate\s+0\.900\s+1\.000\s+passed")
        self.assertIn("verdict: passed (exit 0)", out)
        self.assertEqual(server.completes, [{"gate": {"minPassRate": 0.9}}])
        self.assertEqual(server.requests[0]["key"], "tk_test")

    def test_failed_gate_exits_one(self):
        with GateServer(gate=FAILED_GATE) as server:
            code, out = _run(["eval", "run", _config(self.tmp.name, gate={"minPassRate": 0.9}), "--api-url", server.url], self.home)
        self.assertEqual(code, EXIT_FAILED)
        self.assertIn("verdict: failed (exit 1)", out)

    def test_undecided_gate_exits_two_unless_allowed(self):
        with GateServer(gate=UNDECIDED_GATE) as server:
            cfg = _config(self.tmp.name, gate={"minPassRate": 0.9, "maxRegressionsVsBaseline": 0})
            code, out = _run(["eval", "run", cfg, "--api-url", server.url], self.home)
            self.assertEqual(code, EXIT_UNDECIDED)
            self.assertIn("verdict: undecided (exit 2)", out)
            code, out = _run(["eval", "run", cfg, "--api-url", server.url, "--allow-undecided"], self.home)
        self.assertEqual(code, EXIT_PASSED)
        self.assertIn("verdict: undecided (exit 0)", out)

    def test_no_gate_is_undecided(self):
        with GateServer() as server:
            code, out = _run(["eval", "run", _config(self.tmp.name), "--api-url", server.url], self.home)
        self.assertEqual(code, EXIT_UNDECIDED)
        self.assertIn("gate: none", out)

    def test_json_output_is_one_document_with_the_gate(self):
        with GateServer(gate=FAILED_GATE) as server:
            code, out = _run(["eval", "run", _config(self.tmp.name, gate={"minPassRate": 0.9}), "--api-url", server.url, "--json"], self.home)
        self.assertEqual(code, EXIT_FAILED)
        document = json.loads(out)
        self.assertEqual(document["verdict"], "failed")
        self.assertEqual(document["gate"], FAILED_GATE)
        self.assertEqual(document["runId"], "run_1")
        self.assertEqual(document["passRate"], 1.0)
        self.assertNotIn("EVALKIT_QUIET", os.environ)

    def test_module_target_runs_through_the_same_client(self):
        with GateServer() as server:
            Path(self.tmp.name, "checkout.py").write_text(
                "RULE = '" + RULE + "'\n"
                "def eval():\n    return {'name': 'from-module', 'data': [{'input': 'q', 'output': 'a'}], 'evaluators': [RULE], 'gate': {'minPassRate': 0.5}}\n"
            )
            code, out = _run(["eval", "run", str(Path(self.tmp.name, "checkout.py")), "--api-url", server.url], self.home)
        self.assertEqual(code, EXIT_PASSED)
        self.assertEqual(server.requests[0]["body"]["name"], "from-module")

    def test_module_may_return_a_finished_result(self):
        with GateServer(gate=FAILED_GATE) as server:
            Path(self.tmp.name, "done.py").write_text(
                "import evalkit\n"
                "def eval():\n    return evalkit.Eval('self-run', [{'input': 'q', 'output': 'a'}], evaluators=['" + RULE + "'], gate={'minPassRate': 0.9}, api_url='" + server.url + "')\n"
            )
            code, out = _run(["eval", "run", str(Path(self.tmp.name, "done.py"))], self.home)
        self.assertEqual(code, EXIT_FAILED)
        self.assertIn("self-run", server.requests[0]["raw"])

    def test_http_error_and_bad_config_exit_three(self):
        with GateServer(mode="create-404") as server:
            code, _ = _run(["eval", "run", _config(self.tmp.name), "--api-url", server.url], self.home)
            self.assertEqual(code, EXIT_ERROR)
            code, _ = _run(["eval", "run", str(Path(self.tmp.name, "absent.json")), "--api-url", server.url], self.home)
            self.assertEqual(code, EXIT_ERROR)
            Path(self.tmp.name, "bad.json").write_text(json.dumps({"name": "n", "data": [], "evaluators": [], "judge": "x"}))
            code, _ = _run(["eval", "run", str(Path(self.tmp.name, "bad.json")), "--api-url", server.url], self.home)
        self.assertEqual(code, EXIT_ERROR)

    def test_without_a_client_a_key_is_required(self):
        saved = evalkit._instance
        evalkit._instance = None
        env = {k: os.environ.pop(k) for k in ("EVALKIT_API_KEY", "EVALKIT_SUBSCRIPTION_KEY") if k in os.environ}
        try:
            code, _ = _run(["eval", "run", _config(self.tmp.name), "--api-url", "http://127.0.0.1:1"], self.home)
        finally:
            evalkit._instance = saved
            os.environ.update(env)
        self.assertEqual(code, EXIT_ERROR)


class TestEvalGateCommand(unittest.TestCase):
    def setUp(self):
        reset_context()
        fresh([])
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name, "home")
        self.home.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_reads_an_existing_run_and_maps_its_gate(self):
        with GateServer(gate=UNDECIDED_GATE) as server:
            _run(["eval", "run", _config(self.tmp.name, gate={"minPassRate": 0.9}), "--api-url", server.url], self.home)
            code, out = _run(["eval", "gate", "run_1", "--api-url", server.url], self.home)
            self.assertEqual(code, EXIT_UNDECIDED)
            self.assertIn("run_1", out)
            self.assertRegex(out, r"maxRegressionsVsBaseline\s+0\s+—\s+undecided")
            code, out = _run(["eval", "gate", "run_1", "--api-url", server.url, "--allow-undecided", "--json"], self.home)
            self.assertEqual(code, EXIT_PASSED)
            self.assertEqual(json.loads(out)["verdict"], "undecided")
            read = [r for r in server.requests if r["body"] is None]
        self.assertEqual([r["path"] for r in read], ["/evaluation-runs/run_1", "/evaluation-runs/run_1"])
        self.assertEqual(read[0]["key"], "tk_test")

    def test_unknown_run_exits_three(self):
        with GateServer() as server:
            code, _ = _run(["eval", "gate", "run_9", "--api-url", server.url], self.home)
        self.assertEqual(code, EXIT_ERROR)
