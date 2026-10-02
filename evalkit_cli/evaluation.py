from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import sys
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .config import ConfigStore
from .otlp import read_base

EXIT_PASSED = 0
EXIT_FAILED = 1
EXIT_UNDECIDED = 2
EXIT_ERROR = 3

DEFAULT_ATTR = "eval"
SPEC_KEYS = (
    "name",
    "data",
    "task",
    "evaluators",
    "collections",
    "judge_model",
    "judge_model_config_id",
    "baseline_job_id",
    "gate",
    "hyperparameters",
    "auto_baseline",
    "batch_size",
    "timeout",
)

_CAMEL = re.compile(r"(?<!^)(?=[A-Z])")
_TARGET = re.compile(r"^(?P<module>.+?)(?::(?P<attr>[A-Za-z_][A-Za-z0-9_]*))?$")


class EvalCliError(Exception):
    pass


def verdict(passed: Optional[bool]) -> str:
    if passed is True:
        return "passed"
    if passed is False:
        return "failed"
    return "undecided"


def exit_code(passed: Optional[bool], allow_undecided: bool = False) -> int:
    if passed is True:
        return EXIT_PASSED
    if passed is False:
        return EXIT_FAILED
    return EXIT_PASSED if allow_undecided else EXIT_UNDECIDED


@dataclass
class RunReport:
    run_id: Optional[str]
    job_id: Optional[str]
    url: Optional[str]
    name: Optional[str]
    total_rows: Optional[int]
    errored_rows: Optional[int]
    indeterminate_rows: Optional[int]
    pass_rate: Optional[float]
    average_score: Optional[float]
    overall_scores: Optional[Dict[str, Optional[float]]]
    gate: Optional[Dict[str, Any]]

    @property
    def passed(self) -> Optional[bool]:
        value = self.gate.get("passed") if isinstance(self.gate, Mapping) else None
        return value if isinstance(value, bool) else None

    @classmethod
    def from_result(cls, result: Any) -> "RunReport":
        return cls(
            run_id=result.run_id,
            job_id=result.job_id,
            url=result.url,
            name=result.name,
            total_rows=result.total_rows,
            errored_rows=result.errored_rows,
            indeterminate_rows=result.indeterminate_rows,
            pass_rate=result.pass_rate,
            average_score=result.average_score,
            overall_scores=result.overall_scores,
            gate=result.gate_result if isinstance(result.gate_result, Mapping) else None,
        )

    @classmethod
    def from_body(cls, body: Mapping[str, Any], run_id: str) -> "RunReport":
        gate = body.get("gateResult")
        return cls(
            run_id=str(body.get("runId") or run_id),
            job_id=body.get("jobId"),
            url=body.get("url"),
            name=body.get("name"),
            total_rows=body.get("totalRows"),
            errored_rows=body.get("erroredRows"),
            indeterminate_rows=body.get("indeterminateRows"),
            pass_rate=body.get("passRate"),
            average_score=body.get("averageScore"),
            overall_scores=body.get("overallScores"),
            gate=dict(gate) if isinstance(gate, Mapping) else None,
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "runId": self.run_id,
            "jobId": self.job_id,
            "url": self.url,
            "name": self.name,
            "totalRows": self.total_rows,
            "erroredRows": self.errored_rows,
            "indeterminateRows": self.indeterminate_rows,
            "passRate": self.pass_rate,
            "averageScore": self.average_score,
            "overallScores": self.overall_scores,
            "gate": self.gate,
            "verdict": verdict(self.passed),
        }


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    return "—"


def _percent(value: Optional[float]) -> str:
    return f"{value * 100:.1f}%" if isinstance(value, (int, float)) else "—"


def render_gate_table(gate: Optional[Mapping[str, Any]]) -> List[str]:
    if not isinstance(gate, Mapping):
        return ["gate: none (no gate was set on this run)"]
    checks = [c for c in gate.get("checks") or [] if isinstance(c, Mapping)]
    rows = [("check", "threshold", "actual", "result")]
    for check in checks:
        rows.append(
            (
                str(check.get("name") or "?"),
                _fmt(check.get("threshold")),
                _fmt(check.get("actual")),
                verdict(check.get("passed") if isinstance(check.get("passed"), bool) else None),
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    lines = ["gate:"]
    for row in rows:
        lines.append("  " + "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
    if not checks:
        lines.append("  (no checks)")
    return lines


def render_report(report: RunReport, code: int) -> List[str]:
    lines = [
        f"run:           {report.url or report.run_id or '—'}",
        f"job:           {report.job_id or '—'}",
        f"rows:          {_fmt(report.total_rows)} (errored {_fmt(report.errored_rows)}, undecided {_fmt(report.indeterminate_rows)})",
        f"pass rate:     {_percent(report.pass_rate)}",
        f"average score: {_fmt(report.average_score)}",
    ]
    for metric, score in sorted((report.overall_scores or {}).items()):
        lines.append(f"  {metric}: {_fmt(score)}")
    lines.extend(render_gate_table(report.gate))
    lines.append(f"verdict: {verdict(report.passed)} (exit {code})")
    return lines


def _snake(key: str) -> str:
    return _CAMEL.sub("_", key).lower()


def normalise_spec(raw: Mapping[str, Any]) -> Dict[str, Any]:
    spec: Dict[str, Any] = {}
    for key, value in raw.items():
        name = _snake(str(key))
        if name == "timeout_ms":
            name, value = "timeout", float(value) / 1000
        if name not in SPEC_KEYS:
            raise EvalCliError(f"unknown Eval field {key!r}; known: {', '.join(SPEC_KEYS)}")
        spec[name] = value
    return spec


def read_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise EvalCliError(f"{path}: data file not found")
    text = path.read_text()
    if path.suffix == ".jsonl":
        rows: Any = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)
        if isinstance(rows, Mapping) and isinstance(rows.get("rows"), list):
            rows = rows["rows"]
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise EvalCliError(f"{path}: expected a JSON array of row objects (or JSONL)")
    return [dict(row) for row in rows]


def load_json_spec(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise EvalCliError(f"{path}: config file not found")
    try:
        raw = json.loads(path.read_text())
    except ValueError as exc:
        raise EvalCliError(f"{path}: not valid JSON ({exc})")
    if not isinstance(raw, Mapping):
        raise EvalCliError(f"{path}: expected a JSON object")
    raw = dict(raw)
    data_file = raw.pop("dataFile", None)
    if data_file is not None:
        if "data" in raw:
            raise EvalCliError(f"{path}: give either data or dataFile, not both")
        raw["data"] = read_rows(path.parent / str(data_file))
    return normalise_spec(raw)


def _import(ref: str, cwd: Path) -> Any:
    path = Path(ref) if Path(ref).is_absolute() else cwd / ref
    if path.suffix == ".py" or path.exists():
        if not path.exists():
            raise EvalCliError(f"{ref}: file not found")
        name = "evalkit_eval_" + re.sub(r"\W", "_", path.stem)
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise EvalCliError(f"{ref}: cannot import")
        module = importlib.util.module_from_spec(spec)
        parent = str(path.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    try:
        return importlib.import_module(ref)
    except ImportError as exc:
        raise EvalCliError(f"{ref}: {exc}")


def load_module_spec(target: str, cwd: Path) -> Any:
    match = _TARGET.match(target)
    if not match:
        raise EvalCliError(f"{target}: expected module.py[:attr] or package.module[:attr]")
    module_ref, attr = match.group("module"), match.group("attr") or DEFAULT_ATTR
    module = _import(module_ref, cwd)
    if not hasattr(module, attr):
        raise EvalCliError(f"{module_ref} has no attribute {attr!r}")
    return getattr(module, attr)


def load_target(target: str, cwd: Path) -> Any:
    if target.endswith(".json"):
        return load_json_spec(Path(target) if Path(target).is_absolute() else cwd / target)
    return load_module_spec(target, cwd)


def _is_result(value: Any) -> bool:
    return all(hasattr(value, attr) for attr in ("gate_result", "pass_rate", "run_id", "job_id"))


def resolve_spec(loaded: Any) -> Tuple[Optional[Dict[str, Any]], Any]:
    if callable(loaded) and not isinstance(loaded, Mapping):
        loaded = loaded()
    if _is_result(loaded):
        return None, loaded
    if isinstance(loaded, Mapping):
        return normalise_spec(loaded), None
    raise EvalCliError("the target must be a mapping of Eval() arguments, a callable returning one, or an EvalResult")


def resolve_key(explicit: Optional[str], store: ConfigStore) -> str:
    return explicit or os.getenv("EVALKIT_API_KEY") or os.getenv("EVALKIT_SUBSCRIPTION_KEY") or store.load().key


def resolve_api_url(explicit: Optional[str], store: ConfigStore) -> str:
    return (explicit or os.getenv("EVALKIT_API_URL") or read_base(store.load().endpoint)).rstrip("/")


def _client(key: str, api_url: str) -> Any:
    import evalkit

    if evalkit._instance is not None:
        return evalkit._instance
    if not key:
        raise EvalCliError("a subscription key is required: pass --key, set EVALKIT_API_KEY, or run `evalkit coding install`")
    return evalkit.init(subscription_key=key, api_url=api_url, base_url=api_url, service_name="evalkit-cli", function_tracing=False)


def _http_detail(exc: urllib.error.HTTPError) -> str:
    text = getattr(exc, "response_text", None) or ""
    try:
        body = json.loads(text)
        message = body.get("message") if isinstance(body, Mapping) else None
        if isinstance(message, list):
            message = "; ".join(str(m) for m in message)
        if message:
            return f"HTTP {exc.code}: {message}"
    except ValueError:
        pass
    return f"HTTP {exc.code}: {text[:300] or exc.reason}"


def _finish(report: RunReport, args: Any, out) -> int:
    code = exit_code(report.passed, args.allow_undecided)
    if args.json:
        print(json.dumps(report.to_json(), indent=2, default=str), file=out)
    else:
        for line in render_report(report, code):
            print(line, file=out)
    return code


def cmd_eval_run(args: Any, store: ConfigStore, out) -> int:
    quiet_before = os.environ.get("EVALKIT_QUIET")
    if args.json:
        os.environ["EVALKIT_QUIET"] = "1"
    try:
        spec, result = resolve_spec(load_target(args.target, Path.cwd()))
        if result is None:
            key = resolve_key(args.key, store)
            api_url = resolve_api_url(args.api_url, store)
            _client(key, api_url)
            import evalkit

            result = evalkit.Eval(api_url=api_url, **spec)
            evalkit.flush()
    except EvalCliError as exc:
        print(f"evalkit eval: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except urllib.error.HTTPError as exc:
        print(f"evalkit eval: the run failed ({_http_detail(exc)})", file=sys.stderr)
        return EXIT_ERROR
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"evalkit eval: the run failed ({exc})", file=sys.stderr)
        return EXIT_ERROR
    finally:
        if args.json:
            if quiet_before is None:
                os.environ.pop("EVALKIT_QUIET", None)
            else:
                os.environ["EVALKIT_QUIET"] = quiet_before
    return _finish(RunReport.from_result(result), args, out)


def cmd_eval_gate(args: Any, store: ConfigStore, out) -> int:
    from evalkit.core.api import get_json

    try:
        key = resolve_key(args.key, store)
        api_url = resolve_api_url(args.api_url, store)
        client = _client(key, api_url)
        body = get_json(
            client,
            f"{api_url}/evaluation-runs/{args.run_id}",
            timeout=30.0,
            span_name="GET /evaluation-runs/:runId",
        )
    except EvalCliError as exc:
        print(f"evalkit eval: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except urllib.error.HTTPError as exc:
        hint = " (no such run, or this server has no GET /evaluation-runs/:runId yet)" if exc.code == 404 else ""
        print(f"evalkit eval: could not read run {args.run_id} ({_http_detail(exc)}){hint}", file=sys.stderr)
        return EXIT_ERROR
    except (OSError, ValueError) as exc:
        print(f"evalkit eval: could not read run {args.run_id} ({exc})", file=sys.stderr)
        return EXIT_ERROR
    if not isinstance(body, Mapping):
        print(f"evalkit eval: run {args.run_id} answered with no JSON object", file=sys.stderr)
        return EXIT_ERROR
    return _finish(RunReport.from_body(body, args.run_id), args, out)
