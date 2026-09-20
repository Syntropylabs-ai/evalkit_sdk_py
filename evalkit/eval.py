from __future__ import annotations

import math
import os
import re
import time
import urllib.error
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)
from weakref import WeakKeyDictionary

from .core.api import post_json

_RULE_ID = re.compile(r"^[0-9a-fA-F]{24}$")
_COUNT_KEYS = ("ok", "unparseable", "judgeError", "skipped", "unsupported")

MAX_ROWS_PER_BATCH = 200

_RETRY_DELAYS = (0.25, 0.75)

_TRUTHY = ("1", "true", "yes", "on")

LocalScore = Union[float, int, bool, None, Mapping[str, Any]]
LocalEvaluator = Callable[[Mapping[str, Any], str], LocalScore]

_rows_endpoint_support: "WeakKeyDictionary[Any, bool]" = WeakKeyDictionary()


class EvalRow(TypedDict, total=False):
    input: str
    output: str
    target: str
    expected: str
    context: List[str]
    metadata: Dict[str, Any]


class EvalResult:
    def __init__(
        self,
        *,
        job_id: Optional[str],
        job_ids: List[str],
        name: str,
        dataset_hash: Optional[str],
        total_rows: int,
        average_score: Optional[float],
        pass_rate: Optional[float],
        counts: Optional[Dict[str, int]],
        indeterminate_rows: Optional[int],
        errored_rows: Optional[int],
        scoring_version: Optional[int],
        overall_scores: Optional[Dict[str, Optional[float]]],
        gate_result: Optional[Dict[str, Any]],
        results: List[Dict[str, Any]],
        comparison: Optional[Dict[str, Any]],
        local_scores: Dict[str, List[Optional[float]]],
        run_id: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        self.job_id = job_id
        self.job_ids = job_ids
        self.name = name
        self.dataset_hash = dataset_hash
        self.total_rows = total_rows
        self.average_score = average_score
        self.pass_rate = pass_rate
        self.counts = counts
        self.indeterminate_rows = indeterminate_rows
        self.errored_rows = errored_rows
        self.scoring_version = scoring_version
        self.overall_scores = overall_scores
        self.gate_result = gate_result
        self.results = results
        self.comparison = comparison
        self.local_scores = local_scores
        self.run_id = run_id
        self.url = url

    @property
    def passed(self) -> Optional[bool]:
        if isinstance(self.gate_result, Mapping):
            return self.gate_result.get("passed")
        return None

    def __repr__(self) -> str:
        return (
            f"EvalResult(job_id={self.job_id!r}, total_rows={self.total_rows!r}, "
            f"average_score={self.average_score!r}, pass_rate={self.pass_rate!r}, "
            f"passed={self.passed!r}, url={self.url!r})"
        )


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_errored(row: Mapping[str, Any]) -> bool:
    return bool(row.get("error")) or row.get("status") == "error"


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = {key: 0 for key in _COUNT_KEYS}
    score_sum = 0.0
    scored = passed = failed = indeterminate = errored = 0
    for row in rows:
        row_counts = row.get("counts") or {}
        for key in _COUNT_KEYS:
            counts[key] += row_counts.get(key) or 0
        if _is_errored(row):
            errored += 1
            continue
        score = row.get("overallScore")
        if _is_finite_number(score):
            score_sum += score
            scored += 1
        verdict = row.get("passed")
        if verdict is True:
            passed += 1
        elif verdict is False:
            failed += 1
        else:
            indeterminate += 1
    decided = passed + failed
    return {
        "averageScore": score_sum / scored if scored else None,
        "passRate": passed / decided if decided else None,
        "counts": counts,
        "indeterminateRows": indeterminate,
        "erroredRows": errored,
    }


def _per_metric_averages(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    tallies: Dict[str, List[float]] = {}
    for row in rows:
        if _is_errored(row):
            continue
        for metric in row.get("metrics") or []:
            metric_name = metric.get("metricName")
            if not metric_name:
                continue
            bucket = tallies.setdefault(metric_name, [])
            score = metric.get("score")
            status = metric.get("status")
            if _is_finite_number(score) and (status is None or status == "ok"):
                bucket.append(score)
    return {metric_name: (sum(values) / len(values) if values else None) for metric_name, values in tallies.items()}


def _split_evaluators(
    evaluators: Sequence[Union[str, LocalEvaluator]],
) -> Tuple[List[str], List[Tuple[str, LocalEvaluator]]]:
    rule_ids: List[str] = []
    local: List[Tuple[str, LocalEvaluator]] = []
    labels: set = set()
    for index, entry in enumerate(evaluators):
        if isinstance(entry, str) and _RULE_ID.match(entry):
            rule_ids.append(entry)
            continue
        if not isinstance(entry, str) and callable(entry):
            label = getattr(entry, "__name__", "") or f"evaluator_{index}"
            if label in labels:
                label = f"{label}_{index}"
            labels.add(label)
            local.append((label, entry))
            continue
        raise ValueError(
            f"evaluators[{index}] = {entry!r} is neither a server rule id (24 hex chars) nor a callable"
        )
    if not rule_ids:
        raise ValueError("evaluators must include at least one server rule id")
    return rule_ids, local


def _build_rows(
    data: Iterable[Mapping[str, Any]],
    task: Optional[Callable[[Mapping[str, Any]], str]],
) -> Tuple[List[Mapping[str, Any]], List[str]]:
    source: List[Mapping[str, Any]] = []
    outputs: List[str] = []
    for index, row in enumerate(data):
        if row.get("input") is None:
            raise ValueError(f"data[{index}] has no 'input'")
        if task is not None:
            output = task(row)
        else:
            output = row.get("output")
            if output is None:
                raise ValueError(f"data[{index}] has no 'output' and no task was given")
        source.append(row)
        outputs.append(output)
    if not source:
        raise ValueError("data is empty")
    return source, outputs


def _local_entry(label: str, value: LocalScore) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, bool):
        return {"name": label, "score": 1.0 if value else 0.0, "passed": value}
    if isinstance(value, (int, float)):
        return {"name": label, "score": float(value)}
    if isinstance(value, Mapping):
        score = value.get("score")
        passed = value.get("passed")
        if isinstance(score, bool):
            score = 1.0 if score else 0.0
        if score is None and isinstance(passed, bool):
            score = 1.0 if passed else 0.0
        if not _is_finite_number(score):
            raise ValueError(
                f"local evaluator {label!r} returned {value!r}; a mapping needs a numeric 'score' or a boolean 'passed'"
            )
        entry: Dict[str, Any] = {"name": str(value.get("name") or label), "score": float(score)}
        if isinstance(passed, bool):
            entry["passed"] = passed
        if value.get("reason") is not None:
            entry["reason"] = str(value["reason"])
        return entry
    raise ValueError(
        f"local evaluator {label!r} returned {value!r}; expected a number, a boolean, None or a mapping"
    )


def _run_local(
    local: Sequence[Tuple[str, LocalEvaluator]],
    source: Sequence[Mapping[str, Any]],
    outputs: Sequence[str],
) -> Tuple[Dict[str, List[Optional[float]]], List[List[Dict[str, Any]]]]:
    scores: Dict[str, List[Optional[float]]] = {label: [] for label, _ in local}
    entries: List[List[Dict[str, Any]]] = []
    for row, output in zip(source, outputs):
        row_entries: List[Dict[str, Any]] = []
        for label, fn in local:
            entry = _local_entry(label, fn(row, output))
            scores[label].append(None if entry is None else entry["score"])
            if entry is not None:
                row_entries.append(entry)
        entries.append(row_entries)
    return scores, entries


def _row_metadata(row: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = row.get("metadata")
    out: Dict[str, Any] = dict(metadata) if isinstance(metadata, Mapping) else {}
    context = row.get("context")
    if context is not None:
        out["context"] = list(context)
    return out


@dataclass
class _RunSpec:
    name: str
    rule_ids: List[str]
    judge_model: Optional[str] = None
    judge_model_config_id: Optional[str] = None
    baseline_job_id: Optional[str] = None
    gate: Optional[Mapping[str, Any]] = None
    hyperparameters: Optional[Mapping[str, Any]] = None
    auto_baseline: Optional[bool] = None

    def _judge_and_hyperparameters(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        if self.judge_model_config_id is not None:
            body["judgeModelConfigId"] = self.judge_model_config_id
        if self.judge_model is not None:
            body["judgeModelName"] = self.judge_model
        if self.hyperparameters is not None:
            body["hyperparameters"] = dict(self.hyperparameters)
        return body

    def create_body(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {"name": self.name, "ruleIds": list(self.rule_ids)}
        body.update(self._judge_and_hyperparameters())
        if self.baseline_job_id is not None:
            body["baselineJobId"] = self.baseline_job_id
        if self.auto_baseline is not None:
            body["autoBaseline"] = bool(self.auto_baseline)
        return body

    def complete_body(self) -> Dict[str, Any]:
        return {"gate": dict(self.gate)} if self.gate is not None else {}

    def legacy_body(self, rows: List[Dict[str, Any]], single: bool) -> Dict[str, Any]:
        body: Dict[str, Any] = {"name": self.name, "ruleIds": list(self.rule_ids), "rows": rows}
        body.update(self._judge_and_hyperparameters())
        if single and self.baseline_job_id is not None:
            body["baselineJobId"] = self.baseline_job_id
        if single and self.gate is not None:
            body["gate"] = dict(self.gate)
        if single and self.auto_baseline is not None:
            body["autoBaseline"] = bool(self.auto_baseline)
        return body


@dataclass
class _RunRows:
    source: List[Mapping[str, Any]]
    outputs: List[str]
    local_scores: Dict[str, List[Optional[float]]]
    local_entries: List[List[Dict[str, Any]]]
    batch_size: int

    def __len__(self) -> int:
        return len(self.outputs)

    def batches(self) -> Iterator[Tuple[int, int]]:
        total = len(self.outputs)
        for start in range(0, total, self.batch_size):
            yield start, min(start + self.batch_size, total)

    def append_payload(self, start: int, stop: int) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for index in range(start, stop):
            row = self.source[index]
            item: Dict[str, Any] = {
                "rowIndex": index,
                "input": row["input"],
                "output": self.outputs[index],
            }
            expected = row.get("expected") if row.get("expected") is not None else row.get("target")
            if expected is not None:
                item["expected"] = expected
            metadata = _row_metadata(row)
            if metadata:
                item["metadata"] = metadata
            if self.local_entries[index]:
                item["localScores"] = self.local_entries[index]
            payload.append(item)
        return payload

    def legacy_payload(self, start: int, stop: int) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for index in range(start, stop):
            row = self.source[index]
            item: Dict[str, Any] = {"input": row["input"], "output": self.outputs[index]}
            target = row.get("target") if row.get("target") is not None else row.get("expected")
            if target is not None:
                item["target"] = target
            if row.get("context") is not None:
                item["context"] = list(row["context"])
            payload.append(item)
        return payload


def _with_retry(send: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    last: Optional[BaseException] = None
    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            return send()
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                raise
            last = exc
        except Exception as exc:
            last = exc
        if attempt < len(_RETRY_DELAYS):
            time.sleep(_RETRY_DELAYS[attempt])
    raise last  # type: ignore[misc]


class _RunApi:
    def __init__(self, client: Any, base: str, timeout: float) -> None:
        self._client = client
        self._base = base
        self._timeout = timeout

    def post(self, path: str, body: Dict[str, Any], span_name: str) -> Dict[str, Any]:
        return post_json(
            self._client, self._base + path, body, timeout=self._timeout, span_name=span_name
        )

    def create(self, spec: _RunSpec) -> Dict[str, Any]:
        return self.post("/evaluation-runs", spec.create_body(), "POST /evaluation-runs")

    def append(self, run_id: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        path = f"/evaluation-runs/{run_id}/rows"
        return _with_retry(
            lambda: self.post(path, {"rows": rows}, "POST /evaluation-runs/:runId/rows")
        )

    def complete(self, run_id: str, spec: _RunSpec) -> Dict[str, Any]:
        return self.post(
            f"/evaluation-runs/{run_id}/complete",
            spec.complete_body(),
            "POST /evaluation-runs/:runId/complete",
        )

    def legacy(self, spec: _RunSpec, rows: List[Dict[str, Any]], single: bool) -> Dict[str, Any]:
        return self.post("/evaluation-runs", spec.legacy_body(rows, single), "POST /evaluation-runs")


def _quiet() -> bool:
    return os.getenv("EVALKIT_QUIET", "").strip().lower() in _TRUTHY


def _announce(url: str) -> None:
    if _quiet():
        return
    print(f"[evalkit] evaluation run: {url}")


def _create_run(api: _RunApi, spec: _RunSpec) -> Tuple[str, Optional[str]]:
    created = api.create(spec)
    run_id = created.get("runId")
    if not run_id:
        raise RuntimeError("[evalkit] POST /evaluation-runs answered without a runId")
    url = created.get("url")
    return str(run_id), (str(url) if url else None)


def _summary_result(
    spec: _RunSpec,
    rows: _RunRows,
    run_id: str,
    url: Optional[str],
    summary: Mapping[str, Any],
) -> EvalResult:
    job_id = summary.get("jobId") or run_id
    total_rows = summary.get("totalRows")
    return EvalResult(
        job_id=job_id,
        job_ids=[job_id],
        name=summary.get("name") or spec.name,
        dataset_hash=summary.get("datasetHash"),
        total_rows=total_rows if isinstance(total_rows, int) else len(rows),
        average_score=summary.get("averageScore"),
        pass_rate=summary.get("passRate"),
        counts=summary.get("counts"),
        indeterminate_rows=summary.get("indeterminateRows"),
        errored_rows=summary.get("erroredRows"),
        scoring_version=summary.get("scoringVersion"),
        overall_scores=summary.get("overallScores"),
        gate_result=summary.get("gateResult"),
        results=list(summary.get("results") or []),
        comparison=summary.get("comparison"),
        local_scores=rows.local_scores,
        run_id=run_id,
        url=url,
    )


def _run_appended(
    api: _RunApi, client: Any, spec: _RunSpec, rows: _RunRows
) -> Optional[EvalResult]:
    run_id, url = _create_run(api, spec)
    if url:
        _announce(url)
    for start, stop in rows.batches():
        try:
            api.append(run_id, rows.append_payload(start, stop))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                _rows_endpoint_support[client] = False
                return None
            raise
    _rows_endpoint_support[client] = True
    return _summary_result(spec, rows, run_id, url, api.complete(run_id, spec))


def _assemble(
    spec: _RunSpec,
    rows: _RunRows,
    sizes: Sequence[int],
    responses: Sequence[Dict[str, Any]],
) -> EvalResult:
    first = responses[0]
    job_ids = [response.get("jobId") for response in responses]
    if len(responses) == 1:
        return EvalResult(
            job_id=first.get("jobId"),
            job_ids=job_ids,
            name=first.get("name") or spec.name,
            dataset_hash=first.get("datasetHash"),
            total_rows=first.get("totalRows"),
            average_score=first.get("averageScore"),
            pass_rate=first.get("passRate"),
            counts=first.get("counts"),
            indeterminate_rows=first.get("indeterminateRows"),
            errored_rows=first.get("erroredRows"),
            scoring_version=first.get("scoringVersion"),
            overall_scores=first.get("overallScores"),
            gate_result=first.get("gateResult"),
            results=list(first.get("results") or []),
            comparison=first.get("comparison"),
            local_scores=rows.local_scores,
        )

    results: List[Dict[str, Any]] = []
    offset = 0
    for size, response in zip(sizes, responses):
        for row in response.get("results") or []:
            row = dict(row)
            if isinstance(row.get("rowIndex"), int):
                row["rowIndex"] += offset
            results.append(row)
        offset += size
    rollup = _aggregate(results)
    return EvalResult(
        job_id=job_ids[0],
        job_ids=job_ids,
        name=spec.name,
        dataset_hash=None,
        total_rows=len(results),
        average_score=rollup["averageScore"],
        pass_rate=rollup["passRate"],
        counts=rollup["counts"],
        indeterminate_rows=rollup["indeterminateRows"],
        errored_rows=rollup["erroredRows"],
        scoring_version=first.get("scoringVersion"),
        overall_scores=_per_metric_averages(results),
        gate_result=None,
        results=results,
        comparison=None,
        local_scores=rows.local_scores,
    )


def _run_legacy(api: _RunApi, spec: _RunSpec, rows: _RunRows) -> EvalResult:
    batches = list(rows.batches())
    single = len(batches) == 1
    sizes: List[int] = []
    responses: List[Dict[str, Any]] = []
    for start, stop in batches:
        sizes.append(stop - start)
        responses.append(api.legacy(spec, rows.legacy_payload(start, stop), single))
    return _assemble(spec, rows, sizes, responses)


def Eval(
    name: str,
    data: Iterable[Mapping[str, Any]],
    task: Optional[Callable[[Mapping[str, Any]], str]] = None,
    *,
    evaluators: Sequence[Union[str, LocalEvaluator]],
    judge_model: Optional[str] = None,
    judge_model_config_id: Optional[str] = None,
    baseline_job_id: Optional[str] = None,
    gate: Optional[Mapping[str, Any]] = None,
    hyperparameters: Optional[Mapping[str, Any]] = None,
    auto_baseline: Optional[bool] = None,
    batch_size: int = MAX_ROWS_PER_BATCH,
    timeout: float = 600.0,
    api_url: Optional[str] = None,
) -> EvalResult:
    from . import _get_instance

    client = _get_instance()
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    rule_ids, local = _split_evaluators(evaluators)
    source, outputs = _build_rows(data, task)
    local_scores, local_entries = _run_local(local, source, outputs)

    spec = _RunSpec(
        name=name,
        rule_ids=rule_ids,
        judge_model=judge_model,
        judge_model_config_id=judge_model_config_id,
        baseline_job_id=baseline_job_id,
        gate=gate,
        hyperparameters=hyperparameters,
        auto_baseline=auto_baseline,
    )
    rows = _RunRows(
        source=source,
        outputs=outputs,
        local_scores=local_scores,
        local_entries=local_entries,
        batch_size=min(batch_size, MAX_ROWS_PER_BATCH),
    )

    base = (api_url or os.getenv("EVALKIT_API_URL") or client._opts.api_url).rstrip("/")
    api = _RunApi(client, base, timeout)

    if _rows_endpoint_support.get(client) is not False:
        result = _run_appended(api, client, spec, rows)
        if result is not None:
            return result
    return _run_legacy(api, spec, rows)
