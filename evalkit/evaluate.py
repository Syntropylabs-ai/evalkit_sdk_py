from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

OFFLINE_METRICS = frozenset(
    [
        "tool_trajectory",
        "tool_f1",
        "tool_correctness",
        "response_match",
        "constraint_compliance",
        "contextual_precision",
        "contextual_recall",
    ]
)


def _lcs_length(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _score_tool_trajectory(
    actual: List[str], expected: List[str]
) -> Tuple[float, Dict[str, Any]]:
    if not expected:
        return 1.0, {"lcs_length": 0, "expected_count": 0}
    lcs = _lcs_length(actual, expected)
    return lcs / len(expected), {"lcs_length": lcs, "expected_count": len(expected)}


def _score_tool_f1(
    actual: List[str], expected: List[str]
) -> Tuple[float, Dict[str, Any]]:
    if not actual and not expected:
        return 1.0, {"precision": 1.0, "recall": 1.0}
    if not actual or not expected:
        return 0.0, {"precision": 0.0, "recall": 0.0}
    actual_set = set(actual)
    expected_set = set(expected)
    intersection = actual_set & expected_set
    precision = len(intersection) / len(actual_set)
    recall = len(intersection) / len(expected_set)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return f1, {"precision": precision, "recall": recall}


def _score_tool_correctness(
    actual: List[str], expected: List[str]
) -> Tuple[float, Dict[str, Any]]:
    if not expected:
        return 1.0, {"called": 0, "total": 0}
    actual_set = set(actual)
    called = sum(1 for t in expected if t in actual_set)
    return called / len(expected), {"called": called, "total": len(expected)}


def _score_response_match(
    output: str, required_terms: List[str]
) -> Tuple[float, Dict[str, Any]]:
    if not required_terms:
        return 1.0, {"matched": [], "missing": []}
    lower = output.lower()
    matched = [t for t in required_terms if t.lower() in lower]
    missing = [t for t in required_terms if t.lower() not in lower]
    return len(matched) / len(required_terms), {"matched": matched, "missing": missing}


def _score_constraint_compliance(
    output: str, constraints: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    import re

    checks: Dict[str, bool] = {}
    results: List[bool] = []

    required_terms = constraints.get("requiredTerms") or constraints.get("required_terms")
    if required_terms:
        lower = output.lower()
        all_present = all(t.lower() in lower for t in required_terms)
        checks["requiredTerms"] = all_present
        results.append(all_present)

    has_citations = constraints.get("hasCitations") if "hasCitations" in constraints else constraints.get("has_citations")
    if has_citations is not None:
        has_cite = bool(re.search(r"\[\d+\]|\(\w[^)]{1,80}\d{4}\)|https?://", output))
        passes = has_cite if has_citations else not has_cite
        checks["hasCitations"] = passes
        results.append(passes)

    slide_count = constraints.get("slideCount") if "slideCount" in constraints else constraints.get("slide_count")
    if slide_count is not None:
        count = len(re.findall(r"slide\s+\d+|section\s+\d+|##\s+\d+\.", output, re.IGNORECASE))
        passes = count == slide_count
        checks["slideCount"] = passes
        results.append(passes)

    if not results:
        return 1.0, {"checks": checks}
    score = sum(results) / len(results)
    return score, {"checks": checks}


def _score_contextual_precision(
    retrieved: List[str], expected: List[str]
) -> Tuple[float, Dict[str, Any]]:
    if not retrieved:
        return 1.0, {"relevant": 0, "total": 0}
    expected_set = set(expected)
    relevant = sum(1 for c in retrieved if c in expected_set)
    return relevant / len(retrieved), {"relevant": relevant, "total": len(retrieved)}


def _score_contextual_recall(
    retrieved: List[str], expected: List[str]
) -> Tuple[float, Dict[str, Any]]:
    if not expected:
        return 1.0, {"covered": 0, "total": 0}
    retrieved_set = set(retrieved)
    covered = sum(1 for c in expected if c in retrieved_set)
    return covered / len(expected), {"covered": covered, "total": len(expected)}


def evaluate(
    output: str,
    *,
    input: Optional[str] = None,
    expected_output: Optional[str] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    expected_tools: Optional[List[str]] = None,
    retrieved_context: Optional[List[str]] = None,
    expected_context: Optional[List[str]] = None,
    constraints: Optional[Dict[str, Any]] = None,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    requested = set(metrics) if metrics else set(OFFLINE_METRICS)

    actual_tool_names = [t["name"] for t in (tool_calls or [])]
    exp_tools = expected_tools or []

    scores: Dict[str, float] = {}
    details: Dict[str, Any] = {}

    if "tool_trajectory" in requested and exp_tools:
        s, d = _score_tool_trajectory(actual_tool_names, exp_tools)
        scores["tool_trajectory"] = s
        details["tool_trajectory"] = d

    if "tool_f1" in requested and (actual_tool_names or exp_tools):
        s, d = _score_tool_f1(actual_tool_names, exp_tools)
        scores["tool_f1"] = s
        details["tool_f1"] = d

    if "tool_correctness" in requested and exp_tools:
        s, d = _score_tool_correctness(actual_tool_names, exp_tools)
        scores["tool_correctness"] = s
        details["tool_correctness"] = d

    required_terms = (constraints or {}).get("requiredTerms") or (constraints or {}).get("required_terms", [])
    if "response_match" in requested and required_terms:
        s, d = _score_response_match(output, list(required_terms))
        scores["response_match"] = s
        details["response_match"] = d

    if "constraint_compliance" in requested and constraints:
        s, d = _score_constraint_compliance(output, constraints)
        scores["constraint_compliance"] = s
        details["constraint_compliance"] = d

    if (
        "contextual_precision" in requested
        and retrieved_context is not None
        and expected_context is not None
    ):
        s, d = _score_contextual_precision(retrieved_context, expected_context)
        scores["contextual_precision"] = s
        details["contextual_precision"] = d

    if (
        "contextual_recall" in requested
        and retrieved_context is not None
        and expected_context is not None
    ):
        s, d = _score_contextual_recall(retrieved_context, expected_context)
        scores["contextual_recall"] = s
        details["contextual_recall"] = d

    try:
        from . import _get_instance

        client = _get_instance()
        span_attrs: Dict[str, Any] = {
            "evalkit.span_type": "eval_result",
            "evalkit.eval_type": "offline",
        }
        for metric, score in scores.items():
            span_attrs[f"evalkit.metric.{metric}"] = score

        end_fn, _ = client.start_span("evaluation", span_attrs)
        end_fn("OK")
    except Exception:
        pass

    return scores
