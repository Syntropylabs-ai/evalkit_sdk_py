from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


def evaluate_simulation(
    simulation_id: str,
    collection_id: str,
    *,
    provider: str,
    model: str,
    api_key: str,
    run_id: Optional[str] = None,
    max_tokens: Optional[int] = None,
    api_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a completed scenario-simulation run against an evaluator collection.

    Runs the project's evaluator collection over every scenario of a simulation
    (server-side) and returns per-scenario, per-criterion scores + reasons. The
    judge is bring-your-own-key: ``provider`` + ``model`` + ``api_key`` are used
    for the judge call only and never stored.

    Args:
        simulation_id: The ``simulation_id`` returned by ``simulate_user``.
        collection_id: Mongo id of the evaluator collection whose rules to run.
        provider: Judge provider, e.g. ``"openai"``, ``"anthropic"``, ``"google"``.
        model: Judge model id, e.g. ``"gpt-4o"``.
        api_key: Judge API key (BYOK).
        run_id: A specific run of the simulation. Defaults to the most recent run.
        max_tokens: Judge output token cap (defaults to the backend's value).
        api_url: Override the control-plane URL (default: the value from ``init``).

    Returns:
        A dict shaped like::

            {
              "simulationId": str,
              "runId": str,
              "scenarios": [
                {
                  "scenarioId": str, "name": str, "traceId": str,
                  "status": "OK" | "ERROR" | "skipped",
                  "overallScore": float, "passed": bool,
                  "metrics": [
                    {"ruleId": str, "ruleName": str, "category": str,
                     "score": float, "passed": bool, "reason": str}
                  ],
                  "error": str   # only when status == "skipped"
                }
              ],
              "aggregate": {
                "averageScore": float, "passRate": float,
                "totalScenarios": int, "evaluatedScenarios": int
              }
            }
    """
    from .. import _get_instance

    client = _get_instance()
    base = (api_url or os.getenv("EVALKIT_API_URL") or client._opts.api_url).rstrip("/")

    body: Dict[str, Any] = {
        "simulationId": simulation_id,
        "collectionId": collection_id,
        "provider": provider,
        "model": model,
        "apiKey": api_key,
    }
    if run_id is not None:
        body["runId"] = run_id
    if max_tokens is not None:
        body["maxTokens"] = max_tokens

    url = base + "/simulations/evaluate"
    payload = json.dumps(body, default=str)
    req = urllib.request.Request(
        url,
        data=payload.encode(),
        headers={
            "Content-Type": "application/json",
            "X-Subscription-Key": client._opts.subscription_key,
        },
        method="POST",
    )

    # urllib is not auto-instrumented, so trace this backend hop explicitly. The span
    # nests under whatever trace is active. Don't log the BYOK api_key in the body.
    import evalkit as _ek

    logged_body = json.dumps({k: v for k, v in body.items() if k != "apiKey"}, default=str)
    attrs: Dict[str, Any] = {
        "evalkit.span_type": "http_call",
        "http.method": "POST",
        "http.url": url,
        "http.request.body": logged_body[:8192],
    }
    try:
        end_span, ctx = client.start_span("POST /simulations/evaluate", attrs)
    except Exception:
        end_span, ctx = None, None

    try:
        if ctx is not None:
            token = _ek._current_span.set(ctx)
        try:
            # Evaluating many scenarios can take minutes; give the judge room.
            with urllib.request.urlopen(req, timeout=300) as resp:
                raw = resp.read()
                attrs["http.status_code"] = resp.getcode()
                attrs["http.response.body"] = raw[:8192].decode("utf-8", errors="replace")
        finally:
            if ctx is not None:
                _ek._current_span.reset(token)
    except urllib.error.HTTPError as exc:
        attrs["http.status_code"] = exc.code
        try:
            attrs["http.response.body"] = exc.read()[:8192].decode("utf-8", errors="replace")
        except Exception:
            pass
        if end_span:
            end_span("ERROR")
        raise
    except Exception as exc:
        attrs["error.message"] = str(exc)
        if end_span:
            end_span("ERROR")
        raise

    if end_span:
        end_span("OK")
    return json.loads(raw)
