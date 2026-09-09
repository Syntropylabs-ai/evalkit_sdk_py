from __future__ import annotations

import os
from typing import Any, Dict, Optional

from ..core.api import post_json


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

    return post_json(
        client,
        base + "/simulations/evaluate",
        body,
        timeout=300,
        span_name="POST /simulations/evaluate",
        redact=("apiKey",),
    )
