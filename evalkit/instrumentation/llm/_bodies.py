"""Shared helpers to serialize raw LLM request/response bodies for the trace.

Each provider patch adds two attributes to its ``llm_call`` envelope:
    gen_ai.request.body   — the request (messages/params/config) as JSON
    gen_ai.response.body  — the response object as JSON

Serialization is defensive: it never raises into the host, falls back to
``str()``, and caps the result (the exporter's fitEvent drops these keys first
anyway). Mirrors the reference shape in ``google_genai.py``.
"""
from __future__ import annotations

import json
from typing import Any, Optional


def _dump(o: Any) -> Any:
    if isinstance(o, (list, tuple)):
        return [_dump(x) for x in o]
    if isinstance(o, dict):
        return {k: _dump(v) for k, v in o.items()}
    if hasattr(o, "model_dump"):
        try:
            return o.model_dump(exclude_none=True)
        except Exception:
            return str(o)
    return o


def _safe_body(obj: Any, limit: int = 2 * 1024 * 1024) -> Optional[str]:
    """Best-effort JSON of a request/response object. Never raises; truncated."""
    if obj is None:
        return None
    try:
        s = json.dumps(obj, default=str)
    except Exception:
        try:
            s = str(obj)
        except Exception:
            return None
    return s if len(s) <= limit else s[:limit] + "…"
