from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

from .core.api import post_json

_KINDS = ("thumbs", "rating", "reward", "text")


def _normalise_value(kind: str, value: Any) -> Union[float, str]:
    if kind == "text":
        if not isinstance(value, str) or not value.strip():
            raise ValueError("text feedback needs a non-empty string value")
        return value
    if kind == "thumbs":
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if value in ("up", "down"):
            return 1.0 if value == "up" else 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{kind} feedback needs a numeric value")
    return float(value)


def feedback(
    trace_id: str,
    kind: str,
    value: Any,
    *,
    comment: Optional[str] = None,
    user_id: Optional[str] = None,
    span_id: Optional[str] = None,
    session_id: Optional[str] = None,
    name: Optional[str] = None,
    api_url: Optional[str] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    from . import _get_instance

    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {', '.join(_KINDS)}")
    if not trace_id:
        raise ValueError("trace_id is required")

    client = _get_instance()
    base = (api_url or os.getenv("EVALKIT_API_URL") or client._opts.api_url).rstrip("/")
    body: Dict[str, Any] = {"traceId": trace_id, "kind": kind, "value": _normalise_value(kind, value)}
    if span_id is not None:
        body["spanId"] = span_id
    if session_id is not None:
        body["sessionId"] = session_id
    if comment is not None:
        body["comment"] = comment
    if user_id is not None:
        body["userId"] = user_id
    if name is not None:
        body["name"] = name
    return post_json(client, base + "/v1/feedback", body, timeout=timeout, span_name="POST /v1/feedback")


def reward(trace_id: str, value: float, **kwargs: Any) -> Dict[str, Any]:
    return feedback(trace_id, "reward", value, **kwargs)
