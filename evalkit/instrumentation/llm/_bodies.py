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


def _safe_body(obj: Any, limit: Optional[int] = None) -> Optional[str]:
    """A request or response body as JSON text.

    Kept whole by default: the client cuts a value over the per-value limit when the span is queued,
    keeping its start and its end, so a long conversation keeps its newest messages, and records the
    original size as ``evalkit.truncated.<attribute>``. ``limit`` cuts it the same way here.
    """
    if obj is None:
        return None
    try:
        s = json.dumps(obj, default=str)
    except Exception:
        try:
            s = str(obj)
        except Exception:
            return None
    if limit is not None:
        from ...core.limits import cut_middle

        s = cut_middle(s, limit)[0]
    return s
