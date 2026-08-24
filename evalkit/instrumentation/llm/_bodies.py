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
