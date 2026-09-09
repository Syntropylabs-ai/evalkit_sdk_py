from __future__ import annotations

from typing import Any, Optional

_ID_ATTRS = ("pk", "id", "identity", "username", "email")


def user_id_of(candidate: Any) -> Optional[str]:
    if candidate is None:
        return None
    try:
        if getattr(candidate, "is_authenticated", False) is not True:
            return None
        for attr in _ID_ATTRS:
            value = getattr(candidate, attr, None)
            if value:
                return str(value)
    except Exception:
        return None
    return None
