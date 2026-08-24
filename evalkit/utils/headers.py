from __future__ import annotations

import re
from typing import Any, Dict

SENSITIVE_RE = re.compile(
    r"^(authorization|cookie|set-cookie|x-api-key|api-key|"
    r"x-auth-token|proxy-authorization|x-secret|x-access-token|"
    r"x-subscription-key|token|password)$",
    re.IGNORECASE,
)


def sanitize_headers(headers: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not headers:
        return out
    items = headers.items() if hasattr(headers, "items") else headers
    for k, v in items:
        if not SENSITIVE_RE.match(str(k)):
            out[str(k)] = str(v)
    return out


def make_traceparent(trace_id: str, span_id: str) -> str:
    return f"00-{trace_id}-{span_id}-01"


def parse_traceparent(header: str) -> tuple[str | None, str | None]:
    parts = header.split("-")
    if len(parts) == 4 and parts[0] == "00" and len(parts[1]) == 32 and len(parts[2]) == 16:
        return parts[1], parts[2]
    return None, None


def path_from_url(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).path or "/"
    except Exception:
        return url
