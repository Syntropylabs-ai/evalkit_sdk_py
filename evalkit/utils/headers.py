from __future__ import annotations

import re
from typing import Any, Dict

MASK = "***"

SENSITIVE_RE = re.compile(
    r"^(authorization|cookie|set-cookie|x-api-key|api-key|"
    r"x-auth-token|proxy-authorization|x-secret|x-access-token|"
    r"x-subscription-key|subscription-key|token|password)$",
    re.IGNORECASE,
)

_capture_sensitive = False


def set_capture_sensitive_headers(enabled: bool) -> None:
    global _capture_sensitive
    _capture_sensitive = bool(enabled)


def capture_sensitive_headers() -> bool:
    return _capture_sensitive


def sanitize_headers(headers: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not headers:
        return out
    keep_sensitive = _capture_sensitive
    items = headers.items() if hasattr(headers, "items") else headers
    for k, v in items:
        if keep_sensitive or not SENSITIVE_RE.match(str(k)):
            out[str(k)] = str(v)
        else:
            out[str(k)] = MASK
    return out


def make_traceparent(trace_id: str, span_id: str) -> str:
    return f"00-{trace_id}-{span_id}-01"


_TRACEPARENT_RE = re.compile(
    r"^[ \t]*([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})(-.*)?[ \t]*$"
)
_ZERO_TRACE_ID = "0" * 32
_ZERO_SPAN_ID = "0" * 16


def parse_traceparent(header: str) -> tuple[str | None, str | None]:
    """The trace id and parent span id of a W3C traceparent header, or (None, None) when it is invalid.

    By the W3C Trace Context rules: lowercase hex only, an all-zero trace or span id is invalid,
    version ff is invalid, version 00 has exactly four fields, and a later version may add fields
    after the flags, which are ignored so that a newer caller still joins the trace.
    """
    if not isinstance(header, str) or len(header) > 512:
        return None, None
    match = _TRACEPARENT_RE.match(header)
    if match is None:
        return None, None
    version, trace_id, span_id, _flags, rest = match.groups()
    if version == "ff" or (version == "00" and rest):
        return None, None
    if trace_id == _ZERO_TRACE_ID or span_id == _ZERO_SPAN_ID:
        return None, None
    return trace_id, span_id


def client_span_name(method: str) -> str:
    """An outbound HTTP span's name: the method alone, as OpenTelemetry names a client span whose URL
    template is unknown. The URL, ids and all, stays in `http.url`; a name carrying the path would
    make every order id or charge id a separate operation."""
    return (method or "GET").upper()


def path_from_url(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).path or "/"
    except Exception:
        return url
