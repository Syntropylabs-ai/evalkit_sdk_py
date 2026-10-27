"""Which answers make an incoming request's span an error.

By default every 4xx and 5xx answer does, and an exception the server answers with one is recorded on
the span: a request the server refused is shown as a failure. ``server_error_status_codes`` replaces the
400-599 default with codes (``503``) and inclusive ranges (``"500-599"``): ``["500-599"]`` follows
OpenTelemetry's HTTP conventions, where a 4xx is the caller's mistake (a scanner probing ``/.env``, an
expired token, a body that fails validation), so its span keeps the status code, stays OK and records no
exception. Outgoing calls keep a 4xx as an error whatever the setting: there the call failed for the app.
The rules are the TypeScript SDK's. A value that cannot be read is skipped with one warning and the
default stays; init() never fails over it.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional, Tuple

_log = logging.getLogger("evalkit")

DEFAULT_SERVER_ERROR_STATUS_CODES: Tuple[str, ...] = ("400-599",)

_Range = Tuple[int, int]
_DEFAULT_RANGES: Tuple[_Range, ...] = ((400, 599),)
_RANGE = re.compile(r"^\s*(\d{3})\s*-\s*(\d{3})\s*$")
_SINGLE = re.compile(r"^\s*(\d{3})\s*$")

_ranges: Tuple[_Range, ...] = _DEFAULT_RANGES


def _is_code(n: int) -> bool:
    return 100 <= n <= 599


def _range_of(spec: Any) -> Optional[_Range]:
    if isinstance(spec, bool):
        return None
    if isinstance(spec, int):
        return (spec, spec) if _is_code(spec) else None
    if not isinstance(spec, str):
        return None
    found = _RANGE.match(spec)
    if found:
        low, high = int(found.group(1)), int(found.group(2))
        return (low, high) if _is_code(low) and _is_code(high) and low <= high else None
    found = _SINGLE.match(spec)
    if found and _is_code(int(found.group(1))):
        return (int(found.group(1)), int(found.group(1)))
    return None


def parse_server_error_status_codes(value: Any) -> Tuple[Optional[Tuple[_Range, ...]], Optional[str]]:
    """``value`` as inclusive ranges, and None; or None and why it cannot be read."""
    if not isinstance(value, (list, tuple, set, frozenset)):
        return None, (
            'server_error_status_codes must be a list of status codes (503) or ranges ("500-599") '
            f"(got {type(value).__name__})"
        )
    ranges = []
    for spec in value:
        found = _range_of(spec)
        if found is None:
            shown = repr(spec) if isinstance(spec, (str, int)) and not isinstance(spec, bool) else type(spec).__name__
            return None, (
                'server_error_status_codes entries must be status codes from 100 to 599 or ranges such as '
                f'"500-599" (got {shown})'
            )
        ranges.append(found)
    return tuple(ranges), None


def configure(value: Any = None) -> None:
    """The codes in force from here on; None, or a value that cannot be read, leaves 400-599."""
    global _ranges
    if value is None:
        _ranges = _DEFAULT_RANGES
        return
    ranges, problem = parse_server_error_status_codes(value)
    if problem is not None or ranges is None:
        _log.warning("evalkit.init(): %s; server error status codes stay 500-599.", problem)
        _ranges = _DEFAULT_RANGES
        return
    _ranges = ranges


def reset() -> None:
    """Back to 400-599."""
    global _ranges
    _ranges = _DEFAULT_RANGES


def is_server_error_status(code: Any) -> bool:
    """Whether a server that answered ``code`` failed."""
    if isinstance(code, bool) or not isinstance(code, int):
        return False
    for low, high in _ranges:
        if low <= code <= high:
            return True
    return False


def server_span_status(code: Any) -> str:
    """The status of a server span that answered ``code``."""
    return "ERROR" if is_server_error_status(code) else "OK"


def raised_request_failed(code: Optional[int]) -> bool:
    """Whether a request whose handling raised failed the server, given the status it was answered with.

    ``None`` is a request that got no answer at all (cancelled), and a status below 400 one whose response
    had started when it broke off: both are the server's failure. An answer outside the codes is not,
    whatever raised (with ``["500-599"]``, a client error).
    """
    if not isinstance(code, int) or isinstance(code, bool) or code < 400:
        return True
    return is_server_error_status(code)


# The exceptions web frameworks raise to answer a request with a status, by the module and qualified name
# of a class in their MRO, so none of the frameworks has to be imported to tell them apart, and the
# attribute holding the status, or the status the class always answers with.
_HTTP_ANSWERS = {
    ("starlette.exceptions", "HTTPException"): "status_code",
    ("werkzeug.exceptions", "HTTPException"): "code",
    ("litestar.exceptions.http_exceptions", "HTTPException"): "status_code",
    ("django.http.response", "Http404"): 404,
    ("django.core.exceptions", "PermissionDenied"): 403,
    ("django.core.exceptions", "SuspiciousOperation"): 400,
    ("django.core.exceptions", "BadRequest"): 400,
}


def http_answer_status(exc: Any) -> Optional[int]:
    """The status a web framework answers ``exc`` with, when ``exc`` is one of its own HTTP answers
    (FastAPI's and Starlette's ``HTTPException``, Werkzeug's, Litestar's, Django's ``Http404``,
    ``PermissionDenied``, ``SuspiciousOperation`` and ``BadRequest``); None for any other exception."""
    try:
        for klass in type(exc).__mro__:
            found = _HTTP_ANSWERS.get((getattr(klass, "__module__", None), getattr(klass, "__qualname__", None)))
            if found is None:
                continue
            status = getattr(exc, found, None) if isinstance(found, str) else found
            if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599:
                return status
            return None
    except Exception:
        pass
    return None
