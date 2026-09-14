from __future__ import annotations

import json
import re
from typing import Any, Optional

from ..utils.headers import MASK, SENSITIVE_RE
from .types import TraceEnvelope

SECRET_KEY_RE = re.compile(
    r"(?:^|[._-])("
    r"api[_-]?key|apikey|secret|password|passwd|authorization|credentials?|"
    r"subscription[_-]?key|x-subscription-key|cookie|set-cookie|"
    r"private[_-]?key|access[_-]?key|client[_-]?secret"
    r")(?:$|[._-])",
    re.IGNORECASE,
)

SECRET_TOKEN_KEY_RE = re.compile(r"(?:^|[._-])token$", re.IGNORECASE)

SECRET_PATTERNS = (
    (re.compile(r"tk_live_[0-9a-f]{8,}"), "tk_live_" + MASK),
    (re.compile(r"tk_test_[0-9a-f]{8,}"), "tk_test_" + MASK),
    (re.compile(r"sk-[A-Za-z0-9_-]{16,}"), "sk-" + MASK),
    (re.compile(r"(Bearer)\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE), r"\1 " + MASK),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AKIA" + MASK),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "ghp_" + MASK),
)

HEADER_ATTRIBUTES = frozenset({"http.request.headers", "http.response.headers"})

_MAX_DEPTH = 12


def is_secret_key(key: str, header_like: bool = False) -> bool:
    if SECRET_KEY_RE.search(key) or SECRET_TOKEN_KEY_RE.search(key):
        return True
    return bool(header_like and SENSITIVE_RE.match(key))


def scrub_string(value: str) -> str:
    out = value
    for pattern, replacement in SECRET_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def _scrub_json_string(value: str, header_like: bool, depth: int) -> Optional[str]:
    trimmed = value.lstrip()
    if not trimmed or trimmed[0] not in "{[":
        return None
    try:
        parsed = json.loads(value)
    except Exception:
        return None
    if not isinstance(parsed, (dict, list)):
        return None
    scrubbed = scrub_value(parsed, header_like, depth + 1)
    if scrubbed == parsed:
        return None
    try:
        return json.dumps(scrubbed)
    except Exception:
        return None


def scrub_value(value: Any, header_like: bool = False, depth: int = 0) -> Any:
    if depth > _MAX_DEPTH:
        return value
    if isinstance(value, str):
        as_json = _scrub_json_string(value, header_like, depth)
        return scrub_string(value) if as_json is None else as_json
    if isinstance(value, dict):
        return {
            k: MASK if is_secret_key(str(k), header_like) else scrub_value(v, header_like, depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        scrubbed = [scrub_value(v, header_like, depth + 1) for v in value]
        return tuple(scrubbed) if isinstance(value, tuple) else scrubbed
    return value


def _scrub_attrs(attrs: Any) -> None:
    if not isinstance(attrs, dict):
        return
    for key in list(attrs.keys()):
        name = str(key)
        if is_secret_key(name):
            attrs[key] = MASK
        else:
            attrs[key] = scrub_value(attrs[key], name in HEADER_ATTRIBUTES)


def scrub_secrets(envelope: TraceEnvelope) -> TraceEnvelope:
    if isinstance(envelope.prompt, str):
        envelope.prompt = scrub_string(envelope.prompt)
    if isinstance(envelope.completion, str):
        envelope.completion = scrub_string(envelope.completion)
    if isinstance(envelope.status_message, str):
        envelope.status_message = scrub_string(envelope.status_message)
    _scrub_attrs(envelope.attributes)
    for event in (envelope.events or ()):
        _scrub_attrs(getattr(event, "attributes", None))
    return envelope
