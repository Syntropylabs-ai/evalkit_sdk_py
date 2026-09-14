from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

from .types import TraceEnvelope

_log = logging.getLogger("evalkit")

MaskHook = Callable[[TraceEnvelope], Optional[TraceEnvelope]]

_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("0", "false", "no", "off")

CAPTURE_CONTENT_ENV_VARS = (
    "EVALKIT_CAPTURE_CONTENT",
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
)

CAPTURE_SECRETS_ENV_VARS = ("EVALKIT_CAPTURE_SECRETS", "EVALKIT_CAPTURE_SENSITIVE_HEADERS")

CAPTURE_SENSITIVE_HEADERS_ENV_VARS = CAPTURE_SECRETS_ENV_VARS

CAPTURE_STREAM_BODIES_ENV_VARS = ("EVALKIT_CAPTURE_STREAM_BODIES",)

CONTENT_ATTRIBUTES = frozenset({
    "gen_ai.request.body",
    "gen_ai.response.body",
    "gen_ai.response.thinking",
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.call.result",
    "tool.arguments",
    "tool.result",
    "function.args",
    "function.kwargs",
    "function.result",
    "db.statement",
    "db.query",
    "db.params",
    "retriever.query",
    "log.message",
    "request.body",
    "response.body",
    "http.request.body",
    "http.response.body",
    "http.request.headers",
    "http.response.headers",
    "evalkit.prompt",
    "evalkit.completion",
})

CONTENT_STRIPPED_ATTRIBUTE = "evalkit.content_captured"


def _env_flag(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    _log.warning("%s=%r is not a boolean; ignoring it and leaving the default in place", name, raw)
    return None


def _resolve_flag(explicit: Optional[bool], env_vars: tuple, default: bool) -> bool:
    if explicit is not None:
        return explicit
    for name in env_vars:
        flag = _env_flag(name)
        if flag is not None:
            return flag
    return default


def resolve_capture_content(explicit: Optional[bool]) -> bool:
    return _resolve_flag(explicit, CAPTURE_CONTENT_ENV_VARS, True)


def resolve_capture_secrets(explicit: Optional[bool]) -> bool:
    return _resolve_flag(explicit, CAPTURE_SECRETS_ENV_VARS, False)


resolve_capture_sensitive_headers = resolve_capture_secrets


def resolve_capture_stream_bodies(explicit: Optional[bool]) -> bool:
    return _resolve_flag(explicit, CAPTURE_STREAM_BODIES_ENV_VARS, True)


def _strip_attrs(attrs: Any) -> None:
    if not isinstance(attrs, dict):
        return
    for key in [k for k in attrs if k in CONTENT_ATTRIBUTES]:
        attrs.pop(key, None)


def strip_content(envelope: TraceEnvelope) -> TraceEnvelope:
    envelope.prompt = None
    envelope.completion = None
    attrs = envelope.attributes
    if isinstance(attrs, dict):
        _strip_attrs(attrs)
    else:
        attrs = {}
        envelope.attributes = attrs
    attrs[CONTENT_STRIPPED_ATTRIBUTE] = False
    for event in (envelope.events or ()):
        _strip_attrs(getattr(event, "attributes", None))
    return envelope


def apply_mask(mask: MaskHook, envelope: TraceEnvelope) -> Optional[TraceEnvelope]:
    try:
        masked = mask(envelope)
    except Exception as exc:
        _log.warning(
            "mask hook raised %s: %s — dropping this span, because exporting it would defeat the hook's purpose",
            type(exc).__name__, exc,
        )
        return None
    if masked is None:
        return None
    if not isinstance(masked, TraceEnvelope):
        _log.warning(
            "mask hook returned %s instead of a TraceEnvelope or None — dropping this span",
            type(masked).__name__,
        )
        return None
    return masked
