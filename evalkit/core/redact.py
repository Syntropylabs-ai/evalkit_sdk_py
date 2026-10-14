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
    "gen_ai.response.refusal",
    # The OpenTelemetry GenAI semantic conventions' content attributes, current and earlier ones.
    # Other instrumentation in the process writes them on the spans the bridge forwards, whatever
    # its own content switch says, so capture off has to remove them here.
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.system_instructions",
    "gen_ai.prompt",
    "gen_ai.completion",
    "gen_ai.retrieval.query.text",
    "gen_ai.retrieval.documents",
    "gen_ai.evaluation.explanation",
    # The same payloads as other GenAI tracers write them under plain names.
    "input.value",
    "output.value",
    "llm.prompts",
    "llm.prompt_template.template",
    "llm.prompt_template.variables",
    "all_messages_events",
    "final_result",
})

# Payloads other GenAI tracers write under their own prefix (<prefix>.observation.input and the
# like), matched by ending so that no tracer has to be listed. None of the SDK's own attributes
# end this way.
CONTENT_ATTRIBUTE_SUFFIXES = (
    ".observation.input",
    ".observation.output",
    ".trace.input",
    ".trace.output",
    ".entity.input",
    ".entity.output",
    ".component.input",
    ".component.output",
    ".spanInputs",
    ".spanOutputs",
    ".all_messages",
)

# Message and document lists that tracers flatten into one attribute per field, such as
# gen_ai.prompt.0.content and llm.input_messages.0.message.content. Under these prefixes the
# fields named in CONTENT_FIELDS hold the text; the others (role, tool name, call id, finish
# reason, score) describe it and stay.
CONTENT_LIST_PREFIXES = (
    "gen_ai.prompt.",
    "gen_ai.completion.",
    "llm.input_messages.",
    "llm.output_messages.",
    "retrieval.documents.",
    "reranker.input_documents.",
    "reranker.output_documents.",
    "embedding.embeddings.",
)
CONTENT_FIELDS = frozenset({"content", "contents", "text", "arguments", "result", "value", "message", "query", "url"})

# The GenAI conventions' message events (gen_ai.user.message, gen_ai.choice and the rest) carry the
# message under these generic names rather than under a gen_ai.* key. The events themselves stay,
# so a trace still shows that a message was there and who sent it.
MESSAGE_EVENT_PREFIX = "gen_ai."
MESSAGE_EVENT_FIELDS = frozenset({"content", "message", "tool_calls", "body", "parts"})

CONTENT_STRIPPED_ATTRIBUTE = "evalkit.content_captured"


def is_content_attribute(key: str) -> bool:
    """Whether a span or event attribute holds content that capture_content=False withholds."""
    if key in CONTENT_ATTRIBUTES or key.endswith(CONTENT_ATTRIBUTE_SUFFIXES):
        return True
    return key.startswith(CONTENT_LIST_PREFIXES) and key.rsplit(".", 1)[-1] in CONTENT_FIELDS


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


def _strip_attrs(attrs: Any, message_event: bool = False) -> None:
    if not isinstance(attrs, dict):
        return
    for key in [
        k for k in attrs
        if is_content_attribute(str(k)) or (message_event and k in MESSAGE_EVENT_FIELDS)
    ]:
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
        name = getattr(event, "name", None)
        _strip_attrs(
            getattr(event, "attributes", None),
            isinstance(name, str) and name.startswith(MESSAGE_EVENT_PREFIX),
        )
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
