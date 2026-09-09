from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .ids import span_id, trace_id
from .otlp import json_text, log_record
from .spans import AGENT, OPERATION, STATUS_ERROR, STATUS_OK, TOOL_CALL, attributes, make_span, trace_payload

Record = Tuple[str, Dict[str, Any]]
TRACES = "traces"
LOGS = "logs"
ATTRIBUTION = "hooks"
ENRICHMENT_PREFIX = "evalkit"
ENRICHMENT_EVENT = "enrichment"
ROLE_MAIN = "main"
ROLE_SUBAGENT = "subagent"
SUBAGENT_ID_FIELDS = ("subagent_id", "agent_id", "subagentId")
SUBAGENT_TYPE_FIELDS = ("subagent_type", "agent_type", "subagentType")
FAILURE_FIELDS = ("error", "error_message", "failure")
MS_IN_NS = 1_000_000


@dataclass(frozen=True)
class ToolEvent:
    label: str
    name_fields: Sequence[str] = ()
    args_fields: Sequence[str] = ()
    result_fields: Sequence[str] = ()
    id_fields: Sequence[str] = ()
    duration_fields: Sequence[str] = ("duration_ms", "duration")
    failed: bool = False


@dataclass(frozen=True)
class TurnSpec:
    vendor: str
    agent_name: str
    session_fields: Sequence[str]
    turn_fields: Sequence[str]
    model_fields: Sequence[str]
    prompt_event: str
    prompt_fields: Sequence[str]
    completion_events: Sequence[str]
    completion_fields: Sequence[str]
    tools: Dict[str, ToolEvent]
    session_events: Sequence[str] = ()
    subagent_start: Optional[str] = None
    subagent_stop: Optional[str] = None
    nested: Sequence[str] = ()


CURSOR = TurnSpec(
    vendor="cursor",
    agent_name="Cursor",
    session_fields=("conversation_id",),
    turn_fields=("generation_id",),
    model_fields=("model",),
    prompt_event="beforeSubmitPrompt",
    prompt_fields=("prompt",),
    completion_events=("afterAgentResponse", "stop"),
    completion_fields=("text", "response"),
    session_events=("sessionStart", "sessionEnd"),
    subagent_start="subagentStart",
    subagent_stop="subagentStop",
    tools={
        "postToolUse": ToolEvent("tool", ("tool_name",), ("tool_input",), ("tool_output",), ("tool_use_id", "tool_call_id")),
        "postToolUseFailure": ToolEvent("tool", ("tool_name",), ("tool_input",), ("tool_output",), ("tool_use_id", "tool_call_id"), failed=True),
        "afterShellExecution": ToolEvent("Shell", ("tool_name",), ("command",), ("output",), ("tool_use_id", "call_id")),
        "afterMCPExecution": ToolEvent("MCP", ("tool_name",), ("tool_input", "arguments"), ("tool_output", "output"), ("tool_use_id", "tool_call_id")),
        "afterFileEdit": ToolEvent("Edit", ("tool_name",), ("file_path", "edits"), (), ("tool_use_id",)),
    },
)

WINDSURF = TurnSpec(
    vendor="windsurf",
    agent_name="Windsurf Cascade",
    session_fields=("trajectory_id",),
    turn_fields=("execution_id",),
    model_fields=("model_name", "model"),
    prompt_event="pre_user_prompt",
    prompt_fields=("prompt", "user_prompt", "message", "text"),
    completion_events=("post_cascade_response",),
    completion_fields=("response", "text", "cascade_response", "message"),
    tools={
        "post_run_command": ToolEvent("Command", ("tool_name",), ("command",), ("output",), ("tool_call_id", "call_id")),
        "post_write_code": ToolEvent("WriteCode", ("tool_name",), ("file_path", "path", "code"), ("diff",), ("tool_call_id", "call_id")),
        "post_read_code": ToolEvent("ReadCode", ("tool_name",), ("file_path", "path"), (), ("tool_call_id", "call_id")),
        "post_mcp_tool_use": ToolEvent("MCP", ("tool_name",), ("tool_input", "arguments"), ("tool_output", "output"), ("tool_call_id", "call_id")),
    },
    nested=("tool_info",),
)

SPECS: Dict[str, TurnSpec] = {CURSOR.vendor: CURSOR, WINDSURF.vendor: WINDSURF}


def events(vendor: str) -> Tuple[str, ...]:
    spec = SPECS[vendor]
    ordered = [*spec.session_events, spec.prompt_event, *spec.completion_events, *spec.tools]
    ordered += [name for name in (spec.subagent_start, spec.subagent_stop) if name]
    return tuple(ordered)


@dataclass
class Turn:
    trace: str
    root: str
    session: str
    turn: str
    capture_full: bool
    now_ns: int


def _scopes(payload: Dict[str, Any], nested: Sequence[str]) -> List[Dict[str, Any]]:
    out = [payload]
    for name in nested:
        value = payload.get(name)
        if isinstance(value, dict):
            out.append(value)
    return out


def _value(payload: Dict[str, Any], fields: Sequence[str], nested: Sequence[str] = ()) -> Any:
    for scope in _scopes(payload, nested):
        for name in fields:
            found = scope.get(name)
            if found not in (None, ""):
                return found
    return None


def _text(payload: Dict[str, Any], fields: Sequence[str], nested: Sequence[str] = ()) -> Optional[str]:
    found = _value(payload, fields, nested)
    if found is None:
        return None
    return found if isinstance(found, str) else json_text(found)


def _millis(payload: Dict[str, Any], fields: Sequence[str], nested: Sequence[str] = ()) -> int:
    found = _value(payload, fields, nested)
    try:
        return max(int(float(found)), 0)
    except (TypeError, ValueError):
        return 0


def _base(spec: TurnSpec, payload: Dict[str, Any], turn: Turn) -> Dict[str, Any]:
    values: Dict[str, Any] = {
        "agent.vendor": spec.vendor,
        "gen_ai.conversation.id": turn.session,
        "agent.turn.id": turn.turn,
        "agent.role": ROLE_MAIN,
        "gen_ai.agent.name": spec.agent_name,
        "evalkit.attribution": ATTRIBUTION,
    }
    if turn.capture_full:
        values["user.id"] = _text(payload, ("user_email", "user_id"))
    return values


def _subagent(values: Dict[str, Any], payload: Dict[str, Any]) -> Optional[str]:
    identifier = _text(payload, SUBAGENT_ID_FIELDS)
    kind = _text(payload, SUBAGENT_TYPE_FIELDS)
    if not identifier and not kind:
        return None
    values["agent.role"] = ROLE_SUBAGENT
    values["agent.subagent.id"] = identifier
    values["agent.subagent.type"] = kind
    if kind:
        values["gen_ai.agent.name"] = kind
    return span_id(identifier) if identifier else None


def _turn_root(spec: TurnSpec, payload: Dict[str, Any], turn: Turn) -> List[Record]:
    prompt = _text(payload, spec.prompt_fields)
    values = _base(spec, payload, turn)
    values["evalkit.span_type"] = AGENT
    values["gen_ai.operation.name"] = OPERATION[AGENT]
    values["gen_ai.request.model"] = _text(payload, spec.model_fields, spec.nested)
    name = "turn"
    if turn.capture_full and prompt:
        values["prompt"] = prompt
        name = prompt
    span = make_span(turn.trace, turn.root, None, name, turn.now_ns, turn.now_ns, values)
    return [(TRACES, trace_payload(spec.vendor, [span]))]


def _enrichment(spec: TurnSpec, turn: Turn, span: str, values: Dict[str, Any]) -> Record:
    merged = {
        "evalkit.enrichment": True,
        "agent.vendor": spec.vendor,
        "gen_ai.conversation.id": turn.session,
        "agent.turn.id": turn.turn,
        "evalkit.end_time_unix_nano": turn.now_ns,
        **values,
    }
    record = log_record(spec.vendor, ENRICHMENT_EVENT, attributes(merged), prefix=ENRICHMENT_PREFIX, trace=turn.trace, span=span)
    return (LOGS, record)


def _turn_completion(spec: TurnSpec, payload: Dict[str, Any], turn: Turn) -> List[Record]:
    values: Dict[str, Any] = {}
    if turn.capture_full:
        values["completion"] = _text(payload, spec.completion_fields)
    return [_enrichment(spec, turn, turn.root, values)]


def _subagent_root(spec: TurnSpec, payload: Dict[str, Any], turn: Turn) -> List[Record]:
    values = _base(spec, payload, turn)
    child = _subagent(values, payload)
    if not child:
        return []
    kind = values.get("agent.subagent.type") or "subagent"
    parent_call = _text(payload, ("tool_call_id", "tool_use_id"))
    values["evalkit.span_type"] = AGENT
    values["gen_ai.operation.name"] = OPERATION[AGENT]
    values["gen_ai.request.model"] = _text(payload, ("subagent_model", *spec.model_fields))
    values["agent.subagent.is_parallel_worker"] = payload.get("is_parallel_worker")
    if turn.capture_full:
        values["prompt"] = _text(payload, ("task", "prompt", "description"))
    parent = span_id(parent_call) if parent_call else turn.root
    span = make_span(turn.trace, child, parent, f"subagent:{kind}", turn.now_ns, turn.now_ns, values)
    return [(TRACES, trace_payload(spec.vendor, [span]))]


def _subagent_completion(spec: TurnSpec, payload: Dict[str, Any], turn: Turn) -> List[Record]:
    identifier = _text(payload, SUBAGENT_ID_FIELDS)
    kind = _text(payload, SUBAGENT_TYPE_FIELDS)
    values: Dict[str, Any] = {
        "agent.role": ROLE_SUBAGENT,
        "agent.subagent.id": identifier,
        "agent.subagent.type": kind,
        "agent.subagent.total_tool_uses": _value(payload, ("tool_call_count",)),
        "agent.subagent.message_count": _value(payload, ("message_count",)),
        "agent.subagent.status": _text(payload, ("status", "final_status")),
    }
    if turn.capture_full:
        values["completion"] = _text(payload, ("summary", "last_assistant_message", "text"))
    if not identifier:
        record = log_record(spec.vendor, "subagent_stop", attributes(values), trace=turn.trace)
        return [(LOGS, record)]
    return [_enrichment(spec, turn, span_id(identifier), values)]


def _tool_span(spec: TurnSpec, event: str, tool: ToolEvent, payload: Dict[str, Any], turn: Turn) -> List[Record]:
    duration_ms = _millis(payload, tool.duration_fields, spec.nested)
    name = _text(payload, tool.name_fields, spec.nested) or tool.label
    call_id = _text(payload, tool.id_fields, spec.nested)
    span = span_id(call_id) if call_id else span_id(turn.turn, event, str(turn.now_ns))
    values = _base(spec, payload, turn)
    parent = _subagent(values, payload) or turn.root
    failed = tool.failed or _value(payload, FAILURE_FIELDS) is not None
    values.update(
        {
            "evalkit.span_type": TOOL_CALL,
            "gen_ai.operation.name": OPERATION[TOOL_CALL],
            "gen_ai.tool.name": name,
            "gen_ai.tool.call.id": call_id,
            "duration_ms": duration_ms or None,
            "success": not failed,
        }
    )
    if turn.capture_full:
        values["tool.arguments"] = _value(payload, tool.args_fields, spec.nested)
        values["tool.result"] = _value(payload, tool.result_fields, spec.nested)
    status = STATUS_ERROR if failed else STATUS_OK
    start = turn.now_ns - duration_ms * MS_IN_NS
    return [(TRACES, trace_payload(spec.vendor, [make_span(turn.trace, span, parent, name, start, turn.now_ns, values, status)]))]


def _session_log(spec: TurnSpec, event: str, payload: Dict[str, Any], turn: Turn) -> List[Record]:
    values = {
        "agent.vendor": spec.vendor,
        "gen_ai.conversation.id": turn.session,
        "gen_ai.agent.name": spec.agent_name,
        "evalkit.noise": True,
        "duration_ms": _value(payload, ("duration_ms",)),
        "final_status": _text(payload, ("final_status",)),
    }
    return [(LOGS, log_record(spec.vendor, event, attributes(values)))]


def build(vendor: str, event: str, payload: Dict[str, Any], capture_full: bool, now_ns: int) -> List[Record]:
    spec = SPECS.get(vendor)
    if spec is None:
        return []
    session = _text(payload, spec.session_fields) or ""
    identifier = _text(payload, spec.turn_fields) or ""
    turn = Turn(trace_id(session, identifier), span_id(identifier), session, identifier, capture_full, now_ns)
    if event in spec.session_events:
        return _session_log(spec, event, payload, turn)
    if not identifier:
        return []
    if event == spec.prompt_event:
        return _turn_root(spec, payload, turn)
    if event in spec.completion_events:
        return _turn_completion(spec, payload, turn)
    if event == spec.subagent_start:
        return _subagent_root(spec, payload, turn)
    if event == spec.subagent_stop:
        return _subagent_completion(spec, payload, turn)
    tool = spec.tools.get(event)
    if tool is not None:
        return _tool_span(spec, event, tool, payload, turn)
    return []
