from __future__ import annotations

from typing import Any, Dict, List, Optional

from .otlp import attr

SCOPE = "evalkit-cli"
KIND_INTERNAL = 1
STATUS_UNSET = 0
STATUS_OK = 1
STATUS_ERROR = 2
OPERATION_NAME_MAX = 80

AGENT = "agent"
LLM_CALL = "llm_call"
TOOL_CALL = "tool_call"
OPERATION = {AGENT: "invoke_agent", LLM_CALL: "chat", TOOL_CALL: "execute_tool"}


def attributes(values: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [attr(key, value) for key, value in values.items() if value is not None and value != ""]


def make_span(
    trace: str,
    span: str,
    parent: Optional[str],
    name: str,
    start_ns: int,
    end_ns: int,
    values: Dict[str, Any],
    status: int = STATUS_UNSET,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "traceId": trace,
        "spanId": span,
        "name": name[:OPERATION_NAME_MAX],
        "kind": KIND_INTERNAL,
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(end_ns),
        "attributes": attributes(values),
    }
    if parent:
        out["parentSpanId"] = parent
    if status:
        out["status"] = {"code": status}
    return out


def resource(vendor: str) -> Dict[str, Any]:
    return {"attributes": attributes({"service.name": vendor, "agent.vendor": vendor, "telemetry.sdk.name": SCOPE})}


def trace_payload(vendor: str, spans: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"resourceSpans": [{"resource": resource(vendor), "scopeSpans": [{"scope": {"name": SCOPE}, "spans": spans}]}]}
