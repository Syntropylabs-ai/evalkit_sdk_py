from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class SpanEvent:
    name: str
    timestamp: str
    attributes: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name, "timestamp": self.timestamp}
        if self.attributes:
            d["attributes"] = self.attributes
        return d


@dataclass
class TraceEnvelope:
    """Matches the Go trace-service TraceEnvelope exactly (camelCase JSON)."""

    trace_id: str
    span_id: str
    created_at: str

    span_type: Literal["llm_call", "tool_call", "log", "http_call", "function_call", "db_query", "eval_result"] = "log"
    parent_span_id: Optional[str] = None
    span_kind: Optional[Literal["CLIENT", "SERVER", "INTERNAL", "PRODUCER", "CONSUMER"]] = None
    status: Optional[Literal["OK", "ERROR", "UNSET"]] = None
    status_message: Optional[str] = None
    service_name: Optional[str] = None
    operation: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    prompt: Optional[str] = None
    completion: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None
    events: Optional[List[SpanEvent]] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    latency_ms: Optional[int] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    environment: Optional[str] = None
    app_version: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    device_id: Optional[str] = None
    trace_project_id: Optional[str] = None
    sdk_version: Optional[str] = None
    is_simulation: bool = False
    simulation_id: Optional[str] = None
    scenario_id: Optional[str] = None
    run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "spanType": self.span_type,
            "createdAt": self.created_at,
        }
        if self.parent_span_id:
            d["parentSpanId"] = self.parent_span_id
        if self.span_kind:
            d["spanKind"] = self.span_kind
        if self.status:
            d["status"] = self.status
        if self.status_message:
            d["statusMessage"] = self.status_message
        if self.service_name:
            d["serviceName"] = self.service_name
        if self.operation:
            d["operation"] = self.operation
        if self.model:
            d["model"] = self.model
        if self.provider:
            d["provider"] = self.provider
        if self.prompt is not None:
            d["prompt"] = self.prompt
        if self.completion is not None:
            d["completion"] = self.completion
        if self.attributes:
            d["attributes"] = self.attributes
        if self.events:
            d["events"] = [e.to_dict() for e in self.events]
        if self.tokens_in is not None:
            d["tokensIn"] = self.tokens_in
        if self.tokens_out is not None:
            d["tokensOut"] = self.tokens_out
        if self.latency_ms is not None:
            d["latencyMs"] = self.latency_ms
        if self.start_time:
            d["startTime"] = self.start_time
        if self.end_time:
            d["endTime"] = self.end_time
        if self.environment:
            d["environment"] = self.environment
        if self.app_version:
            d["appVersion"] = self.app_version
        if self.user_id:
            d["userId"] = self.user_id
        if self.session_id:
            d["sessionId"] = self.session_id
        if self.device_id:
            d["deviceId"] = self.device_id
        if self.trace_project_id:
            d["traceProjectId"] = self.trace_project_id
        if self.sdk_version:
            d["sdkVersion"] = self.sdk_version
        if self.is_simulation:
            d["isSimulation"] = True
        if self.simulation_id:
            d["simulationId"] = self.simulation_id
        if self.scenario_id:
            d["scenarioId"] = self.scenario_id
        if self.run_id:
            d["runId"] = self.run_id
        return d


@dataclass
class EvalKitOptions:
    subscription_key: str
    base_url: str = "https://api.syntropylabs.ai"
    api_url: str = "https://api.syntropylabs.ai"
    service_name: str = "unknown"
    debug: bool = False
    max_export_batch_size: int = 512
    scheduled_delay_ms: int = 5000
    max_queue_size: int = 4096
    environment: Optional[str] = None
    app_version: Optional[str] = None
    device_id: Optional[str] = None
