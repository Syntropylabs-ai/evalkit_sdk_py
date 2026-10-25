from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional


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

    trace_id: str
    span_id: str
    created_at: str

    span_type: Literal[
        "llm_call", "tool_call", "log", "http_call", "function_call", "db_query", "eval_result", "agent", "internal"
    ] = "log"
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
    cache_read_tokens: Optional[int] = None
    cache_creation_tokens: Optional[int] = None
    latency_ms: Optional[int] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    environment: Optional[str] = None
    app_version: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    device_id: Optional[str] = None
    agent_name: Optional[str] = None
    trace_project_id: Optional[str] = None
    sdk_version: Optional[str] = None
    is_simulation: bool = False
    simulation_id: Optional[str] = None
    scenario_id: Optional[str] = None
    run_id: Optional[str] = None
    # The exception that failed this span. Never sent as it is: the client records it by the shared
    # exception contract (status, an `exception` event, `evalkit.exception.*` attributes) when the span
    # is queued, and lets go of it then, so no traceback, and none of the frames it holds, outlives that.
    exception: Optional[BaseException] = field(default=None, repr=False, compare=False)
    # Whether that exception propagated out of the span (None: it did).
    exception_escaped: Optional[bool] = field(default=None, repr=False, compare=False)

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
        if self.cache_read_tokens is not None:
            d["cacheReadTokens"] = self.cache_read_tokens
        if self.cache_creation_tokens is not None:
            d["cacheCreationTokens"] = self.cache_creation_tokens
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
        if self.agent_name:
            d["agentName"] = self.agent_name
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
    # About how many bytes the spans waiting to be sent may hold; past it the oldest spans that are not
    # model calls go first.
    max_queue_bytes: int = 32 * 1024 * 1024
    # The most of one HTTP request or response body a server or client span keeps.
    max_body_bytes: int = 64 * 1024
    # The most of one value (a prompt, a completion, an attribute) a span keeps, in UTF-8 bytes.
    max_value_bytes: int = 16 * 1024 * 1024
    environment: Optional[str] = None
    app_version: Optional[str] = None
    device_id: Optional[str] = None
    capture_content: bool = True
    capture_secrets: bool = False
    capture_stream_bodies: bool = True
    mask: Optional[Callable[["TraceEnvelope"], Optional["TraceEnvelope"]]] = None
    export_gzip: bool = True
    # Spans a second each traced function may record; 0 records every call.
    function_rate_limit: float = 100.0
    # The lowest level a log record needs to be recorded as a log span, and how many a second may be.
    log_level: int = logging.WARNING
    log_rate_limit: float = 50.0
    # The share of traces kept; model calls, tool calls and failed spans are kept whatever it is.
    sample_rate: float = 1.0
    # Module prefixes that are, or are not, the application's own code in an exception's frames.
    in_app_include: Optional[List[str]] = None
    in_app_exclude: Optional[List[str]] = None
    # Requests, outgoing calls and functions kept out of traces, as given; core.ignore compiles them.
    ignore_routes: Optional[List[Any]] = None
    ignore_urls: Optional[List[Any]] = None
    ignore_functions: Optional[List[Any]] = None
    # Whether health and readiness probes are traced like any request.
    trace_health_checks: bool = False

    @property
    def capture_sensitive_headers(self) -> bool:
        return self.capture_secrets
