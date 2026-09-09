from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass(frozen=True)
class SpanContext:
    trace_id: str
    span_id: str
    session_id: Optional[str] = None
    device_id: Optional[str] = None
    user_id: Optional[str] = None


@dataclass(frozen=True)
class Identity:
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    device_id: Optional[str] = None


NO_IDENTITY = Identity()

_current_identity: ContextVar[Identity] = ContextVar(
    "evalkit_identity", default=NO_IDENTITY
)


def current_identity() -> Identity:
    return _current_identity.get()


def set_identity(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    device_id: Optional[str] = None,
) -> Token:
    if user_id is None and session_id is None and device_id is None:
        return _current_identity.set(NO_IDENTITY)
    previous = _current_identity.get()
    return _current_identity.set(
        Identity(
            user_id=user_id if user_id is not None else previous.user_id,
            session_id=session_id if session_id is not None else previous.session_id,
            device_id=device_id if device_id is not None else previous.device_id,
        )
    )


def reset_identity(token: Token) -> None:
    try:
        _current_identity.reset(token)
    except Exception:
        pass


_current_span: ContextVar[Optional[SpanContext]] = ContextVar(
    "evalkit_span", default=None
)

_in_http_span: ContextVar[bool] = ContextVar("evalkit_in_http_span", default=False)


@dataclass
class SimulationContext:
    simulation_id: str
    run_id: str
    scenario_id: str
    scenario_name: Optional[str] = None
    persona: Optional[str] = None
    turn: int = 0
    tags: Optional[List[str]] = None
    collector: List[Any] = field(default_factory=list)


_current_simulation: ContextVar[Optional[SimulationContext]] = ContextVar(
    "evalkit_simulation", default=None
)


@dataclass(frozen=True)
class PromptRef:
    name: str
    version: int

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    def attributes(self) -> dict:
        return {"prompt.ref": self.ref, "prompt.name": self.name, "prompt.version": self.version}


_current_prompt: ContextVar[Optional[PromptRef]] = ContextVar("evalkit_prompt", default=None)

_ambient_trace_id: ContextVar[Optional[str]] = ContextVar(
    "evalkit_ambient_trace_id", default=None
)


_otel_available: Optional[bool] = None


def _otel_span_ctx() -> Optional[SpanContext]:
    global _otel_available
    if _otel_available is False:
        return None
    try:
        from opentelemetry import trace as _ot
        _otel_available = True
        sc = _ot.get_current_span().get_span_context()
        if sc and sc.is_valid:
            return SpanContext(trace_id=format(sc.trace_id, "032x"), span_id=format(sc.span_id, "016x"))
    except Exception:
        _otel_available = False
        return None
    return None


def current_span_ctx() -> Optional[SpanContext]:
    return _current_span.get() or _otel_span_ctx()


def get_or_create_ambient_trace_id() -> str:
    otel = _otel_span_ctx()
    if otel is not None:
        return otel.trace_id
    from ..core.client import _new_trace_id
    tid = _ambient_trace_id.get()
    if tid is None:
        tid = _new_trace_id()
        _ambient_trace_id.set(tid)
    return tid


def get_or_create_ambient_ctx() -> SpanContext:
    from ..core.client import _new_span_id
    return SpanContext(trace_id=get_or_create_ambient_trace_id(), span_id=_new_span_id())
