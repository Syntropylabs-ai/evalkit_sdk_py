from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field, replace
from typing import Any, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class SpanContext:
    trace_id: str
    span_id: str
    session_id: Optional[str] = None
    device_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_name: Optional[str] = None


# Spellings other instrumentations use for the same ids, in resolution order. Our own key comes
# first so a span tagged by EvalKit is never overridden by a looser convention on the same span.
SESSION_ID_KEYS = ("evalkit.session_id", "session.id", "conversation.id", "gen_ai.conversation.id", "session_id")
USER_ID_KEYS = ("evalkit.user_id", "user.id", "enduser.id", "user_id")
AGENT_NAME_KEYS = ("evalkit.agent_name", "gen_ai.agent.name", "agent.name")


def first_attr(attributes: Optional[Mapping[str, Any]], keys: tuple) -> Optional[str]:
    if not attributes:
        return None
    for key in keys:
        value = attributes.get(key)
        if isinstance(value, str) and value:
            return value
    return None


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


_current_agent: ContextVar[Optional[str]] = ContextVar("evalkit_agent", default=None)


def current_agent() -> Optional[str]:
    return _current_agent.get()


def set_agent_name(name: Optional[str]) -> Token:
    return _current_agent.set(name or None)


def reset_agent(token: Token) -> None:
    try:
        _current_agent.reset(token)
    except Exception:
        pass


_current_span: ContextVar[Optional[SpanContext]] = ContextVar(
    "evalkit_span", default=None
)

_in_http_span: ContextVar[bool] = ContextVar("evalkit_in_http_span", default=False)


class _Suppression:
    """Tracing held off while one health probe is answered.

    A probe has no trace of its own, so a span opened while answering it (a database ping, a
    traced function, a log line) would be a root trace of its own on every poll. A task or thread
    the handler starts copies the context and shares this object, and it is switched off once the
    probe is answered, so work that runs on after the response is traced as usual.
    """

    __slots__ = ("active",)

    def __init__(self) -> None:
        self.active = True


_suppression: ContextVar[Optional[_Suppression]] = ContextVar("evalkit_suppression", default=None)


def tracing_suppressed() -> bool:
    held = _suppression.get()
    return held is not None and held.active


def suppress_tracing() -> Tuple[Token, _Suppression]:
    held = _Suppression()
    return _suppression.set(held), held


def end_suppression(token: Optional[Token], held: Optional[_Suppression]) -> None:
    if held is not None:
        held.active = False
    if token is not None:
        try:
            _suppression.reset(token)
        except Exception:
            pass

# Set by a framework integration while one of its model calls runs (LlamaIndex's LLM spans): the
# SDK's own record of the provider request inside it is handed to that call, which sends the two
# as one span, instead of being queued as a second llm_call for the same request. Returns whether
# it took the envelope.
_llm_call_owner: ContextVar[Optional[Any]] = ContextVar("evalkit_llm_call_owner", default=None)


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


def child_span_ctx(trace_id: str, span_id: str) -> SpanContext:
    """The context of a span started under the current one: its own ids, its parent's identity and agent.

    An instrumented call is the current span while it runs, and whatever starts inside it (the
    HTTP client, another OpenTelemetry library) reads its session, user and agent from here;
    bare ids would drop them to the process-wide defaults.
    """
    parent = _current_span.get()
    if parent is None or parent.trace_id != trace_id:
        return SpanContext(trace_id=trace_id, span_id=span_id)
    return replace(parent, span_id=span_id)


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
