"""Instrumentation for Google ADK (Agent Development Kit).

ADK's LLM calls already surface as llm_call spans for free: every Gemini call
ADK makes internally goes through google.genai, which EvalKit patches at the
class level (see google_genai.py). Two things don't come for free and are
added here:

  - A `function_call` span (operation "run_agent") wrapping each agent turn,
    sourced from BaseAgent.run_async — the one method every agent (root or
    delegated sub-agent) funnels through, since LlmAgent and friends override
    `_run_async_impl`, not `run_async` itself. Wrapping it once therefore
    gives full agent-hierarchy visibility (root + nested sub-agents) with a
    single patch. Shaped to match the dashboard's existing "run_agent"
    narrative convention (see trace-analysis.service.ts's `firstArg` /
    `extractAgentText`): `function.args` carries the user turn text and
    `function.result` is an `AgentTurnResult(text=..., tool_calls=...)` repr —
    the exact shape evalkit's own simulation framework already produces, so
    ADK runs get the identical narrative rendering for free.

  - A `tool_call` span per tool execution, sourced from LlmAgent's
    before_tool_callback / after_tool_callback — ADK's public, documented
    extension point for exactly this purpose. Deliberately not hooking ADK's
    internal tracing plumbing (google.adk.telemetry._instrumentation,
    google.adk.flows.llm_flows.functions): that surface has already changed
    shape between recent ADK releases, while the callback fields on LlmAgent
    are a stable, user-facing API. EvalKit's callbacks are prepended/appended
    around whatever the caller already configured, so nothing is overridden
    or double-registered, and the before-hook always returns None so it can
    never be mistaken for a response override (ADK skips the real tool call
    if any before_tool_callback returns truthy).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List

from ..._version import SDK_VERSION
from ...core.client import _new_span_id, _now_iso, _safe_io
from ...core.context import SpanContext, _current_span, current_span_ctx, get_or_create_ambient_trace_id
from ...core.types import TraceEnvelope
from ...simulation.types import AgentTurnResult

if TYPE_CHECKING:
    from ...core.client import EvalKitClient

# id(tool_context) -> time.time() at before_tool_callback time. Safe to key by
# identity: ADK creates a fresh ToolContext per tool invocation, and the
# before/after pair for one call always shares that same instance.
_tool_call_starts: Dict[int, float] = {}


def _iso_from_epoch(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _evalkit_before_tool(*, tool: Any = None, args: Any = None, tool_context: Any = None, **_: Any) -> None:
    try:
        _tool_call_starts[id(tool_context)] = time.time()
    except Exception:
        pass
    return None


def _make_after_tool(ek: "EvalKitClient"):
    def _evalkit_after_tool(
        *, tool: Any = None, args: Any = None, tool_context: Any = None, tool_response: Any = None, **_: Any
    ) -> None:
        try:
            start = _tool_call_starts.pop(id(tool_context), None)
            now = time.time()
            latency_ms = int((now - start) * 1000) if start else 0
            start_ts = _iso_from_epoch(start) if start else _now_iso()
            end_ts = _now_iso()

            ctx = current_span_ctx()
            trace_id = ctx.trace_id if ctx else get_or_create_ambient_trace_id()
            parent_span_id = ctx.span_id if ctx else None

            name = getattr(tool, "name", None) or type(tool).__name__
            attrs: Dict[str, Any] = {
                **ek._meta_attrs(),
                "evalkit.span_type": "tool_call",
                "gen_ai.tool.name": str(name),
                "gen_ai.tool.description": str(getattr(tool, "description", "") or ""),
            }
            if args is not None:
                attrs["gen_ai.tool.call.arguments"] = _safe_io(args)
            if tool_response is not None:
                attrs["gen_ai.tool.call.result"] = _safe_io(tool_response)

            ek.enqueue(TraceEnvelope(
                trace_id=trace_id,
                span_id=_new_span_id(),
                parent_span_id=parent_span_id,
                span_type="tool_call",
                status="OK",
                operation=str(name),
                service_name=ek._opts.service_name,
                attributes=attrs,
                latency_ms=latency_ms,
                start_time=start_ts,
                end_time=end_ts,
                created_at=start_ts,
                environment=ek._opts.environment,
                app_version=ek._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
        except Exception:
            pass
        return None

    return _evalkit_after_tool


def _install_tool_call_tracing(ek: "EvalKitClient", LlmAgent: Any) -> None:
    before_prop = LlmAgent.__dict__.get("canonical_before_tool_callbacks")
    after_prop = LlmAgent.__dict__.get("canonical_after_tool_callbacks")
    if before_prop is None or after_prop is None:
        return
    if getattr(before_prop.fget, "_evalkit", False):
        return

    original_before = before_prop.fget
    original_after = after_prop.fget
    after_hook = _make_after_tool(ek)

    def patched_before_getter(self: Any) -> List[Any]:
        return [_evalkit_before_tool, *original_before(self)]

    def patched_after_getter(self: Any) -> List[Any]:
        return [*original_after(self), after_hook]

    patched_before_getter._evalkit = True  # type: ignore[attr-defined]
    LlmAgent.canonical_before_tool_callbacks = property(patched_before_getter)
    LlmAgent.canonical_after_tool_callbacks = property(patched_after_getter)


async def _traced_agent_run(
    ek: "EvalKitClient",
    original: AsyncIterator[Any],
    agent_name: str,
    user_text: str,
) -> AsyncIterator[Any]:
    ctx_parent = current_span_ctx()
    trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
    span_id = _new_span_id()
    parent_span_id = ctx_parent.span_id if ctx_parent else None
    start_ms = time.time() * 1000
    ts = _now_iso()

    # Publish the span context so nested spans (LLM calls, tool calls, and
    # sub-agent run_agent spans) nest correctly under this agent turn.
    span_ctx = SpanContext(trace_id=trace_id, span_id=span_id)
    token = _current_span.set(span_ctx)

    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    try:
        async for event in original:
            try:
                content = getattr(event, "content", None)
                for part in (getattr(content, "parts", None) or []):
                    t = getattr(part, "text", None)
                    if t:
                        text_parts.append(t)
                get_fc = getattr(event, "get_function_calls", None)
                if callable(get_fc):
                    for fc in get_fc():
                        tool_calls.append({
                            "name": getattr(fc, "name", None),
                            "args": dict(getattr(fc, "args", None) or {}),
                        })
            except Exception:
                pass
            yield event

        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)
        result = AgentTurnResult(text="".join(text_parts), tool_calls=tool_calls)

        attrs: Dict[str, Any] = {
            **ek._meta_attrs(),
            "evalkit.span_type": "function_call",
            "function.args": _safe_io([user_text]),
            "function.result": _safe_io(result),
            "gen_ai.agent.name": agent_name,
        }
        ek.enqueue(TraceEnvelope(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            span_type="function_call",
            status="OK",
            operation="run_agent",
            service_name=ek._opts.service_name,
            attributes=attrs,
            latency_ms=latency,
            start_time=ts,
            end_time=end_ts,
            created_at=ts,
            environment=ek._opts.environment,
            app_version=ek._opts.app_version,
            sdk_version=SDK_VERSION,
        ))
    except Exception as exc:
        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)
        attrs = {
            **ek._meta_attrs(),
            "evalkit.span_type": "function_call",
            "function.args": _safe_io([user_text]),
            "gen_ai.agent.name": agent_name,
        }
        ek.enqueue(TraceEnvelope(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            span_type="function_call",
            status="ERROR",
            status_message=str(exc),
            operation="run_agent",
            service_name=ek._opts.service_name,
            attributes=attrs,
            latency_ms=latency,
            start_time=ts,
            end_time=end_ts,
            created_at=ts,
            environment=ek._opts.environment,
            app_version=ek._opts.app_version,
            sdk_version=SDK_VERSION,
        ))
        raise
    finally:
        _current_span.reset(token)


def _install_agent_run_tracing(ek: "EvalKitClient", BaseAgent: Any) -> None:
    if getattr(BaseAgent.run_async, "_evalkit", False):
        return

    original_run_async = BaseAgent.run_async

    def traced_run_async(self: Any, parent_context: Any, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        user_text = ""
        try:
            uc = getattr(parent_context, "user_content", None)
            parts = getattr(uc, "parts", None) or []
            user_text = "".join(getattr(p, "text", "") or "" for p in parts)
        except Exception:
            pass
        agent_name = getattr(self, "name", None) or type(self).__name__
        original_gen = original_run_async(self, parent_context, *args, **kwargs)
        return _traced_agent_run(ek, original_gen, agent_name, user_text)

    traced_run_async._evalkit = True  # type: ignore[attr-defined]
    BaseAgent.run_async = traced_run_async  # type: ignore[method-assign]


def patch_google_adk(ek: "EvalKitClient") -> None:
    """Auto-instruments Google ADK. No-op if google-adk is not installed.

    Safe to call multiple times — idempotent via the ``_evalkit`` sentinel,
    matching every other instrumentation in this package.
    """
    try:
        from google.adk.agents.llm_agent import LlmAgent
    except ImportError:
        return

    try:
        _install_tool_call_tracing(ek, LlmAgent)
    except Exception:
        pass

    try:
        from google.adk.agents.base_agent import BaseAgent
        _install_agent_run_tracing(ek, BaseAgent)
    except Exception:
        pass
