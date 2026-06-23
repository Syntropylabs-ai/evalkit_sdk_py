"""Instrumentation for the Claude Agent SDK (claude_agent_sdk).

The Claude Agent SDK runs the actual Anthropic API call in a subprocess (the
Claude Code CLI), so EvalKit's normal in-process anthropic patch never fires.
Instead, we wrap the two public entry points that the SDK exposes back to Python:

  • ``query()`` — the stateless async-generator API
  • ``ClaudeSDKClient.receive_response()`` — the stateful multi-turn API

Both yield a ``ResultMessage`` as their final item, which carries token counts,
cost, latency, stop reason, and model usage. We read those fields and emit an
``llm_call`` TraceEnvelope — giving EvalKit full visibility without any change
to application code.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, AsyncIterator, List, Optional

from ..._version import SDK_VERSION
from ...core.client import _new_span_id, _now_iso
from ...core.context import SpanContext, _current_span, current_span_ctx, get_or_create_ambient_trace_id
from ...core.types import SpanEvent, TraceEnvelope
from .._toolcalls import from_anthropic_blocks
from ._bodies import _dump, _safe_body

if TYPE_CHECKING:
    from ...core.client import EvalKitClient

_PROMPT_MAX = 32768
_COMPLETION_MAX = 32768


def _get_int(obj: Any, *keys: str) -> int:
    for k in keys:
        v = obj.get(k) if isinstance(obj, dict) else getattr(obj, k, None)
        if v is not None:
            return int(v)
    return 0


def _get_attr(obj: Any, *names: str) -> Any:
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


async def _traced_gen(
    ek: "EvalKitClient",
    original: AsyncIterator[Any],
    prompt: str,
    model_hint: str,
) -> AsyncIterator[Any]:
    """Yield messages from *original*, emitting a llm_call span on completion."""
    ctx_parent = current_span_ctx()
    trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
    span_id = _new_span_id()
    parent_span_id = ctx_parent.span_id if ctx_parent else None
    start_ms = time.time() * 1000
    ts = _now_iso()

    # Publish the span context so child spans (tool calls, sub-agents) nest here.
    span_ctx = SpanContext(trace_id=trace_id, span_id=span_id)
    token = _current_span.set(span_ctx)

    model = model_hint
    completion = ""
    events: List[SpanEvent] = []

    try:
        async for message in original:
            msg_type = type(message).__name__

            if msg_type == "AssistantMessage":
                m = _get_attr(message, "model")
                if m:
                    model = str(m)
                for block in getattr(message, "content", []):
                    if getattr(block, "type", None) == "text":
                        completion += getattr(block, "text", "")
                events.extend(from_anthropic_blocks(getattr(message, "content", []), _now_iso()))

            elif msg_type == "ResultMessage":
                usage = _get_attr(message, "usage") or {}
                tokens_in = _get_int(usage, "input_tokens", "inputTokens")
                tokens_out = _get_int(usage, "output_tokens", "outputTokens")
                end_ts = _now_iso()
                latency = int(time.time() * 1000 - start_ms)

                cost = _get_attr(message, "total_cost_usd", "totalCostUsd")
                stop_reason = _get_attr(message, "stop_reason", "stopReason")
                duration_api_ms = _get_attr(message, "duration_api_ms", "durationApiMs")

                attrs: dict = {**ek._meta_attrs(), "gen_ai.request.model": model}
                if cost is not None:
                    attrs["llm.cost_usd"] = str(cost)
                if stop_reason:
                    attrs["gen_ai.finish_reason"] = str(stop_reason)
                if duration_api_ms is not None:
                    attrs["llm.api_latency_ms"] = str(duration_api_ms)
                num_turns = _get_attr(message, "num_turns", "numTurns")
                if num_turns is not None:
                    attrs["llm.num_turns"] = str(num_turns)

                request_body = _safe_body({"prompt": prompt, "model": model})
                if request_body:
                    attrs["gen_ai.request.body"] = request_body
                response_body = _safe_body(_dump(message))
                if response_body:
                    attrs["gen_ai.response.body"] = response_body

                ek.enqueue(TraceEnvelope(
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    span_type="llm_call",
                    status="OK",
                    operation="claude_agent_sdk.query",
                    service_name=ek._opts.service_name,
                    provider="anthropic",
                    model=model,
                    prompt=prompt[:_PROMPT_MAX],
                    completion=completion[:_COMPLETION_MAX],
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    latency_ms=latency,
                    start_time=ts,
                    end_time=end_ts,
                    created_at=ts,
                    attributes=attrs,
                    events=events or None,
                    environment=ek._opts.environment,
                    app_version=ek._opts.app_version,
                    sdk_version=SDK_VERSION,
                ))

            yield message

    except Exception as exc:
        end_ts = _now_iso()
        latency = int(time.time() * 1000 - start_ms)
        err_request_body = _safe_body({"prompt": prompt, "model": model})
        ek.enqueue(TraceEnvelope(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            span_type="llm_call",
            status="ERROR",
            status_message=str(exc),
            operation="claude_agent_sdk.query",
            service_name=ek._opts.service_name,
            provider="anthropic",
            model=model,
            prompt=prompt[:_PROMPT_MAX],
            latency_ms=int(time.time() * 1000 - start_ms),
            start_time=ts,
            end_time=_now_iso(),
            created_at=ts,
            attributes={**ek._meta_attrs(),
                        **({"gen_ai.request.body": err_request_body} if err_request_body else {})},
            environment=ek._opts.environment,
            app_version=ek._opts.app_version,
            sdk_version=SDK_VERSION,
        ))
        raise
    finally:
        _current_span.reset(token)


def patch_claude_agent_sdk(ek: "EvalKitClient") -> None:
    """Monkey-patch ``claude_agent_sdk.query`` and ``ClaudeSDKClient.receive_response``
    to emit ``llm_call`` spans sourced from ``ResultMessage`` usage data.

    Safe to call multiple times — idempotent via the ``_evalkit`` sentinel.
    No-op if ``claude_agent_sdk`` is not installed.
    """
    try:
        import claude_agent_sdk as _sdk
    except ImportError:
        return

    # ── patch query() ──────────────────────────────────────────────────────────
    if not getattr(_sdk.query, "_evalkit", False):
        _original_query = _sdk.query

        def _traced_query(
            *,
            prompt: str = "",
            options: Any = None,
            **kwargs: Any,
        ) -> AsyncIterator[Any]:
            model_hint = "unknown"
            if options is not None:
                model_hint = str(getattr(options, "model", None) or model_hint)
            original = _original_query(prompt=prompt, options=options, **kwargs)
            return _traced_gen(ek, original, prompt, model_hint)

        _traced_query._evalkit = True  # type: ignore[attr-defined]
        _sdk.query = _traced_query

    # ── patch ClaudeSDKClient.receive_response() ───────────────────────────────
    cls = getattr(_sdk, "ClaudeSDKClient", None)
    if cls is None:
        return

    if getattr(cls.receive_response, "_evalkit", False):
        return

    _original_receive = cls.receive_response

    async def _traced_receive(self: Any) -> AsyncIterator[Any]:
        prompt = str(getattr(self, "_last_message", "") or "")
        model_hint = "unknown"
        opts = getattr(self, "_options", None) or getattr(self, "options", None)
        if opts is not None:
            model_hint = str(getattr(opts, "model", None) or model_hint)
        async for msg in _traced_gen(ek, _original_receive(self), prompt, model_hint):
            yield msg

    _traced_receive._evalkit = True  # type: ignore[attr-defined]
    cls.receive_response = _traced_receive
