"""Single source of truth for tool-call extraction.

Every provider/agent instrumentation funnels through here to turn a raw LLM
response into a list of ``gen_ai.tool.call`` ``SpanEvent`` objects. Previously
this logic was copy-pasted across ~18 sites and drifted out of sync; centralizing
it means a fix or a new field lands in exactly one place.

The emitted attribute keys MUST stay in lock-step with
``EvalKitClient._maybe_emit_tool_spans`` (core/client.py), which promotes each
event into a first-class child ``tool_call`` span:
    gen_ai.tool.name            -> tool.name
    gen_ai.tool.call.arguments  -> tool.arguments
    gen_ai.tool.call.id         -> (kept as-is)
    gen_ai.tool.call.result     -> tool.result

Nothing here may ever raise into the host application: every extractor wraps its
body in try/except and degrades to ``[]``.
"""
from __future__ import annotations

import json
from typing import Any, List, Optional

from ..core.types import SpanEvent

_ARGS_MAX = 32768


def _truncate(value: Any, limit: int = _ARGS_MAX) -> str:
    try:
        text = value if isinstance(value, str) else str(value)
    except Exception:
        return "<unserializable>"
    return text if len(text) <= limit else text[:limit] + "…"


def _to_json(value: Any) -> str:
    """Serialize tool arguments. Strings pass through unchanged (they are already
    JSON from OpenAI-style providers); everything else is json-dumped."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except Exception:
        return str(value)


def tool_event(
    *,
    name: Any,
    arguments: Any = "",
    call_id: Any = "",
    result: Optional[Any] = None,
    ts: str,
) -> Optional[SpanEvent]:
    """Build ONE ``gen_ai.tool.call`` SpanEvent — the only place these are made.

    Returns ``None`` for a falsy ``name`` so callers uniformly skip nameless calls.
    ``gen_ai.tool.call.result`` is included only when ``result`` is not None.
    """
    try:
        if not name:
            return None
        attrs = {
            "gen_ai.tool.name": str(name),
            "gen_ai.tool.call.arguments": _truncate(_to_json(arguments)),
            "gen_ai.tool.call.id": str(call_id or ""),
        }
        if result is not None:
            attrs["gen_ai.tool.call.result"] = _truncate(_to_json(result))
        return SpanEvent(name="gen_ai.tool.call", timestamp=ts, attributes=attrs)
    except Exception:
        return None


# ── Shape 1: OpenAI Chat Completions (also Groq / LiteLLM / Mistral) ───────────

def from_openai_chat(result: Any, ts: str) -> List[SpanEvent]:
    """choices[*].message.tool_calls[*] with .id, .function.name, .function.arguments"""
    events: List[SpanEvent] = []
    try:
        for choice in (getattr(result, "choices", None) or []):
            msg = getattr(choice, "message", None)
            for tc in (getattr(msg, "tool_calls", None) or []):
                fn = getattr(tc, "function", None)
                ev = tool_event(
                    name=getattr(fn, "name", None),
                    arguments=getattr(fn, "arguments", ""),
                    call_id=getattr(tc, "id", ""),
                    ts=ts,
                )
                if ev is not None:
                    events.append(ev)
    except Exception:
        pass
    return events


# ── Shape 2: Anthropic content blocks (also Vertex Claude / Claude Agent SDK) ──

def from_anthropic_blocks(blocks: Any, ts: str) -> List[SpanEvent]:
    """blocks where type=="tool_use" with .id, .name, .input. Handles dict or object."""
    events: List[SpanEvent] = []
    try:
        for block in (blocks or []):
            if isinstance(block, dict):
                btype = block.get("type")
                name = block.get("name")
                bid = block.get("id", "")
                binput = block.get("input", {})
            else:
                btype = getattr(block, "type", None)
                name = getattr(block, "name", None)
                bid = getattr(block, "id", "")
                binput = getattr(block, "input", {})
            if btype != "tool_use":
                continue
            ev = tool_event(name=name, arguments=binput, call_id=bid, ts=ts)
            if ev is not None:
                events.append(ev)
    except Exception:
        pass
    return events


# ── Shape 3: Google GenAI parts ───────────────────────────────────────────────

def from_google_parts(result: Any, ts: str) -> List[SpanEvent]:
    """candidates[*].content.parts[*].function_call with .name, .args (no native id)."""
    events: List[SpanEvent] = []
    try:
        for candidate in (getattr(result, "candidates", None) or []):
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            for part in (getattr(content, "parts", None) or []):
                fc = getattr(part, "function_call", None)
                if fc is None:
                    continue
                raw_args = getattr(fc, "args", None) or {}
                try:
                    args = dict(raw_args)
                except Exception:
                    args = raw_args
                ev = tool_event(
                    name=getattr(fc, "name", None),
                    arguments=args,
                    call_id=f"call_{id(fc):016x}",
                    ts=ts,
                )
                if ev is not None:
                    events.append(ev)
    except Exception:
        pass
    return events


# ── Shape 4: OpenAI Responses API ─────────────────────────────────────────────

def _responses_citations(output_items: list) -> str:
    """URL citations live on message-type items under content[*].annotations[*]."""
    citations: List[str] = []
    try:
        for item in output_items:
            if getattr(item, "type", None) != "message":
                continue
            for block in (getattr(item, "content", None) or []):
                for ann in (getattr(block, "annotations", None) or []):
                    if getattr(ann, "type", None) != "url_citation":
                        continue
                    title = getattr(ann, "title", "") or ""
                    url = getattr(ann, "url", "") or ""
                    if url:
                        citations.append(f"[{title}]({url})" if title else url)
    except Exception:
        pass
    return "\n".join(citations[:10])


def from_openai_responses(items: Any, ts: str) -> List[SpanEvent]:
    """output[*]: type=="function_call" -> {call_id|id, name, arguments};
    type endswith "_call" (e.g. web_search_call) -> {id, type[:-5], action.query},
    with URL citations attached as the call result."""
    events: List[SpanEvent] = []
    try:
        output_items = list(items or [])
        citation_str = _responses_citations(output_items)
        for item in output_items:
            itype = getattr(item, "type", None)
            if itype == "function_call":
                ev = tool_event(
                    name=getattr(item, "name", None),
                    arguments=getattr(item, "arguments", ""),
                    call_id=getattr(item, "call_id", "") or getattr(item, "id", ""),
                    ts=ts,
                )
                if ev is not None:
                    events.append(ev)
            elif itype and isinstance(itype, str) and itype.endswith("_call"):
                action = getattr(item, "action", None)
                query = (getattr(action, "query", "") or "") if action else ""
                ev = tool_event(
                    name=itype[:-5],
                    arguments=query,
                    call_id=getattr(item, "id", ""),
                    result=citation_str or None,
                    ts=ts,
                )
                if ev is not None:
                    events.append(ev)
    except Exception:
        pass
    return events


# ── Shape 5: LangChain / LangGraph AIMessage ──────────────────────────────────

def from_langchain_message(message: Any, ts: str) -> List[SpanEvent]:
    """message.tool_calls[*] — each a dict or object with name/args/id."""
    events: List[SpanEvent] = []
    try:
        for tc in (getattr(message, "tool_calls", None) or []):
            if isinstance(tc, dict):
                name = tc.get("name")
                args = tc.get("args", {})
                tid = tc.get("id", "")
            else:
                name = getattr(tc, "name", None)
                args = getattr(tc, "args", {})
                tid = getattr(tc, "id", "")
            ev = tool_event(name=name, arguments=args, call_id=tid, ts=ts)
            if ev is not None:
                events.append(ev)
    except Exception:
        pass
    return events


def from_langchain_result(response: Any, ts: str) -> List[SpanEvent]:
    """Walk an LLMResult's generations[*][*].message and extract from each."""
    events: List[SpanEvent] = []
    try:
        for batch in (getattr(response, "generations", None) or []):
            for gen in batch:
                msg = getattr(gen, "message", None)
                if msg is not None:
                    events.extend(from_langchain_message(msg, ts))
    except Exception:
        pass
    return events


# ── Auto-detect dispatcher (tests + shape-agnostic callers) ───────────────────

def extract_tool_events(result: Any, ts: str) -> List[SpanEvent]:
    """Best-effort shape detection. Prefer the named extractors when the provider
    is known; use this only when it isn't."""
    try:
        if getattr(result, "choices", None) is not None:
            return from_openai_chat(result, ts)
        if getattr(result, "candidates", None) is not None:
            return from_google_parts(result, ts)
        if getattr(result, "generations", None) is not None:
            return from_langchain_result(result, ts)
        if getattr(result, "output", None) is not None:
            return from_openai_responses(getattr(result, "output"), ts)
        if getattr(result, "tool_calls", None) is not None:
            return from_langchain_message(result, ts)
        if getattr(result, "content", None) is not None:
            return from_anthropic_blocks(getattr(result, "content"), ts)
    except Exception:
        pass
    return []
