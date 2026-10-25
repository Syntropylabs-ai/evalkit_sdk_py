from __future__ import annotations

import json
from typing import Any, List, Optional

from ..core.types import SpanEvent


def _truncate(value: Any, limit: Optional[int] = None) -> str:
    """A tool call's arguments or result as text. Kept whole by default: the client cuts any value
    over the per-value limit, keeping its start and end, when the span is queued."""
    try:
        text = value if isinstance(value, str) else str(value)
    except Exception:
        return "<unserializable>"
    return text if limit is None or len(text) <= limit else text[:limit] + "…"


def _to_json(value: Any) -> str:
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


def from_stream_tool_calls(tool_calls: Any, ts: str) -> List[SpanEvent]:
    events: List[SpanEvent] = []
    try:
        for call in (tool_calls or []):
            ev = tool_event(
                name=call.get("name"),
                arguments=call.get("arguments", ""),
                call_id=call.get("id", ""),
                ts=ts,
            )
            if ev is not None:
                events.append(ev)
    except Exception:
        pass
    return events


def from_openai_chat(result: Any, ts: str) -> List[SpanEvent]:
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


def from_anthropic_blocks(blocks: Any, ts: str) -> List[SpanEvent]:
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
            if btype not in ("tool_use", "server_tool_use"):
                continue
            ev = tool_event(name=name, arguments=binput, call_id=bid, ts=ts)
            if ev is not None:
                events.append(ev)
    except Exception:
        pass
    return events


def from_google_parts(result: Any, ts: str) -> List[SpanEvent]:
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


def _responses_citations(output_items: list) -> str:
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


def from_langchain_message(message: Any, ts: str) -> List[SpanEvent]:
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


def extract_tool_events(result: Any, ts: str) -> List[SpanEvent]:
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
