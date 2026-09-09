from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..core.api import post_json


def _jsonable_tool(t: Any) -> Any:
    if isinstance(t, (str, dict)):
        return t
    if callable(t):
        d: Dict[str, Any] = {"name": getattr(t, "__name__", "tool")}
        doc = (getattr(t, "__doc__", None) or "").strip()
        if doc:
            d["description"] = doc.splitlines()[0]
        return d
    name = getattr(t, "name", None) or getattr(t, "__name__", None)
    return {"name": str(name)} if name else str(t)


def generate_scenarios(
    agent_instructions: str,
    *,
    tools: Optional[List[Any]] = None,
    function_schemas: Optional[List[Any]] = None,
    context: Optional[Any] = None,
    rag: Optional[Dict[str, Any]] = None,
    count: int = 5,
    categories: Optional[List[str]] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
    max_completion_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    api_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    from .. import _get_instance

    client = _get_instance()
    base = (api_url or os.getenv("EVALKIT_API_URL") or client._opts.api_url).rstrip("/")

    body: Dict[str, Any] = {"agentInstructions": agent_instructions, "count": count}
    if tools is not None:
        body["tools"] = [_jsonable_tool(t) for t in tools]
    if function_schemas is not None:
        body["functionSchemas"] = [_jsonable_tool(t) for t in function_schemas]
    if context is not None:
        body["context"] = context
    if rag is not None:
        body["rag"] = rag
    if categories is not None:
        body["categories"] = categories
    if model is not None:
        body["model"] = model
    if provider is not None:
        body["provider"] = provider
    if api_key is not None:
        body["apiKey"] = api_key
    if temperature is not None:
        body["temperature"] = temperature
    if reasoning_effort is not None:
        body["reasoningEffort"] = reasoning_effort
    if max_completion_tokens is not None:
        body["maxCompletionTokens"] = max_completion_tokens
    if max_tokens is not None:
        body["maxTokens"] = max_tokens

    data = post_json(
        client,
        base + "/scenarios/generate",
        body,
        timeout=120,
        span_name="POST /scenarios/generate",
        redact=("apiKey",),
    )
    return data.get("scenarios", [])
