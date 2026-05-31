from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


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

    url = base + "/scenarios/generate"
    payload = json.dumps(body, default=str)
    req = urllib.request.Request(
        url,
        data=payload.encode(),
        headers={
            "Content-Type": "application/json",
            "X-Subscription-Key": client._opts.subscription_key,
        },
        method="POST",
    )

    # urllib is not auto-instrumented, so trace this backend hop explicitly. The span
    # nests under whatever trace is active (e.g. the POST /api/simulate request).
    import evalkit as _ek

    attrs: Dict[str, Any] = {
        "evalkit.span_type": "http_call",
        "http.method": "POST",
        "http.url": url,
        "http.request.body": payload[:8192],
    }
    try:
        end_span, ctx = client.start_span("POST /scenarios/generate", attrs)
    except Exception:
        end_span, ctx = None, None

    try:
        if ctx is not None:
            token = _ek._current_span.set(ctx)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read()
                attrs["http.status_code"] = resp.getcode()
                attrs["http.response.body"] = raw[:8192].decode("utf-8", errors="replace")
        finally:
            if ctx is not None:
                _ek._current_span.reset(token)
    except urllib.error.HTTPError as exc:
        attrs["http.status_code"] = exc.code
        try:
            attrs["http.response.body"] = exc.read()[:8192].decode("utf-8", errors="replace")
        except Exception:
            pass
        if end_span:
            end_span("ERROR")
        raise
    except Exception as exc:
        attrs["error.message"] = str(exc)
        if end_span:
            end_span("ERROR")
        raise

    if end_span:
        end_span("OK")
    data = json.loads(raw)
    return data.get("scenarios", [])
