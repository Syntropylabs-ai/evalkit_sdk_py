from __future__ import annotations

import json
import time
from typing import Any

from .._version import SDK_VERSION
from ..core.client import EvalKitClient, _new_span_id, _now_iso
from ..core.context import _current_span, get_or_create_ambient_trace_id
from ..core.types import TraceEnvelope


def _result_text(result: Any) -> str:
    try:
        content = getattr(result, "content", None)
        if isinstance(content, list):
            parts = []
            for c in content:
                t = getattr(c, "text", None)
                parts.append(t if t is not None else str(c))
            return "\n".join(parts)
    except Exception:
        pass
    try:
        return json.dumps(result, default=str)
    except Exception:
        return str(result)


def patch_mcp(client: EvalKitClient) -> None:
    try:
        from mcp import ClientSession
    except Exception:
        return

    if getattr(ClientSession.call_tool, "_evalkit", False):
        return

    orig = ClientSession.call_tool

    async def traced_call_tool(self: Any, name: str, arguments: Any = None, *args: Any, **kwargs: Any) -> Any:
        # Tracing must never break the tool call: if setup throws, run untraced.
        try:
            ctx = _current_span.get()
            trace_id = ctx.trace_id if ctx else get_or_create_ambient_trace_id()
            parent_span_id = ctx.span_id if ctx else None
            span_id = _new_span_id()
            start_ms = time.time() * 1000
            ts = _now_iso()
            attrs: dict = {
                **client._meta_attrs(),
                "evalkit.span_type": "tool_call",
                "gen_ai.tool.name": str(name),
                "mcp.tool.name": str(name),
            }
            if arguments is not None:
                try:
                    attrs["gen_ai.tool.call.arguments"] = json.dumps(arguments, default=str)[:8192]
                except Exception:
                    pass
        except Exception:
            return await orig(self, name, arguments, *args, **kwargs)

        try:
            result = await orig(self, name, arguments, *args, **kwargs)
        except Exception as e:
            try:
                attrs["error.message"] = str(e)
                client.enqueue(TraceEnvelope(
                    trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                    span_type="tool_call", status="ERROR", status_message=str(e),
                    operation=str(name), service_name=client._opts.service_name,
                    attributes=attrs, latency_ms=int(time.time() * 1000 - start_ms),
                    start_time=ts, end_time=_now_iso(), created_at=ts,
                    environment=client._opts.environment, app_version=client._opts.app_version,
                    sdk_version=SDK_VERSION,
                ))
            except Exception:
                pass
            raise

        try:
            is_error = bool(getattr(result, "isError", False))
            try:
                attrs["gen_ai.tool.call.result"] = _result_text(result)[:8192]
            except Exception:
                pass
            client.enqueue(TraceEnvelope(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id,
                span_type="tool_call", status="ERROR" if is_error else "OK",
                operation=str(name), service_name=client._opts.service_name,
                attributes=attrs, latency_ms=int(time.time() * 1000 - start_ms),
                start_time=ts, end_time=_now_iso(), created_at=ts,
                environment=client._opts.environment, app_version=client._opts.app_version,
                sdk_version=SDK_VERSION,
            ))
        except Exception:
            pass
        return result

    traced_call_tool._evalkit = True  # type: ignore[attr-defined]
    ClientSession.call_tool = traced_call_tool  # type: ignore[method-assign]
