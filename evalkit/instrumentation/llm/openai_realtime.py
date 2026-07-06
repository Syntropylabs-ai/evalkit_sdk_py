"""Trace OpenAI Realtime (voice/realtime over a WebSocket).

Nothing auto-instruments a WebSocket, so we wrap the Realtime connection:
open a ``voice.session`` trace, accumulate the assistant's output transcript, emit
one ``voice.turn`` llm_call span per completed response (text + latency), emit a
``tool_call`` span per function call, and close the session span when the socket
closes. Same span model as the Gemini Live tracer (google_live.py).

The Python API is async:

    client = openai.AsyncOpenAI(...)
    async with client.beta.realtime.connect(model=...) as connection:
        await connection.session.update(...)
        async for event in connection:
            ...

``connect()`` is an *async context manager*. We wrap the connection and forward
async iteration through an observing generator. Event shapes vary by version, so
everything is duck-typed and defensive. Tracing must NEVER break the host: every
wrapper path falls back to the real connection on any error.
"""
from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Optional

from ..._version import SDK_VERSION
from ...core.client import _new_span_id, _now_iso
from ...core.types import SpanEvent, TraceEnvelope
from ._bodies import _dump, _safe_body

if TYPE_CHECKING:
    from ...core.client import EvalKitClient

_MAX_TEXT = 8192


def _base_attrs(model: str) -> dict:
    return {
        "gen_ai.system": "openai",
        "gen_ai.request.model": model,
    }


def _event_type(event: Any) -> str:
    try:
        t = getattr(event, "type", None)
        if isinstance(t, str):
            return t
        if isinstance(event, dict):
            return str(event.get("type", ""))
    except Exception:
        pass
    return ""


def _event_get(event: Any, key: str) -> Any:
    try:
        if isinstance(event, dict):
            return event.get(key)
        return getattr(event, key, None)
    except Exception:
        return None


class _ObservedConnection:
    """Wraps a real Realtime connection: forwards everything, observes events.

    Proxies unknown attributes (``.session``, ``.send``, ``.response``, …) to the
    underlying connection so the host's usage is unchanged. Ends the session span
    when the connection closes.
    """

    # Delta event types that carry assistant text (transcript or text response).
    _DELTA_TYPES = (
        "response.audio_transcript.delta",
        "response.text.delta",
        "response.output_text.delta",
    )
    _DONE_TYPES = ("response.done", "response.completed")

    def __init__(self, client: "EvalKitClient", conn: Any, base: dict, ctx: Any, end_session) -> None:
        self.__dict__["_ek"] = client
        self.__dict__["_conn"] = conn
        self.__dict__["_base"] = base
        self.__dict__["_ctx"] = ctx
        self.__dict__["_end_session"] = end_session
        self.__dict__["_closed"] = False
        self._reset_turn()

    def _reset_turn(self) -> None:
        self.__dict__["_buf"] = []
        self.__dict__["_input_parts"] = []
        self.__dict__["_tool_events"] = []
        self.__dict__["_turn_start"] = time.time() * 1000

    def _emit_tool_call(self, name: Any, args: Any, call_id: Any = None) -> None:
        if not name:
            return
        try:
            if isinstance(args, str):
                args_str = args[:_MAX_TEXT]
            else:
                args_str = json.dumps(args or {})[:_MAX_TEXT]
        except Exception:
            args_str = str(args)[:_MAX_TEXT]
        self._tool_events.append(SpanEvent(
            name="gen_ai.tool.call",
            timestamp=_now_iso(),
            attributes={"gen_ai.tool.name": str(name), "gen_ai.tool.call.arguments": args_str},
        ))
        attrs = {
            **self._base,
            "evalkit.span_type": "tool_call",
            "gen_ai.tool.name": str(name),
            "gen_ai.tool.call.arguments": args_str,
            "tool.name": str(name),
        }
        if call_id:
            attrs["gen_ai.tool.call.id"] = str(call_id)
        try:
            end, _c = self._ek.start_span(str(name), attrs, self._ctx)
            end("OK")
        except Exception:
            pass

    def _scan_output_items(self, response: Any) -> None:
        """Pull function_call output items from a completed response object."""
        try:
            output = _event_get(response, "output") or []
            for item in output:
                itype = _event_get(item, "type")
                if itype in ("function_call", "tool_call"):
                    self._emit_tool_call(
                        _event_get(item, "name"),
                        _event_get(item, "arguments"),
                        _event_get(item, "call_id") or _event_get(item, "id"),
                    )
        except Exception:
            pass

    def _end_turn(self, response: Any = None) -> None:
        try:
            prompt = "\n".join(self._input_parts).strip()
            completion = "".join(self._buf).strip()
            latency_ms = int(time.time() * 1000 - self._turn_start)
            events = self._tool_events
            if response is not None:
                self._scan_output_items(response)
                events = self._tool_events  # _scan_output_items may have appended more
            self._reset_turn()

            if not prompt and not completion:
                return  # skip audio-only turns with no transcript at all

            attrs = dict(self._base)
            usage = _event_get(response, "usage") if response is not None else None
            tokens_in = _event_get(usage, "input_tokens") or 0 if usage else 0
            tokens_out = _event_get(usage, "output_tokens") or 0 if usage else 0
            total_tokens = _event_get(usage, "total_tokens") or 0 if usage else 0
            if usage is not None:
                attrs["gen_ai.usage.total_tokens"] = total_tokens or (tokens_in + tokens_out)
                input_details = _event_get(usage, "input_token_details")
                output_details = _event_get(usage, "output_token_details")
                audio_tokens_in = _event_get(input_details, "audio_tokens") or 0 if input_details else 0
                audio_tokens_out = _event_get(output_details, "audio_tokens") or 0 if output_details else 0
                if audio_tokens_in:
                    attrs["gen_ai.usage.audio_input_tokens"] = audio_tokens_in
                if audio_tokens_out:
                    attrs["gen_ai.usage.audio_output_tokens"] = audio_tokens_out
            resp_body = _safe_body(_dump(response)) if response is not None else None
            if resp_body:
                attrs["gen_ai.response.body"] = resp_body

            ts = _now_iso()
            self._ek.enqueue(TraceEnvelope(
                trace_id=self._ctx.trace_id,
                span_id=_new_span_id(),
                parent_span_id=self._ctx.span_id,
                created_at=ts,
                span_type="llm_call",
                status="OK",
                operation="voice.turn",
                service_name=self._ek._opts.service_name,
                model=self._base.get("gen_ai.request.model"),
                provider="openai-realtime",
                prompt=prompt[:_MAX_TEXT] or None,
                completion=completion[:_MAX_TEXT] or None,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                start_time=ts,
                end_time=ts,
                events=events or None,
                attributes=attrs,
                environment=self._ek._opts.environment,
                app_version=self._ek._opts.app_version,
                session_id=self._ctx.session_id,
                device_id=self._ctx.device_id,
                sdk_version=SDK_VERSION,
            ))
        except Exception:
            pass

    def _observe(self, event: Any) -> None:
        try:
            etype = _event_type(event)
            if etype == "conversation.item.input_audio_transcription.completed":
                transcript = _event_get(event, "transcript")
                if isinstance(transcript, str) and transcript:
                    self._input_parts.append(transcript)
            elif etype in self._DELTA_TYPES:
                delta = _event_get(event, "delta")
                if isinstance(delta, str) and delta:
                    self._buf.append(delta)
            elif etype == "response.function_call_arguments.done":
                self._emit_tool_call(
                    _event_get(event, "name"),
                    _event_get(event, "arguments"),
                    _event_get(event, "call_id"),
                )
            elif etype == "response.output_item.done":
                item = _event_get(event, "item")
                if _event_get(item, "type") in ("function_call", "tool_call"):
                    self._emit_tool_call(
                        _event_get(item, "name"),
                        _event_get(item, "arguments"),
                        _event_get(item, "call_id") or _event_get(item, "id"),
                    )
            elif etype in self._DONE_TYPES:
                self._end_turn(_event_get(event, "response"))
        except Exception:
            pass

    def _finish(self) -> None:
        if self._closed:
            return
        self.__dict__["_closed"] = True
        try:
            self._end_session("OK")
        except Exception:
            pass

    def __aiter__(self):
        inner = self._conn.__aiter__()

        async def _gen():
            try:
                async for event in inner:
                    self._observe(event)
                    yield event
            finally:
                pass

        return _gen()

    async def recv(self, *args: Any, **kwargs: Any):
        recv = getattr(self._conn, "recv", None)
        if recv is None:
            raise AttributeError("recv")
        event = await recv(*args, **kwargs)
        self._observe(event)
        return event

    async def close(self, *args: Any, **kwargs: Any):
        self._finish()
        closer = getattr(self._conn, "close", None)
        if closer is None:
            return None
        res = closer(*args, **kwargs)
        if hasattr(res, "__await__"):
            return await res
        return res

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["_conn"], name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self.__dict__["_conn"], name, value)


class _TracedConnect:
    """Async-context-manager wrapper around ``realtime.connect(...)``.

    Opens the ``voice.session`` trace on ``__aenter__`` and returns an
    ``_ObservedConnection``; ends the session span on ``__aexit__``. Falls back to
    the raw connection if anything in our wrapping fails.
    """

    def __init__(self, client: "EvalKitClient", cm: Any, model: str) -> None:
        self._ek = client
        self._cm = cm
        self._model = model
        self._end_session = None
        self._observed: Optional[_ObservedConnection] = None

    async def __aenter__(self):
        conn = await self._cm.__aenter__()
        try:
            _tid, end, ctx = self._ek.start_trace("voice.session", _base_attrs(self._model))
            self._end_session = end
            self._observed = _ObservedConnection(self._ek, conn, _base_attrs(self._model), ctx, end)
            return self._observed
        except Exception:
            return conn  # fail open — host gets the unwrapped connection

    async def __aexit__(self, exc_type, exc, tb):
        try:
            if self._observed is not None:
                self._observed._finish()
            elif self._end_session is not None:
                self._end_session("ERROR" if exc_type else "OK")
        except Exception:
            pass
        return await self._cm.__aexit__(exc_type, exc, tb)


def _resolve_realtime(openai_client: Any) -> Any:
    """Return the object exposing ``connect`` (client.beta.realtime)."""
    beta = getattr(openai_client, "beta", None)
    realtime = getattr(beta, "realtime", None) if beta is not None else None
    if realtime is None:
        realtime = getattr(openai_client, "realtime", None)
    return realtime


def trace_openai_realtime(
    client: "EvalKitClient",
    openai_client: Any,
    *,
    model: Optional[str] = None,
) -> Any:
    """Explicit drop-in for ``client.beta.realtime.connect(model=)``.

    Returns an async context manager — use it exactly like the original::

        async with evalkit.trace_openai_realtime(openai_client, model=...) as connection:
            ...

    On any error, falls back to the real connect so the host is never broken.
    """
    try:
        realtime = _resolve_realtime(openai_client)
        if realtime is None or not hasattr(realtime, "connect"):
            return realtime.connect(model=model)  # let it raise naturally
        kwargs: dict = {}
        if model is not None:
            kwargs["model"] = model
        cm = realtime.connect(**kwargs)
        return _TracedConnect(client, cm, str(model or "gpt-realtime"))
    except Exception:
        realtime = _resolve_realtime(openai_client)
        return realtime.connect(model=model)


def patch_openai_realtime(client: "EvalKitClient") -> None:
    """Best-effort: wrap ``AsyncRealtime.connect`` so plain
    ``client.beta.realtime.connect(...)`` is traced. Idempotent. Never raises."""
    try:
        from openai.resources.beta.realtime.realtime import AsyncRealtime
    except Exception:
        return
    try:
        original = getattr(AsyncRealtime, "connect", None)
        if original is None or getattr(original, "_evalkit", False):
            return

        def traced_connect(self: Any, *args: Any, **kwargs: Any):
            try:
                model = kwargs.get("model")
                if model is None and args:
                    model = args[0]
                cm = original(self, *args, **kwargs)
                return _TracedConnect(client, cm, str(model or "gpt-realtime"))
            except Exception:
                return original(self, *args, **kwargs)

        traced_connect._evalkit = True  # type: ignore[attr-defined]
        AsyncRealtime.connect = traced_connect
    except Exception:
        pass
