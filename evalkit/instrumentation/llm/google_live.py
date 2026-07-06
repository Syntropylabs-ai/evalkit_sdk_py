"""Trace Google Gemini Live (voice/realtime over a WebSocket).

Nothing auto-instruments a WebSocket, so we wrap the Live session: open a
``voice.session`` trace, accumulate both sides of the conversation (user input
transcription + everything sent via ``send_client_content``/``send_realtime_input``,
and the model's output transcription/text/thinking), emit one ``voice.turn``
llm_call span per completed turn with prompt/completion/tokens/request+response
body — the same shape every other auto-instrumented provider produces — and emit
a ``tool_call`` span per function call. Closes the session span when the socket
closes. Mirrors the TS reference (instrumentation/llm/google-live.ts).

The Python API is async:

    client = genai.Client(...)
    async with client.aio.live.connect(model=..., config=...) as session:
        await session.send_client_content(turns=[...], turn_complete=True)
        async for msg in session.receive():
            ...

``connect()`` is an *async context manager*, so we wrap the object it returns and
forward ``receive()``/``send_*`` through observing wrappers. Tracing must NEVER
break the host: every wrapper path falls back to the real session on any error.
"""
from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, List, Optional

from ..._version import SDK_VERSION
from ...core.client import _new_span_id, _now_iso
from ...core.types import SpanEvent, TraceEnvelope
from ._bodies import _dump, _safe_body
from .google_genai import _extract_prompt

if TYPE_CHECKING:
    from ...core.client import EvalKitClient

_MAX_TEXT = 8192


def _base_attrs(model: str) -> dict:
    return {
        "gen_ai.system": "google",
        "gen_ai.request.model": model,
    }


def _turn_complete(msg: Any) -> bool:
    try:
        sc = getattr(msg, "server_content", None)
        return bool(sc is not None and getattr(sc, "turn_complete", False))
    except Exception:
        return False


def _emit_tool_calls(client: "EvalKitClient", msg: Any, base: dict, parent_ctx: Any) -> List[SpanEvent]:
    """Emit one tool_call span per function call on ``msg.tool_call``.

    Returns the same calls as SpanEvents too, so they can also ride along on the
    turn's own llm_call span (matching how every other provider surfaces tool
    calls), not just as separate child spans.
    """
    events: List[SpanEvent] = []
    try:
        tc = getattr(msg, "tool_call", None)
        if tc is None:
            return events
        for fc in getattr(tc, "function_calls", None) or []:
            name = getattr(fc, "name", None)
            if not name:
                continue
            try:
                args = getattr(fc, "args", None) or {}
                args_str = json.dumps(dict(args))[:_MAX_TEXT]
            except Exception:
                args_str = str(getattr(fc, "args", ""))[:_MAX_TEXT]
            events.append(SpanEvent(
                name="gen_ai.tool.call",
                timestamp=_now_iso(),
                attributes={"gen_ai.tool.name": str(name), "gen_ai.tool.call.arguments": args_str},
            ))
            try:
                end, _ctx = client.start_span(str(name), {
                    **base,
                    "evalkit.span_type": "tool_call",
                    "gen_ai.tool.name": str(name),
                    "gen_ai.tool.call.arguments": args_str,
                    "tool.name": str(name),
                }, parent_ctx)
                end("OK")
            except Exception:
                pass
    except Exception:
        pass
    return events


class _ObservedSession:
    """Wraps a real Live session: forwards everything, observes both directions.

    Holds the session-span ``end`` callback and ends it when the session closes
    (close() or __aexit__). Unknown attributes proxy to the underlying session so
    anything we don't explicitly wrap keeps working unchanged.
    """

    def __init__(self, client: "EvalKitClient", session: Any, base: dict, ctx: Any, end_session) -> None:
        self.__dict__["_ek"] = client
        self.__dict__["_session"] = session
        self.__dict__["_base"] = base
        self.__dict__["_ctx"] = ctx
        self.__dict__["_end_session"] = end_session
        self.__dict__["_closed"] = False
        self._reset_turn()

    def _reset_turn(self) -> None:
        self.__dict__["_input_parts"] = []
        self.__dict__["_output_parts"] = []
        self.__dict__["_thinking_parts"] = []
        self.__dict__["_request_dump"] = []
        self.__dict__["_response_dump"] = []
        self.__dict__["_tool_events"] = []
        self.__dict__["_usage"] = None
        self.__dict__["_turn_start"] = time.time() * 1000

    # ── outbound: capture what the app sends this turn ──────────────────────

    def _capture_outbound(self, kind: str, **fields: Any) -> None:
        try:
            entry = {"type": kind}
            for k, v in fields.items():
                if v is not None:
                    entry[k] = _dump(v)
            self._request_dump.append(entry)
        except Exception:
            pass

    async def send_client_content(self, *args: Any, **kwargs: Any):
        try:
            turns = kwargs.get("turns")
            if turns is not None:
                text = _extract_prompt(turns)
                if text:
                    self._input_parts.append(text)
            self._capture_outbound("client_content", turns=turns, turn_complete=kwargs.get("turn_complete"))
        except Exception:
            pass
        return await self._session.send_client_content(*args, **kwargs)

    async def send_realtime_input(self, *args: Any, **kwargs: Any):
        try:
            text = kwargs.get("text")
            if isinstance(text, str) and text:
                self._input_parts.append(text)
                self._capture_outbound("realtime_input", text=text)
            elif kwargs:
                # Audio/video/activity signals — note the modality, never dump binary data.
                modality = next((k for k in ("audio", "video", "activity_start", "activity_end") if k in kwargs), None)
                if modality:
                    self._capture_outbound("realtime_input", modality=modality)
        except Exception:
            pass
        return await self._session.send_realtime_input(*args, **kwargs)

    async def send_tool_response(self, *args: Any, **kwargs: Any):
        try:
            self._capture_outbound("tool_response", function_responses=kwargs.get("function_responses"))
        except Exception:
            pass
        return await self._session.send_tool_response(*args, **kwargs)

    async def send(self, *args: Any, **kwargs: Any):
        try:
            inp = kwargs.get("input")
            if isinstance(inp, str) and inp:
                self._input_parts.append(inp)
            self._capture_outbound("send", input=inp)
        except Exception:
            pass
        return await self._session.send(*args, **kwargs)

    # ── inbound: observe model output + usage + tool calls ──────────────────

    def _observe(self, msg: Any) -> None:
        try:
            sc = getattr(msg, "server_content", None)
            if sc is not None:
                it = getattr(sc, "input_transcription", None)
                itext = getattr(it, "text", None) if it is not None else None
                if isinstance(itext, str) and itext:
                    self._input_parts.append(itext)

                mt = getattr(sc, "model_turn", None)
                if mt is not None:
                    for p in getattr(mt, "parts", None) or []:
                        t = getattr(p, "text", None)
                        if isinstance(t, str) and t:
                            if getattr(p, "thought", False):
                                self._thinking_parts.append(t)
                            else:
                                self._output_parts.append(t)

                ot = getattr(sc, "output_transcription", None)
                otext = getattr(ot, "text", None) if ot is not None else None
                if isinstance(otext, str) and otext:
                    self._output_parts.append(otext)

            um = getattr(msg, "usage_metadata", None)
            if um is not None:
                self.__dict__["_usage"] = um

            self._tool_events.extend(_emit_tool_calls(self._ek, msg, self._base, self._ctx))

            # Compact response dump — never include raw audio/image bytes.
            try:
                entry: dict = {}
                if sc is not None and getattr(sc, "turn_complete", False):
                    entry["turn_complete"] = True
                if sc is not None and getattr(sc, "interrupted", False):
                    entry["interrupted"] = True
                tc = getattr(msg, "tool_call", None)
                if tc is not None:
                    entry["tool_call"] = _dump(tc)
                if entry:
                    self._response_dump.append(entry)
            except Exception:
                pass

            if _turn_complete(msg):
                self._end_turn()
        except Exception:
            pass

    def _end_turn(self) -> None:
        try:
            prompt = "\n".join(self._input_parts).strip()
            completion = "".join(self._output_parts).strip()
            thinking = "".join(self._thinking_parts).strip()
            latency_ms = int(time.time() * 1000 - self._turn_start)
            usage = self._usage
            request_dump, response_dump = self._request_dump, self._response_dump
            events = self._tool_events
            self._reset_turn()

            if not prompt and not completion:
                return  # skip audio-only turns with no transcript at all

            attrs = dict(self._base)
            req_body = _safe_body(request_dump) if request_dump else None
            resp_body = _safe_body(response_dump) if response_dump else None
            if req_body:
                attrs["gen_ai.request.body"] = req_body
            if resp_body:
                attrs["gen_ai.response.body"] = resp_body
            if thinking:
                attrs["gen_ai.response.thinking"] = thinking[:_MAX_TEXT]

            tokens_in = getattr(usage, "prompt_token_count", 0) or 0 if usage else 0
            tokens_out = getattr(usage, "response_token_count", 0) or 0 if usage else 0
            if usage is not None:
                total = getattr(usage, "total_token_count", 0) or 0
                attrs["gen_ai.usage.total_tokens"] = total or (tokens_in + tokens_out)
                thoughts_tokens = getattr(usage, "thoughts_token_count", 0) or 0
                if thoughts_tokens:
                    attrs["gen_ai.usage.reasoning_tokens"] = thoughts_tokens

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
                provider="google-live",
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

    def _finish(self) -> None:
        if self._closed:
            return
        self.__dict__["_closed"] = True
        try:
            self._end_session("OK")
        except Exception:
            pass

    def receive(self, *args: Any, **kwargs: Any):
        inner = self._session.receive(*args, **kwargs)

        async def _gen():
            try:
                async for msg in inner:
                    self._observe(msg)
                    yield msg
            finally:
                pass

        return _gen()

    async def close(self, *args: Any, **kwargs: Any):
        self._finish()
        closer = getattr(self._session, "close", None)
        if closer is None:
            return None
        res = closer(*args, **kwargs)
        if hasattr(res, "__await__"):
            return await res
        return res

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["_session"], name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self.__dict__["_session"], name, value)


class _TracedConnect:
    """Async-context-manager wrapper around ``live.connect(...)``.

    Opens the ``voice.session`` trace on ``__aenter__`` and returns an
    ``_ObservedSession``; ends the session span on ``__aexit__``. Falls back to
    the raw session if anything in our wrapping fails.
    """

    def __init__(self, client: "EvalKitClient", cm: Any, model: str) -> None:
        self._ek = client
        self._cm = cm
        self._model = model
        self._end_session = None
        self._observed: Optional[_ObservedSession] = None

    async def __aenter__(self):
        session = await self._cm.__aenter__()
        try:
            _tid, end, ctx = self._ek.start_trace("voice.session", _base_attrs(self._model))
            self._end_session = end
            self._observed = _ObservedSession(self._ek, session, _base_attrs(self._model), ctx, end)
            return self._observed
        except Exception:
            return session  # fail open — host gets the unwrapped session

    async def __aexit__(self, exc_type, exc, tb):
        try:
            if self._observed is not None:
                self._observed._finish()
            elif self._end_session is not None:
                self._end_session("ERROR" if exc_type else "OK")
        except Exception:
            pass
        return await self._cm.__aexit__(exc_type, exc, tb)


def _resolve_live(genai_client: Any) -> Any:
    """Return the object exposing ``connect`` (client.aio.live)."""
    aio = getattr(genai_client, "aio", None)
    live = getattr(aio, "live", None) if aio is not None else None
    if live is None:
        live = getattr(genai_client, "live", None)
    return live


def trace_google_live(
    client: "EvalKitClient",
    genai_client: Any,
    *,
    model: Optional[str] = None,
    config: Any = None,
) -> Any:
    """Explicit drop-in for ``client.aio.live.connect(model=, config=)``.

    Returns an async context manager — use it exactly like the original::

        async with evalkit.trace_google_live(genai_client, model=..., config=...) as session:
            ...

    The real socket is opened lazily on ``__aenter__``. On any error, falls back to
    the real connect so the host is never broken.
    """
    try:
        live = _resolve_live(genai_client)
        if live is None or not hasattr(live, "connect"):
            return live.connect(model=model, config=config)  # let it raise naturally
        kwargs: dict = {}
        if model is not None:
            kwargs["model"] = model
        if config is not None:
            kwargs["config"] = config
        cm = live.connect(**kwargs)
        return _TracedConnect(client, cm, str(model or "gemini-live"))
    except Exception:
        live = _resolve_live(genai_client)
        return live.connect(model=model, config=config)


def patch_google_genai_live(client: "EvalKitClient") -> None:
    """Best-effort: wrap ``Live.connect`` so plain ``client.aio.live.connect(...)``
    is traced — including calls made deep inside a framework (e.g. Pipecat) that
    creates and owns its own genai client internally. Idempotent (sentinel attr).
    Never raises."""
    try:
        from google.genai import live as _live_mod
    except Exception:
        return
    try:
        Live = getattr(_live_mod, "AsyncLive", None) or getattr(_live_mod, "Live", None)
        if Live is None:
            return
        original = getattr(Live, "connect", None)
        if original is None or getattr(original, "_evalkit", False):
            return

        def traced_connect(self: Any, *args: Any, **kwargs: Any):
            try:
                model = kwargs.get("model")
                if model is None and args:
                    model = args[0]
                cm = original(self, *args, **kwargs)
                return _TracedConnect(client, cm, str(model or "gemini-live"))
            except Exception:
                return original(self, *args, **kwargs)

        traced_connect._evalkit = True  # type: ignore[attr-defined]
        Live.connect = traced_connect
    except Exception:
        pass
