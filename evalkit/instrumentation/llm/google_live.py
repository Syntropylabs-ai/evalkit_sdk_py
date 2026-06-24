"""Trace Google Gemini Live (voice/realtime over a WebSocket).

Nothing auto-instruments a WebSocket, so we wrap the Live session:
open a ``voice.session`` trace, accumulate the model's output transcript, emit one
``voice.turn`` llm_call span per completed turn (text + latency), emit a
``tool_call`` span per function call, and close the session span when the socket
closes. Mirrors the TS reference (instrumentation/llm/google-live.ts).

The Python API is async:

    client = genai.Client(...)
    async with client.aio.live.connect(model=..., config=...) as session:
        await session.send(...)
        async for msg in session.receive():
            ...

``connect()`` is an *async context manager*, so we wrap the object it returns and
forward ``receive()`` through an observing async generator. Tracing must NEVER
break the host: every wrapper path falls back to the real session on any error.
"""
from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ...core.client import EvalKitClient

_MAX_TEXT = 8192


def _base_attrs(model: str) -> dict:
    return {
        "gen_ai.system": "google",
        "gen_ai.request.model": model,
    }


def _accumulate_text(msg: Any, buf: list) -> None:
    """Append any text the message carries to ``buf`` (best-effort, never raises)."""
    try:
        sc = getattr(msg, "server_content", None)
        if sc is None:
            return
        mt = getattr(sc, "model_turn", None)
        if mt is not None:
            for p in getattr(mt, "parts", None) or []:
                t = getattr(p, "text", None)
                if isinstance(t, str) and t:
                    buf.append(t)
        ot = getattr(sc, "output_transcription", None)
        otext = getattr(ot, "text", None) if ot is not None else None
        if isinstance(otext, str) and otext:
            buf.append(otext)
    except Exception:
        pass


def _turn_complete(msg: Any) -> bool:
    try:
        sc = getattr(msg, "server_content", None)
        return bool(sc is not None and getattr(sc, "turn_complete", False))
    except Exception:
        return False


def _emit_tool_calls(client: "EvalKitClient", msg: Any, base: dict, parent_ctx: Any) -> None:
    """Emit one tool_call span per function call on ``msg.tool_call``."""
    try:
        tc = getattr(msg, "tool_call", None)
        if tc is None:
            return
        for fc in getattr(tc, "function_calls", None) or []:
            name = getattr(fc, "name", None)
            if not name:
                continue
            try:
                args = getattr(fc, "args", None) or {}
                args_str = json.dumps(dict(args))[:_MAX_TEXT]
            except Exception:
                args_str = str(getattr(fc, "args", ""))[:_MAX_TEXT]
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


class _ObservedSession:
    """Wraps a real Live session: forwards everything, observes ``receive()``.

    Holds the session-span ``end`` callback and ends it when the session closes
    (close() or __aexit__). Unknown attributes proxy to the underlying session so
    ``send()`` and friends keep working unchanged.
    """

    def __init__(self, client: "EvalKitClient", session: Any, base: dict, ctx: Any, end_session) -> None:
        self.__dict__["_ek"] = client
        self.__dict__["_session"] = session
        self.__dict__["_base"] = base
        self.__dict__["_ctx"] = ctx
        self.__dict__["_end_session"] = end_session
        self.__dict__["_closed"] = False
        self.__dict__["_buf"] = []
        self.__dict__["_turn_start"] = time.time() * 1000

    def _end_turn(self) -> None:
        try:
            text = "".join(self._buf).strip()
            ms = int(time.time() * 1000 - self._turn_start)
            self._buf.clear()
            self.__dict__["_turn_start"] = time.time() * 1000
            if not text:
                return  # skip audio-only turns with no transcript
            end, _c = self._ek.start_span("voice.turn", {
                **self._base,
                "evalkit.span_type": "llm_call",
                "evalkit.completion": text[:_MAX_TEXT],
                "llm.turn_ms": ms,
            }, self._ctx)
            end("OK")
        except Exception:
            pass

    def _observe(self, msg: Any) -> None:
        try:
            _accumulate_text(msg, self._buf)
            _emit_tool_calls(self._ek, msg, self._base, self._ctx)
            if _turn_complete(msg):
                self._end_turn()
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
    is traced. Idempotent (sentinel attr). Never raises."""
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
