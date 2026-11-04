"""The reference bot's WebSocket server, for either wire format. It plays the bot's side; each
connection gets its own RefBotSession and writes to the shared truth log."""

from __future__ import annotations

import base64
import itertools
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

from ..audio import SynthSpec, mulaw_decode, mulaw_encode, pcm16_from_bytes, pcm16_to_bytes, synth_speech
from ..transport.base import DEFAULT_SESSION_PARAM, FORMATS, MAX_MESSAGE_BYTES, TWILIO_RATE
from ..transport.pcm16 import clear_message as pcm16_clear
from ..transport.twilio import bot_clear_message, bot_mark_message, bot_media_message
from .behaviour import Behaviour, Speech
from .session import Out, RefBotSession
from .truth import CallTruth, TruthLog

log = logging.getLogger("evalkit.voicetest.refbot")


class RefBotServer:
    def __init__(
        self,
        behaviour: Behaviour,
        fmt: str,
        host: str = "127.0.0.1",
        port: int = 19500,
        truth_path: Optional[str] = None,
        sample_rate: Optional[int] = None,
        session_query_param: str = DEFAULT_SESSION_PARAM,
    ) -> None:
        if fmt not in FORMATS:
            raise ValueError(f"format must be one of {FORMATS}")
        self.behaviour = behaviour
        self.fmt = fmt
        self.host = host
        self.port = int(port)
        self.default_rate = TWILIO_RATE if fmt == "twilio" else int(sample_rate or behaviour.sample_rate)
        self.session_query_param = session_query_param
        self.truth = TruthLog(truth_path)
        self._calls = itertools.count()
        self._cache: Dict[Tuple[Speech, int], Any] = {}
        self._server = None
        self.sessions: List[RefBotSession] = []

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}/"

    def render(self, speech: Speech, rate: int):
        key = (speech, rate)
        pcm = self._cache.get(key)
        if pcm is None:
            pcm = synth_speech(SynthSpec.from_json(speech.synth_json()), rate)
            self._cache[key] = pcm
        return pcm

    async def start(self) -> "RefBotServer":
        from websockets.asyncio.server import serve

        # Rendering mid-call would stall the event loop and delay the bot's sends.
        for speech in self.behaviour.all_speech():
            self.render(speech, self.default_rate)
        self._server = await serve(
            self._handle,
            self.host,
            self.port,
            compression=None,
            max_size=MAX_MESSAGE_BYTES,
            ping_interval=None,
        )
        log.info("reference bot (%s) listening on %s", self.fmt, self.url)
        return self

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self.truth.close()

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        await self._server.serve_forever()

    async def __aenter__(self) -> "RefBotServer":
        return await self.start()

    async def __aexit__(self, *exc) -> None:
        await self.close()

    # -- per connection ----------------------------------------------------------------------

    async def _handle(self, ws) -> None:
        from websockets.exceptions import ConnectionClosed

        call = next(self._calls)
        path = getattr(getattr(ws, "request", None), "path", "/") or "/"
        query = parse_qs(urlsplit(path).query)
        query_session = (query.get(self.session_query_param) or [None])[0]
        conn = _Connection(self, ws, call, query_session)
        try:
            async for message in ws:
                if not await conn.on_message(message):
                    break
        except ConnectionClosed:
            pass
        except Exception:
            log.exception("reference bot call %d failed", call)
        finally:
            conn.finish()


class _Connection:
    def __init__(self, server: RefBotServer, ws, call: int, query_session: Optional[str]) -> None:
        self.server = server
        self.ws = ws
        self.call = call
        self.query_session = query_session
        self.session: Optional[RefBotSession] = None
        self.stream_sid = ""
        self.truth = CallTruth(server.truth, call, server.default_rate, query_session)
        self.truth.write("connect", format=server.fmt, query_session=query_session)
        self.reason = "closed"

    def _begin(self, rate: int, inband_session: Optional[str]) -> List[Out]:
        self.truth.rate = rate
        self.truth.session = inband_session or self.query_session
        self.truth.write(
            "start_received",
            sample_rate=rate,
            query_session=self.query_session,
            inband_session=inband_session,
        )
        self.session = RefBotSession(self.server.behaviour, self.server.fmt, rate, self.truth,
                                     self.server.render)
        self.server.sessions.append(self.session)
        return self.session.start()

    async def on_message(self, message) -> bool:
        if self.server.fmt == "twilio":
            return await self._twilio(message)
        return await self._pcm16(message)

    async def _twilio(self, message) -> bool:
        if isinstance(message, bytes):
            self.truth.write("ignored", detail="binary frame")
            return True
        try:
            obj = json.loads(message)
        except ValueError:
            self.truth.write("ignored", detail="text frame that is not JSON")
            return True
        event = obj.get("event") if isinstance(obj, dict) else None
        if event == "connected":
            return True
        if event == "start":
            start = obj.get("start") or {}
            self.stream_sid = str(start.get("streamSid") or obj.get("streamSid") or "")
            params = start.get("customParameters") or {}
            await self._send(self._begin(TWILIO_RATE, params.get("evalkit_session")))
            return True
        if event == "media":
            if self.session is None:
                await self._send(self._begin(TWILIO_RATE, None))
            payload = base64.b64decode((obj.get("media") or {}).get("payload", ""))
            await self._send(self.session.on_audio(mulaw_decode(payload)))
            return True
        if event == "mark":
            if self.session is not None:
                self.session.on_mark(str((obj.get("mark") or {}).get("name")))
            return True
        if event == "stop":
            self.reason = "stop"
            self.truth.write("stop_received")
            return False
        self.truth.write("ignored", detail=f"event {str(event)[:40]!r}")
        return True

    async def _pcm16(self, message) -> bool:
        if isinstance(message, (bytes, bytearray, memoryview)):
            if self.session is None:
                await self._send(self._begin(self.server.default_rate, None))
            await self._send(self.session.on_audio(pcm16_from_bytes(bytes(message))))
            return True
        try:
            obj = json.loads(message)
        except ValueError:
            obj = None
        if isinstance(obj, dict) and obj.get("type") == "start" and self.session is None:
            rate = int(obj.get("sample_rate") or self.server.default_rate)
            await self._send(self._begin(rate, obj.get("session")))
            return True
        self.truth.write("ignored", detail="text frame")
        return True

    async def _send(self, outs: List[Out]) -> None:
        for kind, value in outs:
            if self.server.fmt == "twilio":
                if kind == "audio":
                    data = json.dumps(bot_media_message(self.stream_sid, mulaw_encode(value)))
                elif kind == "mark":
                    data = json.dumps(bot_mark_message(self.stream_sid, value))
                elif kind == "clear":
                    data = json.dumps(bot_clear_message(self.stream_sid))
                else:
                    data = json.dumps(value)
            else:
                if kind == "audio":
                    data = pcm16_to_bytes(value)
                elif kind == "clear":
                    data = json.dumps(pcm16_clear())
                elif kind == "text":
                    data = json.dumps(value)
                else:
                    continue  # pcm16 has no marks
            await self.ws.send(data)

    def finish(self) -> None:
        if self.session is not None:
            self.session.finish(self.reason)
        else:
            self.truth.write("closed", reason=self.reason)
