"""`twilio`: Twilio Media Streams compatible framing (CONTRACT section 3.1).

We play Twilio's side of the stream. The message builders here are also used by the reference bot,
which plays the bot's side, so both ends share one definition of the wire format.
"""

from __future__ import annotations

import base64
import json
import uuid
from typing import Any, Dict, Optional, Union

from ..audio import mulaw_decode, mulaw_encode
from .base import BotAudio, BotClear, BotEvent, BotMark, BotOther, Transport, TWILIO_RATE


def new_sid(prefix: str) -> str:
    return prefix + uuid.uuid4().hex


# Caller (Twilio) side -----------------------------------------------------------------------


def connected_message() -> Dict[str, Any]:
    return {"event": "connected", "protocol": "Call", "version": "1.0.0"}


def start_message(
    seq: int, stream_sid: str, account_sid: str, call_sid: str, session: str
) -> Dict[str, Any]:
    return {
        "event": "start",
        "sequenceNumber": str(seq),
        "start": {
            "streamSid": stream_sid,
            "accountSid": account_sid,
            "callSid": call_sid,
            "tracks": ["inbound"],
            "customParameters": {"evalkit_session": session},
            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": TWILIO_RATE, "channels": 1},
        },
        "streamSid": stream_sid,
    }


def caller_media_message(seq: int, chunk: int, t_ms: int, payload: bytes, stream_sid: str):
    return {
        "event": "media",
        "sequenceNumber": str(seq),
        "media": {
            "track": "inbound",
            "chunk": str(chunk),
            "timestamp": str(t_ms),
            "payload": base64.b64encode(payload).decode("ascii"),
        },
        "streamSid": stream_sid,
    }


def caller_mark_message(seq: int, name: str, stream_sid: str) -> Dict[str, Any]:
    # Twilio's own echo also carries a sequenceNumber; bots that track it keep working.
    return {"event": "mark", "sequenceNumber": str(seq), "streamSid": stream_sid, "mark": {"name": name}}


def stop_message(seq: int, stream_sid: str, account_sid: str, call_sid: str) -> Dict[str, Any]:
    return {
        "event": "stop",
        "sequenceNumber": str(seq),
        "streamSid": stream_sid,
        "stop": {"accountSid": account_sid, "callSid": call_sid},
    }


# Bot side -----------------------------------------------------------------------------------


def bot_media_message(stream_sid: str, payload: bytes) -> Dict[str, Any]:
    return {
        "event": "media",
        "streamSid": stream_sid,
        "media": {"payload": base64.b64encode(payload).decode("ascii")},
    }


def bot_mark_message(stream_sid: str, name: str) -> Dict[str, Any]:
    return {"event": "mark", "streamSid": stream_sid, "mark": {"name": name}}


def bot_clear_message(stream_sid: str) -> Dict[str, Any]:
    return {"event": "clear", "streamSid": stream_sid}


def parse_bot_message(message: Union[str, bytes]) -> BotEvent:
    """A message from the bot, as the caller sees it."""
    if isinstance(message, bytes):
        return BotOther(detail=f"binary frame of {len(message)} bytes")
    try:
        obj = json.loads(message)
    except ValueError:
        return BotOther(detail="text frame that is not JSON")
    if not isinstance(obj, dict):
        return BotOther(detail="JSON frame that is not an object")
    event = obj.get("event")
    if event == "media":
        try:
            payload = base64.b64decode(obj["media"]["payload"], validate=True)
        except (KeyError, TypeError, ValueError):
            return BotOther(detail="media frame without a valid base64 payload")
        return BotAudio(pcm=mulaw_decode(payload), nbytes=len(payload))
    if event == "clear":
        return BotClear()
    if event == "mark":
        name = (obj.get("mark") or {}).get("name") if isinstance(obj.get("mark"), dict) else None
        if not isinstance(name, str):
            return BotOther(detail="mark frame without mark.name")
        return BotMark(name=name)
    return BotOther(detail=f"event {str(event)[:40]!r}")


class TwilioTransport(Transport):
    format = "twilio"

    def __init__(self, target, session: str) -> None:
        super().__init__(target, session)
        self.stream_sid = new_sid("MZ")
        self.account_sid = new_sid("AC")
        self.call_sid = new_sid("CA")
        self._seq = 0
        self._chunk = 0

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def send_start(self) -> Optional[dict]:
        await self._send(json.dumps(connected_message()))
        start = start_message(
            self._next_seq(), self.stream_sid, self.account_sid, self.call_sid, self.session
        )
        await self._send(json.dumps(start))
        return start

    async def send_frame(self, pcm, t_ms: int) -> None:
        self._chunk += 1
        msg = caller_media_message(
            self._next_seq(), self._chunk, t_ms, mulaw_encode(pcm), self.stream_sid
        )
        await self._send(json.dumps(msg))
        self._frames_sent += 1

    async def send_mark(self, name: str) -> bool:
        await self._send(json.dumps(caller_mark_message(self._next_seq(), name, self.stream_sid)))
        return True

    async def send_stop(self) -> Optional[dict]:
        msg = stop_message(self._next_seq(), self.stream_sid, self.account_sid, self.call_sid)
        await self._send(json.dumps(msg))
        return msg

    def parse(self, message: Union[str, bytes]) -> BotEvent:
        return parse_bot_message(message)
