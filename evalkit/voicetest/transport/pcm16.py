"""`pcm16`: raw signed 16-bit little-endian PCM in binary frames (CONTRACT section 3.2)."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Union

from ..audio import pcm16_from_bytes, pcm16_to_bytes
from .base import BotAudio, BotClear, BotEvent, BotOther, Transport


def start_message(session: str, sample_rate: int) -> Dict[str, Any]:
    return {"type": "start", "session": session, "sample_rate": int(sample_rate)}


def clear_message() -> Dict[str, Any]:
    return {"type": "clear"}


def parse_text(message: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(message)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


class Pcm16Transport(Transport):
    format = "pcm16"

    def __init__(self, target, session: str) -> None:
        super().__init__(target, session)
        self._odd = b""  # a bot may split a sample across frames

    async def send_start(self) -> Optional[dict]:
        if not self.target.send_start:
            return None
        msg = start_message(self.session, self.rate)
        await self._send(json.dumps(msg))
        return msg

    async def send_frame(self, pcm, t_ms: int) -> None:
        await self._send(pcm16_to_bytes(pcm))
        self._frames_sent += 1

    def parse(self, message: Union[str, bytes]) -> BotEvent:
        if isinstance(message, (bytes, bytearray, memoryview)):
            data = self._odd + bytes(message)
            whole = len(data) - (len(data) % 2)
            self._odd = data[whole:]
            return BotAudio(pcm=pcm16_from_bytes(data[:whole]), nbytes=len(message))
        obj = parse_text(message)
        if obj is None:
            return BotOther(detail="text frame that is not a JSON object")
        if obj.get("type") == "clear":
            return BotClear()
        if "type" not in obj and "event" in obj:
            # Twilio Media Streams messages name an `event`; saying so makes a format mix-up plain.
            return BotOther(detail=f"event {str(obj.get('event'))[:40]!r}")
        return BotOther(detail=f"type {str(obj.get('type'))[:40]!r}")
