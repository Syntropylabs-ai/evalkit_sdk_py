"""Reading one call's recording directory (CONTRACT section 4): ``recording.wav``, ``events.jsonl`` and
``call.json``.

Only the standard library is used, so re-analysing a recording needs nothing but the VAD.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import wave
from array import array
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_log = logging.getLogger("evalkit.voicetest")

RECORDING_RATE = 16000
WAV_NAME = "recording.wav"
EVENTS_NAME = "events.jsonl"
CALL_NAME = "call.json"


@dataclass
class RecordingDir:
    path: str
    call: Dict[str, Any]
    events: List[Dict[str, Any]]
    caller_pcm: bytes
    bot_pcm: bytes
    sample_rate: int
    duration_ms: int
    problems: List[str] = field(default_factory=list)

    @property
    def transport(self) -> Optional[str]:
        target = self.call.get("target") if isinstance(self.call.get("target"), dict) else {}
        return self.call.get("format") or self.call.get("transport") or target.get("format")

    @property
    def call_sample_rate(self) -> Optional[int]:
        target = self.call.get("target") if isinstance(self.call.get("target"), dict) else {}
        value = self.call.get("sample_rate") or target.get("sample_rate")
        return int(value) if isinstance(value, (int, float)) else None

    @property
    def session(self) -> Optional[str]:
        for key in ("session", "session_token", "session_id", "token"):
            value = self.call.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    @property
    def mode(self) -> str:
        """``dynamic`` when the call asked for its lines during the call (CONTRACT section 4.1)."""
        script = self.call.get("script") if isinstance(self.call.get("script"), dict) else {}
        return "dynamic" if script.get("mode") == "dynamic" else "scripted"

    @property
    def vad_version(self) -> Optional[str]:
        vad = self.call.get("vad")
        if isinstance(vad, dict) and isinstance(vad.get("version"), str):
            return vad["version"]
        for key in ("vad.version", "vad_version"):
            if isinstance(self.call.get(key), str):
                return self.call[key]
        return None

    @property
    def t0_epoch_ms(self) -> Optional[float]:
        """The call start as Unix epoch ms, from whichever spelling ``call.json`` uses."""
        for key in ("t0_epoch_ms", "t0_unix_ms", "t0_wall_ms", "t0_ms"):
            value = self.call.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        for key in ("t0", "t0_wall", "t0_epoch", "t0_unix"):
            value = self.call.get(key)
            parsed = _epoch_ms(value)
            if parsed is not None:
                return parsed
        return None


def _epoch_ms(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        # A wall time in seconds is below 1e11 until the year 5138; in ms it is above it since 1973.
        return float(value) * 1000.0 if value < 1e11 else float(value)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp() * 1000.0
    return None


def _read_events(path: str, problems: List[str]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        problems.append(f"{EVENTS_NAME} is missing")
        return events
    with open(path, "r", encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                problems.append(f"{EVENTS_NAME}:{number} is not JSON")
                continue
            if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                problems.append(f"{EVENTS_NAME}:{number} has no type")
                continue
            if not isinstance(event.get("t"), (int, float)):
                problems.append(f"{EVENTS_NAME}:{number} ({event['type']}) has no t")
                continue
            events.append(event)
    # Stable: events logged at the same sample time keep the order they were written in.
    events.sort(key=lambda e: e["t"])
    return events


def _split_channels(frames: bytes, channels: int) -> List[bytes]:
    samples = array("h")
    samples.frombytes(frames[: len(frames) - (len(frames) % 2)])
    if sys.byteorder == "big":
        samples.byteswap()
    out = []
    for c in range(channels):
        channel = samples[c::channels]
        if sys.byteorder == "big":
            channel.byteswap()
        out.append(channel.tobytes())
    return out


def load_recording_dir(path: str) -> RecordingDir:
    problems: List[str] = []
    call_path = os.path.join(path, CALL_NAME)
    call: Dict[str, Any] = {}
    if os.path.exists(call_path):
        with open(call_path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        if isinstance(loaded, dict):
            call = loaded
        else:
            problems.append(f"{CALL_NAME} is not an object")
    else:
        problems.append(f"{CALL_NAME} is missing")

    with wave.open(os.path.join(path, WAV_NAME), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError(f"{WAV_NAME} must be 16-bit PCM, got {width * 8}-bit")
    if channels != 2:
        raise ValueError(f"{WAV_NAME} must have 2 channels (caller, bot), got {channels}")
    if rate != RECORDING_RATE:
        raise ValueError(f"{WAV_NAME} must be {RECORDING_RATE} Hz, got {rate}")
    caller_pcm, bot_pcm = _split_channels(frames, channels)
    duration_ms = (len(caller_pcm) // 2) * 1000 // rate

    declared = call.get("duration_ms")
    if isinstance(declared, (int, float)) and abs(declared - duration_ms) > 20:
        problems.append(f"{CALL_NAME} duration_ms {declared} differs from the recording's {duration_ms}")

    events = _read_events(os.path.join(path, EVENTS_NAME), problems)
    for problem in problems:
        _log.warning("%s: %s", path, problem)
    return RecordingDir(
        path=path,
        call=call,
        events=events,
        caller_pcm=caller_pcm,
        bot_pcm=bot_pcm,
        sample_rate=rate,
        duration_ms=duration_ms,
        problems=problems,
    )
