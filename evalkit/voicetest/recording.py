"""One call's output directory (CONTRACT section 4): `recording.wav`, `events.jsonl`, `call.json`.

The WAV is written as the call runs, so a crash still leaves the audio up to that point. Both
channels pass through the same deterministic resampler to 16 kHz, so a sample's position in the
file is its position on the call's sample clock.
"""

from __future__ import annotations

import json
import logging
import os
import time
import wave
from typing import Any, Dict, List, Optional

from .audio import RECORDING_RATE, Resampler, samples_to_ms

log = logging.getLogger("evalkit.voicetest.recording")

RECORDING_NAME = "recording.wav"
EVENTS_NAME = "events.jsonl"
CALL_NAME = "call.json"
CHANNELS = ("caller_sent", "bot_heard")


def monotonic_ms() -> float:
    return round(time.monotonic() * 1000.0, 3)


class EventLog:
    """Appends one JSON object per line. Every event carries `t` (integer ms on the sample clock),
    `wall` (monotonic ms when it was logged) and `type`. Lines are in logging order, which is wall
    order; `t` can step back slightly, e.g. a mark echo is logged after the frame it fell in."""

    def __init__(self, path: str, rate: int, keep: bool = True) -> None:
        self.path = str(path)
        self.rate = int(rate)
        self._file = open(self.path, "w", encoding="utf-8")
        self._keep = keep
        self.events: List[Dict[str, Any]] = []

    def log(self, type_: str, sample: Optional[int] = None, t: Optional[int] = None, **fields):
        if sample is not None:
            t = samples_to_ms(sample, self.rate)
        if t is None:
            raise ValueError("an event needs a sample position or t")
        record: Dict[str, Any] = {"t": int(t), "wall": monotonic_ms(), "type": type_}
        record.update(fields)
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        if self._keep:
            self.events.append(record)
        return record

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


class CallRecorder:
    """recording.wav: 2 channels, 16 kHz, PCM16. Channel 0 is the caller as sent, channel 1 the
    bot as heard. Input arrives at the wire rate; the two channels are written in lockstep."""

    def __init__(self, path: str, wire_rate: int) -> None:
        import numpy as np

        self.path = str(path)
        self.wire_rate = int(wire_rate)
        self._resamplers = [Resampler(wire_rate, RECORDING_RATE), Resampler(wire_rate, RECORDING_RATE)]
        self._pending = [np.zeros(0, dtype=np.int16), np.zeros(0, dtype=np.int16)]
        self._wav = wave.open(self.path, "wb")
        self._wav.setnchannels(2)
        self._wav.setsampwidth(2)
        self._wav.setframerate(RECORDING_RATE)
        self.frames_written = 0
        self._closed = False

    def add_caller(self, pcm) -> None:
        self._add(0, pcm)

    def add_bot(self, pcm) -> None:
        self._add(1, pcm)

    def _add(self, channel: int, pcm) -> None:
        import numpy as np

        out = self._resamplers[channel].process(pcm)
        if len(out):
            self._pending[channel] = np.concatenate([self._pending[channel], out])
        self._write_aligned()

    def _write_aligned(self, limit: Optional[int] = None) -> None:
        import numpy as np

        n = min(len(self._pending[0]), len(self._pending[1]))
        if limit is not None:
            n = min(n, max(0, limit - self.frames_written))
        if n <= 0:
            return
        both = np.empty((n, 2), dtype="<i2")
        both[:, 0] = self._pending[0][:n]
        both[:, 1] = self._pending[1][:n]
        self._wav.writeframes(both.tobytes())
        self._pending = [self._pending[0][n:], self._pending[1][n:]]
        self.frames_written += n

    def finish(self, total_wire_samples: int) -> int:
        """Close the file at exactly the call length; returns its frame count at 16 kHz."""
        import numpy as np

        if self._closed:
            return self.frames_written
        target = (int(total_wire_samples) * RECORDING_RATE) // self.wire_rate
        for ch in (0, 1):
            tail = self._resamplers[ch].flush()
            buf = np.concatenate([self._pending[ch], tail])
            need = max(0, target - self.frames_written)
            if len(buf) < need:
                buf = np.concatenate([buf, np.zeros(need - len(buf), dtype=np.int16)])
            self._pending[ch] = buf[:need]
        self._write_aligned(limit=target)
        self._wav.close()
        self._closed = True
        return self.frames_written

    def abort(self) -> None:
        if not self._closed:
            self._wav.close()
            self._closed = True


def write_call_json(out_dir: str, call: Dict[str, Any]) -> str:
    path = os.path.join(str(out_dir), CALL_NAME)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(call, f, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)
    return path
