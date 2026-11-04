"""Audio primitives for the voice test caller and the reference bot.

Everything here is deterministic: the same input always gives the same samples, so a call can be
replayed and a calibration run compared sample for sample. numpy and webrtcvad are imported inside
the functions that need them, because this package ships as an optional SDK extra and importing it
must never fail.

Conventions: PCM is a 1-D numpy int16 array, mono. Positions are absolute sample indices at the
stated rate. A frame is 20 ms.
"""

from __future__ import annotations

import functools
import math
import wave
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

FRAME_MS = 20
SUPPORTED_RATES = (8000, 16000, 24000)
RECORDING_RATE = 16000


def frame_samples(rate: int, ms: int = FRAME_MS) -> int:
    """Samples in one frame of `ms` at `rate`; rates and frame lengths must divide evenly."""
    n = rate * ms
    if n % 1000:
        raise ValueError(f"{ms} ms is not a whole number of samples at {rate} Hz")
    return n // 1000


def samples_to_ms(samples: int, rate: int) -> int:
    """Integer ms on the sample clock. Floors, so a position never reads later than it is."""
    return (int(samples) * 1000) // rate


def ms_to_samples(ms: float, rate: int) -> int:
    return int(round(ms * rate / 1000.0))


# --------------------------------------------------------------------------------------------
# G.711 mu-law. Bit-exact with the classic Sun g711.c reference (and CPython's audioop, which is
# gone in 3.13): the 16-bit sample is reduced to 14 bits, biased by 33 and clipped at 8159.
# --------------------------------------------------------------------------------------------

_ULAW_SEG_END = (0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF)


@functools.lru_cache(maxsize=1)
def _ulaw_decode_table():
    import numpy as np

    u = ~np.arange(256, dtype=np.int32) & 0xFF
    t = ((u & 0x0F) << 3) + 0x84
    t = t << ((u & 0x70) >> 4)
    out = np.where(u & 0x80, 0x84 - t, t - 0x84)
    out.setflags(write=False)
    return out.astype(np.int16)


def mulaw_encode(pcm) -> bytes:
    """int16 PCM -> G.711 mu-law bytes, one byte per sample."""
    import numpy as np

    x = np.asarray(pcm, dtype=np.int16).astype(np.int32) >> 2
    neg = x < 0
    mag = np.minimum(np.where(neg, -x, x), 8159) + 33
    seg = np.searchsorted(np.asarray(_ULAW_SEG_END, dtype=np.int32), mag, side="left")
    code = (seg << 4) | ((mag >> (seg + 1)) & 0x0F)
    code = np.where(seg >= 8, 0x7F, code)
    mask = np.where(neg, 0x7F, 0xFF)
    return (code ^ mask).astype(np.uint8).tobytes()


def mulaw_decode(data: bytes):
    """G.711 mu-law bytes -> int16 PCM."""
    import numpy as np

    codes = np.frombuffer(bytes(data), dtype=np.uint8)
    return _ulaw_decode_table()[codes]


def pcm16_to_bytes(pcm) -> bytes:
    import numpy as np

    return np.asarray(pcm, dtype="<i2").tobytes()


def pcm16_from_bytes(data: bytes):
    import numpy as np

    if len(data) % 2:
        raise ValueError("PCM16 payload has an odd number of bytes")
    return np.frombuffer(bytes(data), dtype="<i2").astype(np.int16)


# --------------------------------------------------------------------------------------------
# Resampling: a fixed Kaiser-windowed sinc, applied polyphase. The filter is centred (zero phase),
# so output sample n sits exactly at input time n * src / dst; timings survive the conversion. The
# streaming form gives the same samples as the one-shot form however the input is chunked.
# --------------------------------------------------------------------------------------------

_ZERO_CROSSINGS = 16
_ROLLOFF = 0.92
_KAISER_BETA = 8.6


@functools.lru_cache(maxsize=32)
def _polyphase(src: int, dst: int):
    import numpy as np

    g = math.gcd(src, dst)
    up, down = dst // g, src // g
    half = int(math.ceil(_ZERO_CROSSINGS * max(1.0, down / up)))  # input samples per side
    cutoff = 0.5 * min(src, dst) * _ROLLOFF / (src * up)  # cycles per upsampled sample
    taps = np.arange(-half + 1, half + 1)  # input offsets d: x[q + d]
    phases = np.arange(up)
    k = phases[:, None] - taps[None, :] * up  # filter index relative to the centre
    width = half * up
    window = np.i0(_KAISER_BETA * np.sqrt(np.clip(1.0 - (k / width) ** 2, 0.0, 1.0))) / np.i0(
        _KAISER_BETA
    )
    table = 2.0 * cutoff * np.sinc(2.0 * cutoff * k) * window
    table /= table.sum(axis=1, keepdims=True)  # unity DC gain on every phase
    table.setflags(write=False)
    return up, down, half, taps, table


class Resampler:
    """Streaming rational resampler between any two integer rates (8, 16 and 24 kHz in practice)."""

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        self.src_rate = int(src_rate)
        self.dst_rate = int(dst_rate)
        self._identity = self.src_rate == self.dst_rate
        self._buf = None  # float64 input history
        self._base = 0  # absolute input index of _buf[0]
        self._total_in = 0
        self._next_out = 0
        self._flushed = False
        if not self._identity:
            self._up, self._down, self._half, self._taps, self._table = _polyphase(
                self.src_rate, self.dst_rate
            )
            self._base = -self._half

    @property
    def total_out(self) -> int:
        return self._next_out

    def process(self, pcm):
        """Feed input samples; returns every output sample that is now fully determined."""
        import numpy as np

        if self._flushed:
            raise RuntimeError("resampler already flushed")
        x = np.asarray(pcm, dtype=np.int16)
        if self._identity:
            self._total_in += len(x)
            self._next_out += len(x)
            return x.copy()
        if self._buf is None:
            self._buf = np.zeros(self._half, dtype=np.float64)
        self._buf = np.concatenate([self._buf, x.astype(np.float64)])
        self._total_in += len(x)
        # output n needs input up to q + half, q = n * down // up
        last_needed = self._total_in - 1 - self._half
        if last_needed < 0:
            return np.zeros(0, dtype=np.int16)
        n_end = (last_needed * self._up) // self._down + 1
        n_end = min(n_end, (self._total_in * self._up) // self._down)
        return self._emit(n_end)

    def flush(self):
        """Finish the stream: the input is taken as zero past its end."""
        import numpy as np

        if self._flushed:
            return np.zeros(0, dtype=np.int16)
        self._flushed = True
        if self._identity:
            return np.zeros(0, dtype=np.int16)
        if self._buf is None:
            self._buf = np.zeros(self._half, dtype=np.float64)
        self._buf = np.concatenate([self._buf, np.zeros(self._half + 1, dtype=np.float64)])
        return self._emit((self._total_in * self._up) // self._down)

    def _emit(self, n_end: int):
        import numpy as np

        n0 = self._next_out
        if n_end <= n0:
            return np.zeros(0, dtype=np.int16)
        n = np.arange(n0, n_end, dtype=np.int64)
        t = n * self._down
        q = t // self._up
        p = t % self._up
        idx = (q - self._base)[:, None] + self._taps[None, :]
        y = np.einsum("ij,ij->i", self._buf[idx], self._table[p])
        self._next_out = n_end
        keep_from = (n_end * self._down) // self._up - self._half + 1 - self._base
        if keep_from > 0:
            self._buf = self._buf[keep_from:]
            self._base += keep_from
        return np.clip(np.rint(y), -32768, 32767).astype(np.int16)


def resample(pcm, src_rate: int, dst_rate: int):
    """One-shot resampling; output length is floor(len * dst / src)."""
    import numpy as np

    r = Resampler(src_rate, dst_rate)
    return np.concatenate([r.process(pcm), r.flush()])


# --------------------------------------------------------------------------------------------
# Framing
# --------------------------------------------------------------------------------------------


def split_frames(pcm, rate: int, ms: int = FRAME_MS):
    """Cut PCM into whole frames, zero-padding the last one. Returns shape (n_frames, frame)."""
    import numpy as np

    n = frame_samples(rate, ms)
    x = np.asarray(pcm, dtype=np.int16)
    count = -(-len(x) // n)
    out = np.zeros(count * n, dtype=np.int16)
    out[: len(x)] = x
    return out.reshape(count, n)


class FrameAssembler:
    """Collects chunks of any size and hands back whole frames, keeping the remainder."""

    def __init__(self, rate: int, ms: int = FRAME_MS) -> None:
        self.frame = frame_samples(rate, ms)
        self._pending = None

    def push(self, pcm) -> List:
        import numpy as np

        x = np.asarray(pcm, dtype=np.int16)
        if self._pending is not None and len(self._pending):
            x = np.concatenate([self._pending, x])
        whole = (len(x) // self.frame) * self.frame
        self._pending = x[whole:]
        return [x[i : i + self.frame] for i in range(0, whole, self.frame)]


# --------------------------------------------------------------------------------------------
# WAV input and output (PCM16 only; that is all the contract uses)
# --------------------------------------------------------------------------------------------


DEFAULT_TRIM_DBFS = -54.0


def read_wav(path: str, rate: int, trim: bool = True, trim_dbfs: float = DEFAULT_TRIM_DBFS):
    """Read a PCM16 WAV as mono at `rate`. With `trim`, leading and trailing samples quieter than
    `trim_dbfs` are cut, because a line's boundaries are taken as exact speech boundaries."""
    with wave.open(str(path), "rb") as w:
        return _wav_to_line(w, str(path), rate, trim, trim_dbfs)


def read_wav_bytes(data: bytes, rate: int, trim: bool = True, trim_dbfs: float = DEFAULT_TRIM_DBFS,
                   name: str = "audio"):
    """`read_wav` for a WAV file already in memory."""
    import io

    with wave.open(io.BytesIO(data), "rb") as w:
        return _wav_to_line(w, name, rate, trim, trim_dbfs)


def _wav_to_line(w, name: str, rate: int, trim: bool, trim_dbfs: float):
    import numpy as np

    if w.getsampwidth() != 2:
        raise ValueError(f"{name}: only 16-bit PCM WAV is supported")
    channels = w.getnchannels()
    src_rate = w.getframerate()
    raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype="<i2").astype(np.int32)
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    x = np.clip(np.rint(x), -32768, 32767).astype(np.int16)
    return line_audio(x, src_rate, rate, trim, trim_dbfs)


def line_audio(pcm, src_rate: int, rate: int, trim: bool = True, trim_dbfs: float = DEFAULT_TRIM_DBFS):
    """Mono int16 at `src_rate` as a caller line at `rate`: resampled, then, with `trim`, cut to the
    first and last sample louder than `trim_dbfs`, since those are logged as the exact speech
    boundaries."""
    import numpy as np

    x = np.asarray(pcm, dtype=np.int16)
    if src_rate != rate:
        x = resample(x, src_rate, rate)
    if trim and len(x):
        threshold = 32768.0 * 10 ** (trim_dbfs / 20.0)
        loud = np.flatnonzero(np.abs(x.astype(np.int32)) > threshold)
        x = x[loud[0] : loud[-1] + 1] if len(loud) else x[:0]
    return x


def write_wav(path: str, pcm, rate: int) -> None:
    """Write mono int16, or (n, channels) int16 interleaved."""
    import numpy as np

    x = np.asarray(pcm, dtype=np.int16)
    channels = 1 if x.ndim == 1 else x.shape[1]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(x.astype("<i2").tobytes())


# --------------------------------------------------------------------------------------------
# The synthetic speech-like signal. A voiced harmonic buzz shaped by moving vowel formants, with a
# falling pitch contour and syllable-rate amplitude modulation. WebRTC VAD reads it as speech on
# every frame at 8 and 16 kHz (tests/test_audio.py), and it starts and ends in digital silence, so
# its boundaries are exact.
# --------------------------------------------------------------------------------------------

_VOWELS = (
    (730.0, 1090.0, 2440.0),
    (270.0, 2290.0, 3010.0),
    (300.0, 870.0, 2240.0),
    (530.0, 1840.0, 2480.0),
    (570.0, 840.0, 2410.0),
    (440.0, 1020.0, 2240.0),
    (660.0, 1720.0, 2410.0),
)
_FORMANT_BW = (90.0, 110.0, 170.0)
_FORMANT_GAIN = (1.0, 0.6, 0.35)


@dataclass(frozen=True)
class SynthSpec:
    ms: int
    seed: int = 0
    f0: float = 130.0
    level_dbfs: float = -20.0  # RMS of the voiced part
    syllable_ms: Tuple[int, int] = (150, 260)
    dip: float = 0.18  # envelope floor between syllables, linear

    @classmethod
    def from_json(cls, obj: dict) -> "SynthSpec":
        if not isinstance(obj, dict) or "ms" not in obj:
            raise ValueError("a synth spec needs at least {\"ms\": <duration>}")
        known = {"ms", "seed", "f0", "level_dbfs", "syllable_ms", "dip"}
        extra = set(obj) - known
        if extra:
            raise ValueError(f"unknown synth fields: {sorted(extra)}")
        kwargs = dict(obj)
        if "syllable_ms" in kwargs:
            lo, hi = kwargs["syllable_ms"]
            kwargs["syllable_ms"] = (int(lo), int(hi))
        spec = cls(**kwargs)
        if spec.ms <= 0:
            raise ValueError("synth ms must be positive")
        return spec


def synth_speech(spec: SynthSpec, rate: int):
    """Render the synthetic signal for `spec` at `rate` as int16."""
    import numpy as np

    rng = np.random.default_rng(int(spec.seed))
    n = int(round(spec.ms * rate / 1000.0))
    if n <= 0:
        return np.zeros(0, dtype=np.int16)
    t = np.arange(n) / float(rate)

    bounds = [0]
    lo_ms, hi_ms = spec.syllable_ms
    while bounds[-1] < n:
        bounds.append(bounds[-1] + int(rng.uniform(lo_ms, hi_ms) * rate / 1000.0))
    bounds[-1] = n
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < int(0.06 * rate):
        bounds.pop(-2)  # no runt syllable at the end
    count = len(bounds) - 1

    env = np.zeros(n)
    attack, release = int(0.012 * rate), int(0.030 * rate)
    centres = np.zeros(count)
    for i in range(count):
        a, b = bounds[i], bounds[i + 1]
        length = b - a
        seg = np.ones(length)
        aa, rr = max(1, min(attack, length // 3)), max(1, min(release, length // 3))
        start_floor = 0.0 if i == 0 else spec.dip
        end_floor = 0.0 if i == count - 1 else spec.dip
        seg[:aa] = start_floor + (1 - start_floor) * 0.5 * (1 - np.cos(np.pi * np.arange(aa) / aa))
        seg[length - rr :] = end_floor + (1 - end_floor) * 0.5 * (
            1 + np.cos(np.pi * np.arange(1, rr + 1) / rr)
        )
        env[a:b] = seg * rng.uniform(0.7, 1.0)
        centres[i] = (a + b) / 2.0

    vowels = np.array([_VOWELS[v] for v in rng.integers(0, len(_VOWELS), size=count)])
    idx = np.arange(n)
    if count == 1:
        tracks = [np.full(n, vowels[0, j]) for j in range(3)]
        accent = np.full(n, rng.uniform(0.94, 1.08))
    else:
        tracks = [np.interp(idx, centres, vowels[:, j]) for j in range(3)]
        accent = np.interp(idx, centres, rng.uniform(0.94, 1.08, size=count))
    duration = max(t[-1], 1e-9)
    f0t = (
        spec.f0
        * (1.0 + 0.12 * (0.5 - t / duration))
        * (1.0 + 0.02 * np.sin(2 * np.pi * 5.0 * t + rng.uniform(0, 2 * np.pi)))
        * accent
    )
    phase = 2 * np.pi * np.cumsum(f0t) / rate
    top = min(3800.0, 0.45 * rate)
    out = np.zeros(n)
    for k in range(1, int(top / (spec.f0 * 0.85)) + 1):
        fk = k * f0t
        amp = np.zeros(n)
        for j in range(3):
            amp += _FORMANT_GAIN[j] / (1.0 + ((fk - tracks[j]) / _FORMANT_BW[j]) ** 2)
        amp *= (spec.f0 / np.maximum(fk, 1.0)) ** 0.6 + 0.02
        amp[fk >= top] = 0.0
        out += amp * np.sin(k * phase)
    out *= env

    voiced = env > 0.5
    rms = float(np.sqrt(np.mean(out[voiced] ** 2))) if voiced.any() else 0.0
    if rms > 0:
        out *= (10 ** (spec.level_dbfs / 20.0) * 32767.0) / rms
    peak = float(np.max(np.abs(out)))
    if peak > 32000.0:
        out *= 32000.0 / peak
    return np.rint(out).astype(np.int16)


# --------------------------------------------------------------------------------------------
# Live speech tracking. Used only for live decisions (when the caller speaks next, when the
# reference bot replies); the measurement is the post-call analysis, which has its own VAD rules.
# WebRTC VAD keeps flagging frames as voiced for 60-100 ms after speech stops (its internal
# hangover), so edges are refined to 5 ms blocks by energy relative to the utterance's level.
# --------------------------------------------------------------------------------------------

_VAD_RATES = (8000, 16000, 32000, 48000)
_EDGE_BLOCK_MS = 5
_EDGE_DB_BELOW = 30.0


@dataclass
class SpeechEvent:
    kind: str  # "onset" | "offset"
    sample: int  # position at the tracker's input rate
    detected_at: int  # input position at which the tracker knew


@dataclass
class SpeechTracker:
    """Streaming speech start/stop detection on one channel.

    `onset_frames` consecutive voiced 20 ms frames start speech (the start is the first of them);
    `offset_ms` of consecutive unvoiced frames end it (the end is the end of the last voiced frame).
    With `refine_edges`, both edges move to the 5 ms block where the energy crosses a level 30 dB
    below the utterance's typical level, searched only inside the VAD's own span, so background
    noise can never extend speech.
    """

    rate: int
    aggressiveness: int = 2
    onset_frames: int = 3
    offset_ms: int = 300
    refine_edges: bool = True
    speaking: bool = False
    onset_sample: Optional[int] = None
    last_voiced_end: Optional[int] = None
    position: int = 0
    _vad: object = field(default=None, repr=False)
    _vad_rate: int = field(default=0, repr=False)
    _resampler: Optional[Resampler] = field(default=None, repr=False)
    _frames: Optional[FrameAssembler] = field(default=None, repr=False)
    _frame_index: int = field(default=0, repr=False)
    _run_voiced: int = field(default=0, repr=False)
    _run_unvoiced: int = field(default=0, repr=False)
    _history: list = field(default_factory=list, repr=False)  # (frame index, block energies)
    _level: list = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        import webrtcvad

        self._vad = webrtcvad.Vad(int(self.aggressiveness))
        self._vad_rate = self.rate if self.rate in _VAD_RATES else RECORDING_RATE
        if self._vad_rate != self.rate:
            self._resampler = Resampler(self.rate, self._vad_rate)
        self._frames = FrameAssembler(self._vad_rate)

    def _to_input(self, vad_samples: int) -> int:
        return (vad_samples * self.rate) // self._vad_rate

    @property
    def onset_pending(self) -> bool:
        """Voiced frames have begun but are still too few to start speech: an onset may follow."""
        return not self.speaking and self._run_voiced > 0

    def speech_ms(self, now_sample: int) -> float:
        """How long the current speech has lasted, or 0 when not speaking."""
        if not self.speaking or self.onset_sample is None:
            return 0.0
        return max(0, now_sample - self.onset_sample) * 1000.0 / self.rate

    def silence_ms(self, now_sample: int) -> Optional[float]:
        """ms since the last voiced audio, or None if no speech has been seen."""
        if self.last_voiced_end is None:
            return None
        return max(0, now_sample - self.last_voiced_end) * 1000.0 / self.rate

    def feed(self, pcm) -> List[SpeechEvent]:
        import numpy as np

        x = np.asarray(pcm, dtype=np.int16)
        self.position += len(x)
        if self._resampler is not None:
            x = self._resampler.process(x)
        events: List[SpeechEvent] = []
        for frame in self._frames.push(x):
            events.extend(self._frame(frame))
        return events

    def _frame(self, frame) -> List[SpeechEvent]:
        import numpy as np

        f = self._frame_index
        self._frame_index += 1
        size = len(frame)
        voiced = bool(self._vad.is_speech(frame.astype("<i2").tobytes(), self._vad_rate))
        block = max(1, self._vad_rate * _EDGE_BLOCK_MS // 1000)
        energies = (frame.astype(np.float64).reshape(-1, block) ** 2).mean(axis=1)
        self._history.append((f, energies))
        keep = self.onset_frames + 12
        if len(self._history) > keep:
            del self._history[: len(self._history) - keep]
        detected = self._to_input((f + 1) * size)
        events: List[SpeechEvent] = []
        if voiced:
            self._run_voiced += 1
            self._run_unvoiced = 0
            self._level.append(float(energies.max()))
            if len(self._level) > 50:
                del self._level[0]
            if not self.speaking and self._run_voiced >= self.onset_frames:
                self.speaking = True
                self.onset_sample = self._to_input(self._onset_edge(f - self.onset_frames + 1, size))
                events.append(SpeechEvent("onset", self.onset_sample, detected))
            if self.speaking:
                self.last_voiced_end = self._to_input(self._offset_edge(f, size))
        else:
            self._run_voiced = 0
            self._run_unvoiced += 1
            if self.speaking and self._run_unvoiced * FRAME_MS >= self.offset_ms:
                self.speaking = False
                events.append(SpeechEvent("offset", int(self.last_voiced_end), detected))
        return events

    def _threshold(self) -> float:
        import numpy as np

        if not self._level:
            return 0.0
        return float(np.median(self._level)) * 10 ** (-_EDGE_DB_BELOW / 10.0)

    def _onset_edge(self, first_frame: int, size: int) -> int:
        if self.refine_edges:
            threshold = self._threshold()
            for f, energies in self._history:
                if first_frame <= f < first_frame + self.onset_frames:
                    per = size // len(energies)
                    for b, e in enumerate(energies):
                        if e > threshold:
                            return f * size + b * per
        return first_frame * size

    def _offset_edge(self, last_voiced_frame: int, size: int) -> int:
        if self.refine_edges:
            threshold = self._threshold()
            best = None
            for f, energies in self._history:
                if last_voiced_frame - 10 <= f <= last_voiced_frame:
                    per = size // len(energies)
                    for b, e in enumerate(energies):
                        if e > threshold:
                            best = f * size + (b + 1) * per
            if best is not None:
                return max(best, self._to_input_inverse(self.onset_sample))
        return (last_voiced_frame + 1) * size

    def _to_input_inverse(self, sample: Optional[int]) -> int:
        if sample is None:
            return 0
        return (int(sample) * self._vad_rate) // self.rate


def iter_frames(pcm, rate: int, ms: int = FRAME_MS) -> Iterator:
    """Iterate over whole frames of `pcm` (the last one zero-padded)."""
    for row in split_frames(pcm, rate, ms):
        yield row
