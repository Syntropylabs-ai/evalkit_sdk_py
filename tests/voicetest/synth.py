"""Synthetic calls with exact ground truth, for testing the post-call analysis.

A copy of the voice-caller calibration generator (``apps/voice-caller/calibration/synth.py``), kept
beside the tests that need it because the calibration scripts stay outside the SDK.

Each call is written as a recording directory the way the caller writes one (CONTRACT section 4), plus
``truth.json``: where every bot segment really starts and ends, and the value each metric must take.
The speech is a voiced harmonic buzz with syllable-like amplitude modulation and moving formants: the
VAD treats it like speech, and its boundaries are known to the sample, which no recording of a real
voice can give.

The expected values come from the generator's own plan (the delay it chose, the stop time it chose),
and are then checked against the CONTRACT section 5 definitions applied to the exact segments, so a
disagreement between what was planned and what the definitions say fails at generation time instead
of hiding inside the calibration's error figures.
"""

from __future__ import annotations

import argparse
import json
import os
import wave
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evalkit.voicetest.analysis.metrics import compute_call, compute_turns, timeline_from_events

RATE = 16000
SPMS = RATE // 1000
MERGE_GAP_MS = 300
MIN_SEGMENT_MS = 100
MAX_WORD_GAP_MS = 150
SILENCE_MIN_MS = 3000
CALLER_ERROR_BUDGET_MS = 50
GENERATOR_VERSION = "synthetic-1"
# A fixed epoch base keeps every generated directory byte-identical for the same seed.
EPOCH_BASE_MS = 1_793_000_000_000.0
MONOTONIC_BASE_MS = 5_000_000.0
FRAME_MS = 20
# The caller's default wait rule (evalkit.voicetest.script): bot speech, then this much silence.
WAIT_SILENCE_MS = 700
WAIT_MAX_MS = 15000

_VOWELS = [(730, 1090), (270, 2290), (300, 870), (530, 1840), (660, 1720), (440, 1020), (490, 1350), (390, 1990)]


# --- the signal -----------------------------------------------------------------------------------------

def _words(rng: Any, dur_ms: int) -> List[Tuple[int, int, List[Tuple[int, int]]]]:
    """Word spans in ms from the utterance start, each with its syllables. Gaps between words are true
    silence of at most MAX_WORD_GAP_MS, well under the 300 ms the segment rule bridges."""
    words: List[Tuple[int, int, List[Tuple[int, int]]]] = []
    t = 0
    while t < dur_ms:
        start = t
        syllables: List[Tuple[int, int]] = []
        for _ in range(int(rng.integers(1, 4))):
            d = int(rng.integers(140, 300))
            if dur_ms - (t + d) < 200:
                d = dur_ms - t
            syllables.append((t, t + d))
            t += d
            if t >= dur_ms:
                break
        words.append((start, t, syllables))
        if t >= dur_ms:
            break
        gap = int(rng.integers(60, MAX_WORD_GAP_MS + 1))
        if dur_ms - (t + gap) < 200:
            s, _, syl = words[-1]
            syl[-1] = (syl[-1][0], dur_ms)
            words[-1] = (s, dur_ms, syl)
            break
        t += gap
    return words


def render_speech(rng: Any, dur_ms: int, f0: float, peak_dbfs: float) -> Tuple[Any, List[Tuple[int, int]]]:
    """A speech-like utterance of exactly ``dur_ms``: its first word starts at 0 and its last word ends
    at ``dur_ms``. Returns the samples and the word spans (ms, relative)."""
    import numpy as np

    n = dur_ms * SPMS
    t = np.arange(n) / RATE
    span = max(float(t[-1]), 1e-3)
    contour = f0 * (
        1.0
        + 0.05 * np.sin(2 * np.pi * 0.6 * t + rng.uniform(0, 6.28))
        + 0.02 * np.sin(2 * np.pi * 2.7 * t + rng.uniform(0, 6.28))
    ) * (1.0 - 0.1 * t / span)
    phase = 2 * np.pi * np.cumsum(contour) / RATE
    words = _words(rng, dur_ms)
    syllables = [syl for _, _, syls in words for syl in syls]
    centers = np.array([(a + b) / 2.0 * SPMS for a, b in syllables])
    harmonics = max(1, int(3800 // (f0 * 1.12)))
    k = np.arange(1, harmonics + 1)[:, None]
    weights = []
    for _ in syllables:
        f1, f2 = _VOWELS[int(rng.integers(0, len(_VOWELS)))]
        freq = k[:, 0] * f0
        w = (1.0 / k[:, 0]) * (
            0.15
            + np.exp(-(((freq - f1) / 150.0) ** 2))
            + 0.7 * np.exp(-(((freq - f2) / 200.0) ** 2))
            + 0.3 * np.exp(-(((freq - 2600.0) / 300.0) ** 2))
        )
        weights.append(w)
    weight_matrix = np.array(weights).T  # harmonics x syllables
    samples = np.arange(n)
    x = np.zeros(n)
    offsets = rng.uniform(0, 6.28, size=harmonics)
    for h in range(harmonics):
        amp = np.interp(samples, centers, weight_matrix[h]) if len(centers) > 1 else weight_matrix[h, 0]
        x += amp * np.sin((h + 1) * phase + offsets[h])

    env = np.zeros(n)
    attack, release = 25 * SPMS, 30 * SPMS
    for w_start, w_end, syls in words:
        a, b = w_start * SPMS, w_end * SPMS
        seg = np.ones(b - a)
        for s_start, s_end in syls:
            sa, sb = s_start * SPMS - a, s_end * SPMS - a
            u = np.linspace(0.0, 1.0, sb - sa, endpoint=False)
            seg[sa:sb] = 0.35 + 0.65 * np.sin(np.pi * u) ** 0.7
        length = b - a
        ramp_in = min(attack, length // 2)
        ramp_out = min(release, length // 2)
        seg[:ramp_in] *= np.sin(0.5 * np.pi * np.arange(ramp_in) / ramp_in) ** 2
        seg[length - ramp_out:] *= np.cos(0.5 * np.pi * np.arange(1, ramp_out + 1) / ramp_out) ** 2
        env[a:b] = seg
    x = x * env + rng.normal(0.0, 1.0, n) * env * 0.04
    peak = float(np.max(np.abs(x))) or 1.0
    x *= (10.0 ** (peak_dbfs / 20.0)) / peak
    return x, [(s, e) for s, e, _ in words]


def background_noise(rng: Any, n: int, rms_dbfs: float) -> Any:
    """Pink-ish noise at a set RMS level: the hiss and hum of a bot's line or room."""
    import numpy as np

    white = rng.normal(0.0, 1.0, n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / RATE)
    spectrum *= 1.0 / np.sqrt(np.maximum(freqs, 50.0))
    noise = np.fft.irfft(spectrum, n)
    rms = float(np.sqrt(np.mean(noise ** 2))) or 1.0
    return noise * (10.0 ** (rms_dbfs / 20.0)) / rms


def _lowpass(cutoff_hz: float, taps: int = 101) -> Any:
    import numpy as np

    m = np.arange(taps) - (taps - 1) / 2.0
    h = np.sinc(2.0 * cutoff_hz / RATE * m) * np.hamming(taps)
    return h / h.sum()


def _to_int16(x: Any) -> Any:
    import numpy as np

    return np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)


def mulaw_encode(pcm: Any) -> Any:
    """G.711 mu-law, bit-exact with the reference encoder."""
    import numpy as np

    pcm = pcm.astype(np.int32)
    sign = np.where(pcm < 0, 0x80, 0)
    magnitude = np.minimum(np.abs(pcm), 32635) + 0x84
    exponent = np.floor(np.log2(np.maximum(magnitude >> 7, 1))).astype(np.int32)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa)) & 0xFF


def mulaw_decode(codes: Any) -> Any:
    import numpy as np

    codes = (~codes.astype(np.int32)) & 0xFF
    sign = codes & 0x80
    exponent = (codes >> 4) & 0x07
    mantissa = codes & 0x0F
    magnitude = (((mantissa << 3) + 0x84) << exponent) - 0x84
    return np.where(sign != 0, -magnitude, magnitude).astype(np.int16)


def through_8k(x: Any, mulaw: bool) -> Any:
    """What an 8 kHz leg does to 16 kHz audio: band-limit, decimate, optionally mu-law, upsample.

    Odd-length linear-phase filters applied centred add no delay, so boundaries stay where they were.
    """
    import numpy as np

    y8 = np.convolve(x, _lowpass(3600.0), mode="same")[::2]
    if mulaw:
        y8 = mulaw_decode(mulaw_encode(_to_int16(y8))).astype(np.float64) / 32768.0
    up = np.zeros(len(y8) * 2)
    up[::2] = y8
    y = np.convolve(up, 2.0 * _lowpass(3800.0), mode="same")
    if len(y) < len(x):
        y = np.concatenate([y, np.zeros(len(x) - len(y))])
    return y[: len(x)]


# --- the plan ---------------------------------------------------------------------------------------------

@dataclass
class TurnPlan:
    line_ms: int
    # The caller's wait after the bot's previous speech ends (end-of-turn detection), before this line.
    gap_ms: int = 800
    # None: the bot does not answer; the caller waits ``no_reply_wait_ms`` and moves on.
    reply_delay_ms: Optional[int] = 600
    reply_parts: List[int] = field(default_factory=lambda: [1500])
    # Pauses between reply parts; over 3,000 ms is dead air the analysis must report.
    reply_pauses: List[int] = field(default_factory=list)
    # Bot talk-over: a short backchannel (offset into the line, duration) ...
    backchannel: Optional[Tuple[int, int]] = None
    # ... or the reply starting this long before the caller's line ends.
    jump_in_ms: Optional[int] = None
    # A planned interruption of the previous reply, after this much of its speech.
    interrupt_after_bot_ms: Optional[int] = None
    # How long the bot keeps talking after the interrupting line starts, and whether it stops with a
    # ``clear`` (dropping queued audio) or by ceasing to send.
    stop_after_ms: int = 300
    stop_via_clear: bool = False
    # The caller starts this line while the bot is still talking, unplanned (the bot does not stop).
    barge_in_after_bot_ms: Optional[int] = None
    no_reply_wait_ms: int = 4000
    # Late caller frames: (ms after the line starts, late_ms).
    late_frames: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class CallPlan:
    name: str
    seed: int
    format: str = "pcm16"
    sample_rate: int = 16000
    turns: List[TurnPlan] = field(default_factory=list)
    # Bot speaks first: (start ms after connect, duration ms).
    greeting: Optional[Tuple[int, int]] = None
    lead_in_ms: int = 600
    noise_dbfs: Optional[float] = None
    # The bot sends each utterance faster than real time (the playback buffer turns it into the heard
    # timeline).
    bursty: bool = False
    burst_speed: float = 8.0
    bot_f0: float = 120.0
    caller_f0: float = 205.0
    cases: List[str] = field(default_factory=list)


@dataclass
class _Utterance:
    start: int
    dur: int
    kind: str
    turn: Optional[int]
    seed: int
    peak_dbfs: float
    cut: Optional[int] = None
    via_clear: bool = False
    words: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def heard_end(self) -> int:
        return self.start + self.dur if self.cut is None else min(self.cut, self.start + self.dur)


def _frame_ceil(ms: int) -> int:
    """The caller starts lines and logs waits on 20 ms frame boundaries."""
    return -(-int(ms) // FRAME_MS) * FRAME_MS


def _merge(spans: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for s, e in sorted(spans):
        if out and s - out[-1][1] < MERGE_GAP_MS:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return [(s, e) for s, e in out if e - s >= MIN_SEGMENT_MS]


class _Builder:
    def __init__(self, plan: CallPlan):
        import numpy as np

        self.plan = plan
        self.rng = np.random.default_rng(plan.seed)
        self.events: List[Dict[str, Any]] = []
        self.utterances: List[_Utterance] = []
        self.lines: List[Dict[str, Any]] = []
        self.expected: List[Dict[str, Any]] = []

    def event(self, t: int, kind: str, **fields: Any) -> None:
        jitter = float(self.rng.uniform(0.0, 2.0))
        record = {"t": int(t), "wall": round(MONOTONIC_BASE_MS + t + jitter, 3), "type": kind}
        record.update(fields)
        self.events.append(record)

    def utterance(self, start: int, dur: int, kind: str, turn: Optional[int]) -> _Utterance:
        u = _Utterance(
            start=int(start),
            dur=int(dur),
            kind=kind,
            turn=turn,
            seed=int(self.rng.integers(0, 2**31 - 1)),
            peak_dbfs=float(self.rng.uniform(-12.0, -6.0)),
        )
        self.utterances.append(u)
        return u

    def _begin_wait(self, index: int, at: int) -> None:
        """A wait before line ``index``, logged as the caller logs it: with the line that follows."""
        turns = self.plan.turns
        fields: Dict[str, Any] = {"index": index}
        planned = index < len(turns) and index > 0 and turns[index].interrupt_after_bot_ms is not None
        if planned:
            fields.update(reason="interrupt", after_bot_ms=turns[index].interrupt_after_bot_ms, max_ms=WAIT_MAX_MS)
        elif index < len(turns) and turns[index].barge_in_after_bot_ms is not None:
            fields.update(reason="fixed", ms=0)
        else:
            fields.update(reason="bot_done", silence_ms=WAIT_SILENCE_MS, max_ms=WAIT_MAX_MS)
        if index >= len(turns):
            fields["end"] = True
        self.event(at, "wait_start", **fields)
        if planned:
            self.event(at, "interrupt_planned", index=index, after_bot_ms=turns[index].interrupt_after_bot_ms)

    def build(self) -> None:
        plan = self.plan
        self.event(0, "connect", format=plan.format, sample_rate=plan.sample_rate)
        self.event(0, "start_sent")
        self._begin_wait(0, 0)
        if plan.greeting is not None:
            g_start, g_dur = plan.greeting
            bot_free = self.utterance(g_start, g_dur, "greeting", None).heard_end
        else:
            bot_free = plan.lead_in_ms - plan.turns[0].gap_ms
        previous_reply: List[_Utterance] = []
        previous_answered = True
        for i, turn in enumerate(plan.turns):
            planned = turn.interrupt_after_bot_ms is not None and i > 0
            barge = turn.barge_in_after_bot_ms is not None and i > 0
            line_start: Optional[int] = None
            reason = "bot_done" if previous_answered else "max_wait"
            if (planned or barge) and previous_reply:
                after = turn.interrupt_after_bot_ms if planned else turn.barge_in_after_bot_ms
                candidate = _frame_ceil(previous_reply[0].start + int(after))
                if any(u.start < candidate < u.heard_end for u in previous_reply):
                    line_start = candidate
                    reason = "interrupt" if planned else "fixed"
                elif planned:
                    reason = "max_wait"
            if line_start is None:
                line_start = _frame_ceil(bot_free + turn.gap_ms)
            self.event(line_start, "wait_end", index=i, reason=reason)
            line_end = line_start + turn.line_ms
            if planned and reason == "interrupt":
                cut = line_start + turn.stop_after_ms
                for u in previous_reply:
                    if u.start < line_start < u.heard_end:
                        u.cut = cut
                        u.via_clear = turn.stop_via_clear
                    elif u.start >= line_start:
                        u.cut = u.start  # never sent: the bot gave up the rest of its reply
            self.event(line_start, "think", index=i, ms=0)
            text = f"synthetic caller line {i}"
            self.event(line_start, "line_start", index=i, text=text)
            self.event(line_end, "line_end", index=i, text=text)
            self.lines.append({"index": i, "start_ms": line_start, "end_ms": line_end, "text": text})
            for offset, late in turn.late_frames:
                frame_t = _frame_ceil(line_start + offset)
                self.event(frame_t, "pacing", late_ms=round(float(late), 1), frame=frame_t // FRAME_MS)
            self._begin_wait(i + 1, line_end)
            # The bot never plays two utterances at once: a reply waits for the bot's earlier speech
            # (still going after an unplanned barge-in) to end.
            earlier_end = max((u.heard_end for u in self.utterances), default=0)

            if turn.backchannel is not None:
                off, dur = turn.backchannel
                self.utterance(line_start + off, dur, "backchannel", i)
            reply: List[_Utterance] = []
            if turn.reply_delay_ms is not None or turn.jump_in_ms is not None:
                start = line_end - turn.jump_in_ms if turn.jump_in_ms is not None else line_end + turn.reply_delay_ms
                if earlier_end > start - 600:
                    start = earlier_end + 600
                for p, dur in enumerate(turn.reply_parts):
                    reply.append(self.utterance(start, dur, "reply", i))
                    start += dur + (turn.reply_pauses[p] if p < len(turn.reply_pauses) else 0)
                bot_free = max(u.heard_end for u in reply)
                previous_answered = True
            else:
                bot_free = line_end + turn.no_reply_wait_ms - plan.turns[min(i + 1, len(plan.turns) - 1)].gap_ms
                previous_answered = False
            # A barged-in reply keeps going; the next line starts after the bot's last word.
            bot_free = max(bot_free, max((u.heard_end for u in self.utterances), default=0))
            previous_reply = reply

        end = _frame_ceil(bot_free + int(self.rng.integers(WAIT_SILENCE_MS, WAIT_SILENCE_MS + 400)))
        self.event(end, "wait_end", index=len(plan.turns), reason="bot_done")
        self.event(end, "stop_sent")
        self.event(end, "closed", by="caller")
        self.duration_ms = end

    def playback_events(self) -> None:
        """``bot_chunk`` and ``clear`` events consistent with the heard timeline."""
        plan = self.plan
        chunk_ms = 20 if plan.format == "twilio" else 100
        bytes_per_ms = 8 if plan.format == "twilio" else plan.sample_rate * 2 // 1000
        for u in self.utterances:
            if u.heard_end <= u.start:
                continue
            sent_until = u.start + u.dur
            if u.cut is not None and not plan.bursty:
                sent_until = min(sent_until, u.cut)
            k = 0
            while True:
                offset = k * chunk_ms
                if u.start + offset >= sent_until:
                    break
                size = min(chunk_ms, sent_until - (u.start + offset))
                if plan.bursty:
                    arrival = u.start + int(offset / plan.burst_speed)
                else:
                    arrival = u.start + offset
                if u.cut is not None and arrival >= u.cut:
                    break
                queued = offset + size - (arrival - u.start)
                self.event(
                    arrival,
                    "bot_chunk",
                    bytes=size * bytes_per_ms,
                    arrival_wall=None,
                    queued_ms=int(queued),
                    heard_ms=u.start + offset,
                    samples=size * plan.sample_rate // 1000,
                )
                k += 1
            if u.cut is not None and u.via_clear and u.cut < u.start + u.dur:
                sent = min(u.start + u.dur, u.start + int((u.cut - u.start) * plan.burst_speed)) if plan.bursty else u.cut
                self.event(u.cut, "clear", dropped_ms=int(max(0, sent - u.cut)))
        for e in self.events:
            if e["type"] == "bot_chunk":
                e["arrival_wall"] = e["wall"]
        self.events.sort(key=lambda e: e["t"])

    def render(self) -> Tuple[Any, Any, List[Tuple[int, int]]]:
        import numpy as np

        plan = self.plan
        n = self.duration_ms * SPMS
        caller = np.zeros(n)
        bot = np.zeros(n)
        heard_words: List[Tuple[int, int]] = []
        for line in self.lines:
            rng = np.random.default_rng(plan.seed * 1009 + line["index"])
            x, _ = render_speech(rng, line["end_ms"] - line["start_ms"], plan.caller_f0, -8.0)
            a = line["start_ms"] * SPMS
            caller[a : a + len(x)] += x
        for u in self.utterances:
            end = u.heard_end
            if end <= u.start:
                continue
            x, words = render_speech(np.random.default_rng(u.seed), u.dur, plan.bot_f0, u.peak_dbfs)
            keep = (end - u.start) * SPMS
            a = u.start * SPMS
            bot[a : a + keep] += x[:keep]
            for s, e in words:
                if u.start + s < end:
                    heard_words.append((u.start + s, min(u.start + e, end)))
        if plan.noise_dbfs is not None:
            bot += background_noise(self.rng, n, plan.noise_dbfs)
        if plan.format == "twilio" or plan.sample_rate == 8000:
            mulaw = plan.format == "twilio"
            caller = through_8k(caller, mulaw)
            bot = through_8k(bot, mulaw)
        return caller, bot, heard_words


# --- expected values --------------------------------------------------------------------------------------

def _expected(builder: _Builder, truth_segments: List[Tuple[int, int]]) -> Dict[str, Any]:
    """Every metric's value from the plan, computed without the analysis code."""
    plan = builder.plan
    lines = builder.lines
    late = [(e["t"], e["late_ms"]) for e in builder.events if e["type"] == "pacing"]
    turns = []
    for n, line in enumerate(lines):
        tp = plan.turns[n]
        lo, le = line["start_ms"], line["end_ms"]
        hi = lines[n + 1]["start_ms"] if n + 1 < len(lines) else builder.duration_ms
        replies = [u for u in builder.utterances if u.turn == n and u.kind == "reply" and u.heard_end > u.start]
        if tp.jump_in_ms is not None or not replies:
            delay = None
        else:
            delay = replies[0].start - le
        if tp.backchannel is not None:
            talk = (1, tp.backchannel[1])
        elif tp.jump_in_ms is not None:
            talk = (1, tp.jump_in_ms)
        else:
            talk = (0, 0)
        in_progress = [s for s in truth_segments if s[0] < lo < s[1]]
        planned = tp.interrupt_after_bot_ms is not None and n > 0
        stop = (in_progress[-1][1] - lo) if (planned and in_progress) else None
        caller_overlap = 0
        if not planned and in_progress:
            caller_overlap = sum(max(0, min(s[1], le) - lo) for s in in_progress)
        long_pauses = [p for p in tp.reply_pauses if p > SILENCE_MIN_MS] if replies else []
        bot_ms = sum(max(0, min(s[1], hi) - max(s[0], lo)) for s in truth_segments)
        unhealthy = any(ms > CALLER_ERROR_BUDGET_MS for t, ms in late if lo <= t < hi)
        turns.append(
            {
                "index": n,
                "response_delay_ms": delay,
                "talk_over_count": talk[0],
                "talk_over_ms": talk[1],
                "interruption_planned": planned,
                "interruption_stop_ms": stop,
                "caller_overlap_ms": caller_overlap,
                "silence_ms": sum(long_pauses),
                "bot_speech_ms": bot_ms,
                "caller_unhealthy": unhealthy,
            }
        )
    return {"first_word_ms": truth_segments[0][0] if truth_segments else None, "turns": turns}


def _check_against_definitions(builder: _Builder, truth_segments: List[Tuple[int, int]], expected: Dict[str, Any]) -> None:
    timeline = timeline_from_events(builder.events, builder.duration_ms)
    turns = compute_turns(timeline, truth_segments)
    call = compute_call(timeline, truth_segments, turns)
    problems = []
    if call["first_word_ms"] != expected["first_word_ms"]:
        problems.append(f"first_word_ms {call['first_word_ms']} != planned {expected['first_word_ms']}")
    for got, want in zip(turns, expected["turns"]):
        for key, value in want.items():
            if got[key] != value:
                problems.append(f"turn {want['index']} {key}: definitions give {got[key]}, plan says {value}")
    if problems:
        raise AssertionError(f"{builder.plan.name}: the plan and the CONTRACT definitions disagree: " + "; ".join(problems))


# --- writing ----------------------------------------------------------------------------------------------

def _write_wav(path: str, caller: Any, bot: Any) -> None:
    import numpy as np

    frames = np.empty(len(caller) * 2, dtype="<i2")
    frames[0::2] = _to_int16(caller)
    frames[1::2] = _to_int16(bot)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(frames.tobytes())


def write_call(plan: CallPlan, out_dir: str, vad_version: str = "webrtc-v1") -> Dict[str, Any]:
    """Writes one synthetic call's recording directory and ``truth.json``; returns the truth."""
    builder = _Builder(plan)
    builder.build()
    builder.playback_events()
    caller, bot, heard_words = builder.render()
    truth_segments = _merge(heard_words)
    expected = _expected(builder, truth_segments)
    _check_against_definitions(builder, truth_segments, expected)

    os.makedirs(out_dir, exist_ok=True)
    _write_wav(os.path.join(out_dir, "recording.wav"), caller, bot)
    with open(os.path.join(out_dir, "events.jsonl"), "w", encoding="utf-8") as fh:
        for event in builder.events:
            fh.write(json.dumps(event, separators=(",", ":")) + "\n")
    session = f"synthetic:{plan.name}:0:0"
    t0_epoch_ms = EPOCH_BASE_MS + plan.seed * 600_000.0
    late = [e["late_ms"] for e in builder.events if e["type"] == "pacing"]
    call = {
        "contract": "v1",
        "caller_version": GENERATOR_VERSION,
        "vad": {"version": vad_version},
        "target": {
            "url": "ws://synthetic.invalid/bot",
            "format": plan.format,
            "sample_rate": plan.sample_rate,
            "headers": [],
            "session_query_param": "evalkit_session",
        },
        "format": plan.format,
        "sample_rate": plan.sample_rate,
        "session": session,
        "script": {"name": plan.name, "lines": [{"text": line["text"], "synth": {}} for line in builder.lines]},
        "t0_epoch_ms": t0_epoch_ms,
        "t0_iso": datetime.fromtimestamp(t0_epoch_ms / 1000.0, timezone.utc).isoformat(),
        "t0_monotonic_ms": MONOTONIC_BASE_MS,
        "duration_ms": builder.duration_ms,
        "frames_sent": builder.duration_ms // FRAME_MS,
        "ended_by": "script",
        "pacing": {"frames": builder.duration_ms // FRAME_MS, "late_over_5ms": len(late), "max_late_ms": max(late, default=0.0)},
        "recording": {"file": "recording.wav", "sample_rate": RATE, "channels": ["caller_sent", "bot_heard"]},
        "errors": [],
    }
    with open(os.path.join(out_dir, "call.json"), "w", encoding="utf-8") as fh:
        json.dump(call, fh, indent=2)
        fh.write("\n")
    truth = {
        "generator": GENERATOR_VERSION,
        "plan": asdict(plan),
        "cases": list(plan.cases),
        "bot_segments": [list(s) for s in truth_segments],
        "caller_lines": [[line["start_ms"], line["end_ms"]] for line in builder.lines],
        "expected": expected,
    }
    with open(os.path.join(out_dir, "truth.json"), "w", encoding="utf-8") as fh:
        json.dump(truth, fh, indent=2)
        fh.write("\n")
    return truth


# --- a calibration set ------------------------------------------------------------------------------------

FORMATS = [("twilio", 8000), ("pcm16", 16000), ("pcm16", 8000), ("pcm16", 24000)]
CASES = (
    "delays", "bot_first", "talk_over", "interruption", "clear", "bursts", "silence", "pacing", "noise",
    "8k_source", "barge_in", "no_reply", "interruption_bot_silent",
)


def make_plan(k: int, seed: int) -> CallPlan:
    """Call ``k`` of a calibration set: every case appears in many calls, in every format."""
    import numpy as np

    rng = np.random.default_rng(seed * 7919 + k)
    fmt, rate = FORMATS[k % len(FORMATS)]
    cases = ["delays"]
    plan = CallPlan(name=f"call{k:03d}", seed=seed * 1000 + k, format=fmt, sample_rate=rate)
    plan.bot_f0 = float(rng.uniform(95.0, 150.0))
    plan.caller_f0 = float(rng.uniform(180.0, 240.0))
    if rate == 8000:
        cases.append("8k_source")
    if k % 3 == 0:
        plan.greeting = (int(rng.integers(200, 1500)), int(rng.integers(1200, 3000)))
        cases.append("bot_first")
    if k % 2 == 1:
        plan.noise_dbfs = float(rng.uniform(-55.0, -45.0))
        cases.append("noise")
    if k % 2 == 0:
        plan.bursty = True
        plan.burst_speed = float(rng.uniform(4.0, 12.0))
        cases.append("bursts")

    n_turns = int(rng.integers(5, 8))
    turns = []
    for i in range(n_turns):
        if k < 4 and i == 0:
            delay = [0, 3000, 0, 3000][k]
        else:
            delay = int(rng.integers(0, 3001))
        parts = [int(rng.integers(800, 2600))]
        if rng.random() < 0.4:
            parts.append(int(rng.integers(600, 1800)))
        pauses = [int(rng.integers(600, 1500)) for _ in parts[1:]]
        turns.append(
            TurnPlan(
                line_ms=int(rng.integers(900, 2800)),
                gap_ms=int(rng.integers(500, 1500)),
                reply_delay_ms=delay,
                reply_parts=parts,
                reply_pauses=pauses,
            )
        )
    # Distinct turns for the special cases; the first turn stays plain so turn 1 can interrupt it.
    special = list(rng.permutation(np.arange(1, n_turns)))

    def take(fits: Any = None) -> Optional[int]:
        for pos in range(len(special) - 1, -1, -1):
            if fits is None or fits(int(special[pos])):
                return int(special.pop(pos))
        return None

    def plain(i: int) -> bool:
        t = turns[i]
        return (
            t.interrupt_after_bot_ms is None and t.barge_in_after_bot_ms is None and t.jump_in_ms is None
            and t.backchannel is None and t.reply_delay_ms is not None
            and not any(p > SILENCE_MIN_MS for p in t.reply_pauses)
        )

    if k % 3 != 2:
        i = take()
        if i is not None:
            prev = turns[i - 1]
            prev.reply_parts = [int(rng.integers(2600, 4200))]
            prev.reply_pauses = []
            prev.jump_in_ms = None
            prev.backchannel = None
            t = turns[i]
            t.interrupt_after_bot_ms = int(rng.integers(300, 1500))
            t.stop_after_ms = int(rng.integers(160, 1100))
            t.line_ms = max(t.line_ms, t.stop_after_ms + 900)
            t.stop_via_clear = plan.bursty or bool(rng.random() < 0.5)
            cases.append("interruption")
            if t.stop_via_clear:
                cases.append("clear")
            # The interrupted reply's own turn must not also carry a talk-over or silence case.
            if (i - 1) in special:
                special.remove(i - 1)
    if k % 6 == 5:
        i = take()
        if i is not None:
            turns[i - 1].reply_parts = [int(rng.integers(800, 1200))]
            turns[i - 1].reply_pauses = []
            turns[i].interrupt_after_bot_ms = int(rng.integers(2000, 2600))
            cases.append("interruption_bot_silent")
            if (i - 1) in special:
                special.remove(i - 1)
    if k % 2 == 0:
        i = take()
        if i is not None:
            t = turns[i]
            if rng.random() < 0.5:
                t.line_ms = max(t.line_ms, 1800)
                dur = int(rng.integers(300, 500))
                t.backchannel = (int(rng.integers(200, t.line_ms - dur - 600)), dur)
            else:
                t.jump_in_ms = int(rng.integers(250, 800))
                t.line_ms = max(t.line_ms, t.jump_in_ms + 400)
                t.reply_parts = [t.reply_parts[0]]
                t.reply_pauses = []
            cases.append("talk_over")
    if k % 3 == 1:
        i = take()
        if i is not None:
            t = turns[i]
            t.reply_parts = [int(rng.integers(800, 1600)), int(rng.integers(800, 1600))]
            t.reply_pauses = [int(rng.integers(3400, 6000))]
            cases.append("silence")
    if k % 4 == 3:
        i = take(lambda j: plain(j - 1) and plain(j))
        if i is not None:
            prev = turns[i - 1]
            prev.reply_parts = [int(rng.integers(2600, 4000))]
            prev.reply_pauses = []
            turns[i].barge_in_after_bot_ms = int(rng.integers(500, 1500))
            cases.append("barge_in")
    if k % 6 == 2:
        i = take(lambda j: j + 1 < n_turns and plain(j) and plain(j + 1))
        if i is not None:
            turns[i].reply_delay_ms = None
            turns[i].jump_in_ms = None
            turns[i].backchannel = None
            turns[i + 1].interrupt_after_bot_ms = None
            turns[i + 1].barge_in_after_bot_ms = None
            cases.append("no_reply")
    if k % 3 == 2:
        i = int(rng.integers(0, n_turns))
        turns[i].late_frames = [(int(rng.integers(100, 800)), int(rng.integers(60, 150)))]
        j = int(rng.integers(0, n_turns))
        turns[j].late_frames.append((int(rng.integers(100, 800)), int(rng.integers(6, 40))))
        cases.append("pacing")
    plan.turns = turns
    plan.cases = cases
    return plan


def generate_set(out_dir: str, calls: int, seed: int, vad_version: str = "webrtc-v1") -> List[str]:
    paths = []
    for k in range(calls):
        plan = make_plan(k, seed)
        path = os.path.join(out_dir, plan.name)
        write_call(plan, path, vad_version)
        paths.append(path)
    return paths


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Write synthetic calls with exact ground truth.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--calls", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    paths = generate_set(args.out, args.calls, args.seed)
    print(f"wrote {len(paths)} calls under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
