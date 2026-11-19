"""Measured prosody of the bot channel: pitch, speaking rate, pauses, loudness and clipping (CONTRACT
section 5, "Prosody").

These are measurements, not a verdict on how natural the bot sounds. They find what a signal can show
(a voice that never moves its pitch, a rate no person speaks at, pauses of one fixed length, a clipped
output stage), and leave the limits to the `voice_prosody` evaluator. On 8 kHz telephone audio the
pitch, rate and pause figures still hold, since all three live well below 4 kHz, while loudness and
clipping partly measure the line.

Like the VAD, every setting that can move a figure belongs to a named, frozen version recorded in the
scorecard. numpy is imported only when a recording is analysed.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .intervals import Interval, normalize

DEFAULT_PROSODY_VERSION = "prosody-v1"


@dataclass(frozen=True)
class ProsodySpec:
    version: str
    # Every figure is read on one 10 ms grid: frame k is centred on k * hop_ms from the call start.
    hop_ms: int
    # YIN (de Cheveigne and Kawahara, 2002): the integration window, the search range and the
    # threshold on the cumulative mean normalised difference below which a frame is periodic.
    yin_window_ms: int
    f0_min_hz: float
    f0_max_hz: float
    yin_threshold: float
    # Pitch is reported only over this many voiced frames: fewer cannot carry a range.
    min_voiced_frames: int
    # Windows of the two intensity contours: a short one that places pause edges to within a frame,
    # and a long one, spanning a pitch period at the lowest F0, for loudness and syllable nuclei.
    pause_window_ms: int
    loudness_window_ms: int
    # A frame is silent below the turn's 90th-percentile intensity minus this; a run of silent frames
    # this long inside bot speech is a pause. 150 ms is longer than a stop closure, shorter than the
    # shortest pause a listener hears as one.
    pause_floor_db: float
    pause_min_ms: int
    # A syllable nucleus is a voiced intensity peak at most this far below the turn's speech level,
    # separated from the previous one by a dip of at least syllable_dip_db (after de Jong and Wempe,
    # 2009) and by at least syllable_min_gap_ms.
    syllable_floor_db: float
    syllable_dip_db: float
    syllable_min_gap_ms: int
    # A sample within 0.26 dB of full scale counts as clipped: PCM16 clips at 32767, and a mu-law leg
    # tops out at 32124 before resampling ripples it.
    clip_level: int
    # Stock phrases: the shortest word sequence that counts when two bot turns both say it.
    stock_phrase_words: int

    def describe(self) -> Dict[str, object]:
        return asdict(self)


PROSODY_SPECS: Dict[str, ProsodySpec] = {
    "prosody-v1": ProsodySpec(
        version="prosody-v1",
        hop_ms=10,
        yin_window_ms=25,
        f0_min_hz=60.0,
        f0_max_hz=400.0,
        yin_threshold=0.15,
        min_voiced_frames=10,
        pause_window_ms=20,
        loudness_window_ms=50,
        pause_floor_db=30.0,
        pause_min_ms=150,
        syllable_floor_db=15.0,
        syllable_dip_db=2.0,
        syllable_min_gap_ms=80,
        clip_level=31785,
        stock_phrase_words=4,
    ),
}

_FRAME_BLOCK = 2048


def get_prosody_spec(version: str) -> ProsodySpec:
    try:
        return PROSODY_SPECS[version]
    except KeyError:
        known = ", ".join(sorted(PROSODY_SPECS))
        raise ValueError(f"unknown prosody version {version!r} (known: {known})") from None


# --- the trackers ---------------------------------------------------------------------------------------

def _samples(pcm16: bytes) -> Any:
    import numpy as np

    return np.frombuffer(pcm16[: len(pcm16) - (len(pcm16) % 2)], dtype="<i2").astype(np.float64)


def frame_count(n_samples: int, rate: int, spec: ProsodySpec) -> int:
    hop = rate * spec.hop_ms // 1000
    return 0 if n_samples <= 0 else n_samples // hop + 1


def track_f0(x: Any, rate: int, spec: ProsodySpec, frames: Optional[Any] = None) -> Any:
    """F0 in Hz per ``spec.hop_ms`` frame, NaN where the frame is not periodic.

    YIN, vectorised over frames: the squared difference function from one FFT cross-correlation per
    frame, its cumulative mean normalisation, the first dip under the threshold followed down to its
    minimum, and a parabolic fit for the sub-sample period. Only ``frames`` (indices on the grid) are
    tracked; the rest stay NaN.
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    hop = rate * spec.hop_ms // 1000
    width = rate * spec.yin_window_ms // 1000
    tau_min = max(2, int(rate // spec.f0_max_hz))
    tau_max = int(math.ceil(rate / spec.f0_min_hz))
    span = width + tau_max + 1
    total = frame_count(len(x), rate, spec)
    f0 = np.full(total, np.nan)
    if total == 0 or len(x) < span:
        return f0
    wanted = np.arange(total) if frames is None else np.asarray(frames, dtype=np.int64)
    wanted = wanted[(wanted >= 0) & (wanted < total)]
    if not len(wanted):
        return f0
    n_fft = 1 << int(math.ceil(math.log2(span)))
    offsets = np.arange(span)
    taus = np.arange(tau_max + 1, dtype=np.float64)
    for b in range(0, len(wanted), _FRAME_BLOCK):
        ks = wanted[b : b + _FRAME_BLOCK]
        starts = np.clip(ks * hop - span // 2, 0, len(x) - span)
        seg = x[starts[:, None] + offsets[None, :]]
        # A window that varies by less than one sample step (a DC run, digital silence) has no
        # period, but FFT rounding can still dip its normalised difference under the threshold.
        moving = seg[:, :width].var(axis=1) >= 1.0
        head = np.zeros_like(seg)
        head[:, :width] = seg[:, :width]
        cross = np.fft.irfft(np.conj(np.fft.rfft(head, n_fft)) * np.fft.rfft(seg, n_fft), n_fft)[:, : tau_max + 1]
        sq = np.concatenate([np.zeros((len(ks), 1)), np.cumsum(seg * seg, axis=1)], axis=1)
        energy_lag = sq[:, width : width + tau_max + 1] - sq[:, : tau_max + 1]
        diff = np.maximum(sq[:, width : width + 1] + energy_lag - 2.0 * cross, 0.0)
        diff[:, 0] = 0.0
        running = np.cumsum(diff[:, 1:], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cmnd = np.ones_like(diff)
            cmnd[:, 1:] = diff[:, 1:] * taus[1:] / running
        cmnd[~np.isfinite(cmnd)] = 1.0
        window = cmnd[:, tau_min:tau_max]
        below = window < spec.yin_threshold
        voiced = below.any(axis=1) & moving
        tau = np.argmax(below, axis=1) + tau_min
        rows = np.arange(len(ks))
        # Down the dip to its floor: the first crossing is on the dip's near slope.
        for _ in range(tau_max):
            nxt = np.minimum(tau + 1, tau_max - 1)
            step = voiced & (nxt > tau) & (cmnd[rows, nxt] < cmnd[rows, tau])
            if not step.any():
                break
            tau = np.where(step, nxt, tau)
        lo, mid, hi = cmnd[rows, tau - 1], cmnd[rows, tau], cmnd[rows, np.minimum(tau + 1, tau_max)]
        curve = lo - 2.0 * mid + hi
        with np.errstate(divide="ignore", invalid="ignore"):
            shift = np.where(curve > 0, 0.5 * (lo - hi) / curve, 0.0)
        period = tau + np.clip(shift, -1.0, 1.0)
        f0[ks] = np.where(voiced, rate / period, np.nan)
    return _despike(f0)


def _despike(f0: Any) -> Any:
    """A five-frame median over voiced neighbours: one frame that jumped an octave is put back, a real
    pitch movement, which lasts several frames, is kept."""
    import numpy as np

    if len(f0) < 5:
        return f0
    padded = np.concatenate([np.full(2, np.nan), f0, np.full(2, np.nan)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, 5)
    voiced = np.isfinite(f0)
    out = f0.copy()
    if voiced.any():
        # Every window holds its own voiced centre, so no median is of NaNs alone.
        out[voiced] = np.nanmedian(windows[voiced], axis=1)
    return out


def intensity_db(x: Any, rate: int, spec: ProsodySpec, window_ms: int) -> Any:
    """RMS level in dBFS of a ``window_ms`` window centred on each frame of the grid."""
    import numpy as np

    total = frame_count(len(x), rate, spec)
    if total == 0:
        return np.zeros(0)
    hop = rate * spec.hop_ms // 1000
    half = rate * window_ms // 2000
    sq = np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64) ** 2)])
    centres = np.arange(total) * hop
    lo = np.clip(centres - half, 0, len(x))
    hi = np.clip(centres + half, 0, len(x))
    width = np.maximum(hi - lo, 1)
    power = np.maximum(sq[hi] - sq[lo], 0.0) / width
    return 10.0 * np.log10(power / (32768.0 ** 2) + 1e-12)


# --- per-group figures ------------------------------------------------------------------------------------

def _mask(intervals: Sequence[Interval], total: int, hop_ms: int) -> Any:
    import numpy as np

    mask = np.zeros(total, dtype=bool)
    for start, end in intervals:
        a = max(0, -(-int(start) // hop_ms))
        b = min(total, -(-int(end) // hop_ms))
        if b > a:
            mask[a:b] = True
    return mask


def _runs(flags: Any) -> List[Tuple[int, int]]:
    """Half-open index runs where ``flags`` is true."""
    import numpy as np

    if not len(flags):
        return []
    edges = np.diff(np.concatenate([[0], flags.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _cv(values: Sequence[float]) -> Optional[float]:
    if len(values) < 3:
        return None
    mean = sum(values) / len(values)
    if mean <= 0:
        return None
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return round(math.sqrt(var) / mean, 3)


def _syllables(level: Any, voiced: Any, speech: Any, spec: ProsodySpec, speech_level: float) -> List[int]:
    """Frames of syllable nuclei: voiced local maxima of the loudness contour, near the speech level and
    each set off from the one before by a dip."""
    import numpy as np

    n = len(level)
    if n < 3:
        return []
    near_voice = voiced.copy()
    near_voice[1:] |= voiced[:-1]
    near_voice[:-1] |= voiced[1:]
    peak = np.zeros(n, dtype=bool)
    peak[1:-1] = (level[1:-1] > level[:-2]) & (level[1:-1] >= level[2:])
    peak &= speech & near_voice & (level >= speech_level - spec.syllable_floor_db)
    kept: List[int] = []
    min_gap = max(1, spec.syllable_min_gap_ms // spec.hop_ms)
    for k in np.flatnonzero(peak).tolist():
        if not kept:
            kept.append(k)
            continue
        last = kept[-1]
        dip = float(level[last : k + 1].min())
        separated = min(level[last], level[k]) - dip >= spec.syllable_dip_db and k - last >= min_gap
        if separated:
            kept.append(k)
        elif level[k] > level[last]:
            # One nucleus seen twice: keep its louder frame.
            kept[-1] = k
    return kept


def _empty_figures() -> Dict[str, Any]:
    return {
        "speech_ms": 0,
        "voiced_ms": 0,
        "f0_median_hz": None,
        "f0_p5_hz": None,
        "f0_p95_hz": None,
        "f0_range_st": None,
        "f0_stdev_st": None,
        "syllables": None,
        "syllables_per_s": None,
        "pause_count": 0,
        "pause_longest_ms": None,
        "pause_mean_ms": None,
        "pause_duration_cv": None,
        "phrase_length_cv": None,
        "pauses": [],
        "loudness_stdev_db": None,
        "loudness_range_db": None,
        "peak_dbfs": None,
        "clipping_ratio": None,
        "clipped_samples": 0,
    }


class _Channel:
    """The bot channel's contours on the frame grid, computed once for every group."""

    def __init__(self, x: Any, rate: int, spec: ProsodySpec, speech: Sequence[Interval]):
        import numpy as np

        self.x = x
        self.rate = rate
        self.spec = spec
        self.total = frame_count(len(x), rate, spec)
        self.in_speech = _mask(speech, self.total, spec.hop_ms)
        self.f0 = track_f0(x, rate, spec, np.flatnonzero(self.in_speech))
        self.fine = intensity_db(x, rate, spec, spec.pause_window_ms)
        self.level = intensity_db(x, rate, spec, spec.loudness_window_ms)

    def measure(self, segments: Sequence[Interval]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """The figures of one stretch of bot speech, and the raw values the call-level figures pool."""
        import numpy as np

        spec, hop = self.spec, self.spec.hop_ms
        figures = _empty_figures()
        raw: Dict[str, Any] = {"f0": [], "level": [], "syllables": 0, "phonation_ms": 0, "pauses": [],
                               "phrases": [], "clipped": 0, "samples": 0, "peak": None}
        segs = normalize(segments)
        if not segs or self.total == 0:
            return figures, raw
        in_seg = _mask(segs, self.total, hop)
        span = _mask([(segs[0][0], segs[-1][1])], self.total, hop)
        if not span.any():
            return figures, raw
        level_in = self.fine[in_seg]
        floor = (float(np.percentile(level_in, 90)) if len(level_in) else -120.0) - spec.pause_floor_db
        silent = span & (~in_seg | (self.fine < floor))
        min_run = -(-spec.pause_min_ms // hop)
        pauses: List[Tuple[int, int]] = []
        for a, b in _runs(silent):
            # A silent run touching the span's edge is outside the speech, not a pause in it.
            if b - a >= min_run and a > 0 and span[a - 1] and b < self.total and span[b]:
                # The window smears each edge by half its length beyond the frame step.
                widen = max(0, spec.pause_window_ms - hop) // 2
                pauses.append((a * hop - widen, b * hop + widen))
        speech = span & ~silent
        voiced = speech & np.isfinite(self.f0)
        speech_ms = int(speech.sum()) * hop
        figures["speech_ms"] = speech_ms
        figures["voiced_ms"] = int(voiced.sum()) * hop

        f0 = self.f0[voiced]
        raw["f0"] = f0.tolist()
        if len(f0) >= spec.min_voiced_frames:
            figures.update(_pitch_figures(f0))

        loud = self.level[voiced]
        raw["level"] = loud.tolist()
        if len(loud) >= spec.min_voiced_frames:
            figures["loudness_stdev_db"] = round(float(np.std(loud)), 2)
            figures["loudness_range_db"] = round(float(np.percentile(loud, 95) - np.percentile(loud, 5)), 2)

        speech_level = float(np.percentile(self.level[speech], 90)) if speech.any() else -120.0
        nuclei = _syllables(self.level, voiced, speech, spec, speech_level)
        figures["syllables"] = len(nuclei)
        if speech_ms > 0:
            figures["syllables_per_s"] = round(len(nuclei) * 1000.0 / speech_ms, 2)
        raw["syllables"], raw["phonation_ms"] = len(nuclei), speech_ms

        durations = [e - s for s, e in pauses]
        figures["pause_count"] = len(pauses)
        figures["pauses"] = [[int(s), int(e)] for s, e in pauses]
        if durations:
            figures["pause_longest_ms"] = int(max(durations))
            figures["pause_mean_ms"] = int(round(sum(durations) / len(durations)))
        figures["pause_duration_cv"] = _cv(durations)
        phrases = [(b - a) * hop for a, b in _runs(speech)]
        figures["phrase_length_cv"] = _cv(phrases)
        raw["pauses"], raw["phrases"] = durations, phrases

        samples = np.concatenate([self.x[s * self.rate // 1000 : e * self.rate // 1000] for s, e in segs])
        if len(samples):
            magnitude = np.abs(samples)
            clipped = int((magnitude >= spec.clip_level).sum())
            peak = float(magnitude.max())
            figures["clipped_samples"] = clipped
            figures["clipping_ratio"] = round(clipped / len(samples), 6)
            figures["peak_dbfs"] = round(20.0 * math.log10(max(peak, 1.0) / 32768.0), 2)
            raw["clipped"], raw["samples"], raw["peak"] = clipped, len(samples), peak
        return figures, raw


def _pitch_figures(f0: Any) -> Dict[str, Any]:
    import numpy as np

    median = float(np.median(f0))
    p5, p95 = float(np.percentile(f0, 5)), float(np.percentile(f0, 95))
    semitones = 12.0 * np.log2(f0 / median)
    return {
        "f0_median_hz": round(median, 1),
        "f0_p5_hz": round(p5, 1),
        "f0_p95_hz": round(p95, 1),
        "f0_range_st": round(12.0 * math.log2(p95 / p5), 2),
        "f0_stdev_st": round(float(np.std(semitones)), 2),
    }


def _pooled(raws: Sequence[Dict[str, Any]], spec: ProsodySpec) -> Dict[str, Any]:
    """The call's figures over all of its bot speech, from the groups' raw values."""
    import numpy as np

    figures = _empty_figures()
    f0 = np.array([v for r in raws for v in r["f0"]], dtype=np.float64)
    loud = np.array([v for r in raws for v in r["level"]], dtype=np.float64)
    phonation = sum(r["phonation_ms"] for r in raws)
    figures["speech_ms"] = phonation
    figures["voiced_ms"] = int(len(f0)) * spec.hop_ms
    if len(f0) >= spec.min_voiced_frames:
        figures.update(_pitch_figures(f0))
    if len(loud) >= spec.min_voiced_frames:
        figures["loudness_stdev_db"] = round(float(np.std(loud)), 2)
        figures["loudness_range_db"] = round(float(np.percentile(loud, 95) - np.percentile(loud, 5)), 2)
    syllables = sum(r["syllables"] for r in raws)
    figures["syllables"] = syllables
    if phonation > 0:
        figures["syllables_per_s"] = round(syllables * 1000.0 / phonation, 2)
    durations = [d for r in raws for d in r["pauses"]]
    figures["pause_count"] = len(durations)
    figures.pop("pauses")
    if durations:
        figures["pause_longest_ms"] = int(max(durations))
        figures["pause_mean_ms"] = int(round(sum(durations) / len(durations)))
    figures["pause_duration_cv"] = _cv(durations)
    figures["phrase_length_cv"] = _cv([p for r in raws for p in r["phrases"]])
    samples = sum(r["samples"] for r in raws)
    if samples:
        clipped = sum(r["clipped"] for r in raws)
        peak = max(r["peak"] for r in raws if r["peak"] is not None)
        figures["clipped_samples"] = clipped
        figures["clipping_ratio"] = round(clipped / samples, 6)
        figures["peak_dbfs"] = round(20.0 * math.log10(max(peak, 1.0) / 32768.0), 2)
    return figures


# --- stock phrases ----------------------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9']+")


def stock_phrases(texts: Mapping[int, str], words: int = 4) -> Dict[str, Any]:
    """Word sequences of at least ``words`` words that two or more bot turns both say, longest first.

    A bot that answers every turn with "I understand your concern, let me look into that" sounds
    scripted however good its voice is; a person rarely repeats a whole clause.
    """
    grams: Dict[Tuple[str, ...], set] = {}
    tokens = {turn: _WORD.findall(text.lower()) for turn, text in texts.items() if text and text.strip()}
    for turn, toks in tokens.items():
        for i in range(len(toks) - words + 1):
            grams.setdefault(tuple(toks[i : i + words]), set()).add(turn)
    shared = {g: turns for g, turns in grams.items() if len(turns) >= 2}
    # Overlapping shared n-grams said in the same turns are one longer phrase.
    phrases: List[Dict[str, Any]] = []
    seen: set = set()
    for turn, toks in sorted(tokens.items()):
        i = 0
        while i <= len(toks) - words:
            gram = tuple(toks[i : i + words])
            if gram not in shared:
                i += 1
                continue
            turns = shared[gram]
            j = i + 1
            while j <= len(toks) - words and shared.get(tuple(toks[j : j + words])) == turns:
                j += 1
            phrase = " ".join(toks[i : j + words - 1])
            key = (phrase, tuple(sorted(turns)))
            if key not in seen and not any(phrase in p["phrase"] and set(turns) <= set(p["turns"]) for p in phrases):
                seen.add(key)
                phrases.append({"phrase": phrase, "turns": sorted(turns), "count": len(turns)})
            i = j + words - 1
    phrases.sort(key=lambda p: (-p["count"], -len(p["phrase"]), p["phrase"]))
    return {"status": "measured", "count": len(phrases), "phrases": phrases[:20], "turns_with_text": len(tokens)}


# --- the scorecard section ----------------------------------------------------------------------------------

def compute_prosody(
    bot_pcm: bytes,
    rate: int,
    turns: Sequence[Mapping[str, Any]],
    preamble: Optional[Mapping[str, Any]] = None,
    version: str = DEFAULT_PROSODY_VERSION,
    bot_transcripts: Optional[Mapping[int, str]] = None,
) -> Dict[str, Any]:
    """The scorecard's ``prosody`` section: figures for the greeting, each turn's bot speech and the call.

    ``turns`` and ``preamble`` are the scorecard's own, whose bot segments say where the bot spoke.
    ``bot_transcripts`` (turn index to text) enables the stock phrase count; without it that count is
    reported as skipped.
    """
    spec = get_prosody_spec(version)
    section: Dict[str, Any] = {"version": spec.version, "spec": spec.describe(), "channel": "bot"}
    try:
        import numpy  # noqa: F401
    except ImportError:
        section["error"] = "numpy is not installed (pip install 'syntropylabs-evalkit[voice]')"
        return section

    def segments_of(entry: Optional[Mapping[str, Any]], key: Optional[str]) -> List[Interval]:
        holder = (entry or {}).get(key) if key else entry
        raw = (holder or {}).get("segments") or []
        return [(int(s), int(e)) for s, e in raw]

    greeting = segments_of(preamble, None)
    per_turn = [(turn, segments_of(turn, "bot")) for turn in turns]
    every = normalize(greeting + [s for _, segs in per_turn for s in segs])
    x = _samples(bot_pcm)
    channel = _Channel(x, rate, spec, every)
    raws: List[Dict[str, Any]] = []

    if greeting:
        figures, raw = channel.measure(greeting)
        raws.append(raw)
        section["preamble"] = {"segments": [list(s) for s in greeting], **figures}
    else:
        section["preamble"] = None
    out_turns: List[Dict[str, Any]] = []
    for turn, segs in per_turn:
        figures, raw = channel.measure(segs)
        if segs:
            raws.append(raw)
        out_turns.append({"index": turn.get("index"), "caller_unhealthy": bool(turn.get("caller_unhealthy")), **figures})
    section["turns"] = out_turns
    section["call"] = _pooled(raws, spec)
    texts = {int(k): v for k, v in (bot_transcripts or {}).items() if isinstance(v, str) and v.strip()}
    section["stock_phrases"] = (
        stock_phrases(texts, spec.stock_phrase_words)
        if texts
        else {"status": "skipped", "reason": "no bot transcript yet: it comes with post-call speech-to-text"}
    )
    return section
