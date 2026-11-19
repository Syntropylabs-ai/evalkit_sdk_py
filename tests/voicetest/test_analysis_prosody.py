"""The bot channel's measured prosody (CONTRACT section 5, "Prosody"), calibrated on synthetic voices
whose pitch, syllables and pauses are known exactly.

The voice is a harmonic buzz shaped by moving formants and a syllable envelope, the same kind of signal
the timing calibration uses: the VAD hears it as speech, and every figure prosody reports has a true
value to be checked against.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import pytest

from evalkit.voicetest.analysis import compute_prosody, get_prosody_spec, stock_phrases, track_f0

from .support import require

require("numpy")

RATE = 16000
SPEC = get_prosody_spec("prosody-v1")
_VOWELS = [(730, 1090), (270, 2290), (300, 870), (530, 1840), (660, 1720), (440, 1020)]


def _np():
    import numpy as np

    return np


def buzz(contour, seed: int = 0):
    """A harmonic buzz following ``contour`` (Hz per sample), peak 0.3 of full scale."""
    np = _np()
    rng = np.random.default_rng(seed)
    phase = 2 * np.pi * np.cumsum(contour) / RATE
    harmonics = max(1, int(3800 // (float(np.mean(contour)) * 1.3)))
    x = sum((1.0 / h) * np.sin(h * phase + rng.uniform(0, 6.28)) for h in range(1, harmonics + 1))
    return 0.3 * x / np.max(np.abs(x))


class Voice:
    """One bot utterance with its truth: syllable spans, pause spans and the F0 at every sample."""

    def __init__(self, x, syllables: List[Tuple[int, int]], pauses: List[Tuple[int, int]], f0):
        self.x = x
        self.syllables = syllables
        self.pauses = pauses
        self.f0 = f0


def speak(
    seed: int,
    phrases: Sequence[int],
    pauses: Sequence[int],
    pitch: Optional[Callable[[object, object], object]] = None,
    f0: float = 140.0,
    peak: float = 0.4,
) -> Voice:
    """Phrases of ``phrases[i]`` syllables, separated by ``pauses[i]`` ms of silence. Inside a phrase,
    syllables of 160 to 260 ms run into one another in words of one to three, with word gaps of 40 to
    90 ms: shorter than any pause, as in connected speech. ``pitch(t, phrase_t)`` is the contour in
    semitones about ``f0``."""
    np = _np()
    pitch = pitch or natural
    rng = np.random.default_rng(seed)
    spans: List[Tuple[int, int]] = []
    pause_spans: List[Tuple[int, int]] = []
    phrase_of: List[Tuple[int, int]] = []
    t = 0
    for p, count in enumerate(phrases):
        start = t
        left = count
        while left:
            word = min(left, int(rng.integers(1, 4)))
            for _ in range(word):
                d = int(rng.integers(160, 261))
                spans.append((t, t + d))
                t += d
            left -= word
            if left:
                t += int(rng.integers(40, 91))
        phrase_of.append((start, t))
        if p < len(pauses):
            pause_spans.append((t, t + pauses[p]))
            t += pauses[p]
    n = t * RATE // 1000
    times = np.arange(n) / RATE
    phrase_t = np.zeros(n)
    for start, end in phrase_of:
        a, b = start * RATE // 1000, end * RATE // 1000
        phrase_t[a:b] = np.linspace(0.0, 1.0, b - a, endpoint=False)
    contour = f0 * 2.0 ** (pitch(times, phrase_t) / 12.0)
    phase = 2 * np.pi * np.cumsum(contour) / RATE
    harmonics = max(1, int(3800 // (f0 * 1.5)))
    centres = np.array([(a + b) / 2.0 * RATE / 1000 for a, b in spans])
    weights = []
    for _ in spans:
        f1, f2 = _VOWELS[int(rng.integers(0, len(_VOWELS)))]
        freq = np.arange(1, harmonics + 1) * f0
        weights.append((1.0 / np.arange(1, harmonics + 1)) * (
            0.15 + np.exp(-(((freq - f1) / 150.0) ** 2)) + 0.7 * np.exp(-(((freq - f2) / 200.0) ** 2))
        ))
    weights = np.array(weights).T
    samples = np.arange(n)
    x = np.zeros(n)
    for h in range(harmonics):
        x += np.interp(samples, centres, weights[h]) * np.sin((h + 1) * phase + rng.uniform(0, 6.28))
    env = np.zeros(n)
    for a_ms, b_ms in spans:
        a, b = a_ms * RATE // 1000, b_ms * RATE // 1000
        u = np.linspace(0.0, 1.0, b - a, endpoint=False)
        env[a:b] = np.sin(np.pi * u) ** 0.8
    x = x * env
    x *= peak / np.max(np.abs(x))
    voiced_f0 = np.where(env > 0.2, contour, np.nan)
    return Voice(x, spans, pause_spans, voiced_f0)


def flat(times, phrase_t):
    return 0.0 * times


def natural(times, phrase_t):
    """Declination over each phrase, a rise and fall on its accent, and a slow wander."""
    np = _np()
    accent = 4.0 * np.exp(-(((phrase_t - 0.3) / 0.12) ** 2))
    return 2.0 - 4.0 * phrase_t + accent + 1.0 * np.sin(2 * np.pi * 0.9 * times)


def pcm(x, lead_ms: int = 500, tail_ms: int = 500) -> Tuple[bytes, int]:
    np = _np()
    lead = np.zeros(lead_ms * RATE // 1000)
    tail = np.zeros(tail_ms * RATE // 1000)
    full = np.concatenate([lead, x, tail])
    return np.clip(np.round(full * 32768.0), -32768, 32767).astype("<i2").tobytes(), lead_ms


def one_turn(voice: Voice, segments: Optional[List[Tuple[int, int]]] = None):
    """The voice as a call's only turn, its bot segments where the speech really is."""
    data, lead = pcm(voice.x)
    if segments is None:
        segments = []
        start = voice.syllables[0][0]
        for a, b in voice.pauses:
            segments.append((start, a))
            start = b
        segments.append((start, voice.syllables[-1][1]))
    segs = [[a + lead, b + lead] for a, b in segments]
    turns = [{"index": 0, "caller_unhealthy": False, "bot": {"segments": segs}}]
    return compute_prosody(data, RATE, turns, {"segments": []}), lead


# --- the F0 tracker ------------------------------------------------------------------------------------

@pytest.mark.parametrize("hz", [70.0, 100.0, 140.0, 220.0, 300.0, 380.0])
def test_flat_pitch_is_tracked_to_a_fraction_of_a_percent(hz):
    np = _np()
    contour = np.full(RATE * 2, hz)
    f0 = track_f0(buzz(contour) * 32768.0, RATE, SPEC)[5:-5]
    assert np.isfinite(f0).mean() > 0.98
    error = np.abs(f0[np.isfinite(f0)] - hz) / hz
    assert np.percentile(error, 95) < 0.005


def _contour_error(contour, x):
    np = _np()
    f0 = track_f0(x * 32768.0, RATE, SPEC)
    truth = contour[np.minimum(np.arange(len(f0)) * (RATE // 100), len(contour) - 1)]
    keep = np.isfinite(f0)
    keep[:5] = keep[-5:] = False
    error = np.abs(f0[keep] - truth[keep]) / truth[keep]
    return f0[keep], error, truth[keep]


def _range_st(values):
    np = _np()
    return 12.0 * math.log2(np.percentile(values, 95) / np.percentile(values, 5))


def test_a_one_octave_glide_is_followed_with_no_octave_error():
    np = _np()
    contour = 120.0 * 2.0 ** np.linspace(0.0, 1.0, RATE * 3)
    f0, error, truth = _contour_error(contour, buzz(contour))
    assert np.percentile(error, 95) < 0.01
    assert (error > 0.2).mean() == 0.0
    assert abs(_range_st(f0) - _range_st(truth)) < 0.1


@pytest.mark.parametrize("leg", ["mulaw_8k", "noise_10db"])
def test_tracking_survives_a_phone_leg_and_a_noisy_line(leg):
    np = _np()
    from .synth import background_noise, through_8k

    t = np.arange(RATE * 3) / RATE
    contour = 180.0 * 2.0 ** (3.0 * np.sin(2 * np.pi * 0.7 * t) / 12.0)
    x = buzz(contour, seed=4)
    if leg == "mulaw_8k":
        x = through_8k(x, True)
    else:
        level = 20.0 * math.log10(float(np.sqrt(np.mean(x ** 2))))
        x = x + background_noise(np.random.default_rng(5), len(x), level - 10.0)
    f0, error, _ = _contour_error(contour, x)
    assert len(f0) > 0.95 * (len(t) // 160 - 10)
    assert np.median(error) < 0.01
    assert (error > 0.2).mean() < 0.01


def test_unvoiced_noise_and_silence_have_no_pitch():
    np = _np()
    rng = np.random.default_rng(1)
    x = np.concatenate([np.zeros(RATE), rng.normal(0.0, 0.1, RATE)])
    assert np.isfinite(track_f0(x * 32768.0, RATE, SPEC)).mean() < 0.02


def test_a_loud_constant_level_has_no_pitch():
    np = _np()
    assert not np.isfinite(track_f0(np.full(3 * RATE, 5000.0), RATE, SPEC)).any()


# --- monotone or not -------------------------------------------------------------------------------------

def test_a_flat_pitch_buzz_reads_monotone():
    section, _ = one_turn(speak(3, [9, 8], [500], flat))
    turn = section["turns"][0]
    assert turn["f0_range_st"] < 1.0
    assert turn["f0_stdev_st"] < 0.3
    assert abs(turn["f0_median_hz"] - 140.0) < 2.0


def test_a_natural_pitch_contour_does_not():
    voice = speak(3, [9, 8], [500], natural)
    section, _ = one_turn(voice)
    turn = section["turns"][0]
    np = _np()
    truth = voice.f0[np.isfinite(voice.f0)]
    true_range = 12.0 * math.log2(np.percentile(truth, 95) / np.percentile(truth, 5))
    assert turn["f0_range_st"] > 3.0
    assert abs(turn["f0_range_st"] - true_range) < 0.6
    assert turn["f0_stdev_st"] > 1.0


# --- rate, pauses, loudness and clipping ---------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(6))
def test_speaking_rate_counts_the_syllables(seed):
    voice = speak(seed, [7, 9, 6], [450, 700], natural if seed % 2 else flat)
    section, _ = one_turn(voice)
    turn = section["turns"][0]
    truth = len(voice.syllables)
    assert abs(turn["syllables"] - truth) <= max(1, round(0.15 * truth)), (turn["syllables"], truth)
    speech_ms = sum(b - a for a, b in voice.syllables)
    true_rate = truth * 1000.0 / speech_ms
    assert abs(turn["syllables_per_s"] - true_rate) / true_rate < 0.25


@pytest.mark.parametrize("segments", ["one_segment", "vad"])
def test_pauses_inside_bot_speech_are_found_with_their_length(segments):
    """Found inside one VAD segment by the level alone, and between segments the VAD split at."""
    voice = speak(11, [6, 5, 7, 4], [200, 650, 1200])
    if segments == "vad":
        require("webrtcvad")
        from evalkit.voicetest.analysis import detect_segments, get_spec

        data, lead = pcm(voice.x)
        spans = [(a - lead, b - lead) for a, b in detect_segments(data, RATE, get_spec("webrtc-v2"))]
        assert len(spans) == 3
    else:
        spans = [(voice.syllables[0][0], voice.syllables[-1][1])]
    section, lead = one_turn(voice, spans)
    turn = section["turns"][0]
    assert turn["pause_count"] == 3
    for (a, b), (got_a, got_b) in zip(voice.pauses, turn["pauses"]):
        assert abs((got_b - got_a) - (b - a)) <= 40
        assert abs(got_a - (a + lead)) <= 30
    assert abs(turn["pause_longest_ms"] - 1200) <= 40
    assert turn["pause_duration_cv"] is not None and turn["pause_duration_cv"] > 0.5


def test_evenly_spaced_equal_pauses_read_as_regular():
    voice = speak(12, [5, 5, 5, 5], [500, 500, 500])
    turn = one_turn(voice)[0]["turns"][0]
    assert turn["pause_count"] == 3
    assert turn["pause_duration_cv"] < 0.05


def test_word_gaps_are_not_pauses():
    voice = speak(13, [14], [])
    turn = one_turn(voice)[0]["turns"][0]
    assert turn["pause_count"] == 0


def test_clipping_ratio():
    np = _np()
    voice = speak(2, [8], [])
    clean = one_turn(voice)[0]["turns"][0]
    assert clean["clipping_ratio"] == 0.0
    voice.x = np.clip(voice.x * 4.0, -1.0, 32767 / 32768)
    clipped = one_turn(voice)[0]["turns"][0]
    assert clipped["clipping_ratio"] > 0.01
    assert clipped["peak_dbfs"] > -0.1


def test_loudness_variation_follows_the_level():
    np = _np()
    steady = speak(5, [10], [])
    varied = speak(5, [10], [])
    ramp = np.linspace(0.15, 1.0, len(varied.x))
    varied.x = varied.x * ramp
    a = one_turn(steady)[0]["turns"][0]["loudness_stdev_db"]
    b = one_turn(varied)[0]["turns"][0]["loudness_stdev_db"]
    assert b > a + 2.0


# --- the section ----------------------------------------------------------------------------------------------

def test_section_has_greeting_turns_and_call_figures():
    np = _np()
    greet = speak(1, [6], [], natural)
    reply = speak(2, [8, 6], [600], flat)
    gap = np.zeros(RATE)
    x = np.concatenate([greet.x, gap, reply.x])
    data, lead = pcm(x)
    g_end = lead + greet.syllables[-1][1]
    r0 = lead + len(greet.x) * 1000 // RATE + 1000
    turns = [
        {"index": 0, "bot": {"segments": [[r0 + reply.syllables[0][0], r0 + reply.pauses[0][0]],
                                          [r0 + reply.pauses[0][1], r0 + reply.syllables[-1][1]]]}},
        {"index": 1, "bot": {"segments": []}},
    ]
    section = compute_prosody(data, RATE, turns, {"segments": [[lead, g_end]]})
    assert section["version"] == "prosody-v1" and section["channel"] == "bot"
    assert section["spec"]["yin_threshold"] == SPEC.yin_threshold
    assert section["preamble"]["f0_range_st"] > 3.0
    assert section["turns"][0]["f0_range_st"] < 1.0
    assert section["turns"][0]["pause_count"] == 1
    assert section["turns"][1]["speech_ms"] == 0 and section["turns"][1]["f0_median_hz"] is None
    call = section["call"]
    assert call["syllables"] == section["preamble"]["syllables"] + section["turns"][0]["syllables"]
    assert call["pause_count"] == 1
    assert "pauses" not in call
    assert section["stock_phrases"]["status"] == "skipped"


def test_no_bot_speech_gives_empty_figures():
    data, _ = pcm(_np().zeros(RATE))
    section = compute_prosody(data, RATE, [{"index": 0, "bot": {"segments": []}}], None)
    assert section["preamble"] is None
    assert section["call"]["f0_median_hz"] is None and section["call"]["speech_ms"] == 0


# --- stock phrases ------------------------------------------------------------------------------------------

def test_stock_phrases_repeated_across_turns():
    texts = {
        0: "I understand your concern, let me look into that for you.",
        1: "Thanks. I understand your concern, let me look into that right away.",
        2: "Your refund is on its way.",
        3: "I understand your concern about the delay.",
    }
    found = stock_phrases(texts)
    assert found["status"] == "measured"
    # Each phrase is the longest run of shared words said in the same turns, most-repeated first.
    assert found["phrases"] == [
        {"phrase": "i understand your concern", "turns": [0, 1, 3], "count": 3},
        {"phrase": "let me look into that", "turns": [0, 1], "count": 2},
    ]
    assert stock_phrases({0: "Hello there.", 1: "Goodbye now."})["count"] == 0


def test_stock_phrases_skip_without_a_transcript():
    data, _ = pcm(speak(1, [5], []).x)
    section = compute_prosody(data, RATE, [{"index": 0, "bot": {"segments": [[500, 1500]]}}], None)
    assert section["stock_phrases"] == {
        "status": "skipped", "reason": "no bot transcript yet: it comes with post-call speech-to-text"}
    texted = compute_prosody(data, RATE, [{"index": 0, "bot": {"segments": [[500, 1500]]}}], None,
                             bot_transcripts={0: "one two three four", 1: "one two three four five"})
    assert texted["stock_phrases"]["count"] == 1
