import warnings

from .support import require

require("numpy", "webrtcvad")

import numpy as np
import pytest

from evalkit.voicetest.audio import (
    FrameAssembler,
    Resampler,
    SpeechTracker,
    SynthSpec,
    frame_samples,
    mulaw_decode,
    mulaw_encode,
    resample,
    split_frames,
    synth_speech,
)

ALL_INT16 = np.arange(-32768, 32768, dtype=np.int16)


def _audioop():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            import audioop  # removed in Python 3.13
        except ImportError:
            return None
    return audioop


# -- mu-law ------------------------------------------------------------------------------------


def test_mulaw_bit_exact_with_audioop():
    audioop = _audioop()
    if audioop is None:
        pytest.skip("audioop is not available on this Python")
    assert mulaw_encode(ALL_INT16) == audioop.lin2ulaw(ALL_INT16.tobytes(), 2)
    codes = bytes(range(256))
    ref = np.frombuffer(audioop.ulaw2lin(codes, 2), dtype=np.int16)
    assert np.array_equal(mulaw_decode(codes), ref)


def test_mulaw_round_trip_error_bound():
    x = ALL_INT16.astype(np.int64)
    y = mulaw_decode(mulaw_encode(ALL_INT16)).astype(np.int64)
    err = np.abs(x - y)
    mag = np.abs(x)
    # Within the companding range a code's cell is 1/16 of an octave wide and decodes to its
    # middle, so the error is at most |x|/32 plus the 14-bit truncation and bias.
    inside = mag <= 32635
    assert np.all(err[inside] <= mag[inside] / 32.0 + 9)
    # Past the top code the signal clips at 32124.
    assert np.all(err[~inside] <= 32768 - 32124)
    assert int(err[np.abs(x) < 64].max()) <= 8


def test_mulaw_properties():
    assert mulaw_encode(np.zeros(4, dtype=np.int16)) == b"\xff" * 4
    assert np.all(mulaw_decode(b"\xff\x7f") == 0)
    table = mulaw_decode(bytes(range(256))).astype(np.int64)
    # 0x80-0xFF positive, decreasing with the code; 0x00-0x7F negative mirrors
    assert np.all(np.diff(table[128:]) < 0)
    assert np.array_equal(table[:128], -table[128:])
    t = np.arange(8000) / 8000.0
    sine = np.rint(12000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    out = mulaw_decode(mulaw_encode(sine)).astype(np.float64)
    snr = 10 * np.log10(np.sum(sine.astype(np.float64) ** 2) / np.sum((out - sine) ** 2))
    assert snr > 35.0


# -- resampling -------------------------------------------------------------------------------

PAIRS = [(8000, 16000), (16000, 8000), (24000, 16000), (16000, 24000), (8000, 24000), (24000, 8000)]


@pytest.mark.parametrize("src,dst", PAIRS)
def test_resample_preserves_a_passband_tone(src, dst):
    n = src  # 1 s
    t = np.arange(n) / src
    x = np.rint(8000 * np.sin(2 * np.pi * 1000 * t)).astype(np.int16)
    y = resample(x, src, dst)
    assert len(y) == n * dst // src
    ty = np.arange(len(y)) / dst
    ideal = 8000 * np.sin(2 * np.pi * 1000 * ty)
    margin = dst // 50
    assert np.max(np.abs(y[margin:-margin] - ideal[margin:-margin])) < 2.0


@pytest.mark.parametrize("src,dst", PAIRS)
def test_resample_streaming_equals_one_shot(src, dst):
    rng = np.random.default_rng(7)
    x = rng.integers(-20000, 20000, size=src // 2 + 13).astype(np.int16)
    one = resample(x, src, dst)
    r = Resampler(src, dst)
    parts, i = [], 0
    while i < len(x):
        k = int(rng.integers(1, 900))
        parts.append(r.process(x[i : i + k]))
        i += k
    parts.append(r.flush())
    assert np.array_equal(np.concatenate(parts), one)
    assert np.array_equal(resample(x, src, dst), one)  # deterministic


@pytest.mark.parametrize("src,dst", PAIRS)
def test_resample_keeps_timing(src, dst):
    x = np.zeros(src // 5, dtype=np.int16)
    at = src // 10
    x[at] = 20000
    y = resample(x, src, dst)
    assert abs(int(np.argmax(np.abs(y))) - at * dst / src) <= 1


def test_resample_rejects_aliases_when_downsampling():
    t = np.arange(16000) / 16000.0
    x = np.rint(10000 * np.sin(2 * np.pi * 5000 * t)).astype(np.int16)  # above 4 kHz Nyquist
    y = resample(x, 16000, 8000).astype(np.float64)[200:-200]
    level = 20 * np.log10(np.sqrt(np.mean(y**2)) / (10000 / np.sqrt(2)) + 1e-12)
    assert level < -50.0


def test_resample_identity_copies():
    x = np.arange(100, dtype=np.int16)
    assert np.array_equal(resample(x, 16000, 16000), x)


# -- framing ----------------------------------------------------------------------------------


def test_frame_sizes():
    assert [frame_samples(r) for r in (8000, 16000, 24000)] == [160, 320, 480]
    with pytest.raises(ValueError):
        frame_samples(44100, 15)


def test_split_frames_pads_the_last_frame():
    x = np.arange(1, 400, dtype=np.int16)
    frames = split_frames(x, 16000)
    assert frames.shape == (2, 320)
    assert np.array_equal(frames.reshape(-1)[:399], x)
    assert np.all(frames.reshape(-1)[399:] == 0)


def test_frame_assembler_keeps_the_remainder():
    fa = FrameAssembler(8000)
    x = np.arange(500, dtype=np.int16)
    got = fa.push(x[:100]) + fa.push(x[100:350]) + fa.push(x[350:])
    assert [len(f) for f in got] == [160, 160, 160]
    assert np.array_equal(np.concatenate(got), x[:480])


# -- the synthetic signal ---------------------------------------------------------------------


def _vad_flags(pcm, rate):
    import webrtcvad

    vad = webrtcvad.Vad(2)
    n = rate // 50
    return np.array([vad.is_speech(pcm[i : i + n].tobytes(), rate) for i in range(0, len(pcm) - n + 1, n)])


def test_synth_is_deterministic_and_exact_length():
    a = synth_speech(SynthSpec(ms=1234, seed=5), 16000)
    b = synth_speech(SynthSpec(ms=1234, seed=5), 16000)
    assert np.array_equal(a, b)
    assert len(a) == 1234 * 16
    assert a[0] == 0 and abs(int(a[-1])) < 50
    assert not np.array_equal(a, synth_speech(SynthSpec(ms=1234, seed=6), 16000))


@pytest.mark.parametrize("rate", [8000, 16000, 24000])
@pytest.mark.parametrize("seed,f0", [(0, 110.0), (1, 130.0), (2, 180.0), (3, 220.0)])
def test_synth_reads_as_speech_to_webrtc_vad(rate, seed, f0):
    pcm = synth_speech(SynthSpec(ms=2000, seed=seed, f0=f0), rate)
    if rate == 24000:
        pcm, rate = resample(pcm, 24000, 16000), 16000
    pad = np.zeros(rate // 2, dtype=np.int16)
    flags = _vad_flags(np.concatenate([pad, pcm, pad]), rate)
    inside = flags[25:125]
    assert inside.mean() >= 0.98
    assert flags[25]  # voiced from the very first frame
    assert not flags[:24].any()


def test_synth_survives_mulaw():
    pcm = synth_speech(SynthSpec(ms=1500, seed=4), 8000)
    pcm = mulaw_decode(mulaw_encode(pcm))
    assert _vad_flags(pcm, 8000).mean() >= 0.98


# -- the live speech tracker ------------------------------------------------------------------


@pytest.mark.parametrize("rate", [8000, 16000, 24000])
def test_tracker_finds_edges(rate):
    speech = synth_speech(SynthSpec(ms=1500, seed=3), rate)
    sig = np.concatenate([np.zeros(rate // 2, np.int16), speech, np.zeros(rate, np.int16)])
    tr = SpeechTracker(rate)
    events = []
    for frame in split_frames(sig, rate):
        events += tr.feed(frame)
    assert [e.kind for e in events] == ["onset", "offset"]
    onset_ms = events[0].sample * 1000 / rate
    offset_ms = events[1].sample * 1000 / rate
    assert abs(onset_ms - 500) <= 10
    # WebRTC VAD's own hangover runs ~100 ms past the end; the energy refinement removes it
    assert abs(offset_ms - 2000) <= 10
