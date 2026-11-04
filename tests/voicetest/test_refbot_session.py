"""The reference bot's engine, driven frame by frame without a socket."""

from .support import require

require("numpy", "webrtcvad")

import numpy as np
import pytest

from evalkit.voicetest.audio import SynthSpec, synth_speech
from evalkit.voicetest.refbot import Behaviour, BehaviourError, CallTruth, RefBotSession, TruthLog

RATE = 16000
F = 320


def caller_audio(spans_ms, total_ms):
    """Caller speech (synthetic) in the given [start, end) ms spans, silence elsewhere."""
    x = np.zeros(total_ms * RATE // 1000, dtype=np.int16)
    for i, (a, b) in enumerate(spans_ms):
        speech = synth_speech(SynthSpec(ms=b - a, seed=i), RATE)
        x[a * 16 : a * 16 + len(speech)] = speech
    return x


def run(behaviour_json, spans_ms, total_ms, fmt="pcm16"):
    truth = TruthLog()
    behaviour = Behaviour.from_json(behaviour_json)
    cache = {}

    def render(speech, rate):
        key = (speech, rate)
        if key not in cache:
            cache[key] = synth_speech(SynthSpec.from_json(speech.synth_json()), rate)
        return cache[key]

    s = RefBotSession(behaviour, fmt, RATE, CallTruth(truth, 0, RATE, "tok"), render)
    sent = []  # (frame start sample, out)
    for out in s.start():
        sent.append((0, out))
    audio = caller_audio(spans_ms, total_ms)
    for k in range(len(audio) // F):
        for out in s.on_audio(audio[k * F : (k + 1) * F]):
            sent.append((k * F, out))
    s.finish("test")
    return truth.records, sent


def rec(records, type_, **match):
    return [r for r in records if r["type"] == type_ and all(r.get(k) == v for k, v in match.items())]


def heard(sent, upto):
    """Replay the sends through an ideal caller buffer (zero network delay)."""
    from evalkit.voicetest.transport import PlaybackBuffer

    buf = PlaybackBuffer(RATE)
    out = []
    for at, (kind, value) in sent:
        while buf.head < at:
            out.append(buf.render(buf.head + F)[0])
        if kind == "audio":
            buf.push(value, at)
        elif kind == "clear":
            buf.clear(at)
    while buf.head < upto:
        out.append(buf.render(buf.head + F)[0])
    return np.concatenate(out)


def test_reply_lands_exactly_delay_after_the_detected_end():
    records, sent = run({"turns": [{"delay_ms": 810, "reply_ms": 1000}]}, [(500, 1500)], 4000)
    (start,) = rec(records, "caller_speech_start")
    (end,) = rec(records, "caller_speech_end")
    assert abs(start["ms"] - 500) <= 10 and abs(end["ms"] - 1500) <= 10
    (sp,) = rec(records, "speech_start", kind="reply")
    assert sp["ms"] == pytest.approx(end["ms"] + 810)
    (se,) = rec(records, "speech_end", kind="reply")
    assert se["ms"] == pytest.approx(sp["ms"] + 1000) and se["reason"] == "complete"
    h = heard(sent, 4000 * 16)
    first = int(np.flatnonzero(h)[0])
    assert abs(first - sp["sample"]) <= 2  # the silence prefix places a sub-frame start exactly
    # realtime: never more than one frame plus the lead ahead of the caller
    for at, (kind, value) in sent:
        if kind == "audio":
            assert len(value) <= 2 * F + F


def test_burst_sends_everything_at_once():
    records, sent = run(
        {"burst_chunk_ms": 200, "turns": [{"delay_ms": 600, "reply_ms": 2000, "mode": "burst"}]},
        [(200, 1000)],
        4000,
    )
    audio = [(at, v) for at, (k, v) in sent if k == "audio"]
    assert len({at for at, _ in audio}) == 1  # one frame's processing
    assert sum(len(v) for _, v in audio) >= 2000 * 16
    (sp,) = rec(records, "speech_start", kind="reply")
    assert sp["mode"] == "burst"


def test_interruption_with_clear_cuts_the_burst():
    records, sent = run(
        {"turns": [
            {"delay_ms": 500, "reply_ms": 3000, "mode": "burst", "on_interrupt": "clear", "stop_delay_ms": 240},
        ]},
        [(200, 1000), (2500, 3200)],
        6000,
    )
    (intr,) = rec(records, "interruption")
    assert intr["action"] == "clear" and intr["caller_turn"] == 1
    (clear,) = rec(records, "clear_sent")
    assert clear["ms"] >= intr["ms"] + 240 and clear["ms"] < intr["ms"] + 240 + 20
    (end,) = rec(records, "speech_end", kind="reply", turn=0)
    assert end["reason"] == "cleared" and end["ms"] == clear["ms"]
    assert any(k == "clear" for _, (k, _) in sent)
    h = heard(sent, 6000 * 16)
    assert not h[int(end["sample"]) + 1 : int(end["sample"]) + 8000].any()


def test_interruption_with_stop_in_realtime():
    records, sent = run(
        {"turns": [{"delay_ms": 500, "reply_ms": 3000, "on_interrupt": "stop", "stop_delay_ms": 300}]},
        [(200, 1000), (2500, 3200)],
        6000,
    )
    (intr,) = rec(records, "interruption")
    (stop,) = rec(records, "stop_sending")
    assert stop["ms"] == pytest.approx(intr["ms"] + 300)
    (end,) = rec(records, "speech_end", kind="reply", turn=0)
    assert end["reason"] == "stopped" and end["ms"] == pytest.approx(stop["ms"])
    assert not any(k == "clear" for _, (k, _) in sent)


def test_ignored_interruption_keeps_talking():
    records, _ = run(
        {"turns": [{"delay_ms": 500, "reply_ms": 2500, "on_interrupt": "ignore"}]},
        [(200, 1000), (2000, 2600)],
        6000,
    )
    assert rec(records, "interruption")[0]["action"] == "ignore"
    (end,) = rec(records, "speech_end", kind="reply", turn=0)
    assert end["reason"] == "complete"


def test_silent_turn_talk_over_and_greeting():
    records, _ = run(
        {
            "greeting": {"after_ms": 100, "ms": 600},
            "turns": [
                {"silent": True},
                {"delay_ms": 700, "reply_ms": 500, "talk_over": {"at_ms": 300, "ms": 400}},
            ],
        },
        [(1000, 1800), (3000, 4200)],
        7000,
    )
    (g,) = rec(records, "speech_start", kind="greeting")
    assert g["ms"] == 100.0
    assert rec(records, "silent_turn", turn=0)
    assert not rec(records, "speech_start", kind="reply", turn=0)
    (to,) = rec(records, "speech_start", kind="talk_over", turn=1)
    (cs,) = rec(records, "caller_speech_start", turn=1)
    assert to["ms"] == pytest.approx(cs["ms"] + 300)
    assert rec(records, "speech_start", kind="reply", turn=1)


def test_twilio_marks_follow_each_utterance():
    records, sent = run({"turns": [{"delay_ms": 600, "reply_ms": 700}]}, [(200, 900)], 3000, fmt="twilio")
    marks = [v for _, (k, v) in sent if k == "mark"]
    assert marks == ["reply-0"]
    (m,) = rec(records, "mark_sent")
    (sp,) = rec(records, "speech_start", kind="reply")
    assert m["ms"] == pytest.approx(sp["ms"] + 700)


@pytest.mark.parametrize(
    "bad",
    [
        {"turns": [{"mode": "warp"}]},
        {"turns": [{"on_interrupt": "explode"}]},
        {"onset_ms": 30},
        {"burst_chunk_ms": 5},
        {"greeting": {"after_ms": 0}},
        {"turns": [{"reply_ms": 0}]},
        {"unknown": 1},
    ],
)
def test_rejects_bad_behaviour(bad):
    with pytest.raises(BehaviourError):
        Behaviour.from_json(bad)
