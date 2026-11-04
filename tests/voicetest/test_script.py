import json

from .support import require

require("numpy")

import numpy as np
import pytest

from evalkit.voicetest.audio import write_wav
from evalkit.voicetest.script import ScriptError, load_script, parse_script
from evalkit.voicetest.sources import LineSource, default_sources, render_lines


def test_parse_with_defaults():
    s = parse_script(
        {
            "lines": [
                {"text": "hi", "synth": {"ms": 800}, "wait": {"mode": "fixed", "ms": 300}},
                {"synth": {"ms": 500, "seed": 2}},
                {"synth": {"ms": 500}, "interrupt_after_bot_ms": 900, "wait": {"max_ms": 4000}},
            ]
        }
    )
    a, b, c = s.lines
    assert (a.wait.mode, a.wait.ms, a.text, a.source) == ("fixed", 300, "hi", "synth")
    assert (b.wait.mode, b.wait.silence_ms, b.wait.max_ms) == ("bot_done", 700, 15000)
    assert (c.interrupt_after_bot_ms, c.wait.max_ms) == (900, 4000)
    assert s.end.mode == "bot_done"


def test_a_bot_done_wait_may_give_up_when_no_bot_speech_starts():
    s = parse_script({"lines": [
        {"synth": {"ms": 500}, "wait": {"mode": "bot_done", "start_within_ms": 3000, "max_ms": 15000}},
        {"synth": {"ms": 500}},
    ]})
    first, second = s.lines
    assert (first.wait.mode, first.wait.start_within_ms, first.wait.silence_ms) == ("bot_done", 3000, 700)
    assert second.wait.start_within_ms is None


def test_the_first_line_wait_for_each_first_speaker():
    from evalkit.voicetest.script import FIRST_SPEAKERS, first_line_wait

    assert FIRST_SPEAKERS == ("auto", "bot", "caller")
    assert first_line_wait("auto") == {"mode": "bot_done", "start_within_ms": 3000, "silence_ms": 700,
                                       "max_ms": 15000}
    assert first_line_wait("bot") == {"mode": "bot_done", "silence_ms": 700, "max_ms": 15000}
    assert first_line_wait("caller") == {"mode": "fixed", "ms": 0}
    for speaker in FIRST_SPEAKERS:  # each is a wait the script format takes
        parse_script({"lines": [{"synth": {"ms": 1}, "wait": first_line_wait(speaker)}]})
    with pytest.raises(ValueError):
        first_line_wait("nobody")


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"lines": []},
        {"lines": [{"synth": {"ms": 1}, "wav": "a.wav"}]},
        {"lines": [{"text": "no audio"}]},
        {"lines": [{"synth": {"ms": 1}, "colour": "red"}]},
        {"lines": [{"synth": {"ms": 1}, "wait": {"mode": "fixed"}}]},
        {"lines": [{"synth": {"ms": 1}, "wait": {"mode": "soon"}}]},
        {"lines": [{"synth": {"ms": 1}, "interrupt_after_bot_ms": 5, "wait": {"mode": "fixed", "ms": 1}}]},
        {"lines": [{"synth": {"ms": 1}}], "end": {"silence_ms": -1}},
        {"lines": [{"synth": {"ms": 1}, "wait": {"mode": "fixed", "ms": 0, "start_within_ms": 3000}}]},
        {"lines": [{"synth": {"ms": 1}, "wait": {"start_within_ms": -1}}]},
        {"lines": [{"synth": {"ms": 1}, "interrupt_after_bot_ms": 5, "wait": {"start_within_ms": 3000}}]},
    ],
)
def test_rejects_bad_scripts(bad):
    with pytest.raises(ScriptError):
        parse_script(bad)


def test_wav_lines_are_resampled_and_trimmed(tmp_path):
    rate = 44100
    body = (8000 * np.sin(2 * np.pi * 300 * np.arange(rate) / rate)).astype(np.int16)
    stereo = np.stack([np.concatenate([np.zeros(rate // 2, np.int16), body, np.zeros(rate // 4, np.int16)])] * 2, 1)
    write_wav(str(tmp_path / "line.wav"), stereo, rate)
    (tmp_path / "s.json").write_text(json.dumps({"lines": [{"wav": "line.wav"}, {"wav": {"path": "line.wav", "trim": False}}]}))
    script = load_script(str(tmp_path / "s.json"))
    trimmed, untrimmed = render_lines(script, 16000)
    assert abs(len(trimmed) - 16000) <= 2
    assert abs(len(untrimmed) - int(1.75 * 16000)) <= 2


def test_sources_are_pluggable():
    class Beep(LineSource):
        kind = "beep"

        def render(self, spec, rate, base_dir):
            return np.full(int(spec) * rate // 1000, 1000, dtype=np.int16)

    script = parse_script({"lines": [{"beep": 40}]}, source_kinds=("wav", "synth", "beep"))
    (pcm,) = render_lines(script, 8000, default_sources([Beep()]))
    assert len(pcm) == 320
