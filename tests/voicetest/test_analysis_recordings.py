"""The analysis on recordings written the way the caller writes them, by the calibration generator."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from evalkit.voicetest.analysis import SCORECARD_SCHEMA, analyze, write_scorecard
from tests.voicetest.synth import CallPlan, TurnPlan, make_plan, write_call

from .support import SDK_ROOT

pytest.importorskip("numpy")
pytest.importorskip("webrtcvad")


def _call(tmp_path, k, vad="webrtc-v1"):
    path = str(tmp_path / f"call{k}")
    truth = write_call(make_plan(k, seed=7), path, vad)
    return path, truth


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_v2_meets_the_target_on_every_format(tmp_path, k):
    path, truth = _call(tmp_path, k)
    scorecard = analyze(path, "webrtc-v2")
    assert scorecard["vad"]["version"] == "webrtc-v2"
    for want, got in zip(truth["expected"]["turns"], scorecard["turns"]):
        for metric in ("response_delay_ms", "interruption_stop_ms"):
            assert (want[metric] is None) == (got[metric] is None), (metric, want, got)
            if want[metric] is not None:
                assert abs(got[metric] - want[metric]) <= 50, (metric, want[metric], got[metric])
        assert got["talk_over_count"] == want["talk_over_count"]
        assert got["caller_unhealthy"] == want["caller_unhealthy"]


def test_v1_ends_segments_late_by_the_vads_hangover(tmp_path):
    """The evidence behind webrtc-v2: v1's segment ends trail the truth by about 100 ms."""
    path, truth = _call(tmp_path, 1)
    got = analyze(path, "webrtc-v1")["segments"]["bot"]
    assert len(got) == len(truth["bot_segments"])
    late = [g[1] - t[1] for t, g in zip(truth["bot_segments"], got)]
    assert sum(late) / len(late) > 60
    early = [abs(g[0] - t[0]) for t, g in zip(truth["bot_segments"], got)]
    assert max(early) <= 30


def test_scorecard_schema_and_purity(tmp_path):
    path, _ = _call(tmp_path, 4)
    first = write_scorecard(path)
    with open(first, "rb") as fh:
        one = fh.read()
    os.remove(first)
    write_scorecard(path)
    with open(first, "rb") as fh:
        assert fh.read() == one
    card = json.loads(one)
    assert card["schema"] == SCORECARD_SCHEMA
    assert card["vad"]["version"] == "webrtc-v1"  # what call.json names
    call = card["call"]
    for key in (
        "session", "transport", "sample_rate", "t0_epoch_ms", "duration_ms", "first_word_ms",
        "response_delay_p50_ms", "response_delay_p95_ms", "talk_over_ms", "talk_over_count", "max_silence_ms",
    ):
        assert key in call
    assert call["transport"] == "twilio" and call["sample_rate"] == 8000
    # Turn i is caller line i, counted from 0 (CONTRACT section 5).
    assert [t["index"] for t in card["turns"]] == list(range(len(card["turns"])))
    turn = card["turns"][0]
    for key in (
        "index", "window", "caller", "bot", "response_delay_ms", "talk_over_ms", "talk_over_count",
        "interruption_planned", "interruption_stop_ms", "silence_ms", "bot_speech_ms", "caller_unhealthy",
    ):
        assert key in turn
    health = card["caller_health"]
    assert health["vad_crosscheck"]["lines_disagreeing"] == []
    assert len(health["vad_crosscheck"]["lines"]) == len(card["turns"])


def test_call_json_vad_version_is_used(tmp_path):
    path, _ = _call(tmp_path, 2, vad="webrtc-v2")
    assert analyze(path)["vad"]["version"] == "webrtc-v2"
    with open(os.path.join(path, "call.json"), "r", encoding="utf-8") as fh:
        call = json.load(fh)
    call["vad"] = {"version": "webrtc-v9"}
    with open(os.path.join(path, "call.json"), "w", encoding="utf-8") as fh:
        json.dump(call, fh)
    with pytest.raises(ValueError, match="unknown vad version"):
        analyze(path)


def test_a_bot_that_never_speaks(tmp_path):
    plan = CallPlan(
        name="silent",
        seed=3,
        format="pcm16",
        sample_rate=16000,
        turns=[TurnPlan(line_ms=1200, reply_delay_ms=None), TurnPlan(line_ms=900, reply_delay_ms=None)],
        noise_dbfs=-50.0,
        cases=["no_reply"],
    )
    path = str(tmp_path / "silent")
    write_call(plan, path)
    card = analyze(path)
    assert card["segments"]["bot"] == []
    assert card["call"]["first_word_ms"] is None
    assert card["call"]["response_delay_p50_ms"] is None
    assert all(t["response_delay_ms"] is None and t["bot_speech_ms"] == 0 for t in card["turns"])
    assert card["call"]["max_silence_ms"] == 0


def test_importing_the_analysis_loads_no_numeric_or_vad_module():
    code = (
        "import sys; import evalkit.voicetest.analysis, evalkit.voicetest.trace; "
        "bad = [m for m in ('numpy', 'webrtcvad') if m in sys.modules]; "
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=SDK_ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
