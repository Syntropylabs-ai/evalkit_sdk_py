"""Assembling ``scorecard.json`` from a recording directory.

The scorecard is a pure function of the directory (and of the VAD version, which ``call.json`` names):
no clock, no host path and no randomness go into it, so re-analysing a recording anywhere gives the same
bytes.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

from .intervals import Interval, overlap
from .metrics import (
    CallerLine,
    compute_call,
    compute_caller_health,
    compute_preamble,
    compute_turns,
    timeline_from_events,
)
from .recording_dir import RECORDING_RATE, RecordingDir, load_recording_dir
from .vad import DEFAULT_VAD_VERSION, detect_segments, get_spec

SCORECARD_SCHEMA = "evalkit.voice.scorecard.v1"
SCORECARD_NAME = "scorecard.json"

# How far the channel-0 VAD may sit from the exact line boundaries before a line is reported as
# disagreeing. The VAD's own boundary error (measured by the calibration) is the floor, plus the
# 50 ms caller error budget of CONTRACT section 5.
CROSSCHECK_TOLERANCE_MS = {
    "webrtc-v1": {"start": 100, "end": 200},
    "webrtc-v2": {"start": 100, "end": 100},
}
_DEFAULT_TOLERANCE = {"start": 100, "end": 200}


def crosscheck_caller(
    lines: Sequence[CallerLine], vad_segments: Sequence[Interval], vad_version: str
) -> Dict[str, Any]:
    """Compares the exact ``line_start``/``line_end`` boundaries with what the VAD hears on channel 0.

    A disagreement means the caller's events and its audio do not line up: a caller defect, not the
    bot's.
    """
    tolerance = CROSSCHECK_TOLERANCE_MS.get(vad_version, _DEFAULT_TOLERANCE)
    per_line: List[Dict[str, Any]] = []
    used = set()
    for line in lines:
        span = (line.start_ms, line.end_ms)
        hits = [k for k, seg in enumerate(vad_segments) if overlap(seg, span) > 0]
        used.update(hits)
        if not hits:
            per_line.append({"index": line.index, "matched": False, "agrees": False})
            continue
        start_delta = vad_segments[hits[0]][0] - line.start_ms
        end_delta = vad_segments[hits[-1]][1] - line.end_ms
        per_line.append(
            {
                "index": line.index,
                "matched": True,
                "start_delta_ms": start_delta,
                "end_delta_ms": end_delta,
                "agrees": abs(start_delta) <= tolerance["start"] and abs(end_delta) <= tolerance["end"],
            }
        )
    extra = [seg for k, seg in enumerate(vad_segments) if k not in used]
    starts = [abs(p["start_delta_ms"]) for p in per_line if p["matched"]]
    ends = [abs(p["end_delta_ms"]) for p in per_line if p["matched"]]
    return {
        "tolerance_ms": dict(tolerance),
        "lines": per_line,
        "lines_disagreeing": [p["index"] for p in per_line if not p["agrees"]],
        "start_delta_max_abs_ms": max(starts, default=None),
        "end_delta_max_abs_ms": max(ends, default=None),
        "unmatched_segments": len(extra),
        "unmatched_speech_ms": sum(e - s for s, e in extra),
    }


def build_scorecard(recording: RecordingDir, vad_version: Optional[str] = None) -> Dict[str, Any]:
    version = vad_version or recording.vad_version or DEFAULT_VAD_VERSION
    spec = get_spec(version)
    bot = detect_segments(recording.bot_pcm, RECORDING_RATE, spec)
    caller_vad = detect_segments(recording.caller_pcm, RECORDING_RATE, spec)
    timeline = timeline_from_events(recording.events, recording.duration_ms)
    turns = compute_turns(timeline, bot, spec.onset_tolerance_ms)
    call = {
        "session": recording.session,
        "transport": recording.transport,
        "sample_rate": recording.call_sample_rate,
        "t0_epoch_ms": recording.t0_epoch_ms,
    }
    call.update(compute_call(timeline, bot, turns))
    health = compute_caller_health(timeline, turns)
    health["vad_crosscheck"] = crosscheck_caller(timeline.lines, caller_vad, spec.version)
    return {
        "schema": SCORECARD_SCHEMA,
        "vad": spec.describe(),
        "call": call,
        "preamble": compute_preamble(timeline, bot),
        "turns": turns,
        "caller_health": health,
        "segments": {
            "bot": [list(s) for s in bot],
            "caller": [[line.start_ms, line.end_ms] for line in timeline.lines],
            "caller_vad": [list(s) for s in caller_vad],
        },
        "problems": list(recording.problems),
    }


def analyze(recording_dir: str, vad_version: Optional[str] = None) -> Dict[str, Any]:
    """The scorecard of one recording directory; the VAD version defaults to the one ``call.json`` names."""
    return build_scorecard(load_recording_dir(recording_dir), vad_version)


def write_scorecard(recording_dir: str, vad_version: Optional[str] = None, name: str = SCORECARD_NAME) -> str:
    scorecard = analyze(recording_dir, vad_version)
    path = os.path.join(recording_dir, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(scorecard, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return path
