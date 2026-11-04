"""Post-call analysis: timing measured from the two-channel recording (CONTRACT section 5).

Importing this package loads nothing heavy; the VAD is imported when a recording is analysed.
"""

from __future__ import annotations

from .metrics import (
    CALLER_ERROR_BUDGET_MS,
    SILENCE_MIN_MS,
    CallerLine,
    CallTimeline,
    Wait,
    compute_call,
    compute_caller_health,
    compute_preamble,
    compute_turns,
    timeline_from_events,
)
from .recording_dir import RecordingDir, load_recording_dir
from .scorecard import SCORECARD_NAME, SCORECARD_SCHEMA, analyze, build_scorecard, write_scorecard
from .vad import DEFAULT_VAD_VERSION, VAD_SPECS, VadSpec, detect_segments, get_spec, segments_from_frames

__all__ = [
    "CALLER_ERROR_BUDGET_MS",
    "SILENCE_MIN_MS",
    "CallerLine",
    "CallTimeline",
    "Wait",
    "compute_call",
    "compute_caller_health",
    "compute_preamble",
    "compute_turns",
    "timeline_from_events",
    "RecordingDir",
    "load_recording_dir",
    "SCORECARD_NAME",
    "SCORECARD_SCHEMA",
    "analyze",
    "build_scorecard",
    "write_scorecard",
    "DEFAULT_VAD_VERSION",
    "VAD_SPECS",
    "VadSpec",
    "detect_segments",
    "get_spec",
    "segments_from_frames",
]
