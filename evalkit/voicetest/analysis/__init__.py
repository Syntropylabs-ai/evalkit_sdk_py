"""Post-call analysis: timing and the bot's prosody, measured from the two-channel recording (CONTRACT
section 5).

Importing this package loads nothing heavy; the VAD and numpy are imported when a recording is analysed.
"""

from __future__ import annotations

from .metrics import (
    ANALYSIS_VERSION,
    CALLER_ERROR_BUDGET_MS,
    INTERRUPTION_MAX_WAIT,
    INTERRUPTION_NOT_REACHED,
    INTERRUPTION_REACHED,
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
from .prosody import (
    DEFAULT_PROSODY_VERSION,
    PROSODY_SPECS,
    ProsodySpec,
    compute_prosody,
    get_prosody_spec,
    stock_phrases,
    track_f0,
)
from .recording_dir import RecordingDir, load_recording_dir
from .scorecard import SCORECARD_NAME, SCORECARD_SCHEMA, analyze, build_scorecard, write_scorecard
from .vad import DEFAULT_VAD_VERSION, VAD_SPECS, VadSpec, detect_segments, get_spec, segments_from_frames

__all__ = [
    "ANALYSIS_VERSION",
    "CALLER_ERROR_BUDGET_MS",
    "INTERRUPTION_MAX_WAIT",
    "INTERRUPTION_NOT_REACHED",
    "INTERRUPTION_REACHED",
    "SILENCE_MIN_MS",
    "CallerLine",
    "CallTimeline",
    "Wait",
    "compute_call",
    "compute_caller_health",
    "compute_preamble",
    "compute_turns",
    "timeline_from_events",
    "DEFAULT_PROSODY_VERSION",
    "PROSODY_SPECS",
    "ProsodySpec",
    "compute_prosody",
    "get_prosody_spec",
    "stock_phrases",
    "track_f0",
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
