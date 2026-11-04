"""Black-box testing of a voice bot over WebSocket: a test caller that talks to the bot, records both
sides, measures the call from the recording, and reports it to SyntropyLabs.

Needs the ``voice`` extra (``pip install 'syntropylabs-evalkit[voice]'``: numpy, websockets and
webrtcvad). Importing this package loads none of them and starts nothing; each is imported by the code
that uses it. Logging goes to the ``evalkit.voicetest`` loggers.

- `run(url, format=..., scenario_set=...)`: the SDK run (``evalkit voice-test run``).
- `check(url, format=...)`: one short call to see whether the bot answers and speaks the format.
- `run_planned_call(...)`: one plan call placed and reported, shared with the hosted runner.
- `script_from_plan_call(call)`: a plan call as a `Script` and its line sources.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "run": ".runner",
    "run_async": ".runner",
    "VoiceTestRun": ".runner",
    "VoiceTestError": ".runner",
    "VoiceTestSetupError": ".runner",
    "check": ".check",
    "check_async": ".check",
    "run_planned_call": ".planned_call",
    "CallOutcome": ".planned_call",
    "ERROR_KINDS": ".planned_call",
    "script_from_plan_call": ".plan",
    "prepare_script_audio": ".plan",
    "parse_plan": ".plan",
    "RunPlan": ".plan",
    "PlannedCall": ".plan",
    "PlanError": ".plan",
    "target_from_settings": ".plan",
    "VoiceTestApi": ".api",
    "VoiceTestApiError": ".api",
    "AudioCache": ".audio_cache",
    "AudioError": ".audio_cache",
    "PlanAudioSource": ".sources",
    "Target": ".transport",
    "run_call": ".caller",
    "parse_script": ".script",
    "load_script": ".script",
    "analyze": ".analysis",
    "send_call_trace": ".trace",
    "ingest_client": ".trace",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
