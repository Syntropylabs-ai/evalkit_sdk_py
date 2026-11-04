"""The call script: caller lines, each with its audio and the rule for when to say it, plus an end
rule. The format is documented in README.md ("Script format").

A wait rule applies before its line: it says when the caller starts speaking. Waiting is a live
decision only; every timing that is scored is measured after the call from the recording.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_SILENCE_MS = 700
DEFAULT_MAX_WAIT_MS = 15000
DEFAULT_MAX_CALL_MS = 10 * 60 * 1000
WAIT_MODES = ("bot_done", "fixed")
DEFAULT_SOURCE_KINDS = ("wav", "synth")

# Who opens the call (RUN_API section 2), and the wait before line 0 the API plans for each. A bot that
# greets starts within GREETING_START_WITHIN_MS, so `auto` speaks then rather than sitting through the
# whole greeting allowance with a bot that waits for the caller.
FIRST_SPEAKERS = ("auto", "bot", "caller")
DEFAULT_FIRST_SPEAKER = "auto"
GREETING_START_WITHIN_MS = 3000
_FIRST_LINE_WAITS = {
    "auto": {"mode": "bot_done", "start_within_ms": GREETING_START_WITHIN_MS,
             "silence_ms": DEFAULT_SILENCE_MS, "max_ms": DEFAULT_MAX_WAIT_MS},
    "bot": {"mode": "bot_done", "silence_ms": DEFAULT_SILENCE_MS, "max_ms": DEFAULT_MAX_WAIT_MS},
    "caller": {"mode": "fixed", "ms": 0},
}


class ScriptError(ValueError):
    pass


def first_line_wait(first_speaker: str) -> Dict[str, Any]:
    """The wait rule (script format) before line 0 when `first_speaker` opens the call."""
    if first_speaker not in _FIRST_LINE_WAITS:
        raise ValueError(f"first speaker must be one of {FIRST_SPEAKERS}, not {first_speaker!r}")
    return dict(_FIRST_LINE_WAITS[first_speaker])


@dataclass
class WaitRule:
    """`bot_done`: wait for bot speech followed by `silence_ms` of silence on the heard timeline,
    giving up after `max_ms`. With `start_within_ms`, it also gives up when no bot speech has started
    that long after the wait began; once speech has started, only `silence_ms` and `max_ms` apply.
    `fixed`: wait `ms` after the previous line ended (or the call began)."""

    mode: str = "bot_done"
    silence_ms: int = DEFAULT_SILENCE_MS
    max_ms: int = DEFAULT_MAX_WAIT_MS
    ms: int = 0
    start_within_ms: Optional[int] = None

    @classmethod
    def from_json(cls, obj: Optional[Dict[str, Any]], where: str) -> "WaitRule":
        if obj is None:
            return cls()
        if not isinstance(obj, dict):
            raise ScriptError(f"{where}: a wait rule must be an object")
        extra = set(obj) - {"mode", "silence_ms", "max_ms", "ms", "start_within_ms"}
        if extra:
            raise ScriptError(f"{where}: unknown wait fields {sorted(extra)}")
        mode = obj.get("mode", "fixed" if "ms" in obj else "bot_done")
        if mode not in WAIT_MODES:
            raise ScriptError(f"{where}: wait mode must be one of {WAIT_MODES}")
        start_within = obj.get("start_within_ms")
        rule = cls(
            mode=mode,
            silence_ms=_ms(obj.get("silence_ms", DEFAULT_SILENCE_MS), where, "silence_ms"),
            max_ms=_ms(obj.get("max_ms", DEFAULT_MAX_WAIT_MS), where, "max_ms"),
            ms=_ms(obj.get("ms", 0), where, "ms"),
            start_within_ms=None if start_within is None else _ms(start_within, where, "start_within_ms"),
        )
        if mode == "fixed" and "ms" not in obj:
            raise ScriptError(f"{where}: a fixed wait needs ms")
        if mode == "fixed" and rule.start_within_ms is not None:
            raise ScriptError(f"{where}: start_within_ms belongs to a bot_done wait; a fixed wait waits ms")
        return rule

    def describe(self) -> str:
        return self.mode


@dataclass
class ScriptLine:
    index: int
    text: str
    source: str  # the LineSource kind: "wav", "synth", later "tts"
    spec: Any  # the source's own spec, as written in the script
    wait: WaitRule
    interrupt_after_bot_ms: Optional[int] = None


@dataclass
class Script:
    lines: List[ScriptLine]
    end: WaitRule
    max_call_ms: int = DEFAULT_MAX_CALL_MS
    name: str = ""
    base_dir: str = "."
    raw: Dict[str, Any] = field(default_factory=dict)


def _ms(value: Any, where: str, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ScriptError(f"{where}: {name} must be a non-negative number of ms")
    return int(round(value))


def parse_script(
    obj: Dict[str, Any], base_dir: str = ".", source_kinds: Iterable[str] = DEFAULT_SOURCE_KINDS
) -> Script:
    kinds: Tuple[str, ...] = tuple(source_kinds)
    if not isinstance(obj, dict):
        raise ScriptError("a script must be a JSON object")
    extra = set(obj) - {"name", "lines", "end", "max_call_ms"}
    if extra:
        raise ScriptError(f"unknown script fields {sorted(extra)}")
    raw_lines = obj.get("lines")
    if not isinstance(raw_lines, list) or not raw_lines:
        raise ScriptError("a script needs a non-empty lines list")
    lines: List[ScriptLine] = []
    for i, raw in enumerate(raw_lines):
        where = f"lines[{i}]"
        if not isinstance(raw, dict):
            raise ScriptError(f"{where}: a line must be an object")
        allowed = {"text", "wait", "interrupt_after_bot_ms", *kinds}
        unknown = set(raw) - allowed
        if unknown:
            raise ScriptError(f"{where}: unknown fields {sorted(unknown)} (audio kinds: {kinds})")
        present = [k for k in kinds if k in raw]
        if len(present) != 1:
            raise ScriptError(f"{where}: give exactly one of {kinds}")
        interrupt = raw.get("interrupt_after_bot_ms")
        wait = WaitRule.from_json(raw.get("wait"), where + ".wait")
        if interrupt is not None:
            interrupt = _ms(interrupt, where, "interrupt_after_bot_ms")
            if wait.mode != "bot_done" or wait.start_within_ms is not None:
                raise ScriptError(f"{where}: interrupt_after_bot_ms takes only max_ms as its wait")
        text = raw.get("text", "")
        if not isinstance(text, str):
            raise ScriptError(f"{where}: text must be a string")
        lines.append(
            ScriptLine(
                index=i,
                text=text,
                source=present[0],
                spec=raw[present[0]],
                wait=wait,
                interrupt_after_bot_ms=interrupt,
            )
        )
    end = WaitRule.from_json(obj.get("end"), "end")
    max_call_ms = _ms(obj.get("max_call_ms", DEFAULT_MAX_CALL_MS), "script", "max_call_ms")
    name = obj.get("name", "")
    return Script(
        lines=lines,
        end=end,
        max_call_ms=max_call_ms,
        name=str(name),
        base_dir=str(base_dir),
        raw=json.loads(json.dumps(obj)),
    )


def load_script(path: str, source_kinds: Iterable[str] = DEFAULT_SOURCE_KINDS) -> Script:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return parse_script(obj, base_dir=os.path.dirname(os.path.abspath(path)), source_kinds=source_kinds)
