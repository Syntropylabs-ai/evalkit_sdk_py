"""The call script: caller lines, each with its audio and the rule for when to say it, plus an end
rule. The format is documented in README.md ("Script format").

A wait rule applies before its line: it says when the caller starts speaking. Waiting is a live
decision only; every timing that is scored is measured after the call from the recording.

A dynamic script (``"mode": "dynamic"``, CONTRACT section 4.1) holds only the lines planned before the
call, the opening first; every later line is asked for during the call from the turn endpoint, by the
rules in its ``turns``.
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
SCRIPT_MODES = ("scripted", "dynamic")
# RUN_API section 3.1.1: caller lines after the opening.
DEFAULT_DYNAMIC_TURNS = 8
MAX_DYNAMIC_TURNS = 20

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
class DynamicTurns:
    """How a dynamic call asks for its lines: at most `max` caller lines after the opening, each asked
    for once the bot's answer is over by `wait`; `interrupt_turn` is the line planned to cut into the
    bot's answer, `interrupt_after_bot_ms` into it."""

    max: int = DEFAULT_DYNAMIC_TURNS
    wait: WaitRule = field(default_factory=WaitRule)
    interrupt_turn: Optional[int] = None
    interrupt_after_bot_ms: Optional[int] = None


@dataclass
class Script:
    lines: List[ScriptLine]
    end: WaitRule
    max_call_ms: int = DEFAULT_MAX_CALL_MS
    name: str = ""
    base_dir: str = "."
    raw: Dict[str, Any] = field(default_factory=dict)
    mode: str = "scripted"
    turns: Optional[DynamicTurns] = None

    @property
    def dynamic(self) -> bool:
        return self.mode == "dynamic"


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
    extra = set(obj) - {"name", "lines", "end", "max_call_ms", "mode", "turns"}
    if extra:
        raise ScriptError(f"unknown script fields {sorted(extra)}")
    mode = obj.get("mode", "scripted")
    if mode not in SCRIPT_MODES:
        raise ScriptError(f"mode must be one of {SCRIPT_MODES}, not {mode!r}")
    if "turns" in obj and mode != "dynamic":
        raise ScriptError("turns belongs to a dynamic script (mode: dynamic)")
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
        mode=mode,
        turns=_dynamic_turns(obj.get("turns")) if mode == "dynamic" else None,
    )


def _dynamic_turns(obj: Any) -> DynamicTurns:
    """A dynamic script's ``turns``: ``{"max", "wait", "interrupt": {"turn", "after_bot_ms"}}``, each
    optional."""
    if obj is None:
        return DynamicTurns()
    if not isinstance(obj, dict):
        raise ScriptError("turns: must be an object")
    extra = set(obj) - {"max", "wait", "interrupt"}
    if extra:
        raise ScriptError(f"turns: unknown fields {sorted(extra)}")
    most = obj.get("max", DEFAULT_DYNAMIC_TURNS)
    if isinstance(most, bool) or not isinstance(most, int) or not 1 <= most <= MAX_DYNAMIC_TURNS:
        raise ScriptError(f"turns.max must be a whole number from 1 to {MAX_DYNAMIC_TURNS}")
    wait = WaitRule.from_json(obj.get("wait"), "turns.wait")
    turns = DynamicTurns(max=most, wait=wait)
    interrupt = obj.get("interrupt")
    if interrupt is not None:
        if not isinstance(interrupt, dict) or set(interrupt) - {"turn", "after_bot_ms"}:
            raise ScriptError("turns.interrupt: must be {turn, after_bot_ms}")
        turn = interrupt.get("turn")
        if isinstance(turn, bool) or not isinstance(turn, int) or not 1 <= turn <= most:
            raise ScriptError(f"turns.interrupt.turn must be a caller line from 1 to turns.max ({most})")
        turns.interrupt_turn = turn
        turns.interrupt_after_bot_ms = _ms(interrupt.get("after_bot_ms", 0), "turns.interrupt", "after_bot_ms")
    return turns


def load_script(path: str, source_kinds: Iterable[str] = DEFAULT_SOURCE_KINDS) -> Script:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return parse_script(obj, base_dir=os.path.dirname(os.path.abspath(path)), source_kinds=source_kinds)
