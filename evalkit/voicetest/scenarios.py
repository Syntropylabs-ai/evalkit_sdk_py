"""Inline scenarios: the ``scenarios`` of ``POST /v1/voice-tests`` (RUN_API sections 3.1 and 3.1.1), written
in a YAML or JSON file or passed from Python, for a run that needs no scenario set saved in the console.

A scenario is one of two kinds. A **scripted** one names the caller's exact lines; a **goal** one gives
the caller a goal, and the caller talks its way to it during the call, hearing what the bot says
(RUN_API section 3.4). The wire shape is camelCase::

    {"name": "...", "lines": [{"text": "...", "wait": {...}, "interruptAfterBotMs": 1500}],
     "successCriteria": ["..."]}
    {"name": "...", "goal": "...", "persona": "IMPATIENT", "facts": {"order_number": "4471"},
     "opening": "...", "successCriteria": ["..."], "maxTurns": 8, "interrupt": {"turn": 1, "afterBotMs": 1500}}

What people write is looser, and is normalised here into that shape: a line may be a plain string,
snake_case spellings are accepted (``success_criteria``, ``interrupt_after_bot_ms``, ``silence_ms``,
``max_ms``, ``start_within_ms``, ``mode`` for ``rule``, ``max_turns``, ``after_bot_ms``, ``reveal_when``),
``successCriteria`` may be one string, and a scenario written by `evalkit.generate_scenarios` plays its
``turns`` as its lines, or, written for a live simulated user (no turns), its ``user_goal``,
``hidden_info`` and opening as a goal scenario. The API's limits are checked here as well, so a mistake
is reported with the file, the scenario and the line before anything is sent. Unknown fields are refused.
Standard library only, except PyYAML for a ``.yaml`` file.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

# RUN_API section 3.1. The server checks the same limits; checking them here names the file and line.
MAX_SCENARIOS = 50
MAX_LINES = 40
MAX_LINE_CHARS = 500
MAX_NAME_CHARS = 200
MAX_CRITERIA = 20
MAX_CRITERION_CHARS = 500
# Calls an SDK run may plan: scenarios x repeats.
MAX_SDK_CALLS = 50
MAX_SILENCE_MS = 10_000
MIN_MAX_MS = 1_000
MAX_MAX_MS = 60_000
MAX_FIXED_MS = 60_000
MAX_INTERRUPT_AFTER_BOT_MS = 30_000
# The wait a line gets when it names none: what an interrupting line's maxMs is checked against.
DEFAULT_MAX_WAIT_MS = 15_000
# RUN_API section 3.1.1, a scenario with a goal.
MAX_GOAL_CHARS = 1000
MAX_PERSONA_CHARS = 500
MAX_FACTS = 12
MAX_FACT_CHARS = 500
MAX_OPENING_CHARS = 500
DEFAULT_MAX_TURNS = 8
MAX_TURNS = 20
PERSONAS = ("EXPERT", "NOVICE", "ADVERSARIAL", "VAGUE", "CONTEXT_SWITCHER", "IMPATIENT")

DEFAULT_FILE = "voice-tests.yaml"

_SCENARIO_FIELDS = {"name", "lines", "successCriteria", "success_criteria"}
_GOAL_FIELDS = {"name", "goal", "persona", "facts", "opening", "successCriteria", "success_criteria",
                "maxTurns", "max_turns", "interrupt"}
_LINE_FIELDS = {"text", "wait", "interruptAfterBotMs", "interrupt_after_bot_ms"}
_WAIT_FIELDS = {
    "rule": "rule", "mode": "rule",
    "silenceMs": "silenceMs", "silence_ms": "silenceMs",
    "maxMs": "maxMs", "max_ms": "maxMs",
    "startWithinMs": "startWithinMs", "start_within_ms": "startWithinMs",
    "ms": "ms",
}

PathLike = Union[str, "os.PathLike[str]"]


class ScenarioError(ValueError):
    """Inline scenarios that cannot be sent: the message names the file, the scenario and the line."""


def _ms(value: Any, where: str, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        raise ScenarioError(f"{where}: {name} must be a whole number of ms")
    if not low <= value <= high:
        raise ScenarioError(f"{where}: {name} must be from {low:,} to {high:,} ms, not {value:g}")
    return int(round(value))


def _wait(raw: Any, where: str) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ScenarioError(f"{where}: a wait is a mapping such as {{rule: bot_done, silenceMs: 700}}")
    unknown = sorted(set(raw) - set(_WAIT_FIELDS))
    if unknown:
        raise ScenarioError(f"{where}: unknown wait fields {unknown} (a wait takes rule, silenceMs, maxMs, "
                            "startWithinMs, or ms for a fixed wait)")
    fields: Dict[str, Any] = {}
    for key, value in raw.items():
        canonical = _WAIT_FIELDS[key]
        if canonical in fields:
            raise ScenarioError(f"{where}: {canonical} is given twice")
        fields[canonical] = value
    rule = fields.pop("rule", "fixed" if "ms" in fields else "bot_done")
    if rule == "fixed":
        extra = sorted(set(fields) - {"ms"})
        if extra:
            raise ScenarioError(f"{where}: a fixed wait takes only ms, not {extra}")
        if "ms" not in fields:
            raise ScenarioError(f"{where}: a fixed wait needs ms")
        return {"rule": "fixed", "ms": _ms(fields["ms"], where, "ms", 0, MAX_FIXED_MS)}
    if rule != "bot_done":
        raise ScenarioError(f"{where}: rule must be bot_done or fixed, not {rule!r}")
    if "ms" in fields:
        raise ScenarioError(f"{where}: ms belongs to a fixed wait; a bot_done wait takes silenceMs and maxMs")
    out: Dict[str, Any] = {"rule": "bot_done"}
    if "silenceMs" in fields:
        out["silenceMs"] = _ms(fields["silenceMs"], where, "silenceMs", 0, MAX_SILENCE_MS)
    if "maxMs" in fields:
        out["maxMs"] = _ms(fields["maxMs"], where, "maxMs", MIN_MAX_MS, MAX_MAX_MS)
    if "startWithinMs" in fields:
        limit = out.get("maxMs", DEFAULT_MAX_WAIT_MS)
        out["startWithinMs"] = _ms(fields["startWithinMs"], where, "startWithinMs", 0, limit)
    return out


def _text(raw: Any, where: str) -> str:
    if isinstance(raw, bool) or isinstance(raw, (int, float)) or raw is None:
        shown = "empty" if raw is None else json.dumps(raw)
        raise ScenarioError(f"{where} is {shown}, not text; in YAML, quote a line such as \"Yes\" or \"No\"")
    if not isinstance(raw, str):
        raise ScenarioError(f"{where}: the text must be a string")
    text = raw.strip()
    if not text:
        raise ScenarioError(f"{where}: the text is empty")
    if len(text) > MAX_LINE_CHARS:
        raise ScenarioError(f"{where}: the text is {len(text)} characters; a line takes at most {MAX_LINE_CHARS}")
    return text


def _line(raw: Any, where: str) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {"text": _text(raw, where)}
    unknown = sorted(set(raw) - _LINE_FIELDS)
    if unknown and "text" not in raw:
        raise ScenarioError(f"{where} has no text (found {unknown}); in YAML, a line with a colon in it "
                            "needs quotes, or write it as text: ...")
    if unknown:
        raise ScenarioError(f"{where}: unknown line fields {unknown} (a line takes text, wait and "
                            "interruptAfterBotMs)")
    out: Dict[str, Any] = {"text": _text(raw.get("text"), where)}
    wait = _wait(raw["wait"], where + ".wait") if raw.get("wait") is not None else None
    interrupt = raw.get("interruptAfterBotMs", raw.get("interrupt_after_bot_ms"))
    if "interruptAfterBotMs" in raw and "interrupt_after_bot_ms" in raw:
        raise ScenarioError(f"{where}: interruptAfterBotMs is given twice")
    if interrupt is not None:
        interrupt = _ms(interrupt, where, "interruptAfterBotMs", 0, MAX_INTERRUPT_AFTER_BOT_MS)
        if wait is not None and (wait["rule"] != "bot_done" or "startWithinMs" in wait):
            raise ScenarioError(f"{where}: a line that interrupts waits for the bot to speak, so its wait is "
                                "bot_done without startWithinMs (or leave the wait out)")
        limit = (wait or {}).get("maxMs", DEFAULT_MAX_WAIT_MS)
        if interrupt >= limit:
            raise ScenarioError(f"{where}: interruptAfterBotMs ({interrupt:,}) must be less than the wait's "
                                f"maxMs ({limit:,}), when the caller stops waiting for the bot")
    if wait is not None:
        out["wait"] = wait
    if interrupt is not None:
        out["interruptAfterBotMs"] = interrupt
    return out


def _criteria(raw: Any, where: str) -> List[str]:
    items = [raw] if isinstance(raw, str) else raw
    if not isinstance(items, (list, tuple)):
        raise ScenarioError(f"{where}: successCriteria is a string or a list of strings")
    if len(items) > MAX_CRITERIA:
        raise ScenarioError(f"{where}: {len(items)} success criteria; a scenario takes at most {MAX_CRITERIA}")
    out = []
    for i, item in enumerate(items):
        if not isinstance(item, str) or not item.strip():
            raise ScenarioError(f"{where}[{i}]: a success criterion is non-empty text")
        if len(item.strip()) > MAX_CRITERION_CHARS:
            raise ScenarioError(f"{where}[{i}]: a success criterion takes at most {MAX_CRITERION_CHARS} characters")
        out.append(item.strip())
    return out


def _bounded_text(raw: Any, where: str, what: str, limit: int) -> str:
    if isinstance(raw, bool) or isinstance(raw, (int, float)):
        raise ScenarioError(f"{where}: {what} is {json.dumps(raw)}, not text; in YAML, quote it")
    if not isinstance(raw, str) or not raw.strip():
        raise ScenarioError(f"{where}: {what} must be non-empty text")
    text = raw.strip()
    if len(text) > limit:
        raise ScenarioError(f"{where}: {what} is {len(text):,} characters; it takes at most {limit:,}")
    return text


def _whole(raw: Any, where: str, what: str, low: int, high: int) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ScenarioError(f"{where}: {what} must be a whole number from {low} to {high:,}")
    if not low <= raw <= high:
        raise ScenarioError(f"{where}: {what} must be from {low} to {high:,}, not {raw}")
    return raw


def _one_of(raw: Mapping[str, Any], where: str, *names: str) -> Any:
    given = [n for n in names if n in raw]
    if len(given) > 1:
        raise ScenarioError(f"{where}: {names[0]} is given twice")
    return raw[given[0]] if given else None


def _persona(raw: Any, where: str) -> str:
    text = _bounded_text(raw, where, "persona", MAX_PERSONA_CHARS)
    upper = text.upper().replace(" ", "_").replace("-", "_")
    return upper if upper in PERSONAS else text


def _facts(raw: Any, where: str) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ScenarioError(f"{where}: facts is a mapping of what the caller knows, such as "
                            "{order_number: \"4471\"} or {order_number: {value: \"4471\", revealWhen: ...}}")
    if len(raw) > MAX_FACTS:
        raise ScenarioError(f"{where}: {len(raw)} facts; a scenario takes at most {MAX_FACTS}")
    out: Dict[str, Any] = {}
    for key, fact in raw.items():
        name = _bounded_text(key, where, "a fact's name", MAX_FACT_CHARS)
        at = f"{where}.{name}"
        if not isinstance(fact, Mapping):
            out[name] = _bounded_text(fact, at, "the fact", MAX_FACT_CHARS)
            continue
        unknown = sorted(set(fact) - {"value", "revealWhen", "reveal_when"})
        if unknown:
            raise ScenarioError(f"{at}: unknown fact fields {unknown} (a fact takes value and revealWhen)")
        entry: Dict[str, Any] = {"value": _bounded_text(fact.get("value"), at, "value", MAX_FACT_CHARS)}
        reveal = _one_of(fact, at, "revealWhen", "reveal_when")
        if reveal is not None:
            entry["revealWhen"] = _bounded_text(reveal, at, "revealWhen", MAX_FACT_CHARS)
        out[name] = entry
    return out


def _interrupt(raw: Any, where: str, max_turns: int) -> Dict[str, int]:
    if not isinstance(raw, Mapping):
        raise ScenarioError(f"{where}: interrupt is a mapping such as {{turn: 1, afterBotMs: 1500}}")
    unknown = sorted(set(raw) - {"turn", "afterBotMs", "after_bot_ms"})
    if unknown:
        raise ScenarioError(f"{where}: unknown interrupt fields {unknown} (an interrupt takes turn and afterBotMs)")
    if "turn" not in raw:
        raise ScenarioError(f"{where}: interrupt needs the turn, the caller line (from 1) that cuts in")
    turn = raw["turn"]
    if isinstance(turn, bool) or not isinstance(turn, int) or not 1 <= turn <= max_turns:
        raise ScenarioError(f"{where}: interrupt.turn must be a caller line from 1 to maxTurns ({max_turns}), "
                            f"not {json.dumps(turn)}")
    after = _one_of(raw, where, "afterBotMs", "after_bot_ms")
    if after is None:
        raise ScenarioError(f"{where}: interrupt needs afterBotMs, how far into the bot's answer to cut in")
    return {"turn": turn, "afterBotMs": _ms(after, where, "interrupt.afterBotMs", 0, MAX_INTERRUPT_AFTER_BOT_MS)}


def _from_live(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """A scenario `generate_scenarios` wrote for a live simulated user, as a goal scenario."""
    out: Dict[str, Any] = {"name": raw.get("name"), "goal": raw.get("user_goal")}
    persona = raw.get("persona")
    if isinstance(persona, Mapping):
        persona = persona.get("name") or persona.get("description")
    if isinstance(persona, str) and persona.strip():
        out["persona"] = persona
    hidden = raw.get("hidden_info")
    if isinstance(hidden, Mapping) and hidden:
        facts: Dict[str, Any] = {}
        for key, fact in hidden.items():
            if isinstance(fact, Mapping):
                entry = {"value": fact.get("value")}
                reveal = fact.get("reveal_trigger") or fact.get("revealTrigger")
                if isinstance(reveal, str) and reveal.strip():
                    entry["revealWhen"] = reveal
                facts[key] = entry
            else:
                facts[key] = fact
        out["facts"] = facts
    opening = raw.get("starting_prompt") or raw.get("opening_message")
    if isinstance(opening, str) and opening.strip():
        out["opening"] = opening
    if raw.get("success_criteria"):
        out["successCriteria"] = raw["success_criteria"]
    return out


def _goal_scenario(raw: Mapping[str, Any], name: str, label: str) -> Dict[str, Any]:
    unknown = sorted(set(raw) - _GOAL_FIELDS)
    if unknown:
        raise ScenarioError(f"{label}: unknown scenario fields {unknown} (a scenario with a goal takes name, goal, "
                            "persona, facts, opening, successCriteria, maxTurns and interrupt)")
    out: Dict[str, Any] = {"name": name, "goal": _bounded_text(raw.get("goal"), label, "the goal", MAX_GOAL_CHARS)}
    if raw.get("persona") is not None:
        out["persona"] = _persona(raw["persona"], label)
    if raw.get("facts") is not None:
        facts = _facts(raw["facts"], f"{label}: facts")
        if facts:
            out["facts"] = facts
    if raw.get("opening") is not None:
        out["opening"] = _bounded_text(raw["opening"], label, "the opening", MAX_OPENING_CHARS)
    criteria = _one_of(raw, label, "successCriteria", "success_criteria")
    if criteria is not None:
        kept = _criteria(criteria, f"{label}: successCriteria")
        if kept:
            out["successCriteria"] = kept
    max_turns = _one_of(raw, label, "maxTurns", "max_turns")
    if max_turns is not None:
        out["maxTurns"] = _whole(max_turns, label, "maxTurns", 1, MAX_TURNS)
    if raw.get("interrupt") is not None:
        out["interrupt"] = _interrupt(raw["interrupt"], label, out.get("maxTurns", DEFAULT_MAX_TURNS))
    return out


def _scenario(raw: Any, where: str) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ScenarioError(f"{where}: a scenario is a mapping with a name, and lines or a goal")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ScenarioError(f"{where}: a scenario needs a name")
    name = name.strip()
    if len(name) > MAX_NAME_CHARS:
        raise ScenarioError(f"{where}: the name takes at most {MAX_NAME_CHARS} characters")
    label = f"{where} ({json.dumps(name)})"
    generated = raw.get("lines") is None and "turns" in raw
    if generated and not raw.get("turns") and raw.get("user_goal"):
        return _goal_scenario(_from_live(dict(raw, name=name)), name, label)
    if not generated:
        if raw.get("goal") is not None and raw.get("lines") is not None:
            raise ScenarioError(f"{label}: a scenario has a goal (the caller talks its way to it) or lines (the "
                                "caller says exactly these), never both")
        if raw.get("goal") is not None:
            return _goal_scenario(raw, name, label)
        unknown = sorted(set(raw) - _SCENARIO_FIELDS)
        if unknown:
            goal_only = sorted(set(unknown) & _GOAL_FIELDS)
            hint = (f"; {', '.join(goal_only)} belong to a scenario with a goal" if goal_only
                    else "")
            raise ScenarioError(f"{label}: unknown scenario fields {unknown} (a scenario with lines takes name, "
                                f"lines and successCriteria{hint})")
        if raw.get("lines") is None:
            raise ScenarioError(f"{label}: a scenario needs lines (what the caller says) or a goal (what the "
                                "caller wants)")
    lines = raw.get("turns") if generated else raw.get("lines")
    key = "turns" if generated else "lines"
    if not isinstance(lines, (list, tuple)) or not lines:
        hint = (" (a scenario generated for a live simulated user has a user_goal instead; this one has neither)"
                if generated else "")
        raise ScenarioError(f"{label}: a scenario needs a non-empty {key} list{hint}")
    if len(lines) > MAX_LINES:
        raise ScenarioError(f"{label}: {len(lines)} lines; a scenario takes at most {MAX_LINES}")
    out: Dict[str, Any] = {"name": name,
                           "lines": [_line(line, f"{label}: {key}[{i}]") for i, line in enumerate(lines)]}
    if "successCriteria" in raw and "success_criteria" in raw and not generated:
        raise ScenarioError(f"{label}: successCriteria is given twice")
    criteria = raw.get("successCriteria", raw.get("success_criteria"))
    if criteria is not None:
        kept = _criteria(criteria, f"{label}: successCriteria")
        if kept:
            out["successCriteria"] = kept
    return out


def normalize_scenarios(items: Any, where: str = "scenarios") -> List[Dict[str, Any]]:
    """`items` (mappings as written, or as `generate_scenarios` returns them) in the API's shape.
    Raises ScenarioError naming `where`, the scenario and the line."""
    if isinstance(items, Mapping) and "scenarios" in items:
        extra = sorted(set(items) - {"scenarios"})
        if extra:
            raise ScenarioError(f"{where}: unknown top-level fields {extra} (the file holds scenarios only; the "
                                "bot's URL and format are command-line options)")
        items = items["scenarios"]
    if not isinstance(items, (list, tuple)) or not items:
        raise ScenarioError(f"{where}: no scenarios (expected a list, or a mapping with a scenarios list)")
    if len(items) > MAX_SCENARIOS:
        raise ScenarioError(f"{where}: {len(items)} scenarios; a run takes at most {MAX_SCENARIOS}")
    return [_scenario(raw, f"{where}: scenarios[{i}]") for i, raw in enumerate(items)]


def _parse_yaml(text: str, path: str) -> Any:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        raise ScenarioError(
            f"{path}: reading YAML needs PyYAML: pip install 'syntropylabs-evalkit[voice]' (or pyyaml), "
            "or write the scenarios as JSON") from None
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ScenarioError(f"{path}: not valid YAML: {exc}") from None


def load_scenarios(path: PathLike) -> List[Dict[str, Any]]:
    """The scenarios of a ``.yaml``/``.yml`` or ``.json`` file, in the API's shape (another suffix is read
    as JSON, else YAML). Raises ScenarioError, naming the file, when it cannot be read or used."""
    name = os.fspath(path)
    try:
        with open(name, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise ScenarioError(f"{name}: {exc.strerror or exc}") from None
    suffix = os.path.splitext(name)[1].lower()
    if suffix in (".yaml", ".yml"):
        data = _parse_yaml(text, name)
    else:
        try:
            data = json.loads(text)
        except ValueError as exc:
            if suffix == ".json":
                raise ScenarioError(f"{name}: not valid JSON: {exc}") from None
            data = _parse_yaml(text, name)
    return normalize_scenarios(data, where=name)


def is_inline(items: Any) -> bool:
    """Whether `items` are inline scenarios (mappings) rather than a set's scenario ids (strings)."""
    return isinstance(items, (list, tuple)) and bool(items) and all(isinstance(i, Mapping) for i in items)


# `evalkit voice-test init` ---------------------------------------------------------------------------

STARTER_YAML = """\
# voice-tests.yaml: callers for `evalkit voice-test run --scenarios voice-tests.yaml`
#
# Each scenario is one call. A caller with a goal talks to your bot until the goal is met or it gives
# up, answering what your bot actually says. A caller with lines says exactly those lines.
# Without a file, `evalkit voice-test run --about "what your bot does"` writes the callers for you.
# Limits: 50 scenarios; a goal of 1,000 characters; 20 turns a call; 40 lines, 500 characters each.
scenarios:
  - name: Refund for a duplicate charge
    goal: Get the second charge on order 4471 refunded and find out when the money arrives
    persona: IMPATIENT
    facts:
      order_number: "4471"
    successCriteria:
      - The agent refunds the second charge
      - The agent says when the refund will arrive
    interrupt: { turn: 1, afterBotMs: 1500 }
  - name: Refund outside the return window
    goal: Get money back for headphones bought two months ago that stopped working last week
    persona: NOVICE
    successCriteria:
      - The agent explains the 30-day return policy
      - The agent offers store credit or a repair
  - name: Opening hours, exact lines
    lines:
      - Hi, what are your opening hours on Saturday?
      - Thanks, that's all.
"""


def starter_scenarios() -> List[Dict[str, Any]]:
    """`STARTER_YAML`'s scenarios in the API's shape, for a JSON starter file."""
    return normalize_scenarios([
        {"name": "Refund for a duplicate charge",
         "goal": "Get the second charge on order 4471 refunded and find out when the money arrives",
         "persona": "IMPATIENT",
         "facts": {"order_number": "4471"},
         "successCriteria": ["The agent refunds the second charge", "The agent says when the refund will arrive"],
         "interrupt": {"turn": 1, "afterBotMs": 1500}},
        {"name": "Refund outside the return window",
         "goal": "Get money back for headphones bought two months ago that stopped working last week",
         "persona": "NOVICE",
         "successCriteria": ["The agent explains the 30-day return policy",
                             "The agent offers store credit or a repair"]},
        {"name": "Opening hours, exact lines",
         "lines": ["Hi, what are your opening hours on Saturday?", "Thanks, that's all."]},
    ], where="starter")


def write_starter(path: PathLike = DEFAULT_FILE, force: bool = False) -> str:
    """Writes the starter scenarios to `path` (JSON for a ``.json`` path, else YAML) and returns the
    path. Raises FileExistsError when the file exists and `force` is false."""
    name = os.fspath(path)
    if os.path.exists(name) and not force:
        raise FileExistsError(name)
    if os.path.splitext(name)[1].lower() == ".json":
        text = json.dumps({"scenarios": starter_scenarios()}, indent=2) + "\n"
    else:
        text = STARTER_YAML
    directory = os.path.dirname(os.path.abspath(name))
    os.makedirs(directory, exist_ok=True)
    with open(name, "w", encoding="utf-8") as fh:
        fh.write(text)
    return name


def count_lines(scenarios: Sequence[Mapping[str, Any]]) -> int:
    return sum(len(s.get("lines") or ()) for s in scenarios)


def is_goal(scenario: Mapping[str, Any]) -> bool:
    """A scenario played as a dynamic call: it has a goal, not lines."""
    return scenario.get("goal") is not None


def with_max_turns(scenarios: Sequence[Mapping[str, Any]], max_turns: int) -> List[Dict[str, Any]]:
    """`scenarios` with `max_turns` as the maxTurns of each goal scenario that sets none (``--max-turns``).
    Raises ScenarioError when a scenario's planned interruption falls after it."""
    out: List[Dict[str, Any]] = []
    for i, scenario in enumerate(scenarios):
        scenario = dict(scenario)
        if is_goal(scenario) and "maxTurns" not in scenario:
            interrupt = scenario.get("interrupt") or {}
            if interrupt.get("turn", 0) > max_turns:
                raise ScenarioError(f"scenarios[{i}] ({json.dumps(scenario.get('name'))}): its interrupt is planned "
                                    f"before turn {interrupt['turn']}, after the last of {max_turns} turns "
                                    "(--max-turns)")
            scenario["maxTurns"] = max_turns
        out.append(scenario)
    return out


def describe(scenarios: Optional[Sequence[Mapping[str, Any]]]) -> str:
    if not scenarios:
        return "no scenarios"
    n = len(scenarios)
    head = f"{n} scenario{'s' if n != 1 else ''}"
    goals = sum(1 for s in scenarios if is_goal(s))
    if not goals:
        return f"{head}, {count_lines(scenarios)} lines"
    scripted = n - goals
    parts = [f"{goals} with a goal"]
    if scripted:
        lines = count_lines(scenarios)
        parts.append(f"{scripted} with {lines} line{'s' if lines != 1 else ''}")
    return f"{head}: {', '.join(parts)}"
