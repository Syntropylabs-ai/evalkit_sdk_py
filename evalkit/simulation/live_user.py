"""The simulated user that plays a scenario's turns after its scripted ones.

Each turn is one POST /simulations/user-turn: the platform plays the user from
the scenario's persona, goal and hidden facts, sees the conversation so far
(with the tool calls the agent reported), and answers with the next message or
that the user is done. Each call is recorded as its own span marked
``evalkit.simulation.role: user``, which the platform never judges as the
agent's turn, tool call or output.
"""
from __future__ import annotations

import json
import os
import secrets
import time
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

from .._version import SDK_VERSION
from ..core.api import post_json
from ..core.context import current_span_ctx
from ..core.types import TraceEnvelope
from .types import AgentTurnResult, SimContext, resolve_persona

# The platform plays at most this many user turns of one scenario.
MAX_LIVE_USER_TURNS = 50
SIMULATED_USER_SPAN = "simulated_user"
ROLE_KEY = "evalkit.simulation.role"
_TOOL_FIELD_CAP = 2000


class SimulatedUserError(RuntimeError):
    """The simulated user could not play its turn (no model, a refused key,
    an unusable answer). Ends the scenario as ERROR with the platform's reason."""


def explicit_turns(scenario: Dict[str, Any]) -> List[str]:
    """The user messages the scenario scripts: strings, or {role: user, content} messages."""
    msgs: List[str] = []
    for t in scenario.get("turns") or scenario.get("messages") or []:
        if isinstance(t, str) and t.strip():
            msgs.append(t)
        elif isinstance(t, dict) and t.get("content") and t.get("role", "user") == "user":
            msgs.append(str(t["content"]))
    return msgs


def scripted_turns(scenario: Dict[str, Any]) -> List[str]:
    """The scripted turns, falling back to the opening message alone."""
    msgs = explicit_turns(scenario)
    if not msgs:
        opening = scenario.get("starting_prompt") or scenario.get("opening_message")
        if opening:
            msgs = [str(opening)]
    return msgs


def wants_live_user(scenario: Dict[str, Any], live_user: Optional[bool]) -> bool:
    """live_user=None plays a live user for a scenario that states a goal and
    scripts no turns; a turns-only scenario replays its turns as written."""
    if live_user is not None:
        return bool(live_user)
    has_goal = bool(scenario.get("user_goal") or scenario.get("conversation_plan"))
    return has_goal and not explicit_turns(scenario)


def opening_message(scenario: Dict[str, Any]) -> str:
    scripted = scripted_turns(scenario)
    if scripted:
        return scripted[0]
    return str(scenario.get("user_goal") or scenario.get("name") or "Hello.")


def reported_tool_calls(result: Any) -> List[Dict[str, Any]]:
    """The tool calls an AgentTurnResult reports, in the shape the platform reads:
    {name, arguments?, result?, error?}. Anything else reports none."""
    if not isinstance(result, AgentTurnResult):
        return []
    calls: List[Dict[str, Any]] = []
    for call in result.tool_calls or []:
        if not isinstance(call, dict):
            continue
        name = call.get("name") or call.get("tool")
        if not name:
            continue
        entry: Dict[str, Any] = {"name": str(name)}
        for key, sources in (
            ("arguments", ("arguments", "args", "input")),
            ("result", ("result", "output")),
        ):
            for src in sources:
                if call.get(src) is not None:
                    entry[key] = _capped(call[src])
                    break
        if call.get("error"):
            entry["error"] = str(call["error"])[:_TOOL_FIELD_CAP]
        calls.append(entry)
    return calls


def _capped(value: Any) -> Any:
    try:
        text = value if isinstance(value, str) else json.dumps(value, default=str)
    except Exception:
        text = str(value)
    if len(text) <= _TOOL_FIELD_CAP:
        return value if not isinstance(value, str) else text
    return text[:_TOOL_FIELD_CAP] + "..."


def _scenario_brief(scenario: Dict[str, Any], max_turns: int) -> Dict[str, Any]:
    brief: Dict[str, Any] = {"maxTurns": max_turns}
    persona = scenario.get("persona")
    if isinstance(persona, dict):
        name, desc = resolve_persona(persona)
        brief["persona"] = desc if desc and desc != name else name
    elif isinstance(persona, str) and persona.strip():
        brief["persona"] = persona.strip()
    for key, field in (
        ("userGoal", "user_goal"),
        ("conversationPlan", "conversation_plan"),
        ("name", "name"),
    ):
        value = scenario.get(field)
        if isinstance(value, str) and value.strip():
            brief[key] = value.strip()
    scenario_type = scenario.get("scenario_type") or scenario.get("category")
    if isinstance(scenario_type, str) and scenario_type.strip():
        brief["scenarioType"] = scenario_type.strip()
    if scenario.get("hidden_info"):
        brief["hiddenInfo"] = scenario["hidden_info"]
    fixtures = scenario.get("tool_fixtures") or scenario.get("fixtures")
    if isinstance(fixtures, list) and fixtures:
        brief["fixtures"] = fixtures
    min_turns = scenario.get("min_user_turns")
    if isinstance(min_turns, int) and min_turns > 0:
        brief["minUserTurns"] = min(min_turns, max_turns)
    return brief


class Conversation:
    """One scenario's user side: the scripted turns first, then the turn_events
    scheduled for a turn, then (for a live scenario) the simulated user, until
    the user is done or max_turns user turns have been played."""

    def __init__(
        self,
        client: Any,
        scenario: Dict[str, Any],
        *,
        max_turns: int,
        live: bool,
        user_model: Optional[str],
        api_url: Optional[str],
    ) -> None:
        self.client = client
        self.scenario = scenario
        self.live = live
        self.max_turns = max(1, int(max_turns))
        self.live_cap = min(self.max_turns, MAX_LIVE_USER_TURNS)
        self.user_model = user_model
        self.api_url = api_url
        self.scripted = scripted_turns(scenario)
        self.history: List[Dict[str, Any]] = []
        self.user_messages: List[str] = []
        self.ended_by: Optional[str] = None
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        self._unpriced = False

    @property
    def turns(self) -> int:
        return len(self.user_messages)

    def opening(self) -> Optional[str]:
        if self.scripted:
            return self.scripted[0]
        return opening_message(self.scenario) if self.live else None

    def begin_turn(self, ctx: SimContext, sim: Any, message: str) -> None:
        self.user_messages.append(message)
        sim.turn = ctx.turn = self.turns
        ctx.message = message
        self.history.append({"role": "user", "content": message})

    def end_turn(self, text: str, result: Any) -> None:
        reply: Dict[str, Any] = {"role": "assistant", "content": text}
        calls = reported_tool_calls(result)
        if calls:
            reply["toolCalls"] = calls
        self.history.append(reply)

    def next_message(self) -> Optional[str]:
        """The user's next message after the agent's latest reply, or None
        when the conversation is over. Blocks on the platform for a live turn."""
        n = self.turns + 1
        if n > self.max_turns:
            self.ended_by = "max_turns"
            return None
        if n <= len(self.scripted):
            return self.scripted[n - 1]
        for event in self.scenario.get("turn_events") or []:
            if isinstance(event, dict) and event.get("message") and _int(event.get("at_user_turn")) == n:
                return str(event["message"])
        if not self.live:
            self.ended_by = "script"
            return None
        if n > self.live_cap:
            self.ended_by = "max_turns"
            return None
        message = self._ask(n)
        if message is None:
            self.ended_by = "user"
        return message

    def summary(self) -> Dict[str, Any]:
        usage = dict(self.usage)
        if self._unpriced:
            usage["cost_usd"] = None
        return usage

    def _ask(self, turn: int) -> Optional[str]:
        base = (self.api_url or os.getenv("EVALKIT_API_URL") or self.client._opts.api_url).rstrip("/")
        body: Dict[str, Any] = {
            "scenario": _scenario_brief(self.scenario, self.live_cap),
            "history": self.history,
            "turn": turn,
        }
        if self.user_model:
            body["model"] = self.user_model
        started = time.time()
        start_iso = _iso(started)
        answer: Dict[str, Any] = {}
        failure: Optional[str] = None
        try:
            answer = post_json(
                self.client, base + "/simulations/user-turn", body, timeout=120, span_name="", trace=False
            )
        except urllib.error.HTTPError as exc:
            failure = f"{exc.code} {_platform_message(exc)}"
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        usage = answer.get("usage") if isinstance(answer, dict) else None
        self._count(usage)
        done = bool(answer.get("done")) if failure is None else False
        message = answer.get("message") if failure is None and not done else None
        if failure is None and not done and not (isinstance(message, str) and message.strip()):
            failure = "the platform answered without a message"
        self._record_span(turn, start_iso, started, done, message, usage, failure)
        if failure is not None:
            raise SimulatedUserError(f"simulated user failed at turn {turn}: {failure}"[:500])
        return None if done else str(message)

    def _count(self, usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        self.usage["calls"] += 1
        self.usage["input_tokens"] += int(usage.get("inputTokens") or 0)
        self.usage["output_tokens"] += int(usage.get("outputTokens") or 0)
        cost = usage.get("costUsd")
        if cost is None:
            self._unpriced = True
        else:
            self.usage["cost_usd"] += float(cost)

    def _record_span(
        self,
        turn: int,
        start_iso: str,
        started: float,
        done: bool,
        message: Optional[str],
        usage: Any,
        failure: Optional[str],
    ) -> None:
        """One span per simulated user turn, under the scenario's root. It names
        no agent: the simulated user is the test, not the agent under test."""
        try:
            parent = current_span_ctx()
            if parent is None:
                return
            attrs: Dict[str, Any] = {
                **self.client._meta_attrs(),
                ROLE_KEY: "user",
                "evalkit.simulated_user.turn": turn,
                "evalkit.simulated_user.done": done,
            }
            if self.user_model:
                attrs["evalkit.simulated_user.model"] = self.user_model
            if isinstance(usage, dict):
                attrs["evalkit.simulated_user.input_tokens"] = int(usage.get("inputTokens") or 0)
                attrs["evalkit.simulated_user.output_tokens"] = int(usage.get("outputTokens") or 0)
                if usage.get("costUsd") is not None:
                    attrs["evalkit.simulated_user.cost_usd"] = float(usage["costUsd"])
            if failure is not None:
                attrs["error.message"] = failure[:2000]
            opts = self.client._opts
            self.client.enqueue(TraceEnvelope(
                trace_id=parent.trace_id,
                span_id=secrets.token_hex(8),
                parent_span_id=parent.span_id,
                span_type="function_call",
                status="ERROR" if failure is not None else "OK",
                status_message=failure[:500] if failure is not None else None,
                operation=SIMULATED_USER_SPAN,
                completion=message if isinstance(message, str) else None,
                service_name=opts.service_name,
                attributes=attrs,
                latency_ms=int((time.time() - started) * 1000),
                start_time=start_iso,
                end_time=_iso(time.time()),
                created_at=start_iso,
                environment=opts.environment,
                app_version=opts.app_version,
                session_id=parent.session_id,
                user_id=parent.user_id,
                # Empty, not None: None would inherit the agent under test from the scenario's root.
                agent_name="",
                sdk_version=SDK_VERSION,
            ))
        except Exception:
            pass


def _platform_message(exc: urllib.error.HTTPError) -> str:
    text = getattr(exc, "response_text", None) or ""
    try:
        body = json.loads(text)
        message = body.get("message") if isinstance(body, dict) else None
        if isinstance(message, list):
            message = "; ".join(str(m) for m in message)
        if message:
            return str(message)
    except Exception:
        pass
    return text[:300] or str(exc.reason)


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _iso(ts: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


__all__: Tuple[str, ...] = (
    "Conversation",
    "SimulatedUserError",
    "MAX_LIVE_USER_TURNS",
    "wants_live_user",
)
