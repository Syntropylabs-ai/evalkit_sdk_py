from __future__ import annotations

import inspect
import logging
import secrets
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from ..core.context import SimulationContext, _current_simulation
from ..evaluate import evaluate
from .types import AgentTurnResult, SimContext, resolve_persona

_log = logging.getLogger("evalkit")

Entrypoint = Callable[[SimContext], Union[str, AgentTurnResult, Any]]


@contextmanager
def with_simulation(sim: SimulationContext):
    token = _current_simulation.set(sim)
    try:
        yield sim
    finally:
        _current_simulation.reset(token)


def _resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        import asyncio

        try:
            return asyncio.run(value)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(value)
            finally:
                loop.close()
    return value


def _user_turns(scenario: Dict[str, Any]) -> List[str]:
    raw = scenario.get("turns") or scenario.get("messages") or []
    msgs: List[str] = []
    for t in raw:
        if isinstance(t, str):
            msgs.append(t)
        elif isinstance(t, dict) and t.get("content") and t.get("role", "user") == "user":
            msgs.append(str(t["content"]))
    if not msgs:
        sp = scenario.get("starting_prompt")
        if sp:
            msgs = [str(sp)]
    return msgs


def _tools_from_collector(sim: SimulationContext) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    for env in sim.collector:
        if getattr(env, "span_type", None) != "tool_call":
            continue
        attrs = env.attributes or {}
        name = attrs.get("tool.name") or env.operation
        if name:
            tools.append({"name": name, "args": attrs.get("tool.arguments")})
    return tools


def _score(
    scenario: Dict[str, Any],
    output: str,
    tool_calls: List[Dict[str, Any]],
    metrics: Optional[List[str]],
) -> Dict[str, float]:
    constraints = dict(scenario.get("constraints") or {})
    keywords = scenario.get("target_keywords") or []
    if keywords:
        existing = constraints.get("required_terms") or constraints.get("requiredTerms") or []
        constraints["required_terms"] = list(dict.fromkeys([*existing, *keywords]))
    try:
        return evaluate(
            output=output,
            tool_calls=tool_calls,
            expected_tools=scenario.get("expected_tools") or [],
            constraints=constraints or None,
            metrics=metrics,
        )
    except Exception as exc:
        _log.warning("[evalkit] scoring failed: %s", exc)
        return {}


def _run_scenario(client: Any, entrypoint: Entrypoint, scenario: Dict[str, Any], **cfg: Any) -> Dict[str, Any]:
    name = scenario.get("name", "scenario")
    scenario_id = scenario.get("scenario_id") or ("scn_" + secrets.token_hex(6))
    persona_id, _ = resolve_persona(scenario.get("persona"))
    state = dict((scenario.get("setup") or {}).get("state") or {})
    state["__safe_mode__"] = cfg["safe_mode"]
    ctx = SimContext(message="", session_id="sess_" + secrets.token_hex(6), state=state)

    sim = SimulationContext(
        simulation_id=cfg["simulation_id"],
        run_id=cfg["run_id"],
        scenario_id=scenario_id,
        scenario_name=name,
        persona=persona_id,
        tags=cfg["tags"],
    )

    turn_messages = _user_turns(scenario)[: cfg["max_turns"]]
    trace_id, end_trace, _root = client.start_trace(f"scenario:{name}", session_id=ctx.session_id)

    status, error, turns = "OK", None, 0
    texts: List[str] = []
    explicit_tools: List[Dict[str, Any]] = []
    scores: Dict[str, float] = {}

    with with_simulation(sim):
        try:
            for i, msg in enumerate(turn_messages):
                sim.turn = i + 1
                ctx.turn = i + 1
                ctx.message = msg
                result = _resolve(entrypoint(ctx))
                if isinstance(result, AgentTurnResult):
                    texts.append(result.text)
                    explicit_tools.extend(result.tool_calls or [])
                else:
                    texts.append(str(result))
                turns += 1

            actual_tools = explicit_tools or _tools_from_collector(sim)
            scores = _score(scenario, "\n".join(texts), actual_tools, cfg["metrics"])
        except Exception as exc:
            status, error = "ERROR", str(exc)
            _log.warning("[evalkit] scenario '%s' failed: %s", name, exc)
        finally:
            end_trace(status)

    out: Dict[str, Any] = {
        "scenario_id": scenario_id,
        "name": name,
        "status": status,
        "turns": turns,
        "scores": scores,
    }
    if error:
        out["error"] = error
    return out


def simulate_user(
    entrypoint: Entrypoint,
    scenarios: Iterable[Dict[str, Any]],
    *,
    metrics: Optional[List[str]] = None,
    max_turns: int = 12,
    tags: Optional[List[str]] = None,
    safe_mode: bool = True,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    from .. import _get_instance

    client = _get_instance()
    cfg = {
        "simulation_id": "sim_" + secrets.token_hex(8),
        "run_id": "run_" + secrets.token_hex(8),
        "metrics": metrics,
        "max_turns": max_turns,
        "tags": tags,
        "safe_mode": safe_mode,
    }
    results = [_run_scenario(client, entrypoint, sc, **cfg) for sc in scenarios]
    return {"simulation_id": cfg["simulation_id"], "run_id": cfg["run_id"], "results": results}
