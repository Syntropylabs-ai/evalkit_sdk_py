from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import secrets
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from ..core.context import SimulationContext, _current_simulation, _current_span
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


@contextmanager
def _bind_scenario_root(root: Any):
    token = _current_span.set(root)
    try:
        yield
    finally:
        _current_span.reset(token)


def _run_coro_blocking(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: Dict[str, Any] = {}

    def _worker() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except Exception as exc:
            result["error"] = exc

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        return _run_coro_blocking(value)
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
    span_tools: List[Dict[str, Any]] = []
    event_tools: List[Dict[str, Any]] = []
    for env in sim.collector:
        span_type = getattr(env, "span_type", None)
        attrs = env.attributes or {}
        if span_type == "tool_call":
            name = attrs.get("tool.name") or attrs.get("gen_ai.tool.name") or env.operation
            if name:
                span_tools.append({"name": name, "args": attrs.get("tool.arguments")})
        elif span_type == "llm_call":
            for ev in (getattr(env, "events", None) or []):
                ev_attrs = getattr(ev, "attributes", None) or {}
                name = ev_attrs.get("gen_ai.tool.name")
                if name:
                    event_tools.append({"name": name, "args": ev_attrs.get("gen_ai.tool.call.arguments")})
    return span_tools or event_tools


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


def _begin_scenario(client: Any, scenario: Dict[str, Any], cfg: Dict[str, Any]):
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
    _trace_id, end_trace, root = client.start_trace(f"scenario:{name}", session_id=ctx.session_id, join_trace_id=cfg["trace_id"])
    return name, scenario_id, ctx, sim, turn_messages, end_trace, root


def _finish_scenario(
    scenario: Dict[str, Any],
    scenario_id: str,
    name: str,
    sim: SimulationContext,
    texts: List[str],
    explicit_tools: List[Dict[str, Any]],
    metrics: Optional[List[str]],
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    actual_tools = explicit_tools or _tools_from_collector(sim)
    scores = _score(scenario, "\n".join(texts), actual_tools, metrics)
    return scores, actual_tools


def _result(scenario_id: str, name: str, status: str, turns: int, scores: Dict[str, float], error: Optional[str]) -> Dict[str, Any]:
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


def _run_scenario(client: Any, entrypoint: Entrypoint, scenario: Dict[str, Any], **cfg: Any) -> Dict[str, Any]:
    name, scenario_id, ctx, sim, turn_messages, end_trace, root = _begin_scenario(client, scenario, cfg)
    with _bind_scenario_root(root):
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

                scores, _ = _finish_scenario(scenario, scenario_id, name, sim, texts, explicit_tools, cfg["metrics"])
            except Exception as exc:
                status, error = "ERROR", str(exc)
                _log.warning("[evalkit] scenario '%s' failed: %s", name, exc)
            finally:
                end_trace(status)

        return _result(scenario_id, name, status, turns, scores, error)


async def _run_scenario_async(client: Any, entrypoint: Entrypoint, scenario: Dict[str, Any], **cfg: Any) -> Dict[str, Any]:
    name, scenario_id, ctx, sim, turn_messages, end_trace, root = _begin_scenario(client, scenario, cfg)
    with _bind_scenario_root(root):
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
                    result = entrypoint(ctx)
                    if inspect.isawaitable(result):
                        result = await result
                    if isinstance(result, AgentTurnResult):
                        texts.append(result.text)
                        explicit_tools.extend(result.tool_calls or [])
                    else:
                        texts.append(str(result))
                    turns += 1

                scores, _ = _finish_scenario(scenario, scenario_id, name, sim, texts, explicit_tools, cfg["metrics"])
            except Exception as exc:
                status, error = "ERROR", str(exc)
                _log.warning("[evalkit] scenario '%s' failed: %s", name, exc)
            finally:
                end_trace(status)

        return _result(scenario_id, name, status, turns, scores, error)


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
    from ..core.context import get_or_create_ambient_trace_id

    client = _get_instance()
    cfg = {
        "simulation_id": "sim_" + secrets.token_hex(8),
        "run_id": "run_" + secrets.token_hex(8),
        "metrics": metrics,
        "max_turns": max_turns,
        "tags": tags,
        "safe_mode": safe_mode,
    }
    scenario_list = list(scenarios)

    # One trace per run, one root and one session per scenario inside it.
    cfg["trace_id"] = get_or_create_ambient_trace_id()

    if inspect.iscoroutinefunction(entrypoint):
        async def _run_all() -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for sc in scenario_list:
                out.append(await asyncio.ensure_future(_run_scenario_async(client, entrypoint, sc, **cfg)))
            return out

        results = _run_coro_blocking(_run_all())
    else:
        results = [
            contextvars.copy_context().run(_run_scenario, client, entrypoint, sc, **cfg)
            for sc in scenario_list
        ]

    try:
        client.flush()
    except Exception:
        pass

    return {"simulation_id": cfg["simulation_id"], "run_id": cfg["run_id"], "results": results}
