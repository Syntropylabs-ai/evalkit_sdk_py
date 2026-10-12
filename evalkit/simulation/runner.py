from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import secrets
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from ..core.client import MAX_STATUS_MESSAGE
from ..core.context import SimulationContext, _current_simulation, _current_span
from ..evaluate import evaluate
from .live_user import Conversation, wants_live_user
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


def scenario_session_id(simulation_id: str, run_id: str, scenario_id: str) -> str:
    """One session per scenario execution. A session id names its run and
    scenario, so the platform groups a scenario's spans into one conversation
    and can tell two runs of the same scenario apart."""
    return f"{simulation_id}:{run_id}:{scenario_id}"


def _begin_scenario(client: Any, scenario: Dict[str, Any], cfg: Dict[str, Any]):
    name = scenario.get("name", "scenario")
    scenario_id = scenario.get("scenario_id") or ("scn_" + secrets.token_hex(6))
    persona_id, _ = resolve_persona(scenario.get("persona"))
    state = dict((scenario.get("setup") or {}).get("state") or {})
    state["__safe_mode__"] = cfg["safe_mode"]
    session_id = scenario_session_id(cfg["simulation_id"], cfg["run_id"], scenario_id)
    ctx = SimContext(message="", session_id=session_id, state=state)

    sim = SimulationContext(
        simulation_id=cfg["simulation_id"],
        run_id=cfg["run_id"],
        scenario_id=scenario_id,
        scenario_name=name,
        persona=persona_id,
        tags=cfg["tags"],
    )
    conversation = Conversation(
        client,
        scenario,
        max_turns=cfg["max_turns"],
        live=wants_live_user(scenario, cfg["live_user"]),
        user_model=cfg["user_model"],
        api_url=cfg["api_url"],
    )
    _trace_id, end_trace, root = client.start_trace(
        f"scenario:{name}", session_id=session_id, agent=cfg.get("agent"), join_trace_id=cfg["trace_id"]
    )
    return name, scenario_id, ctx, sim, conversation, end_trace, root


def _end_scenario_root(
    end_trace: Callable[..., None],
    status: str,
    user_messages: List[str],
    texts: List[str],
    failure: Optional[BaseException],
) -> None:
    """Ends the scenario's root span with what the platform lists for the
    scenario: the opening message as its input, the agent's last reply as its
    output, and the error that ended it, so a failed probe or scenario reads
    as why it failed without opening its children."""
    last_reply = texts[-1] if texts else None
    fields: Dict[str, Any] = {
        "prompt": user_messages[0] if user_messages else None,
        "completion": last_reply if isinstance(last_reply, str) else None,
    }
    if failure is not None:
        message = (str(failure) or type(failure).__name__)[:MAX_STATUS_MESSAGE]
        fields["status_message"] = message
        fields["attributes"] = {"error.type": type(failure).__name__, "error.message": message}
    end_trace(status, **fields)


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


def _result(
    scenario_id: str,
    name: str,
    status: str,
    scores: Dict[str, float],
    error: Optional[str],
    conversation: Conversation,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "scenario_id": scenario_id,
        "name": name,
        "status": status,
        "turns": conversation.turns,
        "scores": scores,
    }
    if conversation.live:
        out["transcript"] = [{"role": m["role"], "content": m["content"]} for m in conversation.history]
        out["ended_by"] = conversation.ended_by or "error"
        out["simulated_user_usage"] = conversation.summary()
    if error:
        out["error"] = error
    return out


def _reply_text(result: Any, explicit_tools: List[Dict[str, Any]]) -> str:
    if isinstance(result, AgentTurnResult):
        explicit_tools.extend(result.tool_calls or [])
        return result.text
    return str(result)


def _run_scenario(client: Any, entrypoint: Entrypoint, scenario: Dict[str, Any], **cfg: Any) -> Dict[str, Any]:
    name, scenario_id, ctx, sim, conversation, end_trace, root = _begin_scenario(client, scenario, cfg)
    with _bind_scenario_root(root):
        status, error = "OK", None
        failure: Optional[BaseException] = None
        texts: List[str] = []
        explicit_tools: List[Dict[str, Any]] = []
        scores: Dict[str, float] = {}

        with with_simulation(sim):
            try:
                message = conversation.opening()
                while message is not None:
                    conversation.begin_turn(ctx, sim, message)
                    result = _resolve(entrypoint(ctx))
                    texts.append(_reply_text(result, explicit_tools))
                    conversation.end_turn(texts[-1], result)
                    message = conversation.next_message()

                scores, _ = _finish_scenario(scenario, scenario_id, name, sim, texts, explicit_tools, cfg["metrics"])
            except Exception as exc:
                status, error, failure = "ERROR", str(exc), exc
                _log.warning("[evalkit] scenario '%s' failed: %s", name, exc)
            finally:
                _end_scenario_root(end_trace, status, conversation.user_messages, texts, failure)

        return _result(scenario_id, name, status, scores, error, conversation)


async def _run_scenario_async(client: Any, entrypoint: Entrypoint, scenario: Dict[str, Any], **cfg: Any) -> Dict[str, Any]:
    name, scenario_id, ctx, sim, conversation, end_trace, root = _begin_scenario(client, scenario, cfg)
    with _bind_scenario_root(root):
        status, error = "OK", None
        failure: Optional[BaseException] = None
        texts: List[str] = []
        explicit_tools: List[Dict[str, Any]] = []
        scores: Dict[str, float] = {}

        with with_simulation(sim):
            try:
                message = conversation.opening()
                while message is not None:
                    conversation.begin_turn(ctx, sim, message)
                    result = entrypoint(ctx)
                    if inspect.isawaitable(result):
                        result = await result
                    texts.append(_reply_text(result, explicit_tools))
                    conversation.end_turn(texts[-1], result)
                    # The simulated user's call blocks on the platform; off the loop, in this context.
                    message = await asyncio.to_thread(conversation.next_message)

                scores, _ = _finish_scenario(scenario, scenario_id, name, sim, texts, explicit_tools, cfg["metrics"])
            except Exception as exc:
                status, error, failure = "ERROR", str(exc), exc
                _log.warning("[evalkit] scenario '%s' failed: %s", name, exc)
            finally:
                _end_scenario_root(end_trace, status, conversation.user_messages, texts, failure)

        return _result(scenario_id, name, status, scores, error, conversation)


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
    agent: Optional[str] = None,
    user_model: Optional[str] = None,
    live_user: Optional[bool] = None,
    api_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Plays each scenario against `entrypoint`, one root span and one session
    per scenario. `agent` names the agent under test on every span of the run,
    so the run's traces list under that agent.

    A scenario's scripted turns are played first. After them, a live scenario
    (live_user=True, or by default one with a user_goal or conversation_plan
    and no turns) asks the platform's simulated user for each next message,
    until the user is done or max_turns user turns were played (at most 50 for
    the live user). `user_model` names the model configuration the simulated
    user runs on, by name or id; `model` is used when it is not given."""
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
        "agent": agent.strip() if isinstance(agent, str) and agent.strip() else None,
        "user_model": (user_model or model or "").strip() or None,
        "live_user": live_user,
        "api_url": api_url,
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
