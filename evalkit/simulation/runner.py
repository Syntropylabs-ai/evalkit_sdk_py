from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import secrets
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

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


@contextmanager
def _scenario_trace(name: str):
    """One OpenTelemetry parent span per scenario so a framework's own OTel spans
    (e.g. Google ADK invocations) and EvalKit's native spans share ONE trace per
    scenario instead of each turn starting a new root. start_trace() called inside
    joins this span's trace id. No-op when the OTel SDK isn't active."""
    span_cm = None
    try:
        from ..core.otel_provider import otel_active
        if otel_active():
            from opentelemetry import trace as _ot
            span_cm = _ot.get_tracer("evalkit").start_as_current_span(f"scenario:{name}")
    except Exception:
        span_cm = None
    if span_cm is None:
        yield
        return
    with span_cm:
        yield


def _run_coro_blocking(coro: Any) -> Any:
    """Run a coroutine to completion from sync code, even if an event loop is
    already running in this thread (in which case we run it on a private loop in
    a worker thread so we never collide with the caller's loop)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # A loop is already running in this thread — run on a dedicated thread.
    result: Dict[str, Any] = {}

    def _worker() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except Exception as exc:  # surface to caller
            result["error"] = exc

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _resolve(value: Any) -> Any:
    """Resolve an entrypoint return value for the synchronous driver: await it
    if it is a coroutine/awaitable, otherwise return as-is."""
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
    """Reconstruct the agent's actual tool calls from spans captured during the
    scenario. Handles both explicit ``tool_call`` spans and the ``gen_ai.tool.call``
    events that the LLM patchers (OpenAI / Anthropic / Google / ADK) attach to the
    ``llm_call`` span — so deterministic tool metrics work without the user having
    to return an AgentTurnResult."""
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
    # Prefer first-class tool_call spans; fall back to the LLM span's tool events
    # (avoids double-counting now that enqueue promotes events → tool_call spans).
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
    trace_id, end_trace, _root = client.start_trace(f"scenario:{name}", session_id=ctx.session_id)
    return name, scenario_id, ctx, sim, turn_messages, end_trace


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
    with _scenario_trace(scenario.get("name", "scenario")):
        name, scenario_id, ctx, sim, turn_messages, end_trace = _begin_scenario(client, scenario, cfg)

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
    with _scenario_trace(scenario.get("name", "scenario")):
        name, scenario_id, ctx, sim, turn_messages, end_trace = _begin_scenario(client, scenario, cfg)

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

    # Pin ONE ambient trace ID before any scenario runs. Every copy_context() /
    # ensure_future() below inherits this ID from the parent context, so all
    # scenarios — and any pre-run ops like generate_scenarios() — land in the
    # same trace instead of each minting an isolated ID.
    get_or_create_ambient_trace_id()

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
