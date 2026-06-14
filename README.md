# EvalKit — Python SDK

Tracing and evaluation for LLM apps. A single `init()` call auto-instruments your
LLM clients, HTTP calls, database queries, and logging, then streams traces to
[Syntropy Labs](https://syntropylabs.ai).

```bash
pip install syntropylabs-evalkit
```

> Installs as `syntropylabs-evalkit`; you import it as `evalkit`.

## Contents

- [Quick start](#quick-start)
- [What gets traced](#what-gets-traced)
- [Web frameworks](#web-frameworks)
- [Trace your own code](#trace-your-own-code)
- [Manual spans](#manual-spans)
- [Offline evaluation](#offline-evaluation)
- [Scenario simulation](#scenario-simulation)
- [Configuration](#configuration)

## Quick start

```python
import evalkit

evalkit.init(
    subscription_key="tk_live_...",   # Dashboard → Settings → Tracing
    service_name="my-service",
)

# Every OpenAI / Anthropic / HTTP / DB call from here on is traced automatically.
from openai import OpenAI

resp = OpenAI().chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

Call `init()` once, as early as possible. Trace context (including trace IDs)
propagates across threads and async tasks automatically — no manual wiring.

## What gets traced

| Category    | Captured automatically                                                |
| ----------- | --------------------------------------------------------------------- |
| LLM clients | OpenAI, Anthropic, Bedrock, Cohere, Google (GenAI / Vertex), Mistral  |
| Frameworks  | LangChain / LangGraph, LiteLLM, Claude Agent SDK                      |
| HTTP        | `requests`, `httpx`, `aiohttp` — method, URL, status, latency         |
| Databases   | SQLAlchemy, psycopg, asyncpg, PyMongo, Redis — query text + latency   |
| Your code   | Every function in your app's source tree (APM) — on by default        |

## Web frameworks

```python
# FastAPI / Starlette
from evalkit import EvalKitMiddleware
app.add_middleware(EvalKitMiddleware)

# Flask
evalkit.instrument_flask(app)

# Django — add to MIDDLEWARE
"evalkit.EvalKitDjangoMiddleware"
```

## Trace your own code

Function tracing is **on by default**: `init()` wraps every function in your app's
own source tree as it imports — one `function_call` span each, with input, output,
and latency. Third-party libraries are never touched.

```python
# Disable it
evalkit.init(..., function_tracing=False)   # or env EVALKIT_FUNCTION_TRACE=false

# Trace sibling packages outside the caller's directory
evalkit.init(..., trace_packages=["support_bot", "workers"])
```

Need finer control? Opt in explicitly — a function, a tool, a class, or a module:

```python
@evalkit.trace_function()           # → function_call span
def do_work(x):
    return x * 2

@evalkit.trace_tool()               # → tool_call span (counts toward tool metrics)
def search_web(query: str):
    return run_search(query)

@evalkit.traced                     # → every method of the class
class OrderService:
    def place(self, order): ...
    def cancel(self, id): ...

import myapp
evalkit.trace_package(myapp)        # → every function across the whole package
```

> A client-side tool the model calls only shows its **output** if you wrap it with
> `trace_tool` — the SDK sees the model's request, not your function's return value.
> Server-side tools (e.g. OpenAI `web_search`) and LangChain tools are automatic.

## Manual spans

```python
end, ctx = evalkit.start_span("my-operation", {"key": "value"})
try:
    ...  # your work
    end("ok")
except Exception:
    end("error")
    raise
```

## Offline evaluation

Deterministic, local scoring — no judge-model cost. Results are pushed as an
`eval_result` span.

```python
scores = evalkit.evaluate(
    output="Your return window is 30 days.",
    input="What is the return policy?",
    expected_tools=["search_knowledge_base"],
    tool_calls=[{"name": "search_knowledge_base"}],
    constraints={"required_terms": ["return", "30"]},
)
# → {"tool_trajectory": 1.0, "tool_f1": 1.0, "tool_correctness": 1.0,
#    "response_match": 1.0, "constraint_compliance": 1.0}
```

`evaluate()` returns only the metrics applicable to the inputs you pass: tool metrics
from `tool_calls` / `expected_tools`, `response_match` / `constraint_compliance` from
`constraints`, and `contextual_precision` / `contextual_recall` from
`retrieved_context` / `expected_context`.

## Scenario simulation

Generate synthetic-user scenarios from your agent's prompt and tools, replay each one
against your real agent, then grade the run with LLM-as-judge evaluators.

**1. Generate** scenarios (bring your own key for the generation call):

```python
scenarios = evalkit.generate_scenarios(
    agent_instructions=SYSTEM_PROMPT,
    tools=["search_kb", "lookup_order", "create_ticket"],
    count=5,
    provider="anthropic",                 # or "openai" / "google"
    api_key="sk-ant-...",
    model="claude-haiku-4-5-20251001",
)
```

**2. Simulate** — replay each scenario against your real agent:

```python
def entrypoint(ctx: evalkit.SimContext) -> evalkit.AgentTurnResult:
    # ctx.message    — the synthetic user's message for this turn
    # ctx.session_id — stable per scenario; use it to keep multi-turn context
    reply, tools_used = run_my_agent(ctx.session_id, ctx.message)
    return evalkit.AgentTurnResult(text=reply, tool_calls=[{"name": t} for t in tools_used])

report = evalkit.simulate_user(entrypoint, scenarios, tags=["ci"])
print(report["simulation_id"], report["run_id"])
```

**3. Evaluate** the run against an evaluator collection (BYOK judge). Per-scenario,
per-criterion scores come back with reasons, and also appear in the dashboard:

```python
result = evalkit.evaluate_simulation(
    report["simulation_id"],
    collection_id="665f0c...",            # Dashboard → Evaluators → Collections
    provider="openai",
    model="gpt-4o",
    api_key="sk-...",
    max_tokens=1024,                      # optional judge output cap
    # run_id="run_...",                   # optional; defaults to the latest run
)

print(result["aggregate"])                # {"averageScore": ..., "passRate": ...}
for scn in result["scenarios"]:
    print(scn["name"], scn["overallScore"], scn["passed"])
    for m in scn["metrics"]:
        print("  -", m["ruleName"], m["score"], m["reason"])
```

### Out-of-process agents (Claude Agent SDK)

The Claude Agent SDK runs the model call in a subprocess, so the in-process patch
can't see it. EvalKit instead wraps `claude_agent_sdk.query()` and
`ClaudeSDKClient.receive_response()`, reading token/cost/latency from the
`ResultMessage`. This is automatic via `init()` when `claude_agent_sdk` is installed;
call `evalkit.patch_claude_agent_sdk()` explicitly if you install it later.

## Configuration

```python
evalkit.init(
    subscription_key="tk_live_...",
    service_name="my-service",
    base_url="https://api.syntropylabs.ai",   # trace ingest (default)
    api_url="https://api.syntropylabs.ai",    # control plane (default)
    environment="production",                 # production | staging | development
    debug=False,                              # log exports to stdout
    function_tracing=True,                    # auto-trace your functions (default)
    trace_packages=None,                      # extra sibling packages to trace
)
```

Traces are batched and exported in the background. Flush before exit if needed:

```python
evalkit.flush()
```

## Links

- Website: https://syntropylabs.ai
- Documentation: https://syntropylabs.ai/docs

## License

Proprietary — © 2026 Syntropy Labs. All rights reserved. See [LICENSE](LICENSE).
