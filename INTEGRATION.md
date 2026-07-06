# EvalKit Python SDK — Integration Guide

`syntropylabs-evalkit` —

- **Package:** `syntropylabs-evalkit`
- **Version:** 0.2.8
- **Requires:** Python ≥ 3.9

> One rule the SDK never breaks: **tracing must never break your app.** Every patch, span, and export is wrapped so a failure degrades to a no-op and your code keeps running.

---

## 1. Install

```bash
pip install syntropylabs-evalkit
```

Install with the integrations you use (optional extras):

```bash
pip install "syntropylabs-evalkit[openai]"      # or anthropic, google, cohere, mistral, groq, bedrock, litellm, langchain
pip install "syntropylabs-evalkit[all]"         # every provider integration
pip install "syntropylabs-evalkit[otel]"        # OpenTelemetry bridge (community auto-instrumentation)
```

---

## 2. Quick start

Call `init()` once, as early as possible (before importing/creating your LLM clients
and web app), then use your code normally — LLM calls, HTTP, DB, and your own
functions are traced automatically.

```python
import evalkit

evalkit.init(
    subscription_key="sk_live_...",     # required
    service_name="my-agent",
    environment="production",
)

# From here on, OpenAI/Anthropic/etc. calls, outbound HTTP, DB queries,
# and your own module-level functions are auto-traced and shipped to EvalKit.
```

`init()` is **idempotent** — calling it again returns the existing client. Every
top-level helper raises `RuntimeError("[evalkit] call evalkit.init() before using the SDK")`
if used before `init()`.

---

## 3. Configuration

```python
evalkit.init(
    subscription_key,                              # str, REQUIRED — sent as X-Subscription-Key
    base_url="https://api.syntropylabs.ai",        # trace ingest endpoint base
    api_url="https://api.syntropylabs.ai",         # control-plane base (scenarios/simulation)
    service_name="unknown",                        # stamped on every span
    debug=False,                                    # verbose exporter logging
    max_export_batch_size=512,                      # flush trigger
    scheduled_delay_ms=5000,                        # background flush interval
    max_queue_size=4096,                            # ring buffer cap (oldest dropped when full)
    max_body_bytes=10 * 1024 * 1024,               # cap on captured HTTP req/resp bodies
    environment=None,                               # e.g. "production" / "staging"
    app_version=None,                               # e.g. "1.4.2"
    device_id=None,
    function_tracing=True,                          # auto-trace your own functions at import time
    trace_packages=None,                            # extra top-level packages to function-trace
)
```

### Environment variables
| Var | Effect |
|-----|--------|
| `EVALKIT_API_URL` | Overrides `api_url` for `generate_scenarios` / `evaluate_simulation`. |
| `EVALKIT_FUNCTION_TRACE` | Set to `false` to disable import-time function tracing. |

---

## 4. What gets traced automatically

After `init()`, the SDK auto-instruments everything installed in your environment:

- **LLM providers** (class-level, every client instance): OpenAI (Chat Completions + Responses API), Anthropic (Messages + streaming), Google GenAI (`google.genai`) + legacy `google.generativeai`, Cohere, AWS Bedrock (Converse), Mistral, Groq, LiteLLM, LangChain / LangGraph, Claude Agent SDK. Each emits an `llm_call` span with `model`, `provider`, `prompt`, `completion`, token usage, latency, tool calls, and streaming accumulation.
- **HTTP clients**: `requests`, `httpx`, `aiohttp` → `http_call` spans, with W3C `traceparent` propagation injected outbound.
- **Databases**: redis, pymongo/motor, asyncpg, psycopg v3, pymysql, SQLAlchemy → `db_query` spans.
- **MCP** (`mcp.ClientSession.call_tool`) → `tool_call` spans.
- **Logging**: each `logging` record ≥ DEBUG becomes a child `log` span, with `trace_id` injected into records.
- **Your own functions**: module-level functions in your source tree are wrapped as `function_call` spans at import time (disable with `function_tracing=False`).
- **Thread/executor context**: trace context propagates into `Thread` / `ThreadPoolExecutor` so background work doesn't orphan.

> Dedup: if the matching OpenTelemetry community instrumentation package is installed, the SDK steps aside and lets OTel own that library (no double spans).

Re-run HTTP/logging instrumentation manually if needed:

```python
evalkit.auto_instrument(logging=True, requests=True, httpx=True)
```

---

## 5. Manual LLM client patching

For clients created in a way the auto-patch can't reach, or to bind a specific
parent span, patch the instance directly:

```python
from openai import OpenAI
import evalkit

client = OpenAI()
evalkit.patch_openai_client(client)            # async: patch_async_openai_client
```

Available helpers:

```
patch_openai_client / patch_async_openai_client
patch_anthropic_client / patch_async_anthropic_client
patch_bedrock_client(bedrock_runtime_client)
patch_anthropic_vertex_client
patch_vertex_generative_model(model)
patch_cohere_client(client)
patch_google_ai_model(model)        patch_google_genai()        # class-level
patch_langchain()  ->  callback handler
patch_litellm()
patch_claude_agent_sdk()
patch_mistral_client / patch_async_mistral_client
patch_sqlalchemy_engine / patch_async_sqlalchemy_engine
```

### Realtime / voice (WebSocket)

Trace streaming voice sessions — `voice.session` / `voice.turn` / `tool_call` spans:

```python
import evalkit

# Google Gemini Live
async with evalkit.trace_google_live(genai_client, model="gemini-2.0-flash-live") as session:
    ...

# OpenAI Realtime
conn = evalkit.trace_openai_realtime(openai_client, model="gpt-realtime")
```

---

## 6. Web framework integration

Add the middleware so inbound requests become `http_call` server spans, continue
upstream `traceparent`, and inject `X-Trace-Id` into responses.

**FastAPI / Starlette**
```python
from fastapi import FastAPI
from evalkit import EvalKitMiddleware

app = FastAPI()
app.add_middleware(EvalKitMiddleware)
```

**Django** (`settings.py`)
```python
MIDDLEWARE = [
    "evalkit.EvalKitDjangoMiddleware",
    # ... your other middleware
]
```

**Flask**
```python
from flask import Flask
import evalkit

app = Flask(__name__)
evalkit.instrument_flask(app)
```

**Litestar**
```python
from litestar import Litestar
from evalkit import create_litestar_middleware

app = Litestar(route_handlers=[...], middleware=[create_litestar_middleware()])
```

---

## 7. Manual tracing

Use these when you want explicit spans around code the auto-tracer doesn't cover.

**Decorators** (sync + async):
```python
import evalkit

@evalkit.trace_function()              # -> function_call span
def build_prompt(user_id): ...

@evalkit.trace_tool()                  # -> first-class tool_call span (counts toward tool metrics)
def search_docs(query): ...

@evalkit.traced                        # class decorator: wraps every method
class Agent: ...
```

Trace whole modules / packages:
```python
import mypkg.tools
evalkit.trace_module(mypkg.tools)      # wrap every function defined in the module
evalkit.trace_package(mypkg)           # recurse into the package
```

**Imperative spans:**
```python
# Generic span
end, ctx = evalkit.start_span("retrieve", attributes={"k": 5})
try:
    ...
finally:
    end("OK")                          # or "ERROR"

# Root / log span
trace_id, end, ctx = evalkit.start_trace("nightly-job")
end("OK")

# HTTP span
trace_id, end, ctx = evalkit.start_http_trace(
    "call-vendor", method="POST", url="https://api.vendor.com/x", request_body={...})
end("OK", status_code=200, response_body={...})

# Nest spans under an explicit context
with evalkit.with_trace(ctx):
    ...

evalkit.current_trace_id()             # active trace id (for logging/correlation)
```

`capture_io=True` (default on decorators) records args/results as span attributes
(truncated). Set `capture_io=False` for sensitive payloads.

---

## 8. Offline evaluation — `evaluate()`

Deterministic metrics computed **locally, no LLM call**. Returns
`{metric_name: score}` (0.0–1.0) and, if initialised, also emits an `eval_result`
span to the dashboard.

```python
scores = evalkit.evaluate(
    output="The capital of France is Paris [1].",
    input="What is the capital of France?",
    tool_calls=[{"name": "search"}, {"name": "summarize"}],
    expected_tools=["search", "summarize"],
    retrieved_context=["doc_a", "doc_b"],
    expected_context=["doc_a"],
    constraints={"requiredTerms": ["Paris"], "hasCitations": True},
    metrics=None,        # default: all applicable
)
# -> {"tool_trajectory": 1.0, "tool_f1": 1.0, "tool_correctness": 1.0,
#     "response_match": 1.0, "constraint_compliance": 1.0,
#     "contextual_precision": 0.5, "contextual_recall": 1.0}
```

| Metric | Needs | Measures |
|--------|-------|----------|
| `tool_trajectory` | `expected_tools` | order-aware tool match (LCS / len expected) |
| `tool_f1` | `expected_tools` | F1 over tool-name sets |
| `tool_correctness` | `expected_tools` | recall: fraction of expected tools called |
| `response_match` | `constraints.requiredTerms` | fraction of required terms present |
| `constraint_compliance` | `constraints` | requiredTerms / hasCitations / slideCount checks |
| `contextual_precision` | `retrieved_context` + `expected_context` | retrieved ∩ expected / retrieved |
| `contextual_recall` | `retrieved_context` + `expected_context` | expected ∩ retrieved / expected |

Each metric is computed only when its inputs are present. Constraint keys accept
snake_case or camelCase.

---

## 9. Scenario generation — `generate_scenarios()`

Generate test scenarios for an agent from its instructions + tools (control-plane,
bring-your-own-key).

```python
scenarios = evalkit.generate_scenarios(
    agent_instructions="You are a support bot for an e-commerce store.",
    tools=[...],                       # tool objects / callables (auto-serialized)
    function_schemas=[...],            # or raw JSON schemas
    context={"brand": "Acme"},
    rag={"enabled": True},
    count=5,
    categories=["happy_path", "edge_case", "adversarial"],
    provider="openai", model="gpt-4o", api_key="sk-...",   # BYOK judge/generator
    temperature=0.7,
    reasoning_effort="medium",         # for reasoning models
    max_completion_tokens=2000,
)
# -> list of scenario dicts ready to feed into simulate_user()
```

---

## 10. User simulation — `simulate_user()`

Drives a multi-turn synthetic user against your agent locally, tags every span
with the simulation, and scores each scenario with `evaluate()`.

Your **entrypoint** receives a `SimContext` per turn and returns text or an
`AgentTurnResult`:

```python
from evalkit import SimContext, AgentTurnResult
import evalkit

def my_agent(ctx: SimContext) -> AgentTurnResult:
    # ctx.message  -> the simulated user's message this turn
    # ctx.session_id, ctx.state (dict, persists across turns), ctx.turn (int)
    reply, tools = run_my_agent(ctx.message, ctx.state)
    return AgentTurnResult(
        text=reply,
        tool_calls=[{"name": t} for t in tools],   # optional; else read from traced spans
        retrieved_context=[...],                    # optional, for RAG metrics
    )

result = evalkit.simulate_user(
    my_agent,
    scenarios,                          # from generate_scenarios() or hand-written
    metrics=["tool_correctness", "response_match"],
    max_turns=12,
    tags=["release-1.4"],
    safe_mode=True,
)
# -> {"simulation_id": "sim_...", "run_id": "run_...",
#     "results": [{"scenario_id", "name", "status", "turns", "scores", "error?"}, ...]}
```

`entrypoint` may be sync **or** async. A hand-written scenario dict can include:

```python
{
  "name": "refund flow",
  "turns": ["I want a refund", "order #1234"],   # or "messages", or "starting_prompt"
  "expected_tools": ["lookup_order", "issue_refund"],
  "constraints": {"requiredTerms": ["refund"]},
  "target_keywords": ["refund", "processed"],
  "persona": {"name": "frustrated customer"},
  "setup": {"state": {"account": "premium"}},
}
```

All spans produced during a scenario are tagged `isSimulation=true` with the
`simulationId` / `runId` / `scenarioId`, so they’re filterable in the dashboard.

---

## 11. Simulation evaluation (LLM judge) — `evaluate_simulation()`

Run your project’s evaluator collection (LLM-as-judge) over a completed simulation,
server-side. The judge is bring-your-own-key and never stored.

```python
report = evalkit.evaluate_simulation(
    simulation_id=result["simulation_id"],
    collection_id="<evaluator collection id from your dashboard>",
    provider="openai", model="gpt-4o", api_key="sk-...",   # judge BYOK
    run_id=result["run_id"],            # optional; default = latest run
    max_tokens=None,
)
# -> per-scenario, per-criterion scores + reasons + aggregate (averageScore, passRate, ...)
```

---

## 12. OpenTelemetry interop

The SDK is OTel-native in two directions:

- **Inbound (always on):** EvalKit spans automatically nest under, and share the
  `trace_id` of, any active OpenTelemetry span (if `opentelemetry-api` is importable).
- **Outbound bridge (opt-in, `pip install "syntropylabs-evalkit[otel]"`):** community
  OTel auto-instrumentations are enabled (minus the LLM libs EvalKit hand-instruments),
  and their spans are mapped to EvalKit envelopes and shipped through the same pipeline.

W3C `traceparent` is the wire format for distributed propagation — emitted on
outbound HTTP and parsed on inbound requests.

---

## 13. Flushing & shutdown

Spans are batched and exported by a background daemon thread; `atexit` flushes on
exit. To force a synchronous flush (e.g. in short-lived scripts / serverless):

```python
evalkit.flush()
```

---

## 14. Notes & troubleshooting

- **Nothing showing up?** Confirm `init()` runs before clients/app are created, and that `subscription_key` is valid. Three consecutive `401/403` responses permanently disable the exporter for the process (a bad key never self-heals) — fix the key and restart.
- **Duplicate spans?** You likely patched a client that’s also auto-instrumented; remove the manual `patch_*` call (auto-patch already covers it).
- **Sensitive payloads:** set `capture_io=False` on decorators and lower `max_body_bytes`; captured HTTP/DB bodies are size-capped and oversized spans are trimmed before send.
- **Async/threads:** context propagates into threads and executors automatically; if you spawn processes, call `init()` in each.
- **Span types you’ll see:** `llm_call`, `tool_call`, `http_call`, `db_query`, `function_call`, `log`, `eval_result`.
