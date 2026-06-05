# EvalKit Python SDK

LLM observability and tracing for Python apps. One `init()` call auto-instruments
your LLM clients, HTTP calls, database queries, and logging — then streams traces to
[Syntropy Labs](https://syntropylabs.ai).

## Installation

```bash
pip install syntropylabs-evalkit
```

Optional provider extras:

```bash
pip install "syntropylabs-evalkit[openai]"      # OpenAI
pip install "syntropylabs-evalkit[anthropic]"   # Anthropic
pip install "syntropylabs-evalkit[all]"         # everything
```

> The PyPI package is `syntropylabs-evalkit`, but you import it as `evalkit`.

## Quickstart

```python
import evalkit

evalkit.init(
    subscription_key="sk_...",       # your Syntropy Labs key
    service_name="my-service",
)

# That's it — your OpenAI / Anthropic / HTTP / DB calls are now traced automatically.
from openai import OpenAI

client = OpenAI()
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

`init()` sets up auto-instrumentation for you. Context (including trace IDs)
propagates automatically across threads — no manual wiring required.

## Web frameworks

```python
# FastAPI / Starlette
from evalkit import EvalKitMiddleware
app.add_middleware(EvalKitMiddleware)

# Flask
import evalkit
evalkit.instrument_flask(app)

# Django — add to MIDDLEWARE
"evalkit.EvalKitDjangoMiddleware"
```

## Manual spans

```python
import evalkit

end, ctx = evalkit.start_span("my-operation", {"key": "value"})
try:
    ...  # your work
finally:
    end("ok")

# Or as a decorator
@evalkit.trace_function()
def do_work(x):
    return x * 2
```

## SQLAlchemy

```python
import evalkit
evalkit.patch_sqlalchemy_engine(engine)
```

## Evaluation

Score agent outputs locally — no judge-model cost, results appear as `eval_result` spans:

```python
import evalkit

scores = evalkit.evaluate(
    output="Your return window is 30 days.",
    input="What is the return policy?",
    expected_tools=["search_knowledge_base"],
    tool_calls=[{"name": "search_knowledge_base"}],
    constraints={"required_terms": ["return", "30"]},
)
# → {"tool_trajectory_f1": 1.0, "required_terms": 1.0, ...}
```

## Scenario simulation

Generate realistic synthetic-user scenarios from your agent's system prompt and tool list, then run each scenario against your real agent and score the results automatically:

```python
import evalkit

evalkit.init(subscription_key="tk_live_...", service_name="my-agent")

# Step 1 — generate scenarios server-side (BYOK: your own key for the generation call)
scenarios = evalkit.generate_scenarios(
    agent_instructions=SYSTEM_PROMPT,
    tools=["search_kb", "lookup_order", "create_ticket"],
    count=5,
    provider="anthropic",           # "openai" or "google" also supported
    api_key="sk-ant-...",           # BYOK key for generation model
    model="claude-haiku-4-5-20251001",
)

# Step 2 — simulate each scenario against your real agent and score it
def entrypoint(ctx: evalkit.SimContext) -> evalkit.AgentTurnResult:
    # ctx.message    — the synthetic user's turn message
    # ctx.session_id — stable per-scenario, use it to keep multi-turn context
    reply, tools_used = run_my_agent(ctx.session_id, ctx.message)
    return evalkit.AgentTurnResult(
        text=reply,
        tool_calls=[{"name": t} for t in tools_used],
    )

report = evalkit.simulate_user(entrypoint, scenarios, tags=["ci"])
# Results appear in Dashboard → Simulations
print("Simulation ID:", report["simulation_id"])
```

### Out-of-process agents (Claude Agent SDK)

The Claude Agent SDK runs the Anthropic call in a subprocess, so the normal in-process patch can't observe it. EvalKit wraps `claude_agent_sdk.query()` and `ClaudeSDKClient.receive_response()` instead, reading token/cost/latency from the `ResultMessage` the SDK already returns. This happens automatically via `init()` when `claude_agent_sdk` is installed. To call it explicitly:

```python
evalkit.patch_claude_agent_sdk()
```

## Flushing

Traces are batched and exported in the background. Flush before exit if needed:

```python
evalkit.flush()
```

## Links

- Website: https://syntropylabs.ai
- Documentation: https://syntropylabs.ai/docs

## License

Proprietary — © 2026 Syntropy Labs. All rights reserved. See [LICENSE](LICENSE).
