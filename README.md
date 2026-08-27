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
- [Streaming latency (time to first token)](#streaming-latency-time-to-first-token)
- [Streaming HTTP response bodies](#streaming-http-response-bodies)
- [Request and response headers](#request-and-response-headers)
- [Web frameworks](#web-frameworks)
- [Trace your own code](#trace-your-own-code)
- [Manual spans](#manual-spans)
- [Offline evaluation](#offline-evaluation)
- [Scenario simulation](#scenario-simulation)
- [Privacy: content capture and masking](#privacy-content-capture-and-masking)
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
| LLM clients | OpenAI, Anthropic, Bedrock (`boto3` and `aiobotocore`), Cohere, Google (GenAI / Vertex), Mistral |
| Frameworks  | LangChain / LangGraph, LiteLLM, Claude Agent SDK                      |
| HTTP        | `requests`, `httpx`, `aiohttp` — method, URL, status, latency         |
| Databases   | SQLAlchemy, psycopg, asyncpg, PyMongo, Redis — query text + latency   |
| Your code   | Every function in your app's source tree (APM) — on by default        |

### Spans from other OpenTelemetry instrumentors

Third-party OTel spans are bridged into the same pipeline. A bridged span keeps its kind, its error
description, its events, and its GenAI token and cache counts, and it fills the `prompt`,
`completion`, `environment`, `appVersion`, `userId`, `sessionId`, `deviceId` and `sdkVersion` columns
from the matching `evalkit.*` attribute — `evalkit.prompt`, `evalkit.session_id`, and so on — the
same convention the TypeScript SDK uses. Setting those on your own OTel span is all it takes to make
it filterable.

Token counts are read from both the current and the deprecated semantic-convention spellings
(`gen_ai.usage.input_tokens` or `gen_ai.usage.prompt_tokens`), and cache counts from
`gen_ai.usage.cache_read_tokens`, `gen_ai.usage.cache_read.input_tokens` or
`gen_ai.usage.cached_tokens`. Providers disagree on whether cached tokens are already part of the
input count; when the input count is smaller than the cache buckets it is treated as exclusive of
them and they are added in, so the cost breakdown is comparable across providers.

A span that looks like an LLM call (`gen_ai.*`) but carries neither `evalkit.span_type` nor
`evalkit.sdk_version` is treated as a second copy of a call EvalKit already reported: it is filed as
`function_call` and its token and cache counts are dropped rather than double-counted. Set
`EVALKIT_COUNT_FOREIGN_LLM=1` when that other instrumentor is the only one reporting the call. Note
that it is those two attributes specifically — an `evalkit.user_id` on its own does not exempt a span
from the rule.

## Streaming latency (time to first token)

Pass `stream=True` and the span carries `gen_ai.server.time_to_first_token_ms` — an integer, in
milliseconds. Nothing to enable.

Four rules, and they are the same in all four EvalKit SDKs so the number is comparable across
languages:

- **The clock starts before the request is dispatched**, not at the first chunk. It covers span
  setup, connection and queueing, because that is the wait a user actually experiences.
- **Only generated text marks it** — a content delta, or a thinking/reasoning delta from an
  extended-thinking model. Reasoning text marks the clock while still being captured separately as
  `gen_ai.response.thinking` rather than as the completion.
- **Tool-argument deltas do not mark it.** A response that is only a tool call has no first token,
  so the attribute is absent rather than zero.
- **It is not recorded when a stream fails**, so a truncated stream cannot skew the percentiles.

Covered: OpenAI (sync + async, chat and `auto` instrumentation), Anthropic (sync + async), Bedrock
(`converse_stream` / `invoke_model_with_response_stream`, `boto3` and `aiobotocore`), Google GenAI
(`generate_content_stream`, sync + async), Vertex AI (`GenerativeModel.generate_content(stream=True)`
and Anthropic-on-Vertex `messages.create(stream=True)`), LiteLLM, and Ollama.

A stream that dies mid-flight is reported as `status="ERROR"` with the provider's message, and keeps
the partial completion and whatever usage the provider had already reported — the tokens were billed
whether or not the stream finished, and the truncated text is usually the only clue as to why it
stopped. A caller that simply stops reading early is **not** a failure: the span stays `OK` and
keeps its TTFT.

## Streaming HTTP response bodies

Outgoing HTTP calls are captured with their bodies, and a streamed response is no exception. The
tracer cannot read the body itself — that would consume the stream out from under your code — so it
tees instead: every chunk you pull is copied into a capped buffer on its way to you, byte for byte.

Because the body is not known when the headers arrive, **the span is emitted when the stream ends**,
not when the response is returned. Whichever of these happens first wins, and it happens exactly
once: the iterator is exhausted, the response is closed or released, the response is garbage
collected, or `evalkit.flush()` runs. The last one is the backstop, and it is registered at exit, so
a response you never read still produces a span rather than vanishing.

Deferring matters because the trace store is append-only. There is no update path for a span that
has already been written, so "emit now and patch in the body later" would mean a second row for the
same call — and a second row means the call's cost is counted twice, permanently.

The three tee points are the ones every other accessor funnels through, so it does not matter how
you read the body:

| Library | Teed at | Also covers |
| --- | --- | --- |
| `requests` | `iter_content` | `.text`, `.content`, `.json()`, `iter_lines` |
| `httpx` | `iter_bytes` / `aiter_bytes` | `read`, `aread`, `iter_text`, `iter_lines`, `aiter_*` |
| `aiohttp` | a proxy over `resp.content` | `read()`, `text()`, `json()`, `iter_chunked`, `iter_any`, `async for` |

For `aiohttp` this is the first time response bodies are captured at all — the tracer previously
recorded only headers, because there was no safe moment to read.

Deferring splits one timing into three, since a streamed call's duration is no longer the same
question as its responsiveness:

- `latency_ms` — the whole stream, first byte of the request to last byte of the body.
- `http.server.time_to_headers_ms` — until the server responded. This is what `latency_ms` used to
  mean for streamed calls.
- `http.server.time_to_first_byte_ms` — until the first body byte, the HTTP analogue of TTFT.

A partial capture is never presented as a whole one:

- `http.response.streamed` — the body was pulled incrementally, rather than read in one go.
- `http.response.stream.complete` — the stream ran to exhaustion. `false` means the response was
  closed, abandoned, or still open at flush, so the body is a prefix.
- `http.response.stream.chunks` — how many chunks arrived, counted even past the buffer cap.
- `http.response.body.bytes` — the true size, which can exceed the captured body.
- `http.response.body.truncated` — the buffer hit `max_body_bytes` and the rest was dropped.

Buffers are bounded by `max_body_bytes`, and at most 1024 streams are tracked at once; if an
application leaks responses faster than that, the oldest is emitted early rather than dropped.

To go back to emitting as soon as the headers land, with no body for streamed responses:

```python
evalkit.init(subscription_key="tk_live_...", capture_stream_bodies=False)
# or
export EVALKIT_CAPTURE_STREAM_BODIES=false
```

## Request and response headers

HTTP and web-framework spans carry both request and response headers, including `Authorization`,
`Cookie` and `x-api-key`. They are captured because an auth failure is close to undebuggable without
them, and stripping them silently was worse than the alternative.

They are content, so `capture_content=False` removes them along with every other payload, and a
`mask` callback sees them under `http.request.headers` / `http.response.headers` if you want to
rewrite rather than drop. To strip just the credential headers and keep the rest — `traceparent`
included, so parent linkage survives:

```python
evalkit.init(subscription_key="tk_live_...", capture_sensitive_headers=False)
# or
export EVALKIT_CAPTURE_SENSITIVE_HEADERS=false
```

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

## Privacy: content capture and masking

Two independent controls. `capture_content` is the coarse switch; `mask` is the scalpel.

```python
evalkit.init(
    subscription_key="tk_live_...",
    capture_content=False,
)
```

`capture_content=False` keeps every metric and drops every payload. You still get tokens, cost,
latency, TTFT, model, provider, finish reason, status, errors, tool *names* and call IDs, span
hierarchy, `db.system` / `db.operation` / row counts. You lose prompts, completions, thinking text,
request and response bodies, tool arguments and results, traced-function arguments and return
values, SQL text and bound parameters, retriever queries, and log message bodies. Spans that were
stripped carry `evalkit.content_captured=false`, so the dashboard can tell "capture is off" apart
from "the SDK failed to read the body".

It can also be set by environment variable, which is useful when the same image ships to several
environments:

```bash
export EVALKIT_CAPTURE_CONTENT=false
# or the standard OpenTelemetry variable, which EvalKit also honours:
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=false
```

Precedence is `capture_content=` argument, then `EVALKIT_CAPTURE_CONTENT`, then
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`, then on. Note that EvalKit's switch is
deliberately broader than the OpenTelemetry one: the standard variable covers GenAI message content
only, and turning it off here also removes database statements and traced-function arguments.

For anything finer, pass a `mask` callback. It runs on every envelope immediately before it is
queued for export, whichever instrumentation produced it — including spans that arrived through the
OpenTelemetry bridge from third-party instrumentors.

```python
import re

CARD = re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b")

def mask(span):
    if span.prompt:
        span.prompt = CARD.sub("<card>", span.prompt)
    if span.span_type == "db_query":
        return None          # return None to drop the span entirely
    return span

evalkit.init(subscription_key="tk_live_...", mask=mask)
```

Mutate the envelope and return it, or return `None` to drop it. If the hook raises, or returns
anything other than an envelope or `None`, the span is **dropped** and a warning is logged — a mask
that exists to keep content inside the process fails closed rather than exporting unmasked. It is
the one place in the SDK that does; everywhere else instrumentation failures are swallowed and
tracing degrades instead of your app.

When both controls are set, `capture_content` is applied first, so `mask` sees the already-stripped
envelope.

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
    capture_content=None,                     # None = env var, else True/False
    capture_sensitive_headers=None,           # None = env var, else True/False
    capture_stream_bodies=None,               # None = env var, else True/False
    mask=None,                                # callable(envelope) -> envelope | None
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
