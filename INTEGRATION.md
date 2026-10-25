# EvalKit Python SDK integration guide

EvalKit is the SyntropyLabs SDK for Python. It traces your LLM app and sends the traces to [SyntropyLabs](https://syntropylabs.ai). It can also replay test scenarios against your agent and score the answers.

The package is `syntropylabs-evalkit`, at version 0.3.2, and you import it as `evalkit`. It needs Python 3.9 or later. The [README](README.md) is the full reference. It also covers identity, evaluation runs, feedback, prompts and the `evalkit` command.

> Tracing never breaks your app. Every patch, span and export is wrapped, so a failure in tracing becomes a no-op and your code keeps running. If your `mask` function raises, EvalKit drops that span, so content you meant to hide is not sent. Calls to the SyntropyLabs API are the exception: `Eval()`, `feedback()`, `prompts.get()`, `generate_scenarios()` and `evaluate_simulation()` raise when the request fails.

## Install

```bash
pip install syntropylabs-evalkit
```

Extras install the client libraries that EvalKit instruments:

```bash
pip install "syntropylabs-evalkit[openai]"      # or anthropic, google, cohere, mistral, groq, bedrock, litellm, langchain
pip install "syntropylabs-evalkit[all]"         # every provider integration
pip install "syntropylabs-evalkit[otel]"        # OpenTelemetry bridge (community auto-instrumentation)
```

The extras are `openai`, `anthropic`, `google`, `cohere`, `mistral`, `groq`, `bedrock`, `litellm`, `ollama`, `langchain`, `langgraph`, `pinecone`, `qdrant`, `chroma`, `weaviate`, `llamaindex`, `elevenlabs`, `assemblyai`, `otel` and `all`. Each extra installs its library, for example `google-genai` for `google` and `boto3` for `bedrock`. `all` installs every one of them.

You do not need an extra for a library you already have. EvalKit instruments each supported library that it can import when `init()` runs.

The `otel` extra adds nothing to the core install. Its two packages, `opentelemetry-sdk` and `opentelemetry-instrumentation`, are core dependencies, and the OpenTelemetry bridge is part of every install. See [OpenTelemetry interop](#opentelemetry-interop).

## Quick start

Call `init()` once, as early as you can. Do it before you import or create your LLM clients and your web app. From then on, LLM calls, outbound HTTP requests, database queries and your own functions are traced:

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

Subscription keys start with `tk_live_`. Yours is on the Environments page of the SyntropyLabs console. To open it, choose Settings in the sidebar footer, then Environments under This project.

A second call to `init()` returns the client from the first call and ignores its arguments.

Until `init()` has run, most SDK functions raise `RuntimeError("[evalkit] call evalkit.init() before using the SDK")`. The identity and agent helpers, `session()`, `current_trace_id()`, `with_trace()` and `evaluate()` work without it.

## Configuration

`init()` takes these options. Each value shown is the default:

```python
evalkit.init(
    subscription_key,                              # str, REQUIRED (sent as X-Subscription-Key)
    base_url="https://api.syntropylabs.ai",        # trace ingest endpoint base
    api_url="https://api.syntropylabs.ai",         # control-plane base (scenarios/simulation)
    service_name="unknown",                        # stamped on every span
    debug=False,                                    # verbose exporter logging
    max_export_batch_size=512,                      # flush trigger
    scheduled_delay_ms=5000,                        # background flush interval
    max_queue_size=4096,                            # queue cap (oldest non-model-call span dropped when full)
    max_queue_bytes=32 * 1024 * 1024,               # about how many bytes the queue may hold
    max_body_bytes=64 * 1024,                       # cap on captured HTTP req/resp bodies
    max_value_bytes=None,                           # per-value cap, 16 MiB (head and tail kept)
    environment=None,                               # e.g. "production" / "staging"
    app_version=None,                               # e.g. "1.4.2"
    device_id=None,
    function_tracing=True,                          # auto-trace your own functions at import time
    trace_packages=None,                            # extra top-level packages to function-trace
    function_rate_limit=100,                        # spans a second per traced function (0: all)
    log_level=None,                                 # WARNING, or EVALKIT_LOG_LEVEL
    log_rate_limit=50,                              # log spans a second
    sample_rate=None,                               # share of traces kept, or OTEL_TRACES_SAMPLER
    capture_unhandled=False,                        # record uncaught exceptions
    in_app_include=None, in_app_exclude=None,       # module prefixes that are / are not your code
)
```

`init()` never raises. An invalid option makes the SDK a no-op with one warning naming the problem, and `EVALKIT_DISABLED=true` or `OTEL_SDK_DISABLED=true` turns it off entirely. The README covers every option under [Configuration](README.md#configuration).

`base_url` is where spans go. The SDK posts them to `{base_url}/v1/ingest`, with your key in the `X-Subscription-Key` header.

`api_url` is the base for calls to the SyntropyLabs API. These are `Eval()`, feedback and rewards, prompts, `generate_scenarios()` and `evaluate_simulation()`.

With `debug=True`, the exporter logs each export, retry and auth failure at `DEBUG` level on the `evalkit` logger.

`init()` also takes `capture_content`, `capture_secrets` (or its alias `capture_sensitive_headers`), `capture_stream_bodies` and `mask`. These control which content leaves your process. See [Content capture and masking](README.md#content-capture-and-masking) in the README.

### Environment variables

`EVALKIT_API_URL` overrides the `api_url` you pass to `init()`, for calls to the SyntropyLabs API. It does not change where spans go. An `api_url` argument passed to one of those functions wins over both.

`EVALKIT_FUNCTION_TRACE=false` turns off the tracing of your own functions. Only the exact value `false` has this effect.

The README lists the other variables under [Environment variables](README.md#environment-variables).

## What gets traced

After `init()`, EvalKit instruments every supported library that is installed. LLM clients are patched on the class, so every client instance is traced.

- LLM clients: OpenAI (Chat Completions and Responses), Anthropic (Messages, streaming included), Google GenAI (`google.genai`) and the older `google.generativeai`, Cohere, Mistral, Groq and Ollama. Anthropic on Vertex (`AnthropicVertex`) and Google GenAI in Vertex mode are covered too.
- AWS Bedrock: the runtime's Converse and InvokeModel calls, streaming included, and the agent runtime's InvokeAgent, InvokeInlineAgent, Retrieve and RetrieveAndGenerate calls.
- Frameworks: LiteLLM, LangChain and LangGraph, LlamaIndex, Google ADK and the Claude Agent SDK.
- Vector stores: Pinecone, Qdrant, Chroma and Weaviate.
- Voice: OpenAI audio and Realtime, ElevenLabs, AssemblyAI and Google GenAI Live.
- HTTP clients: `requests`, `httpx` and `aiohttp`, as `http_call` spans.
- Databases: Redis, PyMongo (Motor included), asyncpg, psycopg 3, psycopg2 and PyMySQL, as `db_query` spans.
- MCP: `mcp.ClientSession.call_tool`, as `tool_call` spans.
- Logging: records from Python's `logging` module, as `log` spans.
- Your own functions: public module-level functions in your source tree, as `function_call` spans.

Each LLM call becomes an `llm_call` span with the model, provider, prompt, completion, token usage, latency and tool calls. A streamed response is collected into the same span.

EvalKit also adds a W3C `traceparent` header to each outgoing HTTP request. With `requests` and `aiohttp`, it adds the header when you pass the headers as a plain dict or pass none.

A log record becomes a span when it reaches the root logger. The root level is `WARNING` unless your app changes it, so by default only warnings and errors are recorded. EvalKit changes no handler, formatter or level, and adds `trace_id` and `span_id` to every record for a format that asks for them.

The source tree is the directory of the file that calls `init()`. Modules imported after `init()` are covered too. Turn this off with `function_tracing=False`.

Trace context carries into `threading.Thread` and `ThreadPoolExecutor`. Work that you hand to a thread stays in its trace.

Some libraries need a call of your own. The Vertex AI SDK's `GenerativeModel` needs `patch_vertex_generative_model(model)`. SQLAlchemy needs `patch_sqlalchemy_engine(engine)`, unless EvalKit creates the tracer provider, in which case its OpenTelemetry instrumentor traces it.

EvalKit owns the libraries it patches. [How duplicates are handled](#how-duplicates-are-handled) explains what happens to OpenTelemetry spans for the same calls.

`auto_instrument()` runs the whole setup again, LLM patches included:

```python
evalkit.auto_instrument(logging=True, requests=True, httpx=True)
```

A patch that is already in place is not applied twice. Passing `False` for a flag skips that step on this run. It does not undo what `init()` installed, so these flags cannot turn instrumentation off.

## Patch a client by hand

Use a patch function for a client that the class-level patches did not reach. The Vertex AI SDK's `GenerativeModel` always needs one.

```python
from openai import OpenAI
import evalkit

client = OpenAI()
evalkit.patch_openai_client(client)            # async: patch_async_openai_client
```

These are the helpers:

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

A patch function skips a client that is already wrapped. Once `init()` has patched a library's classes, the helper for that library does nothing, and its `parent_ctx` argument has no effect. To put such calls under a span you choose, run them inside `with evalkit.with_trace(ctx):`, as shown in [Imperative spans](#imperative-spans).

`evalkit` also exports `patch_ollama()`, `patch_pinecone()`, `patch_qdrant()`, `patch_chroma()`, `patch_weaviate()`, `patch_llamaindex()`, `patch_openai_audio()`, `patch_elevenlabs()`, `patch_assemblyai()`, `patch_google_genai_live()` and `patch_openai_realtime()`. They take no arguments. `init()` applies the same patches when the library is installed.

### Realtime voice sessions

A realtime voice session becomes a `voice.session` trace. Each turn is a `voice.turn` span of type `llm_call`, and each tool the model calls is a `tool_call` span named after the tool.

`init()` traces these sessions without any wrapper. It patches `connect()` for Gemini Live on the async `google.genai` client (`client.aio.live.connect()`) and for OpenAI Realtime on the async OpenAI client (`client.beta.realtime.connect()`).

`trace_google_live()` and `trace_openai_realtime()` wrap one connection explicitly. Both return an async context manager:

```python
import evalkit

# Google Gemini Live
async with evalkit.trace_google_live(genai_client, model="gemini-2.0-flash-live") as session:
    ...

# OpenAI Realtime
conn = evalkit.trace_openai_realtime(openai_client, model="gpt-realtime")
```

Use the wrappers only where `init()` has not patched `connect()`. On a connection that `init()` already traces, a wrapper records a second `voice.session` span.

## Web frameworks

Add the middleware for your framework. Each inbound request then becomes an `http_call` span named after its method and path, such as `GET /orders`. The middleware continues an incoming `traceparent` header and adds `X-Trace-Id` and `traceparent` headers to the response.

For FastAPI or Starlette, add `EvalKitMiddleware`:

```python
from fastapi import FastAPI
from evalkit import EvalKitMiddleware

app = FastAPI()
app.add_middleware(EvalKitMiddleware)
```

For Django, add `evalkit.EvalKitDjangoMiddleware` to `MIDDLEWARE` in `settings.py`:

```python
MIDDLEWARE = [
    "evalkit.EvalKitDjangoMiddleware",
    # ... your other middleware
]
```

For Flask, call `instrument_flask()` on the app. It raises `RuntimeError` if `init()` has not run yet:

```python
from flask import Flask
import evalkit

app = Flask(__name__)
evalkit.instrument_flask(app)
```

For Litestar, pass the middleware when you create the app:

```python
from litestar import Litestar
from evalkit import create_litestar_middleware

app = Litestar(route_handlers=[...], middleware=[create_litestar_middleware()])
```

The Django, FastAPI and Starlette middleware also record the id of a signed-in user. See [Identity](README.md#identity) in the README.

## Manual tracing

Use these for code that the automatic tracing does not reach.

### Decorators and modules

The decorators work on sync and async functions:

```python
import evalkit

@evalkit.trace_function()              # -> function_call span
def build_prompt(user_id): ...

@evalkit.trace_tool()                  # -> first-class tool_call span (counts toward tool metrics)
def search_docs(query): ...

@evalkit.traced                        # class decorator: wraps every method
class Agent: ...
```

`traced` wraps only the methods that the class defines itself, and it skips names that start with `__`. The decorators look up the client when they are applied, so call `init()` before the decorated code is imported.

To trace a whole module or package:

```python
import mypkg.tools
evalkit.trace_module(mypkg.tools)      # wrap every function defined in the module
evalkit.trace_package(mypkg)           # recurse into the package
```

`trace_module()` wraps the public functions defined in the module itself. `trace_package()` does the same for the package and imports each of its submodules. Both skip names that start with `_` unless you pass `include_private=True`, and both return the number of functions they wrapped.

The decorators and both module helpers take `capture_io`. With `capture_io=True`, the default, arguments and return values are recorded as JSON and cut to 4096 characters, with `…` at the end. Set `capture_io=False` for sensitive payloads.

### Imperative spans

Open a span yourself when a block of work has no function to decorate:

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

`start_span()` opens a `function_call` span under the current span and returns its `end` function and context. It does not make the new span current. To nest the spans opened in a block under it, wrap the block in `with evalkit.with_trace(ctx):`.

`start_trace()` opens a root span of type `log` and makes it current until you call `end`. Inside another span, it opens a child in that span's trace. `start_http_trace()` opens an `http_call` span in a new trace, and its `end` records the response.

`end()` sends the status string exactly as you pass it. The SDK defines `"OK"`, `"ERROR"` and `"UNSET"`. `current_trace_id()` returns the current trace's id, and it creates one when no trace is active.

The README gives each signature under [Manual spans](README.md#manual-spans).

## Offline evaluation

`evaluate()` scores an output locally with deterministic metrics, and it makes no LLM call. It returns `{metric_name: score}`, with each score from 0.0 to 1.0. It works without `init()`. After `init()`, it also sends the scores as an `eval_result` span.

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

Each metric is computed only when its inputs are present. With `metrics=None`, you get every metric your arguments allow. `input` and `expected_output` are accepted, but no metric reads them.

| Metric | Computed when | Score |
| --- | --- | --- |
| `tool_trajectory` | `expected_tools` is not empty | Longest common subsequence of called and expected tools, over the number expected |
| `tool_f1` | `tool_calls` or `expected_tools` is not empty | F1 of the called and expected tool-name sets |
| `tool_correctness` | `expected_tools` is not empty | Share of expected tools that were called |
| `response_match` | `constraints` has `requiredTerms` | Share of required terms found in the output, ignoring case |
| `constraint_compliance` | `constraints` is not empty | Share of the `requiredTerms`, `hasCitations` and `slideCount` checks that pass |
| `contextual_precision` | `retrieved_context` and `expected_context` are both given | Share of retrieved items that are expected |
| `contextual_recall` | `retrieved_context` and `expected_context` are both given | Share of expected items that were retrieved |

Constraint keys accept snake_case or camelCase, such as `required_terms` or `requiredTerms`. `hasCitations` looks for a marker such as `[1]`, a year in parentheses or a URL. `slideCount` counts mentions such as `slide 3`, `section 3` or `## 3.`.

## Scenario generation

`generate_scenarios()` has the SyntropyLabs API write test scenarios for your agent, from its instructions and tools. You bring your own key for the model that writes them:

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

The request goes to `POST {api_url}/scenarios/generate` with a 120-second timeout. The SDK removes `apiKey` from the request body that it records on the span.

Strings and dicts in `tools` are sent as they are. A callable is sent as its name and the first line of its docstring. `count` defaults to 5. The function also accepts `max_tokens` and `api_url`, and it returns a list of scenario dicts.

## User simulation

`simulate_user()` plays scenarios against your agent in your own process. The user side is scripted: each turn sends the scenario's next message, and no model writes the user's lines.

Your entrypoint is called once per turn with a `SimContext`. It returns a string or an `AgentTurnResult`, and it can be sync or async:

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

`ctx.turn` counts from 1. `ctx.state` lasts for the turns of one scenario, and `ctx.session_id` stays the same for all of them.

If no turn returns `tool_calls`, EvalKit uses the tool calls it traced during the scenario. `simulate_user()` does not read `retrieved_context`, so the scores include no contextual metrics.

`safe_mode`, on by default, reaches your entrypoint as `ctx.state["__safe_mode__"]`. The SDK does nothing else with it. `tags` are recorded on the run's spans as `evalkit.tags`.

A hand-written scenario is a dict like this one:

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

The user turns come from `turns`, or from `messages` when there is no `turns`. Each turn is a string, or a dict with `content` whose `role` is `user`, which is the default role. If neither key gives a message, the scenario plays one turn from `starting_prompt`. At most `max_turns` turns are played, 12 by default.

Each scenario is scored with `evaluate()` against its `expected_tools` and `constraints`, and its `target_keywords` are added to the required terms. `metrics=` limits which metrics run. The persona's name is recorded on the spans as `evalkit.persona`, and it does not change the scripted messages. `setup.state` seeds `ctx.state`.

Each scenario gets a root span named `scenario:<name>` and its own session, and the whole run shares one trace. A scenario whose entrypoint raises gets status `ERROR` and an `error` message, and the run goes on to the next scenario.

Every EvalKit span opened during a scenario carries `isSimulation`, `simulationId`, `runId` and `scenarioId`. You can filter for them in the dashboard. Spans that arrive from other OpenTelemetry instrumentors do not carry these fields.

## Simulation evaluation

`evaluate_simulation()` grades a finished simulation on the server. It runs an evaluator collection from your project, with an LLM as the judge. The judge uses your own provider key, which is never stored.

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

The collection id is on the Evaluators page, under Collections. `provider`, `model` and `api_key` are required, and the SDK sends `run_id` and `max_tokens` only when you pass them.

The request goes to `POST {api_url}/simulations/evaluate` with a 300-second timeout. The SDK removes `apiKey` from the request body that it records on the span.

## OpenTelemetry interop

EvalKit works with OpenTelemetry in both directions. Both are part of every install.

### OpenTelemetry spans as parents

When no EvalKit span is active, a new EvalKit span takes the active OpenTelemetry span as its parent and shares its trace ID. This uses `opentelemetry-api`, which the SDK installs.

### OpenTelemetry spans sent to SyntropyLabs

If your app has no tracer provider, `init()` creates one and makes it the global provider. It then registers every installed OpenTelemetry instrumentor except those on a skip list. The list holds the libraries that EvalKit instruments itself, plus Replicate and Transformers.

If your app already has a tracer provider, `init()` adds EvalKit's span processor to it and registers no instrumentors. Either way, each OpenTelemetry span is mapped to an EvalKit span and sent through the same exporter.

W3C `traceparent` carries the trace between services. EvalKit adds it to outbound HTTP requests, and the web middleware reads it from inbound requests.

### How duplicates are handled

EvalKit owns the libraries it patches. It drops OpenTelemetry spans from the scopes it owns, which cover `requests`, `httpx`, `aiohttp`, the databases it patches and Bedrock calls through botocore.

An LLM span from another instrumentor is kept, but EvalKit files it as `function_call` and drops its token and cache counts, so usage is not counted twice. Set `EVALKIT_COUNT_FOREIGN_LLM=1` when that instrumentor is the only one that reports the call.

The README has the details under [Spans from other OpenTelemetry instrumentors](README.md#spans-from-other-opentelemetry-instrumentors).

## Flushing and shutdown

A background daemon thread exports spans in batches, and the queue is flushed when the program exits. To export the queue synchronously, for example at the end of a short-lived script or a serverless handler, call `flush()`:

```python
evalkit.flush()              # waits at most 5 seconds; flush(timeout=...) to change it
```

It never raises. `evalkit.stats()` reports what is queued and dropped, failed exports, the last error and the circuit breaker's state.

## Troubleshooting

### Nothing shows up

Check that `init()` runs before your clients and your app are created, and that `subscription_key` is valid. Set `debug=True` to log each export, retry and auth failure.

After 3 consecutive 401 or 403 responses, the exporter stops for the rest of the process, and says so in a warning. A bad key never recovers on its own, so fix the key and restart. `evalkit.stats()["last_error"]` holds the last export error, and a receiver that stays down opens a circuit breaker (`circuit_state` `open`) that drops spans instead of retrying into it.

### Duplicate spans

A `patch_*` call on a client that `init()` already patched does nothing, so it cannot double your spans. A second LLM instrumentor does report each call again. See [How duplicates are handled](#how-duplicates-are-handled).

### Sensitive payloads

Set `capture_io=False` on decorators and lower `max_body_bytes`. Recorded HTTP bodies and database statements are capped at `max_body_bytes`, and a span over 8 MiB is trimmed before it is sent. To strip or mask content on every span, use `capture_content` and `mask`, described in [Content capture and masking](README.md#content-capture-and-masking).

### Threads and processes

Trace context carries into threads and thread pools without extra code. If you start other processes, call `init()` in each one.

### Span types

You will see these span types: `llm_call`, `tool_call`, `http_call`, `db_query`, `function_call`, `log` and `eval_result`.
