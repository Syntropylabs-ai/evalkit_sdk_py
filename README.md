# EvalKit Python SDK

EvalKit is the SyntropyLabs SDK for Python LLM applications. It traces what your application does and sends the traces to [SyntropyLabs](https://syntropylabs.ai). It can also run simulated users against your agent and evaluate the answers.

One call to `init()` instruments your LLM clients, HTTP calls, database queries and logging.

## Install

```bash
pip install syntropylabs-evalkit
```

The package is named `syntropylabs-evalkit`, and you import it as `evalkit`. It needs Python 3.9 or later. It also installs the `evalkit` command, which runs evaluations in CI and connects coding agents.

## Contents

- [Quick start](#quick-start)
- [What gets traced](#what-gets-traced)
- [Streaming latency (time to first token)](#streaming-latency-time-to-first-token)
- [Streaming HTTP response bodies](#streaming-http-response-bodies)
- [Request and response headers](#request-and-response-headers)
- [Web frameworks](#web-frameworks)
- [Trace your own code](#trace-your-own-code)
- [Manual spans](#manual-spans)
- [Identity](#identity)
- [Offline evaluation](#offline-evaluation)
- [Evaluation runs](#evaluation-runs)
- [Feedback and rewards](#feedback-and-rewards)
- [Prompts](#prompts)
- [Scenario simulation](#scenario-simulation)
- [Coding agents](#coding-agents)
- [Content capture and masking](#content-capture-and-masking)
- [Configuration](#configuration)

## Quick start

```python
import evalkit

evalkit.init(
    subscription_key="tk_live_...",   # Settings → This project → Environments
    service_name="my-service",
)

# Every OpenAI / Anthropic / HTTP / DB call from here on is traced automatically.
from openai import OpenAI

resp = OpenAI().chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

Your subscription key is on the Environments page of the SyntropyLabs console. To open it, choose Settings in the sidebar footer, then Environments under This project.

Call `init()` once, as early as you can. Trace context, including the trace ID, then carries over to threads and async tasks without extra code.

A second call to `init()` returns the client from the first call and ignores its arguments.

Until `init()` has run, most SDK functions raise `RuntimeError("[evalkit] call evalkit.init() before using the SDK")`. The identity and agent helpers, `session()`, `current_trace_id()`, `with_trace()` and `evaluate()` work without it.

Spans wait in a queue. A background thread exports them in batches, and the queue is flushed when your program exits.

## What gets traced

`init()` instruments these libraries when they are installed. You do not need to call anything else.

- LLM clients: OpenAI (Chat Completions and Responses), Anthropic, Bedrock (`boto3` and `aiobotocore`), Google GenAI, Google AI (`google.generativeai`), Cohere, Mistral, Groq and Ollama. Anthropic on Vertex (`AnthropicVertex`) and Google GenAI in Vertex mode are included.
- Agent frameworks: LangChain and LangGraph (through LangChain callbacks), LiteLLM, LlamaIndex, Google ADK, the Claude Agent SDK, and tool calls from MCP clients.
- Vector stores: Pinecone, Qdrant, Chroma and Weaviate.
- Voice: OpenAI audio and Realtime, ElevenLabs, AssemblyAI and Google GenAI Live.
- HTTP: `requests`, `httpx` and `aiohttp`, with method, URL, status, latency, headers and bodies.
- Databases: Redis, PyMongo (Motor included), asyncpg, psycopg 3 and PyMySQL, with query text and latency.
- Logging: records from Python's `logging` module. See [Logging](#logging).
- Your code: every public module-level function in your app's source tree, for application performance monitoring (APM). This is on by default. See [Trace your own code](#trace-your-own-code).

A few cases need a call of their own, or are not covered:

- The Vertex AI SDK's `GenerativeModel` (from `vertexai`) is not patched automatically. Call `evalkit.patch_vertex_generative_model(model)` on each model.
- SQLAlchemy is traced by its OpenTelemetry instrumentor when EvalKit creates the tracer provider. See [Spans from other OpenTelemetry instrumentors](#spans-from-other-opentelemetry-instrumentors). Otherwise, call `evalkit.patch_sqlalchemy_engine(engine)` or `evalkit.patch_async_sqlalchemy_engine(async_engine)`.
- psycopg2 is not instrumented. psycopg 3 is.
- OpenAI Responses calls made with the async client and `stream=True` are not traced.

### Logging

`init()` adds a handler to the root logger. Each record that passes the root logger's level becomes a `log` span. The root level is `WARNING` unless your app changes it, so by default only warnings and errors are recorded.

The span's operation is `log.<LEVEL>`, for example `log.ERROR`. Records at `ERROR` or above get status `ERROR`. The span has `log.level`, `log.message`, `log.logger`, `log.filename`, `log.lineno` and `log.funcname`. When the record has exception info, it also has `error.type`, `error.message` and `error.stack`.

`init()` also adds `trace_id` and `span_id` to every record. If the root logger has no stream handler, it adds a console handler with the format `%(asctime)s %(levelname)s [%(trace_id)s] %(name)s: %(message)s`. Otherwise it adds the ids to your stream handlers, and it replaces the formatter of any handler whose format lacks `trace_id`.

### Claude Agent SDK

The Claude Agent SDK runs the model in a subprocess, which the in-process patches cannot see. EvalKit wraps `claude_agent_sdk.query()` and `ClaudeSDKClient.receive_response()` and reads the tokens and cost from the final `ResultMessage`.

Each call becomes an `llm_call` span named `claude_agent_sdk.query`, with provider `anthropic`. Besides the tokens, it records the cost as `llm.cost_usd`, plus `gen_ai.finish_reason`, `llm.api_latency_ms` (the SDK's `duration_api_ms`) and `llm.num_turns`. `latency_ms` is the wall-clock time that EvalKit measures.

`init()` replaces `query` on the `claude_agent_sdk` module. A `from claude_agent_sdk import query` that runs before `init()` keeps the original function, so call `init()` first or call `claude_agent_sdk.query(...)`. `receive_response()` is patched on the class, so import order does not matter for it.

`init()` does this when `claude_agent_sdk` is installed. If you install it later, call `evalkit.patch_claude_agent_sdk()`.

### Spans from other OpenTelemetry instrumentors

EvalKit also receives spans from OpenTelemetry instrumentation. How it does so depends on whether your app already has a tracer provider.

If it has none, `init()` creates one with your `service_name` and makes it the global provider. It then registers every installed OpenTelemetry instrumentor except those on a skip list. The list holds the LLM, web framework, HTTP and database libraries that EvalKit instruments itself. It also holds Replicate, Transformers and psycopg2, which EvalKit has no patches for.

SQLAlchemy and urllib3 are not on the skip list, and their instrumentors are installed with the SDK, so their spans arrive this way.

If your app already has a tracer provider, `init()` adds EvalKit's span processor to it and registers no instrumentors. If that provider rejects span processors, EvalKit logs a warning and receives no OpenTelemetry spans from it.

EvalKit drops these OpenTelemetry spans, because they would repeat its own:

- spans from the scopes it owns: Bedrock runtime calls through botocore, the databases it patches, `requests`, `httpx` and `aiohttp`
- ASGI event spans
- spans from the Google ADK scope `gcp.vertex.agent`

A bridged span keeps its kind, its error description, its events, and its GenAI token and cache counts. It fills the `prompt`, `completion`, `environment`, `appVersion`, `userId`, `sessionId`, `deviceId` and `sdkVersion` columns from the matching `evalkit.*` attribute, such as `evalkit.prompt` or `evalkit.session_id`. The TypeScript SDK uses the same convention. To make your own OpenTelemetry span filterable, set those attributes on it.

Token counts are read from the current and the deprecated semantic-convention names, `gen_ai.usage.input_tokens` or `gen_ai.usage.prompt_tokens`. Cache counts are read from `gen_ai.usage.cache_read_tokens`, `gen_ai.usage.cache_read.input_tokens` or `gen_ai.usage.cached_tokens`.

Providers disagree on whether cached tokens are part of the input count. When the input count is smaller than the cache counts, EvalKit treats it as excluding them and adds them in. This keeps the cost breakdown comparable across providers.

Some spans look like an LLM call, because they have `gen_ai.*` attributes, but have neither `evalkit.span_type` nor `evalkit.sdk_version`. EvalKit treats such a span as a second copy of a call it already reported. It files the span as `function_call` and drops its token and cache counts, so they are not counted twice.

Set `EVALKIT_COUNT_FOREIGN_LLM=1` when the other instrumentor is the only one that reports the call. Only those two attributes exempt a span. An `evalkit.user_id` alone does not.

## Streaming latency (time to first token)

When you stream a response, for example with `stream=True`, the span records `gen_ai.server.time_to_first_token_ms`. This is the time to first token (TTFT), as an integer number of milliseconds. You do not need to turn it on.

All four EvalKit SDKs follow the same four rules, so the number is comparable across languages:

1. The clock starts before the request is sent, not at the first chunk. It covers span setup, connection and queueing, because the user waits through all of them.
2. Only generated text stops the clock. That is a content delta, or a thinking or reasoning delta from an extended-thinking model. Reasoning text is recorded separately as `gen_ai.response.thinking`, not as the completion.
3. Tool-argument deltas do not stop the clock. A response that is only a tool call has no first token, so the attribute is absent. It is not zero.
4. TTFT is not recorded when a stream fails, so a truncated stream cannot skew the percentiles.

The Python SDK departs from these rules in a few places:

- An Anthropic `input_json_delta`, which carries tool arguments, stops the clock.
- An OpenAI refusal delta counts as generated text.
- The OpenAI-format tracer behind OpenAI, Groq and LiteLLM reads only content and refusal deltas. A reasoning delta does not stop its clock.
- Ollama thinking text does not stop the clock.
- A Bedrock stream that reports an error event keeps its TTFT, although the span has status `ERROR`.

TTFT is recorded for these streams:

- OpenAI Chat Completions, sync and async, whether patched automatically or with `patch_openai_client` and `patch_async_openai_client`
- Anthropic `messages.create(stream=True)`, sync and async, on Vertex too
- Bedrock `converse_stream` and `invoke_model_with_response_stream`, with `boto3` and `aiobotocore`
- Google GenAI `generate_content_stream`, sync and async
- Google AI (`google.generativeai`) streams
- Vertex AI `GenerativeModel.generate_content(stream=True)`, once the model is patched
- Cohere `chat_stream`, Mistral, Groq, LiteLLM and Ollama

TTFT is not recorded for Anthropic's `messages.stream()` helper or for OpenAI Responses streams. A sync Responses stream is always recorded with status `OK`, even when it fails.

A stream that fails partway is recorded with status `ERROR` and the provider's error message. The span keeps the partial completion and any usage the provider had already reported. Those tokens were billed whether or not the stream finished, and the partial text is often the only clue to why it stopped.

A caller that stops reading early has not failed. Its span stays `OK` and keeps its TTFT.

## Streaming HTTP response bodies

EvalKit records outgoing HTTP calls with their bodies, streamed responses included. Reading a streamed body would consume it before your code sees it. So the tracer tees the stream: each chunk you read is copied into a capped buffer on its way to you, byte for byte.

The body is not known when the headers arrive, so the span is emitted when the stream ends, not when the response is returned. The first of these events ends the stream, and the span is emitted only once:

- the iterator is exhausted
- the response is closed or released
- the response is garbage collected
- `evalkit.flush()` runs

`flush()` is registered to run at exit. A response you never read still produces a span.

The span waits because the trace store is append-only, and a span that has been written cannot be updated. Sending the span first and the body later would write a second row for the same call. The call's cost would then be counted twice, permanently.

The body is recorded as `http.response.body`. Each library is teed at one point that every other way of reading the body goes through, so it does not matter how you read it:

| Library | Teed at | Also covers |
| --- | --- | --- |
| `requests` | `iter_content` | `.text`, `.content`, `.json()`, `iter_lines` |
| `httpx` | `iter_bytes` / `aiter_bytes` | `read`, `aread`, `iter_text`, `iter_lines`, `aiter_*` |
| `aiohttp` | a proxy over `resp.content` | `read()`, `text()`, `json()`, `iter_chunked`, `iter_any`, `async for` |

Teeing applies to a `requests` call made with `stream=True`, to an `httpx` response whose body is still unread when it is returned, and to every `aiohttp` response. It needs `capture_stream_bodies` turned on and `max_body_bytes` above 0.

A streamed call gets three timings, because how long it took and how soon it answered are separate questions:

- `latency_ms` is the whole call, from the first byte of the request to the last byte of the body.
- `http.server.time_to_headers_ms` is the time until the server sent its response headers.
- `http.server.time_to_first_byte_ms` is the time until the first body byte. It is the HTTP counterpart of TTFT.

These attributes show when a capture is partial:

- `http.response.streamed`: the body was read in pieces, not in one go.
- `http.response.stream.complete`: the stream ran to the end. `false` means the response was closed, abandoned or still open at flush, so the body is only a prefix.
- `http.response.stream.chunks`: the number of chunks that arrived, counted past the buffer cap too.
- `http.response.body.bytes`: the full size, which can be larger than the captured body.
- `http.response.body.truncated`: the buffer reached `max_body_bytes`, and the rest was dropped.

Each buffer holds at most `max_body_bytes`, and EvalKit tracks at most 1024 open streams. If an application leaks responses faster than that, the oldest stream's span is emitted early. It is not dropped.

To emit the span as soon as the headers arrive, with no body for streamed responses, turn this off:

```python
evalkit.init(subscription_key="tk_live_...", capture_stream_bodies=False)
# or
export EVALKIT_CAPTURE_STREAM_BODIES=false
```

## Request and response headers

HTTP and web-framework spans record request and response headers in `http.request.headers` and `http.response.headers`. Credential headers keep their names, but their values are recorded as `***`. These include `Authorization`, `Cookie`, `x-api-key`, `x-subscription-key` and the rest of the list in [Secrets are masked before export](#secrets-are-masked-before-export). You can see what was sent, and the key never leaves your process. All other headers, `traceparent` included, are recorded as sent.

Headers count as content, so `capture_content=False` removes them along with every other payload. A `mask` callback also receives them, and it can rewrite them.

To record credential headers in full, for example to debug a 401 from the trace alone, turn on `capture_secrets`:

```python
evalkit.init(subscription_key="tk_live_...", capture_secrets=True)
# or
export EVALKIT_CAPTURE_SECRETS=1
```

The Django middleware leaves some headers out even with `capture_secrets=True`. It never records the request headers `Authorization`, `Cookie`, `X-Api-Key`, `X-Auth-Token`, `Proxy-Authorization`, `X-Secret`, `Token`, `Password` and `X-Subscription-Key`. It never records the response headers `Set-Cookie`, `Authorization`, `X-Api-Key` and `X-Subscription-Key`.

## Web frameworks

Add the middleware for your framework:

```python
# FastAPI / Starlette
from evalkit import EvalKitMiddleware
app.add_middleware(EvalKitMiddleware)

# Flask
evalkit.instrument_flask(app)

# Django: add to MIDDLEWARE
"evalkit.EvalKitDjangoMiddleware"
```

Each middleware records one `http_call` span per request, named after the method and path, such as `GET /orders`. It continues an incoming `traceparent` header. It also adds `X-Trace-Id` and `traceparent` headers to the response.

- `EvalKitStarletteMiddleware` is the same class as `EvalKitMiddleware`, exported under a Starlette name.
- For Litestar, pass the middleware when you create the app: `Litestar(route_handlers=[...], middleware=[create_litestar_middleware()])`.
- `instrument_flask(app)` needs `init()` to have run first. It raises `RuntimeError` otherwise.
- Until `init()` runs, the FastAPI middleware passes requests through untraced. It traces HTTP requests only, so WebSocket and lifespan events pass through as well.
- The Django, FastAPI and Starlette middleware can fill in the user id. See [Identity](#identity).

## Trace your own code

Function tracing is on by default. As your modules are imported, `init()` wraps each public module-level function in your app's source tree. Each call becomes one `function_call` span with its input, output and latency.

The source tree is the directory of the file that called `init()`, or the current working directory if that file cannot be found. Methods, classes, nested functions and names that start with `_` are not wrapped. Installed packages, in `site-packages` and similar directories, are never wrapped.

```python
# Disable it
evalkit.init(..., function_tracing=False)   # or env EVALKIT_FUNCTION_TRACE=false

# Trace sibling packages outside the caller's directory
evalkit.init(..., trace_packages=["support_bot", "workers"])
```

`trace_packages` takes any importable module name, third-party packages included.

For finer control, opt in explicitly. You can trace a function, a tool, a class or a module:

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

- `trace_function(name=None, attributes=None, capture_io=True)` and `trace_tool(...)` take the same arguments. `trace_tool` records a `tool_call` span, which counts toward the tool metrics.
- `traced` wraps the methods a class defines itself, but not inherited ones, and skips names that start with `__`. Span names default to `ClassName.method`, and `prefix=` replaces the class name.
- `trace_module(module)` and `trace_package(package)` return the number of functions they wrapped. They skip names that start with `_` unless you pass `include_private=True`. `trace_package` imports every submodule of the package.
- These functions look up the client when they are applied. Call `init()` before the decorated code is imported.
- With `capture_io=True`, the default, arguments and return values are recorded as JSON and cut to 4096 characters, with `…` at the end. They are stored in `function.args`, `function.kwargs` and `function.result`, or in `tool.arguments` and `tool.result` for a tool.

> A client-side tool that the model calls shows its output only if you wrap it with `trace_tool`. The SDK sees the model's request, not your function's return value. Server-side tools, such as OpenAI's `web_search`, and LangChain tools are traced automatically.

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

`start_span(name, attributes=None, parent_ctx=None)` opens a span under the current span, or under `parent_ctx` when you pass one. It returns an `end` function and the span's context. Call `end(status)` when the work is done. The status defaults to `"OK"`.

The status values the SDK defines are `"OK"`, `"ERROR"` and `"UNSET"`. `end()` sends the string exactly as you pass it.

A span from `start_span` does not become the current span. To make the spans opened inside a block its children, wrap the block in `with evalkit.with_trace(ctx):`.

`start_trace(name, attributes=None, session_id=None, device_id=None, user_id=None, agent=None)` opens a root span and makes it current until you call its `end`. It returns `(trace_id, end, ctx)`. Inside another span, it opens a child in that span's trace. Its `end` also takes the keyword arguments `status_message`, `prompt`, `completion` and `attributes`, so the root span can show what was asked, what was answered and why it failed. The status message is cut to 2000 characters.

`start_http_trace(name, *, method, url, headers=None, request_body=None, attributes=None)` opens an `http_call` span in a new trace. Its `end(status, *, status_code=None, response_body=None, response_headers=None)` records the response.

`current_trace_id()` returns the id of the current trace. If no trace is active, it creates one.

## Identity

Every span can carry three ids. They group spans into sessions and users.

| Field | What it is | Where it shows up |
|---|---|---|
| `session_id` | One conversation or visit | The Sessions view, with the turns in order and the cost and latency of each session |
| `user_id` | The end user your app authenticated | The Users view, with volume, cost and failure rate per user. Per-user rate limits count by it too. |
| `device_id` | The installation or browser | Grouping before sign-in, when there is no user yet |

Set them for one trace:

```python
trace_id, end, ctx = evalkit.start_trace(
    "chat-turn", user_id="u_8123", session_id="conv_55", device_id="ios-9f2",
)
```

Or set them once for everything that follows, such as a request, a worker task or a whole script:

```python
evalkit.set_user(user_id="u_8123", session_id="conv_55")
...
evalkit.set_user()          # clears all three
```

`set_user()` stores the ids in a `contextvars.ContextVar`. A value set inside a request or an `asyncio` task stays in it and never leaks into another one. `evalkit.current_user()` returns the ids in effect.

Fields merge. `set_user(session_id=...)` followed by `set_user(user_id=...)` leaves both set, and a call with no arguments clears all three. `set_user()` returns a token, and `evalkit.clear_user(token)` restores the previous values.

Each field is resolved in this order, and the first value found wins:

1. An argument to `start_trace(...)`.
2. An id in the span's own attributes.
3. The parent span.
4. `set_user()` or `evalkit.session(...)`.
5. For `device_id` only, the `device_id` given to `evalkit.init()`.

Child spans inherit the ids of their trace, so the LLM calls, tool calls, database queries and HTTP calls in a trace all get the same three ids. Spans from other OpenTelemetry libraries keep the ids in effect where the span started, even though a background thread exports them.

Ids can also come from attributes on `start_trace`, `start_span` or a third-party OpenTelemetry span, under any of these names. When a span has more than one, the first name in the list that has a value wins.

| Field | Accepted attribute names |
|---|---|
| session | `evalkit.session_id`, `session.id`, `conversation.id`, `gen_ai.conversation.id`, `session_id` |
| user | `evalkit.user_id`, `user.id`, `enduser.id`, `user_id` |
| agent | `evalkit.agent_name`, `gen_ai.agent.name`, `agent.name` |

The device id is read only from `evalkit.device_id`.

The Django middleware fills `user_id` from `request.user`, and the FastAPI and Starlette middleware fill it from `scope["user"]`. They do this only for an authenticated user, and they use the first of `pk`, `id`, `identity`, `username` and `email` that has a value. A user id you set yourself wins.

Without these ids a trace is still recorded. It just cannot be grouped into a session or attributed to a user.

### Sessions

Wrap one conversation in `evalkit.session()`. Every span opened inside it joins that session:

```python
with evalkit.session("conv_55", user_id="u_8123"):
    answer = agent.run(message)

@evalkit.session("conv_55")        # also a decorator, sync or async
async def handle_turn(message): ...
```

On exit, the ids that were in force before the block come back. Concurrent requests or tasks never see each other's session. `user_id` is optional, and without it the current user is kept. A span that names its own session keeps it.

### Agents

Name the agent in code to group traces, sessions and scores by it:

```python
evalkit.set_agent("support-bot")                       # everything that follows
trace_id, end, ctx = evalkit.start_trace("turn", agent="planner")   # one trace
end_step, _ = evalkit.start_span("research", {"gen_ai.agent.name": "researcher"})  # a sub-agent
```

`set_agent` is scoped like `set_user`. It returns a token for `evalkit.clear_agent(token)`, and `evalkit.current_agent()` reads the current name. Spans carry the name in the attributes `evalkit.agent_name` and `gen_ai.agent.name`, and in the envelope field `agentName`.

Agent versions are detected automatically and listed under the agent's name. A version is defined by the system prompt, the tools, the model and the sampling parameters (temperature and top_p).

## Offline evaluation

`evaluate()` scores an output locally with deterministic metrics. It calls no judge model, so there is no judge-model cost. It works without `init()`. After `init()`, it also sends the scores as an `eval_result` span.

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

It returns only the metrics that your arguments allow:

| Metric | Computed when |
| --- | --- |
| `tool_trajectory` | `expected_tools` is not empty |
| `tool_f1` | `tool_calls` or `expected_tools` is not empty |
| `tool_correctness` | `expected_tools` is not empty |
| `response_match` | `constraints` has `required_terms` (or `requiredTerms`) |
| `constraint_compliance` | `constraints` is not empty |
| `contextual_precision`, `contextual_recall` | both `retrieved_context` and `expected_context` are given |

`metrics=` limits the result to the metrics you name. `input` and `expected_output` are accepted but not used in any score. `constraint_compliance` checks `required_terms`, `has_citations` and `slide_count`, and each key is also accepted in camelCase.

The span is named `evaluation`. It has `evalkit.eval_type` set to `offline` and one `evalkit.metric.<name>` attribute per score. If the span cannot be sent, `evaluate()` still returns the scores.

## Evaluation runs

`evalkit.Eval()` runs your dataset through the platform's LLM-as-judge evaluators from code. The run is saved as an evaluation job. You can see it in the dashboard, compare it with earlier runs and gate on it. The server takes the judge's provider key from your project's model configuration, so the SDK never sends one.

```python
def answer(row):
    return my_agent(row["input"])

def exact_match(row, output):
    return 1.0 if output.strip() == row.get("expected", "").strip() else None

def length_budget(row, output):
    return {"score": len(output) / 400, "passed": len(output) <= 400, "reason": "400 chars"}

result = evalkit.Eval(
    "nightly-regression",
    data=[{"input": "What is the return policy?", "expected": "30 days"}, ...],
    task=answer,                                  # omit if rows already carry "output"
    evaluators=["665f0c1234567890abcdef12", exact_match, length_budget],
    judge_model="gpt-4o",                         # a model configuration name in your project
    baseline_job_id="665f0c1234567890abcdefaa",   # optional: compare against an earlier job
    gate={"minPassRate": 0.9, "maxRegressionsVsBaseline": 0},
)
print(result.url)                                 # printed once as the run starts, too
print(result.job_id, result.average_score, result.pass_rate, result.passed)
if result.passed is not True:                     # None = undecided: fail closed
    sys.exit(1 if result.passed is False else 2)
print(result.local_scores["exact_match"])         # one entry per row, None preserved
for row in result.results:
    print(row["rowIndex"], row["overallScore"], row["passed"], row["status"])
```

`task` turns a row into an output. Leave it out when the rows already have an `output`. `Eval()` also takes `judge_model_config_id`, `auto_baseline`, `hyperparameters`, `batch_size` (default 200), `timeout` (default 600 seconds) and `api_url`.

### Evaluators and collections by id

Every evaluator on the Evaluators page has an id, whether it comes from the gallery or is your own. So does every collection, and the Use in code button on either one copies its id. Put rule ids in `evaluators` and collection ids in `collections`. They mix freely with each other and with local scorers:

```python
import evalkit

evalkit.init(subscription_key="tk_live_...")   # Settings → This project → Environments

rows = [
    {
        "input": "How long do I have to return an item?",
        "output": "You can return it within 30 days of delivery.",
        "expected": "30 days from delivery",
        "context": "Items can be returned within 30 days of delivery.",
    },
    {
        "input": "Do you ship to Canada?",
        "output": "Yes, in 5 to 7 business days.",
        "expected": "Yes, 5-7 business days",
        "context": ["We ship to the US and Canada.", "Canadian orders arrive in 5-7 business days."],
    },
]

result = evalkit.Eval(
    "support-suite",
    data=rows,
    evaluators=["665f0c1234567890abcdef12"],     # rule ids
    collections=["665f0c1234567890abcdefc0"],    # collection ids
    judge_model="gpt-5-mini",   # only needed when a rule has no judge model of its own
)
print(result.url)
print(result.pass_rate, result.average_score, result.overall_scores)
```

- The server expands a collection into its rules, in the collection's order. A rule named twice runs once.
- A single collection is scored the way the dashboard scores it. Its weights, per-rule thresholds, aggregation (mean, weighted or min) and hard gates decide each row. Rule ids and local scorers next to it join as ordinary members at weight 1.
- Two or more collections have no single aggregation to follow. Every rule then runs once at weight 1 and passes on its own threshold, as rule ids alone do.
- Only the evaluators and collections of your key's organization are accepted, managed evaluators from the gallery included. Any other id is refused as if it did not exist: `HTTP Error 404: Evaluation rule(s) not found: <id>` or `Evaluator collection(s) not found: <id>`. A rule that a collection lists but your organization cannot use is reported with its collection.
- A server that predates collections is detected when the run opens. The SDK raises `RuntimeError` before it sends any row, so it never scores only the rule ids.
- A row's `context` holds its retrieved documents, as one string or a list of strings.

### How a run is sent

1. `POST /evaluation-runs` creates the run. It answers `{runId, url, ruleIds}`, where `ruleIds` lists the rules it will score once collections are expanded. The SDK prints the link once while the rows upload and returns it as `result.url`, with `result.run_id`. Set `EVALKIT_QUIET=1` to stop the link from printing.
2. `POST /evaluation-runs/{runId}/rows` appends the rows in batches of at most 200. `batch_size=` sets a smaller batch, a larger value is clamped to 200, and a value below 1 raises `ValueError`. Every row carries its absolute `rowIndex`, so the server recognizes a row it already has. When a batch fails with a network error or a 5xx response, the SDK sends the same payload again, up to 3 attempts in all, after 0.25 and 0.75 seconds. No row is counted twice. A 4xx response raises at once.
3. `POST /evaluation-runs/{runId}/complete` judges every row, applies the gate to the whole run and compares the run with the baseline. However many batches were uploaded, the run has one job, one gate verdict and one comparison.

Row fields are `input`, `output` (or a `task` that produces it), `expected`, `metadata` and `context`. `target` is still accepted in place of `expected`. `context` is sent inside `metadata.context`.

### Local scorers

`evaluators` can mix server rule ids with local scorers. A rule id is a 24-character hex MongoDB id from the Evaluators page. A local scorer is a callable `(row, output)` that runs in your process. Its result goes with the row as `localScores` and appears in the run results with the evaluator source `local`.

A scorer can return:

- a number, which is the score
- a bool, which gives a score of 1 or 0 and a pass or fail verdict
- `None`, which means the scorer does not apply to the row, so nothing is sent for it
- a mapping `{"score": ..., "passed": ..., "reason": ...}`, which sets all three

Any other return value raises `ValueError`. So does a run that has no server rule id in `evaluators` and no collection id in `collections`.

### Scores and the gate verdict

- `average_score` is the mean over the rows that produced a finite score. `pass_rate` is `passed / (passed + failed)`. Both are `None` when they are undefined, never 0.
- `result.passed` is `gate_result["passed"]`. That is `True`, `False`, or `None` when a check could not be decided. A check is undecided when there is no comparable baseline, when the run never scored a `perMetricMinScore` metric, or when every row errored. `result.passed` is also `None` when no gate was given.
- Gate on `passed is not True`. A check for `passed is False` lets an undecided run through.
- The result also has `name`, `dataset_hash`, `total_rows`, `counts`, `errored_rows`, `indeterminate_rows` and `scoring_version`.

### Errors and older servers

Each request times out after 600 seconds by default (`timeout=`). A 4xx or 5xx response raises `urllib.error.HTTPError`.

A 400 or 404 from `POST /evaluation-runs` itself, such as a validation error or an evaluator id that does not exist, raises `urllib.error.HTTPError`. Its message is the server's reason, and `exc.response_text` holds the first 8192 bytes of the response body. It is never mistaken for an older server.

Some deployments predate the append endpoints, and `POST /evaluation-runs/{runId}/rows` answers 404. The SDK then falls back to one call per batch. Each batch becomes its own job (`result.job_ids`), the totals are recomputed on the client, and with more than one batch `gate_result` and `comparison` are `None`.

Rows sent this way carry `target` and `context` as top-level fields. The SDK detects an older server once per client, not once per batch. A run with `collections` never falls back, because the fallback cannot carry them.

### Run evaluations in CI

The `evalkit eval` command makes the same run from the command line and turns the verdict into the exit code. A pipeline step then fails when the gate fails, and also when the gate cannot be decided.

```bash
evalkit eval run eval.json                          # a JSON config
evalkit eval run evals/checkout.py                  # a module: its `eval` attribute
evalkit eval run evals/checkout.py:smoke --json     # another attribute of it
evalkit eval run evals.checkout:smoke               # an importable module works too
evalkit eval gate <runId>                           # re-read the gate of a run made elsewhere
```

| Exit code | Meaning |
|---|---|
| `0` | `gate_result["passed"]` is `True`. |
| `1` | It is `False`. |
| `2` | Undecided: it is `None`, or the run had no gate. Pass `--allow-undecided` to exit with `0` here. |
| `3` | The run could not be made or read, for example because of a bad config, an HTTP error or a missing key. |

Any other exception is not caught, so Python ends the command with a traceback and exit code 1.

`eval.json` holds the `Eval()` arguments in camelCase, so the same file works with the TypeScript CLI. The keys are `name`, `evaluators`, `collections`, `gate`, `judgeModel`, `baselineJobId`, `autoBaseline`, `hyperparameters`, `batchSize` and `timeoutMs`. The Python CLI also accepts `judgeModelConfigId`, `timeout` in seconds, and the snake_case names. An unknown key is an error.

The rows go either inline as `data` or in a file named by `dataFile`. The file is a `.jsonl` file of rows, a `.json` array, or a JSON object with a `rows` array, and its path is resolved relative to the config. The rows must already contain their outputs:

```json
{
  "name": "checkout · ${GITHUB_SHA}",
  "dataFile": "rows.jsonl",
  "evaluators": ["665f0c1234567890abcdef12"],
  "judgeModel": "gpt-5-mini",
  "autoBaseline": true,
  "gate": { "minPassRate": 0.9, "maxRegressionsVsBaseline": 0 }
}
```

A module target is `path/to/file.py[:attr]` or `package.module[:attr]`, and `attr` defaults to `eval`. The attribute can be:

- a dict of `Eval()` keyword arguments, in snake_case or camelCase
- a callable with no arguments that returns such a dict, which is where a `task` and local scorers go
- the `EvalResult` of a callable that runs `evalkit.Eval()` itself

The key comes from `--key`, then `EVALKIT_API_KEY`, then `EVALKIT_SUBSCRIPTION_KEY`, then the key that `evalkit coding install` stored. The API base comes from `--api-url`, then `EVALKIT_API_URL`, then the coding endpoint with `/v1/otlp` removed. That endpoint is `EVALKIT_OTLP_ENDPOINT`, or the one `evalkit coding install` stored, or `https://api.syntropylabs.ai/v1/otlp`, so the base defaults to `https://api.syntropylabs.ai`.

The command prints the run link, the pass rate, the average score and the average per metric. Then it prints a table with one row per gate check, with columns for the check, its threshold, the actual value and the result (`passed`, `failed` or `undecided`). Last come the verdict and the exit code it maps to.

`--json` prints one object, to paste into a job summary or parse with `jq`. Its keys are `runId`, `jobId`, `url`, `name`, `totalRows`, `erroredRows`, `indeterminateRows`, `passRate`, `averageScore`, `overallScores`, `gate` (with `passed`, `checks` and `evaluatedAt`) and `verdict`. For example, in a GitHub Actions step:

```yaml
- run: evalkit eval run eval.json --json > eval-result.json
  env:
    EVALKIT_API_KEY: ${{ secrets.EVALKIT_API_KEY }}
```

`evalkit eval gate <runId>` reads `GET /evaluation-runs/{runId}` with the same key and applies the same exit codes. Use it when one job runs the evaluation and another decides. Its request times out after 30 seconds.

## Feedback and rewards

`evalkit.feedback()` attaches a human or end-user signal to a trace that your app already produced. The call goes to the control plane with your subscription key. The feedback is stored as a score on the trace, beside the judge scores. RL dataset builds can use it as a reward source.

```python
import evalkit

evalkit.feedback(trace_id, "thumbs", True, comment="answered in one line", user_id="u_42")
evalkit.feedback(trace_id, "rating", 4, session_id="s_9")
evalkit.feedback(trace_id, "text", "too verbose", span_id=span_id)
evalkit.reward(trace_id, 0.8, name="task_reward")
```

| Kind | Value | Stored as |
|---|---|---|
| `thumbs` | `True` or `False`, `"up"` or `"down"`, or 1 or 0 | 1 or 0 |
| `rating` | A number from 1 to 5 | The number |
| `reward` | A number from 0 to 1 | The number |
| `text` | A non-empty string | The text |

The server rejects any other value with a 400 error, such as `rating must be a number in [1, 5]`.

The score's source is `end_user` when you pass `user_id`, and `sdk` otherwise. This applies to every kind.

The score is named `reward` for a reward and `feedback:<kind>` for the other kinds, for example `feedback:thumbs`. Pass `name=` to choose another name. With `span_id`, the score is attached to that span. Without it, the score is attached to the trace.

`evalkit.reward(trace_id, value, **kwargs)` sends a `reward`. Both functions take the optional keyword arguments `comment`, `user_id`, `session_id`, `span_id`, `name`, `api_url` and `timeout` (default 30 seconds). The request goes to `/v1/feedback` on the `api_url` argument, then `EVALKIT_API_URL`, then the `api_url` given to `init()`.

The SDK raises `ValueError` for an unknown kind, a missing `trace_id`, an empty text value, or a rating or reward that is not a number. A bool does not count as a number here. A failed request raises too. Like every control-plane call, the request is recorded as an `http_call` span.

## Prompts

`evalkit.prompts.get()` fetches a versioned prompt from the registry. You can compile its `{{variables}}` and stamp the LLM calls it drives, so the UI shows usage per prompt version. Reads go to the control plane with your subscription key, and the SDK caches each read for `cache_ttl` seconds.

```python
import evalkit

prompt = evalkit.prompts.get("support_agent_system", label="production")
messages = prompt.compile({"customer": "Ada", "plan": "pro"})

with prompt.stamped():
    client.chat.completions.create(model=prompt.config.get("model", "gpt-4o-mini"), messages=messages)

evalkit.Eval("support-golden", data=rows, evaluators=[RULE_ID], hyperparameters=prompt.hyperparameters())
```

- `evalkit.prompts.get(name, *, label=None, version=None, variables=None, api_url=None, cache_ttl=60.0, timeout=30.0)` fetches a prompt. Pass `label` or `version`, not both, or it raises `ValueError`. The server picks the version: the given `version`, else the given `label`, else the `production` label, else the latest version. With `variables`, the server compiles the template, and the SDK compiles it locally if the server did not.
- `prompt.template`, `prompt.variables`, `prompt.config`, `prompt.semver` and `prompt.labels` are what the registry stores.
- `prompt.compile(variables)` substitutes Mustache-style variables locally. Strings go in as they are, numbers and booleans as text, and objects as JSON. A missing variable raises `ValueError`, which names every missing one.
- `with prompt.stamped():`, or `prompt.stamp()` and then `prompt.unstamp()`, stamps every `llm_call` span created in that context. The span gets `prompt.ref` (`name@version`), `prompt.name` and `prompt.version`. Attributes you set yourself win.
- `prompt.hyperparameters()` returns `{"promptName", "promptVersion"}` for `Eval(...)`, so the run links to the prompt version.
- `evalkit.prompts.clear_cache()` drops the cached reads, for tests and hot reloads.

## Scenario simulation

Simulation tests your agent against synthetic users in three steps. You generate scenarios from your agent's prompt and tools, replay each one against your real agent, and grade the run with LLM-as-judge evaluators.

### Generate scenarios

`generate_scenarios()` has the server write test scenarios. You bring your own provider key for the generation call:

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

It returns a list of scenario dicts, and `count` defaults to 5. It also accepts `function_schemas`, `context`, `rag`, `categories`, `temperature`, `reasoning_effort`, `max_completion_tokens`, `max_tokens` and `api_url`.

### Replay them against your agent

`simulate_user()` replays each scenario against an entrypoint that calls your agent:

```python
def entrypoint(ctx: evalkit.SimContext) -> evalkit.AgentTurnResult:
    # ctx.message:    the synthetic user's message for this turn
    # ctx.session_id: stable per scenario; use it to keep multi-turn context
    reply, tools_used = run_my_agent(ctx.session_id, ctx.message)
    return evalkit.AgentTurnResult(text=reply, tool_calls=[{"name": t} for t in tools_used])

report = evalkit.simulate_user(entrypoint, scenarios, tags=["ci"])
print(report["simulation_id"], report["run_id"])
```

The user side is scripted. Each turn's message comes from the scenario's `turns` or `messages`, or from its `starting_prompt` when it has neither. A scenario plays at most `max_turns` turns, 12 by default.

The entrypoint receives a `SimContext` with `message`, `session_id`, `state` and `turn`. It returns a string or an `AgentTurnResult` with `text`, `tool_calls` and `retrieved_context`. It can also be a coroutine function.

Each scenario gets its own root span, named `scenario:<name>`, and its own session. The whole run shares one trace. `agent=` names the agent under test on every span of the run. Each scenario is also scored locally with `evaluate()`, against its `expected_tools`, `constraints` and `target_keywords`, and `metrics=` limits which metrics run. `simulate_user()` returns a dict with `simulation_id`, `run_id` and `results`.

### Grade the run

`evaluate_simulation()` grades a run against an evaluator collection, with a judge model that you supply the key for. `provider`, `model` and `api_key` are required. Scores come back per scenario and per criterion, with reasons, and they also appear in the dashboard:

```python
result = evalkit.evaluate_simulation(
    report["simulation_id"],
    collection_id="665f0c...",            # Evaluators → Collections
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

The request goes to `POST /simulations/evaluate` with a 300-second timeout. The SDK removes `apiKey` from the request body that it records on the span.

## Coding agents

The `evalkit` command that comes with the SDK also connects coding agents. `evalkit coding` points a coding agent's own telemetry at your EvalKit environment. Its sessions then appear as one trace per turn, with the main agent, every subagent and everything they did. It supports Claude Code, Codex, Gemini CLI, Cursor, Windsurf and OpenCode.

```bash
evalkit coding install --vendor claude-code --key tk_live_...
evalkit coding install --vendor codex --key tk_live_...
evalkit coding install --vendor gemini-cli --key tk_live_... --capture full
evalkit coding install --vendor cursor --key tk_live_... --scope project
evalkit coding install --vendor opencode --key tk_live_...
evalkit coding status
evalkit coding doctor
evalkit coding backfill --vendor codex --last 5
evalkit coding uninstall --vendor codex
```

### Install options

- `--key`: your subscription key. Without it, `install` uses `EVALKIT_API_KEY` or the stored key.
- `--endpoint`: the OTLP endpoint. It defaults to the stored endpoint, else `https://api.syntropylabs.ai/v1/otlp`.
- `--tenant`: the tenant id that `doctor` reads traces back with. It is optional. See [Check the setup](#check-the-setup).
- `--scope user|project`: `project` writes into the current directory.
- `--capture minimal|full`: whether prompt, reply and tool text leave your machine. It defaults to the stored mode, else `minimal`.
- `--no-traces`: send log events only. Spans are on by default wherever the vendor has them.
- `--no-subagents`: for Claude Code, write no hooks at all.
- `--dry-run`: print what would be written, and write nothing.
- `--managed`: for Claude Code, print a managed-settings snippet and write nothing.

### Backfill Codex replies

`evalkit coding backfill --vendor codex` reads Codex's own rollout files (`~/.codex/sessions/**/rollout-*.jsonl`). It sends the assistant replies that Codex's telemetry never exports, as one `codex.assistant_message` log record per message. It adds `codex.assistant_reasoning` when the rollout has reasoning text. Each record carries the turn's own OpenTelemetry trace id when the rollout recorded one.

The options are `--session <thread id>`, `--last N`, `--since <ISO>`, `--all`, `--dry-run` and `--force`. The command prints one summary line per session. It keeps a byte offset per file in `~/.config/evalkit/codex-offsets.json`, so running it again never sends the same message twice. It sends no text unless the stored capture mode is `full`.

### Where the configuration is stored

The key and endpoint are stored once in `~/.config/evalkit/config.env`, with mode 0600. `~/.config/evalkit/state.json` records exactly what was written. `uninstall` removes only that and restores any value it replaced. The key is never printed.

### What you get per vendor

| Vendor | Tree per turn | Tool calls | Tool output | Prompt and reply | Subagents | Tokens and cost |
|---|---|---|---|---|---|---|
| `claude-code` | Yes | Yes | With `--capture full` | With `--capture full` | Yes | Yes |
| `codex` | Yes | When Codex emits them | With `--capture full` | With `--capture full` | No | Yes |
| `gemini-cli` | Yes | Yes | No | With `--capture full` | As `agent_call` spans | Tokens, no cost |
| `cursor` | Yes | Yes | With `--capture full` | With `--capture full` | Yes | No |
| `windsurf` | Yes | Yes | With `--capture full` | With `--capture full` | No | No |
| `opencode` | Yes | Yes | With `--capture full` | With `--capture full` | Child sessions | When OpenCode reports them |

- Claude Code: the tree is the agent's own spans, and `--no-traces` falls back to flat log events. Subagents nest under the `Agent` tool call. Tokens are on the span, and the cost comes from the `api_request` event.
- Codex: the tree is the turn span, and the receiver drops Codex's ~150 internal plumbing spans. The prompt comes from Codex and the reply from the rollout file, through the `Stop` hook or `backfill`. Tokens come from `sse_event` and the cost from `turn_cost`.
- Gemini CLI: the tree is made of `agent_call`, `llm_call` and `tool_call` spans.
- Cursor: the hooks build the tree with deterministic span ids. Subagents come from `subagentStart` and `subagentStop`. Cursor's hooks carry neither tokens nor cost.
- Windsurf: the tree is built the same way. Cascade has no subagent events.
- OpenCode: the plugin builds the tree, and child sessions nest under their parent turn. Tokens and cost are recorded when OpenCode reports them on the assistant message.

### What `install` writes

#### Claude Code

`install` writes `env` keys into `~/.claude/settings.json`, or `.claude/settings.json` with `--scope project`. The same file drives the VS Code and JetBrains versions.

- Always: `CLAUDE_CODE_ENABLE_TELEMETRY`, `OTEL_*_EXPORTER=otlp`, `OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf`, the endpoint, an `Authorization=Bearer` header and the resource attribute `evalkit.vendor=claude-code`.
- For span mode, which is on by default: `CLAUDE_CODE_ENHANCED_TELEMETRY_BETA=1`, `OTEL_TRACES_EXPORTER=otlp` and `OTEL_TRACES_EXPORT_INTERVAL=1000`. `--no-traces` removes these three.
- With `--capture full`: `OTEL_LOG_USER_PROMPTS`, `OTEL_LOG_ASSISTANT_RESPONSES`, `OTEL_LOG_TOOL_DETAILS`, `OTEL_LOG_TOOL_CONTENT`, `OTEL_LOG_RAW_API_BODIES` and `CLAUDE_CODE_OTEL_CONTENT_MAX_LENGTH=524288`. Claude Code cuts each body at 60 KB by default, which is smaller than its own system prompt, so the cap is raised to 512 K characters.

In span mode, Claude Code opens one trace per turn. A `claude_code.interaction` root has `llm_request` and `tool` children, and a subagent's work nests under the `Agent` tool's execution span. Content and cost still arrive as log events: `api_request`, `api_request_body`, `api_response_body`, `assistant_response`, `user_prompt`, `tool_result` and `subagent_completed`. The receiver merges them onto the matching span by trace and span id.

Unless you pass `--no-subagents`, `install` also adds `PostToolUse` and `SubagentStop` hooks. They have a 5-second timeout and always exit 0. They post `claude_code.tool_output` and `claude_code.subagent_stop`, which add the output of tools other than Bash and the subagent's last message.

`uninstall` deletes those keys and restores the previous values, including a key you had already set to the value EvalKit writes. It removes only the hooks whose command contains `coding hook --vendor claude-code`. It also remembers files written under the other `--scope`.

#### Codex

`install` writes an `[otel]` table between `# evalkit-managed-begin/end` markers in `~/.codex/config.toml`, or `.codex/config.toml` with `--scope project`. The table points the log, trace and metrics exporters at `<endpoint>/v1/{logs,traces,metrics}`. `--no-traces` writes `trace_exporter = "none"`. `install` refuses to run if an `[otel]` table already exists outside the markers.

With `--capture full`, `install` also merges `Stop` and `SessionEnd` hook entries into `~/.codex/hooks.json` and leaves any other hook alone. `Stop` has a 5-second timeout. `SessionEnd` has 3 seconds, because Codex clamps that event to 3.

Codex exports the turn span natively, with `codex.api_request`, `sse_event` token counts, `turn_cost`, `tool_result` and `user_prompt`. The event name is in the `event.name` attribute.

Codex never exports the assistant's reply, so the hook reads it from the rollout file that the CLI writes anyway. The hook posts `codex.assistant_message` and `codex.assistant_reasoning` with `conversation.id`, `turn.id`, `item.id` and `completion`. It uses the trace id that Codex used for that turn (`event_msg/task_started.trace_id`). Sessions from before 2026-09 lack that id, and the receiver then joins the records by `turn.id`.

Codex marks a newly written hook as untrusted and asks you to review it before it runs. `backfill` covers whatever the hook missed.

`uninstall` removes the block and only the hook entries whose command contains `coding hook --vendor codex`.

#### Gemini CLI

`install` writes a `telemetry` block into `~/.gemini/settings.json`. It sets `otlpProtocol: http`, sets `traces: true` unless you pass `--no-traces`, and leaves `logPrompts` false unless you pass `--capture full`. It also writes the `OTEL_EXPORTER_OTLP_HEADERS` line between `# evalkit-begin/end` markers in `~/.gemini/.env`.

Gemini CLI sends its own spans, with `gen_ai.operation.name` set to `agent_call`, `llm_call` or `tool_call`. It also sends `gemini_cli.*` events, correlated by `prompt_id`.

`uninstall` restores the previous `telemetry` values and removes the marker block.

#### Cursor

`install` writes 12 hook entries into `~/.cursor/hooks.json`. Each one runs `evalkit coding hook --vendor cursor --event ...` with a 5-second timeout. Other hooks are left alone.

Cursor has no telemetry export, so the hook builds the trace itself, with deterministic ids:

- The trace id is `sha256(conversation_id + generation_id)[:32]`.
- The turn's root span id is `sha256(generation_id)[:16]`.
- A tool span id is `sha256(tool_use_id)[:16]`, and the span starts at the current time minus `duration`.
- A subagent span id is `sha256(subagent_id)[:16]`, and its parent is `sha256(tool_call_id)[:16]`.

`beforeSubmitPrompt` opens the turn. `afterAgentResponse` and `stop` complete it with an enrichment record, and `subagentStop` completes the subagent.

`uninstall` removes only the entries whose command contains `coding hook --vendor cursor`. A reinstall also drops entries for events that EvalKit no longer uses.

#### Windsurf

`install` writes 6 hook entries into `~/.codeium/windsurf/hooks.json`, or `.windsurf/hooks.json` with `--scope project`. The trace is built the same way as Cursor's, with `trajectory_id` as the session and `execution_id` as the turn.

`pre_user_prompt` opens the turn. `post_run_command`, `post_write_code`, `post_read_code` and `post_mcp_tool_use` become tool spans, and `post_cascade_response` completes the turn. `uninstall` removes only EvalKit's entries, as it does for Cursor.

#### OpenCode

`install` writes the plugin to `~/.config/opencode/plugins/evalkit.ts`, with no key inside, and adds a `"plugin"` entry to `~/.config/opencode/opencode.json`. With `--scope project`, it writes `.opencode/plugins/evalkit.ts` and `opencode.json` in the current directory.

The plugin turns `session.*`, `message.updated`, `message.part.updated` and `tool.execute.before|after` events into the same span shapes. It reads the key from `~/.config/evalkit/config.env` at run time and posts OTLP JSON.

`uninstall` deletes the plugin file and removes only EvalKit's entry from `"plugin"`.

### Check the setup

`evalkit coding status` prints the endpoint, the masked key, the capture level and the last `doctor` result. For each vendor, it prints the scope, the capture level, whether traces are on and which hooks are installed.

`evalkit coding doctor` posts one synthetic turn: an `agent` root with an `llm_call` child and a `tool_call` child, plus one enrichment record. It then reads `GET <api>/v1/traces/<tenant>/<traceId>` back with the same key, up to 4 times, 0.75 seconds apart. It checks that the merged tree came back with at least 3 spans and a prompt.

The read API scopes traces by tenant. Without a tenant, the tenant in the path is `me`, which a receiver from 2026-09-15 on resolves to the key's tenant. So a plain `evalkit coding doctor` works against a current receiver.

`doctor` takes the tenant from `--tenant`, then `EVALKIT_TENANT_ID`, then the tenant that `install --tenant` stored. Only `install` stores it. Pass a tenant only to a receiver older than 2026-09-15. Such a receiver answers the `me` read with 403, and doctor reports `read back forbidden at <url>: this receiver predates the 'me' alias; pass --tenant <your tenant id> or update the receiver`.

A tenant that the key does not belong to gets `read back forbidden at <url>: the key belongs to a different tenant than '<tenant>'`. Any other failed read ends with `at <url>`. The URL in the message is the one that was read. The `exporter <endpoint>` part at the end of the line is the ingest endpoint.

The `hook` subcommand always exits 0 and never writes to stdout. It gives up after 5 seconds, so it cannot block the editor.

## Content capture and masking

Two separate controls decide what leaves your process. `capture_content` is the coarse switch, and `mask` is the fine one.

```python
evalkit.init(
    subscription_key="tk_live_...",
    capture_content=False,
)
```

`capture_content=False` keeps every metric and drops every payload.

- You keep tokens, cost, latency, TTFT, model, provider, finish reason, status, errors, tool names and call IDs, the span hierarchy, and `db.system`, `db.operation` and row counts.
- You lose prompts, completions, thinking text, request and response bodies and headers, tool arguments and results, traced-function arguments and return values, SQL text and bound parameters, retriever queries, and log message bodies.

Stripped spans carry `evalkit.content_captured=false`, so the dashboard can tell "capture is off" apart from "the SDK failed to read the body".

You can also set it with an environment variable, which helps when the same image ships to several environments:

```bash
export EVALKIT_CAPTURE_CONTENT=false
# or the standard OpenTelemetry variable, which EvalKit also honours:
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=false
```

The setting is resolved in this order: the `capture_content=` argument, then `EVALKIT_CAPTURE_CONTENT`, then `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`. If none is set, capture is on.

EvalKit's switch is broader than the OpenTelemetry one on purpose. The standard variable covers GenAI message content only. When EvalKit reads it, turning it off also removes database statements and traced-function arguments.

For finer control, pass a `mask` callback. It runs on every envelope just before the envelope is queued for export, whichever instrumentation produced it. That includes spans that came through the OpenTelemetry bridge from third-party instrumentors.

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

Change the envelope and return it, or return `None` to drop it. If the callback raises, or returns anything other than an envelope or `None`, the span is dropped and a warning is logged.

A mask exists to keep content inside the process, so it fails closed. It is the only place in the SDK that does. Everywhere else, instrumentation failures are caught, and tracing degrades while your app keeps running.

When both controls are set, `capture_content` applies first, so `mask` sees the stripped envelope. The secret scrubber described next runs between the two.

### Secrets are masked before export

Every envelope goes through a secret scrubber before your `mask` callback runs and before anything leaves the process. The scrubber runs whether or not content capture is on. Turning capture off only ever removes more.

#### By key

The scrubber replaces the value of any object key that matches one of these names with `"***"`:

`api_key`, `api-key`, `apikey`, `secret`, `token`, `password`, `passwd`, `authorization`, `credential`, `credentials`, `subscription_key`, `subscription-key`, `x-subscription-key`, `cookie`, `set-cookie`, `private_key`, `access_key`, `client_secret`

The match is case-insensitive, on the whole key or on a segment delimited by `.`, `_` or `-`. It works recursively through objects and arrays. It also covers attribute values that are JSON strings, which are parsed, scrubbed and serialized again. So `subscriptionKey`, `X-Subscription-Key`, `http.request.api_key` and `credentials.password` are all masked.

`token` matches only as the whole key or the final segment, because it is also a telemetry word. `prompt_token_count`, `gen_ai.server.time_to_first_token_ms`, `input_tokens` and `tokenizer` are metrics, not credentials. Masking them would blank a dashboard column and protect nothing.

Booleans and numbers under a matching key become `"***"` as well, so the shape of the value gives nothing away.

`http.request.headers` and `http.response.headers` also go through the sensitive-header list: `authorization`, `cookie`, `set-cookie`, `x-api-key`, `api-key`, `x-auth-token`, `proxy-authorization`, `x-secret`, `x-access-token`, `x-subscription-key`, `subscription-key`, `token` and `password`.

#### By pattern

Inside any string, the scrubber replaces these patterns with their prefix followed by `***`. That covers attributes, event attributes, the operation, the prompt, the completion and the status message.

| Pattern | Becomes |
| ---------------------------------- | ------------- |
| `tk_live_[0-9a-f]{8,}`             | `tk_live_***` |
| `tk_test_[0-9a-f]{8,}`             | `tk_test_***` |
| `sk-[A-Za-z0-9_-]{16,}`            | `sk-***`      |
| `Bearer\s+[A-Za-z0-9._~+/=-]{8,}`  | `Bearer ***`  |
| `AKIA[0-9A-Z]{16}`                 | `AKIA***`     |
| `ghp_[A-Za-z0-9]{20,}`             | `ghp_***`     |

A string with no secret in it is kept byte for byte, JSON strings included. Nothing is reformatted.

#### URLs

`http.url`, `url.full`, `http.target` and `server.address` are always read as URLs, and `url.query` is read as a bare query string. Any other attribute value that is a single `http://`, `https://`, `ws://` or `wss://` URL and nothing else is read as a URL too, including values nested inside JSON strings.

In a URL, the userinfo is masked: `https://user:pass@host/x` becomes `https://***@host/x`.

A query or fragment parameter has its value replaced by `***` when its key is on the by-key list or the sensitive-header list. The same applies when the key is `key`, `sig` or `signature`, as the whole key or its final segment. Keys are matched case-insensitively after URL-decoding.

Every other parameter keeps its value and position. `?api_key=AKIA…&page=2` becomes `?api_key=***&page=2`, and `?q=hello` is left as it is. A URL inside free text, such as a prompt, a completion, an operation name or a log line, goes only through the pattern list.

#### Identity is not masked

Identity is left unmasked on purpose. Email addresses, user IDs, session IDs and device IDs pass through unchanged, because the Users view, per-user cost and session replay are built on them.

To turn the scrubber off, for example to debug your own auth failure:

```python
evalkit.init(subscription_key="tk_live_...", capture_secrets=True)
# or
export EVALKIT_CAPTURE_SECRETS=1
```

`capture_sensitive_headers=` and `EVALKIT_CAPTURE_SENSITIVE_HEADERS` are aliases of the same switch. When both spellings are set, `capture_secrets` wins. Credential headers are masked unless you turn the switch on.

## Configuration

These are the main `init()` options:

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
    capture_secrets=None,                     # None = env var, else True/False
    capture_sensitive_headers=None,           # alias of capture_secrets
    capture_stream_bodies=None,               # None = env var, else True/False
    mask=None,                                # callable(envelope) -> envelope | None
)
```

`init()` also takes these options:

| Option | Default | Meaning |
| --- | --- | --- |
| `max_export_batch_size` | `512` | When this many spans are waiting, the queue is exported at once. |
| `scheduled_delay_ms` | `5000` | How often, in milliseconds, the background thread exports |
| `max_queue_size` | `4096` | The most spans waiting for export. When the queue is full, the oldest span is dropped. |
| `max_body_bytes` | 10 MiB | The largest HTTP body recorded per request or response |
| `app_version` | `None` | Recorded on every span as `evalkit.app_version` |
| `device_id` | `None` | The device id for spans that have none of their own |

`service_name` defaults to `unknown`. `environment` is recorded on every span as `evalkit.environment`, and every span also carries `evalkit.sdk_version`. With `debug=True`, the exporter logs each export, retry and auth failure at `DEBUG` level on the `evalkit` logger.

Spans are batched and exported in the background, and the queue is flushed at exit. To export the queue at any other point, call `flush()`:

```python
evalkit.flush()
```

### How spans are exported

The SDK sends spans to `{base_url}/v1/ingest` with your key in the `X-Subscription-Key` header. Each request times out after 10 seconds.

- A 5xx response or a network error is retried up to 3 times, after 0.2, 0.6 and 1.8 seconds.
- Any other 4xx response drops the batch with a warning.
- A 401 or 403 is not retried. After 3 in a row, the SDK stops exporting for the rest of the process and logs a warning to check your subscription key. A successful export resets the count.
- A span larger than 8 MiB is trimmed in stages until it fits. First, request and response bodies longer than 1024 characters are replaced by a note. Next, a long prompt and completion are replaced the same way, and the status message is cut to 4096 characters. Then every string attribute is cut to 512 characters. As a last step, the attributes are replaced by a single `trace.truncated` marker, and the events, prompt and completion are removed. A span that still does not fit is dropped.

### Environment variables

| Variable | Effect |
| --- | --- |
| `EVALKIT_API_URL` | Sets the control-plane base for `Eval()`, feedback, prompts, `generate_scenarios()` and `evaluate_simulation()`. It does not change where traces go. |
| `EVALKIT_CAPTURE_CONTENT` | `false` drops payloads. See [Content capture and masking](#content-capture-and-masking). |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | The same, read after `EVALKIT_CAPTURE_CONTENT` |
| `EVALKIT_CAPTURE_SECRETS` | `1` turns off secret masking. |
| `EVALKIT_CAPTURE_SENSITIVE_HEADERS` | An alias of `EVALKIT_CAPTURE_SECRETS`, read after it |
| `EVALKIT_CAPTURE_STREAM_BODIES` | `false` stops teeing streamed HTTP bodies. |
| `EVALKIT_FUNCTION_TRACE` | Exactly `false` turns off function tracing. |
| `EVALKIT_COUNT_FOREIGN_LLM` | `1`, `true` or `yes` keeps the token counts of foreign LLM spans. |
| `EVALKIT_QUIET` | `1`, `true`, `yes` or `on` stops the evaluation-run link from printing. |

An argument to `init()` or to the function wins over its environment variable. The capture variables accept `1`, `true`, `yes` or `on`, and `0`, `false`, `no` or `off`. For any other value, EvalKit logs a warning and keeps the default.

The `evalkit` command also reads `EVALKIT_API_KEY`, `EVALKIT_SUBSCRIPTION_KEY`, `EVALKIT_OTLP_ENDPOINT`, `EVALKIT_CONTENT_CAPTURE` and `EVALKIT_TENANT_ID`. The last four override the values that `evalkit coding install` stored.

## Links

- Website: https://syntropylabs.ai
- Documentation: https://syntropylabs.ai/docs

## License

Apache License 2.0. See [LICENSE](LICENSE).
