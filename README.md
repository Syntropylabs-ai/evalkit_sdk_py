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
- [Identity](#identity)
- [Offline evaluation](#offline-evaluation)
- [Evaluation runs](#evaluation-runs)
- [Prompts](#prompts)
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

HTTP and web-framework spans carry both request and response headers under
`http.request.headers` / `http.response.headers`. Credential headers — `Authorization`, `Cookie`,
`x-api-key`, `x-subscription-key` and the rest of the list in
[Privacy](#privacy-content-capture-and-masking) — keep their names but their values are recorded as
`***`, so you can see what was sent without the key leaving your process. Everything else,
`traceparent` included, is recorded verbatim.

They are also content, so `capture_content=False` removes them along with every other payload, and a
`mask` callback sees them if you want to rewrite rather than drop. To record credential headers in
full instead — the usual reason is debugging a 401 from the trace alone:

```python
evalkit.init(subscription_key="tk_live_...", capture_secrets=True)
# or
export EVALKIT_CAPTURE_SECRETS=1
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

## Identity

Every span can carry three ids. They are what turns a pile of spans into sessions and
users, and they are how per-user limits are counted.

| Field | What it is | Where it shows up |
|---|---|---|
| `session_id` | one conversation, one visit | Sessions view: the turns in order, cost and latency per session |
| `user_id` | the end user your app authenticated | Users view: volume, cost and failure rate per user; per-user rate limits |
| `device_id` | the installation or browser | grouping before sign-in, when there is no user yet |

Set them for one trace:

```python
trace_id, end, ctx = evalkit.start_trace(
    "chat-turn", user_id="u_8123", session_id="conv_55", device_id="ios-9f2",
)
```

Or once for everything that follows — a request, a worker task, a whole script:

```python
evalkit.set_user(user_id="u_8123", session_id="conv_55")
...
evalkit.set_user()          # clears all three
```

- `set_user()` writes to a `contextvars.ContextVar`: a value set inside a request or
  an `asyncio` task stays there and never leaks into another one. Fields merge, so
  `set_user(session_id=...)` followed by `set_user(user_id=...)` leaves both set;
  calling it with no arguments clears them. It returns a token, and
  `evalkit.clear_user(token)` restores what was there before.
- Precedence: explicit `start_trace(...)` arguments, then `set_user`, then the
  `device_id` given to `evalkit.init()`.
- Child spans inherit from the trace — LLM calls, tool calls, DB queries and HTTP
  calls under a trace all land with the same three ids.
- The Django middleware fills `user_id` from `request.user` and the FastAPI /
  Starlette middleware from `scope["user"]`, both only when the user is
  authenticated; anything you set yourself wins.
- `evalkit.current_user()` returns what is in effect right now.

Without these ids a trace is still recorded — it just cannot be grouped into a
session or attributed to a user.

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

## Evaluation runs

Run your dataset through the platform's LLM-as-judge evaluators from code. The run is
persisted as an evaluation job (visible in the dashboard, comparable against earlier
runs) and can be gated. The judge's provider key is resolved server-side from your
project's model configuration — the SDK never sends one.

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
print(result.local_scores["exact_match"])         # one entry per row, None preserved
for row in result.results:
    print(row["rowIndex"], row["overallScore"], row["passed"], row["status"])
```

### How a run is sent

1. `POST /evaluation-runs` creates the run and answers `{runId, url}`. The link is
   printed once while the rows upload — set `EVALKIT_QUIET=1` to silence it — and is
   returned as `result.url` (with `result.run_id`).
2. Rows are appended with `POST /evaluation-runs/{runId}/rows` in batches of at most
   200 rows (`batch_size=`, clamped to the server cap). Every row carries its
   absolute `rowIndex`, so the server recognises a row it already has: the SDK
   retries a failed batch twice (network error or 5xx) with the identical payload and
   nothing is counted twice.
3. `POST /evaluation-runs/{runId}/complete` judges every row, applies the gate over
   the whole run and compares it against the baseline. One job, one gate verdict, one
   comparison, however many batches were uploaded.

Row fields: `input`, `output` (or a `task` that produces it), `expected` (`target` is
still accepted), `metadata`, and `context` — which is sent inside
`metadata.context`.

### Local scorers

`evaluators` mixes server rule ids (24-hex-char Mongo ids from Dashboard →
Evaluators) with local callables `(row, output)`. A callable runs in your process and
its result travels with the row as `localScores`, showing up in the run results with
evaluator source `local`. Return:

- a number — the score,
- a bool — score 1/0 plus a pass/fail verdict,
- `None` — not applicable to this row; nothing is sent for it,
- a mapping `{"score": ..., "passed": ..., "reason": ...}` — all three.

Anything else raises `ValueError`, as does an `evaluators` list without at least one
server rule id.

### Notes

- `average_score` is the mean over rows that produced a finite score; `pass_rate` is
  `passed / (passed + failed)`. Both are `None` when undefined — never 0.
- `result.passed` is `gate_result["passed"]` (`True` / `False` / `None` when a check
  could not be decided), or `None` when no gate was given.
- The client timeout is 600 s per request (`timeout=`); a 4xx/5xx raises
  `urllib.error.HTTPError`.
- Against a deployment older than the append endpoints the SDK falls back to the
  previous single-call behaviour: one job per batch (`result.job_ids`), aggregates
  recomputed client-side, and `gate_result` / `comparison` `None` above one batch.
  The fallback is detected once per client, not once per batch.

## Feedback and rewards

Attach a human or end-user signal to a trace your app already produced. The call goes to the control plane with your subscription key and shows up as a score on the trace, beside judge scores, in Annotate and in calibration.

```python
import evalkit

evalkit.feedback(trace_id, "thumbs", True, comment="answered in one line", user_id="u_42")
evalkit.feedback(trace_id, "rating", 4, session_id="s_9")
evalkit.feedback(trace_id, "text", "too verbose", span_id=span_id)
evalkit.reward(trace_id, 0.8, name="task_reward")
```

| kind | value | stored as |
|---|---|---|
| `thumbs` | `True`/`False`, `"up"`/`"down"`, or 1/0 | numeric 1 or 0 |
| `rating` | any number (your scale) | numeric |
| `reward` | 0..1 | numeric, source `end_user` when `user_id` is given, else `sdk` |
| `text` | non-empty string | text |

`api_url`, `timeout`, `span_id`, `session_id`, `name` are optional keyword arguments. Failures raise; the request is recorded as an `http_call` span like every other control-plane call.

## Prompts

Fetch a versioned prompt from the registry, compile its `{{variables}}`, and stamp the calls it drives so usage shows up per version in the UI. Reads go to the control plane with your subscription key; the SDK caches each read for `cache_ttl` seconds.

```python
import evalkit

prompt = evalkit.prompts.get("support_agent_system", label="production")
messages = prompt.compile({"customer": "Ada", "plan": "pro"})

with prompt.stamped():
    client.chat.completions.create(model=prompt.config.get("model", "gpt-4o-mini"), messages=messages)

evalkit.Eval("support-golden", data=rows, evaluators=[RULE_ID], hyperparameters=prompt.hyperparameters())
```

| | |
|---|---|
| `evalkit.prompts.get(name, *, label=None, version=None, variables=None, api_url=None, cache_ttl=60, timeout=30)` | resolution: `version` → `label` → `production` → latest; `variables` asks the server to compile and falls back to local compilation |
| `prompt.template`, `prompt.variables`, `prompt.config`, `prompt.semver`, `prompt.labels` | what the registry stores |
| `prompt.compile(variables)` | local Mustache-style substitution; strings as-is, numbers and booleans as text, objects as JSON; a missing variable raises `ValueError` naming every missing one |
| `with prompt.stamped():` / `prompt.stamp()` + `prompt.unstamp()` | every `llm_call` span created in that context carries `prompt.ref` (`name@version`), `prompt.name`, `prompt.version`; explicit attributes win |
| `prompt.hyperparameters()` | `{"promptName", "promptVersion"}` for `Eval(...)`, so runs link to the version |
| `evalkit.prompts.clear_cache()` | drop cached reads (tests, hot reloads) |

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

## Coding agents (Claude Code · Codex · Gemini CLI · Cursor · Windsurf · OpenCode)

Installing the SDK also installs the `evalkit` command. It points a coding agent's own telemetry at your EvalKit environment so its sessions appear as one trace per turn — the main agent, every subagent and everything they did:

```bash
evalkit coding install --vendor claude-code --key tk_live_...
evalkit coding install --vendor codex --key tk_live_...
evalkit coding install --vendor gemini-cli --key tk_live_... --capture full
evalkit coding install --vendor cursor --key tk_live_... --scope project
evalkit coding install --vendor opencode --key tk_live_...
evalkit coding status
evalkit coding doctor --tenant <your tenant id>
evalkit coding uninstall --vendor codex
```

Flags: `--endpoint` (default `https://api.syntropylabs.ai/v1/otlp`), `--tenant` (the tenant id `doctor` reads traces back with), `--scope user|project` (project writes into the current directory), `--capture minimal|full` (whether prompt, reply and tool text leave your machine; default minimal), `--no-traces` (keep log events only — spans are on by default wherever the vendor has them), `--no-subagents` (Claude Code: write no hooks at all), `--dry-run`, `--managed` (Claude Code: print a managed-settings snippet instead of writing).

The key and endpoint are stored once in `~/.config/evalkit/config.env` (mode 0600); `~/.config/evalkit/state.json` records exactly what was written so `uninstall` removes only that and restores any value it replaced. The key is never printed.

### What you get per vendor

| Vendor | One tree per turn | Tool calls | Tool output | Prompt & reply | Subagents | Tokens & cost |
|---|---|---|---|---|---|---|
| `claude-code` | yes — the agent's own spans (`--no-traces` falls back to flat log events) | yes | `--capture full` | `--capture full` | yes, nested under the `Agent` tool call | yes (tokens on the span, cost from the `api_request` event) |
| `codex` | yes — the turn span; Codex's ~150 internal plumbing spans are dropped by the receiver | when Codex emits them | `--capture full` | prompt with `--capture full` | no | yes (`sse_event` tokens, `turn_cost`) |
| `gemini-cli` | yes — `agent_call` / `llm_call` / `tool_call` spans | yes | no | `--capture full` | `agent_call` spans | tokens yes, cost no |
| `cursor` | yes — built by the hooks with deterministic span ids | yes | `--capture full` | `--capture full` | yes (`subagentStart` / `subagentStop`) | no — Cursor's hooks carry neither |
| `windsurf` | yes — same construction | yes | `--capture full` | `--capture full` | no (Cascade has no subagent events) | no |
| `opencode` | yes — built by the plugin | yes | `--capture full` | `--capture full` | child sessions nest under their parent turn | when OpenCode reports them on the assistant message |

### What `install` writes

| Vendor | What `install` writes | How data flows | Remove |
|---|---|---|---|
| `claude-code` | `env` keys in `~/.claude/settings.json` (or `.claude/settings.json`): `CLAUDE_CODE_ENABLE_TELEMETRY`, `OTEL_*_EXPORTER=otlp`, `OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf`, endpoint, `Authorization=Bearer` header, resource attribute `evalkit.vendor=claude-code`; span mode is on by default (`CLAUDE_CODE_ENHANCED_TELEMETRY_BETA=1`, `OTEL_TRACES_EXPORTER=otlp`, `OTEL_TRACES_EXPORT_INTERVAL=1000`) and `--no-traces` removes those three; `--capture full` adds `OTEL_LOG_USER_PROMPTS`, `OTEL_LOG_ASSISTANT_RESPONSES`, `OTEL_LOG_TOOL_DETAILS`, `OTEL_LOG_TOOL_CONTENT` and `OTEL_LOG_RAW_API_BODIES` plus `CLAUDE_CODE_OTEL_CONTENT_MAX_LENGTH=524288` — Claude Code cuts each body at 60 KB by default, which is smaller than its own system prompt, so the cap is raised to 512 K characters. The same file drives the VS Code and JetBrains flavours | In span mode Claude Code opens one trace per turn: a `claude_code.interaction` root with `llm_request` and `tool` children, and a subagent's work nested under the `Agent` tool's execution span. Content and cost still arrive as log events (`api_request`, `api_request_body` / `api_response_body`, `assistant_response`, `user_prompt`, `tool_result`, `subagent_completed`) which the receiver merges onto the matching span by trace and span id. Unless `--no-subagents`, `PostToolUse` and `SubagentStop` hooks (5 s timeout, always exit 0) post `claude_code.tool_output` and `claude_code.subagent_stop` so non-Bash tool output and the subagent's last message are covered too | `uninstall` deletes those keys, restores previous values — including a key you had already set to the same value we write — and removes only the hooks whose command contains `coding hook --vendor claude-code`. Files written under the other `--scope` are remembered too |
| `codex` | an `[otel]` table between `# evalkit-managed-begin/end` markers in `~/.codex/config.toml` (or `.codex/config.toml`) with log, **trace** and metrics exporters pointing at `<endpoint>/v1/{logs,traces,metrics}`; `--no-traces` writes `trace_exporter = "none"`; refuses to run if an `[otel]` table already exists outside the markers | Codex's native export: the turn span plus `codex.api_request`, `sse_event` token counts, `turn_cost`, `tool_result`, `user_prompt` (event names live in the `event.name` attribute) | `uninstall` removes the block |
| `gemini-cli` | `telemetry` block in `~/.gemini/settings.json` (`otlpProtocol: http`, `traces: true` unless `--no-traces`, `logPrompts` false unless `--capture full`) and the `OTEL_EXPORTER_OTLP_HEADERS` line between `# evalkit-begin/end` markers in `~/.gemini/.env` | Gemini CLI's native spans (`gen_ai.operation.name` = `agent_call` / `llm_call` / `tool_call`) plus `gemini_cli.*` events correlated by `prompt_id` | `uninstall` restores the previous `telemetry` values and removes the marker block |
| `cursor` | 12 hook entries (`evalkit coding hook --vendor cursor --event …`, 5 s timeout) in `~/.cursor/hooks.json`; other hooks untouched | Cursor has no telemetry export, so the hook **builds the trace itself**: trace id = `sha256(conversation_id + generation_id)[:32]`, turn root span id = `sha256(generation_id)[:16]`, tool span id = `sha256(tool_use_id)[:16]` (start = now − `duration`), subagent span id = `sha256(subagent_id)[:16]` parented to `sha256(tool_call_id)[:16]`. `beforeSubmitPrompt` opens the turn, `afterAgentResponse` / `stop` complete it with an enrichment record, `subagentStop` completes the subagent | `uninstall` removes only entries whose command contains `coding hook --vendor cursor`; a reinstall also drops entries for events we no longer use |
| `windsurf` | 6 hook entries in `~/.codeium/windsurf/hooks.json` (project scope: `.windsurf/hooks.json`) | same construction with `trajectory_id` as the session and `execution_id` as the turn; `post_run_command`, `post_write_code`, `post_read_code` and `post_mcp_tool_use` become tool spans, `post_cascade_response` completes the turn | same |
| `opencode` | `~/.config/opencode/plugins/evalkit.ts` (the plugin, no key inside) and a `"plugin"` entry in `~/.config/opencode/opencode.json`; project scope writes `.opencode/plugins/evalkit.ts` and `opencode.json` in the current directory | the plugin turns `session.*`, `message.updated`, `message.part.updated` and `tool.execute.before|after` into the same span shapes, reads the key from `~/.config/evalkit/config.env` at run time, and posts OTLP JSON | `uninstall` deletes the plugin file and removes only our entry from `"plugin"` |

`status` prints the endpoint, the masked key, the capture level, the last `doctor` result and, per vendor, the scope, capture level, whether traces are on and which hooks are installed. `doctor` posts one synthetic turn — an `agent` root with an `llm_call` and a `tool_call` child, plus one enrichment record — then reads `GET <api>/v1/traces/<tenant>/<traceId>` back with the same key and checks the merged tree came back (3 spans, prompt present); it needs `--tenant` (remembered in `config.env`) because the read API scopes traces by tenant. The `hook` subcommand always exits 0, never writes to stdout, and gives up after 5 seconds so it can never block the editor.

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

### Secrets are masked before export

Every envelope goes through a secret scrubber before your `mask` hook runs and before anything
leaves the process. It runs whether or not content capture is on: turning capture off removes
strictly more, never less.

**By key.** Any object key matching one of these, case-insensitively, as the whole key or as a
dot/underscore/dash-delimited segment, has its value replaced by `"***"` — recursively through
objects and arrays, and through attribute values that are JSON strings, which are parsed, scrubbed
and re-serialized:

`api_key` / `api-key` / `apikey`, `secret`, `token`, `password`, `passwd`, `authorization`,
`credential`, `credentials`, `subscription_key` / `subscription-key`, `x-subscription-key`,
`cookie`, `set-cookie`, `private_key`, `access_key`, `client_secret`.

So `subscriptionKey`, `X-Subscription-Key`, `http.request.api_key` and `credentials.password` are
all masked. `token` is the one word matched only as the whole key or the *final* segment, because it
is also telemetry vocabulary: `prompt_token_count`, `gen_ai.server.time_to_first_token_ms`,
`input_tokens` and `tokenizer` are metrics, not credentials, and masking them would blank out a
dashboard column to protect nothing.
Booleans and numbers under a matching key become `"***"` as well, so the value's shape never hints
at what was there. `http.request.headers` and `http.response.headers` additionally go through the
sensitive-header list (`authorization`, `cookie`, `set-cookie`, `x-api-key`, `api-key`,
`x-auth-token`, `proxy-authorization`, `x-secret`, `x-access-token`, `x-subscription-key`,
`subscription-key`, `token`, `password`).

**By pattern.** Inside any string — attributes, event attributes, prompt, completion, status
message — these are replaced by their own prefix followed by `***`:

| Pattern | Becomes |
| ---------------------------------- | ------------- |
| `tk_live_[0-9a-f]{8,}`             | `tk_live_***` |
| `tk_test_[0-9a-f]{8,}`             | `tk_test_***` |
| `sk-[A-Za-z0-9_-]{16,}`            | `sk-***`      |
| `Bearer\s+[A-Za-z0-9._~+/=-]{8,}`  | `Bearer ***`  |
| `AKIA[0-9A-Z]{16}`                 | `AKIA***`     |
| `ghp_[A-Za-z0-9]{20,}`             | `ghp_***`     |

Strings that contain no secret are left byte for byte, JSON strings included — nothing is
reformatted on the way through.

Identity is deliberately **not** masked. Email addresses, user IDs, session IDs and device IDs pass
through untouched: they are what the Users view, per-user cost and session replay are built on. A
credential is never yours to lose; an identity is the product.

To turn the scrubber off — debugging your own auth failure is the usual reason:

```python
evalkit.init(subscription_key="tk_live_...", capture_secrets=True)
# or
export EVALKIT_CAPTURE_SECRETS=1
```

`capture_sensitive_headers=` and `EVALKIT_CAPTURE_SENSITIVE_HEADERS` are aliases of the same switch,
and their default moved with it: credential headers are masked unless you opt back in.

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
    capture_secrets=None,                     # None = env var, else True/False
    capture_sensitive_headers=None,           # alias of capture_secrets
    capture_stream_bodies=None,               # None = env var, else True/False
    mask=None,                                # callable(envelope) -> envelope | None
)
```

Traces are batched and exported in the background. Flush before exit if needed:

```python
evalkit.flush()
```

Environment variables: `EVALKIT_API_URL` overrides the control-plane base for
`Eval()`, feedback and prompts; `EVALKIT_QUIET=1` silences the evaluation-run link;
`EVALKIT_FUNCTION_TRACE=false` turns off function tracing;
`EVALKIT_CAPTURE_SECRETS=1` turns off secret masking.

## Links

- Website: https://syntropylabs.ai
- Documentation: https://syntropylabs.ai/docs

## License

Proprietary — © 2026 Syntropy Labs. All rights reserved. See [LICENSE](LICENSE).
