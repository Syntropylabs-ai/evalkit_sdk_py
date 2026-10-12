# Changelog

## 0.3.4 — 2026-10-12

### Added
- `simulate_user()` plays a live simulated user. After a scenario's scripted turns, or when it has
  none, the platform plays the user from the scenario's persona, `user_goal`, `hidden_info` and
  tool fixtures, reading the conversation so far, and the user stops once its goal is met or
  `max_turns` user turns were played (at most 50). New parameters: `user_model` (the model
  configuration the user runs on, by name or id; falls back to `model`) and `live_user` (on by
  default for a scenario with a `user_goal` or `conversation_plan` and no `turns`; turns-only
  scenarios replay as before). Tool calls an `AgentTurnResult` reports, with `result` or
  `error`, reach the user as what the agent was told. Each user turn is its own span marked
  `evalkit.simulation.role: user`, never judged as the agent. Live results carry `transcript`,
  `ended_by` and `simulated_user_usage`. Needs the platform's `POST /simulations/user-turn`
  (paid plans with simulations).
- `generate_scenarios()` writes scenarios for the live user by default (persona, `user_goal`,
  `hidden_info`, `success_criteria`, the opening message, no later turns). Pass
  `scripted_turns=True` for every turn written out.

### Changed
- A 429 from the receiver is retried like a 5xx instead of dropping the spans. On a 429 or 503,
  `Retry-After` (seconds or an HTTP-date) sets the wait, capped at 10 s; without it the backoff
  steps (0.2 s, 0.6 s, 1.8 s) are jittered so clients refused together do not retry together.
  One export spends at most 20 s waiting across all its requests, which also bounds the flush at
  exit. A 429 that outlasts the retries stops the rest of that export, as a 5xx does. Every other
  4xx is still final.
- The `evalkit` CLI retries the same answers while the caller has time: a coding-agent hook
  within its 5 s limit, `evalkit coding backfill` for up to 20 s per post. `evalkit coding doctor`
  still tries each request once and reports what it got.

## 0.3.3 — 2026-10-11

### Added
- Export requests are gzip-compressed and sent with `Content-Encoding: gzip`. Pass
  `export_gzip=False` to `init()`, or set `EVALKIT_EXPORT_GZIP=false`, to send plain JSON.
  A receiver without compressed ingest, such as a trace service deployed before it, answers 415,
  or 400 because it reads the gzip bytes as JSON. The SDK then sends the same request
  uncompressed, and unless that is refused with 400 too, it sends every later request
  uncompressed and logs one warning. Such a receiver loses no spans: it costs one extra request
  per process. A 400 to both the compressed and the plain request is about the spans, so
  compression stays on.
- Spans inside a LangGraph node carry where they ran: `langgraph.node`, `langgraph.step` (an
  integer), `langgraph.triggers` and `langgraph.checkpoint_ns`, copied from the metadata
  LangGraph gives each run. This covers chain, chat-model, LLM, tool and retriever spans.
  `langgraph.triggers` is a compact JSON array in a string, such as `["branch:to:agent"]`, the
  same string the TypeScript SDK writes. `langchain.tags` is unchanged.
- `Eval()` rows take `trace_id`, `session_id`, `messages`, `tools` and `system_prompt`, so
  agent evaluators can score the agent's turns and tool calls. `trace_id` or `session_id` points
  at a trace or session already recorded in the key's project and environment. `messages` (OpenAI
  chat format) carries the conversation itself, with `tools` (OpenAI tool definitions) and
  `system_prompt` beside it. They are sent as the top-level row keys `traceId`, `sessionId`,
  `messages`, `tools` and `systemPrompt` when present, not inside `metadata`. A non-string or
  empty id, or `messages` or `tools` that is not a list, raises `TypeError` or `ValueError`
  naming the row before the run opens.

### Changed
- The queue is exported in requests of at most `max_export_batch_size` spans and 4 MiB of JSON
  before compression. It went out as one request of up to `max_queue_size` (4096) spans, so a
  batch of large spans could pass the 10 MiB the ingest proxy accepts and be refused whole.
- A 413 splits the request in half and sends each half again, down to a single span. A span
  that is still refused is dropped with a warning that names it and gives its size in bytes. A
  413 used to drop the whole queue with a warning.
- When a request gets no answer, or only 5xx answers, after its 3 retries, the rest of that
  export is dropped instead of being sent into the same failure, so a flush at exit still waits
  out one round of retries. A 401 or 403 ends the export the same way, so one export counts as
  one auth failure, as before.
- Outbound HTTP spans from `requests`, `httpx` and `aiohttp` are named by the method alone
  (`POST`), as OpenTelemetry names a client span whose URL template is unknown, and carry span
  kind `CLIENT`. They were named by the path, `POST /charges/ch_demo_4821/refunds`, so every id
  was an operation of its own. The full URL, ids and query included, stays in `http.url`. The
  request spans of the framework integrations carry span kind `SERVER`.

### Fixed
- The secret scrubber no longer masks `gen_ai.server.time_to_first_token` and
  `gen_ai.server.time_per_output_token` on bridged OpenTelemetry spans, nor per-token prices
  such as `input_cost_per_token`. A key ending in `token` was read as a credential, so those
  values arrived as `***` and the span lost its time to first token. Every other key ending in
  `token` (`access_token`, `refresh_token`, `id_token`, `csrf_token`, `token`) is still masked.
  Usage counts and `max_tokens` end in `tokens` and were never masked.
- With `debug=True`, an export whose 2xx response was not JSON was sent again as if it had
  failed, so the receiver could store its spans up to four times.
- On a FastAPI that traces requests itself (its built-in OpenTelemetry telemetry, in 0.143 and
  not in 0.118), `EvalKitMiddleware` exported every request as two traces: FastAPI's server
  span with its `fastapi.dependencies`, `fastapi.endpoint` and `fastapi.serialization` spans in
  one, the SDK's request span and everything under it in the other. The request span now takes
  the ids of FastAPI's server span, so a request is one trace, and FastAPI's own spans of the
  request are not exported; a span another OpenTelemetry library opens inside the endpoint hangs
  off the request span. The same holds under Starlette's `OpenTelemetryMiddleware` and the
  contrib ASGI and FastAPI instrumentations. Pinning FastAPI below its built-in telemetry is no
  longer needed.
- A `requests` call was recorded twice, once by the SDK and once as a `GET` or `POST` span from
  OpenTelemetry's urllib3 instrumentation, which this package installs. The urllib3 span's
  traceparent also replaced the SDK's, so the service called joined a trace that was never
  exported. The urllib3 instrumentation now stays out of a call the SDK records itself; urllib3
  used directly, as botocore and boto3 use it, is still traced.
- Request spans of `EvalKitMiddleware` (FastAPI and Starlette), `instrument_flask` and
  `EvalKitDjangoMiddleware` are named by the route template, `POST /charges/{charge_id}/refunds`,
  `GET /orders/<int:order_id>`, with `http.route` set, instead of by the raw path, which gave the
  Services table one row per id. A FastAPI route of an included router keeps the include prefix.
  A request that matched no route keeps its raw path. The raw URL stays in `http.url`.
- Spans an agent framework records on the OpenTelemetry GenAI conventions (Strands Agents, and
  any framework that sets `gen_ai.operation.name`) were all stored as `function_call` with their
  tokens dropped, so a Strands trace showed no agents, no tool calls and no model calls of its
  own. They are now typed by their operation:
  - `invoke_agent` (and the multi-agent `invoke_swarm` / `invoke_graph`) is an `agent` span named
    by `gen_ai.agent.name`, else by its span name, with the user's message as its prompt and the
    agent's reply as its completion. Its usage totals the model calls under it, so it carries no
    tokens.
  - `execute_tool` is a `tool_call` named by the tool, with `tool.name`, `tool.arguments` (from
    `gen_ai.tool.call.arguments` or the `gen_ai.tool.message` event) and `tool.result` (from
    `gen_ai.tool.call.result` or the `gen_ai.choice` event, content blocks read as their text). A
    `gen_ai.tool.status` of `error` makes it an error span.
  - `chat`, `generate_content` and `text_completion` are an `llm_call` with the model, the
    messages, the completion and the tokens, unless the SDK recorded the same request itself (its
    own `llm_call` ran inside the framework's span, or the framework's span inside it). That copy
    is an `internal` span without tokens, so the call is counted once.
    `EVALKIT_COUNT_FOREIGN_LLM=1` still counts both.
  - `execute_event_loop_cycle` is an `internal` span, never a tool or model call.
  A `gen_ai.*` span that names no operation keeps its `function_call` type.
- A message whose content is a list of blocks (OpenAI parts, Anthropic or Bedrock blocks, which
  Strands and other agent frameworks send) was stored in the prompt as Python's repr of the
  list, `user: [{'text': 'Hi', 'type': 'text'}]`. Text blocks are now stored as their text, a
  tool-use or other structured block as compact JSON, and an image or file as `[image]`-style
  markers. This applies to the OpenAI, Anthropic, LiteLLM, Ollama and LangChain prompts.

## 0.3.2 — 2026-10-02

### Added
- `Eval(..., collections=[...])` runs evaluator collections from the dashboard by id, alone or
  beside rule ids and local scorers in `evaluators`. The server expands each collection into
  its rules; a run of one collection is scored the way the dashboard scores it (its weights,
  thresholds, aggregation and hard gates), and two or more run every rule once at weight 1.
  `evaluators` is now optional when `collections` is given, and `evalkit eval run` configs
  accept `collections`. Needs a server that supports collections: an older one is detected
  when the run opens and raises `RuntimeError` before any row is sent.
- `simulate_user(..., agent="name")` names the agent under test on every span of the run, so the
  run's traces list under that agent.
- The `end` that `start_trace` returns also takes `status_message`, `prompt`, `completion` and
  `attributes` as keyword arguments, so a root span can show what was asked, what was answered
  and why it failed. The status message is cut to 2000 characters. Attributes given to `end`
  win over the ones given to `start_trace`.

### Changed
- A simulated scenario's session id names its simulation, run and scenario
  (`<simulation_id>:<run_id>:<scenario_id>`), so two runs of one scenario stay apart on the
  dashboard. It was a random `sess_` id.
- A scenario's root span records the scenario's first message as its prompt and the agent's last
  reply as its completion. When the scenario fails, the error is its status message and its
  `error.type` and `error.message` attributes, so the scenario list shows why it failed.

### Fixed
- A row's `context` given as one string is sent as one document. It was split into one
  entry per character, so a grounding evaluator judged the answer against single letters.
- An HTTP error from an evaluation run reads as the server's reason
  (`HTTP Error 404: Evaluation rule(s) not found: ...`). It used to read
  `HTTP Error 404: Not Found`. It is still a `urllib.error.HTTPError` with the same `code` and `response_text`.

## 0.3.1 — 2026-09-26

### Fixed
- OpenTelemetry spans that another library opens inside an instrumented model call (the HTTP
  span httpx starts under an OpenAI, Anthropic, Gemini, Bedrock, Ollama or other provider call)
  keep the caller's session, user and agent. They were exported with none, so a conversation's
  HTTP spans fell out of its session and a sub-agent's out of that agent.
- OpenAI chat, OpenAI Responses and Anthropic model-call spans record the declared tool names
  (`gen_ai.request.tools`) and the sampling parameters (`gen_ai.request.temperature`, `top_p`,
  `max_tokens`). The dashboard tells agent versions apart by system prompt, tools, model and
  sampling, and without these a changed tool set never showed as a new version. The first spans
  after upgrading start a new version entry for each agent, since the parameters are now part of it.
- An Anthropic `system` prompt passed as a list of blocks (as prompt caching requires) is
  recorded as its text in the span's prompt. It was recorded as the list's Python repr.

## 0.3.0 — 2026-09-26

Sessions and named agents. Traces now group into conversations and agents on the dashboard
(Traces › Sessions, Traces › Agents), and conversations can be evaluated as a whole.

### Added
- `evalkit.session(session_id, user_id=None)`: a context manager and decorator (sync and async)
  that puts every span opened inside it in that session, then restores the previous identity.
  Scoped to the current task, so concurrent conversations never mix.
- `evalkit.set_agent(name)`, `evalkit.clear_agent(token)`, `evalkit.current_agent()`, and
  `start_trace(..., agent="planner")`. Spans carry the envelope field `agentName` and the
  attributes `evalkit.agent_name` and `gen_ai.agent.name`.
- The session and user id are read from span attributes under every common spelling:
  `session.id`, `conversation.id`, `gen_ai.conversation.id`, `session_id`, `evalkit.session_id`,
  and `user.id`, `enduser.id`, `evalkit.user_id`. Child spans inherit their trace's session.
- The agent name is read from `evalkit.agent_name`, `gen_ai.agent.name` or `agent.name`.
- A Bedrock Agent's session id reaches the span's session.

### Changed
- A root `start_trace` always starts a new trace. Consecutive turns handled in one thread or task
  used to share the first turn's trace id, so a whole session showed as one trace.
  The client's `start_trace(..., join_trace_id=...)` (on the `EvalKitClient` that `init()`
  returns) joins an existing trace on purpose.
- OpenTelemetry spans from other libraries keep the session, user and agent in force when they
  started (they were read on the export thread, which has none). A span started inside an
  EvalKit span joins that EvalKit trace. It used to be exported as a separate trace.
