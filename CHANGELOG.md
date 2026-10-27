# Changelog

## 0.3.7 — 2026-10-27

Every server middleware now decides which answers make a request's span an error by one rule, and the
rule can be chosen. By default every 4xx and 5xx answer is an error, as before.

### Behaviour changes
- A route whose last segment names a probe is a health probe, and untraced by default, only when no
  segment before it names a record: `/health`, `/api/v2/health` and `/actuator/health/readiness` are
  probes, while `/datasets/<id>/health`, `/users/42/healthz` and a template such as
  `/datasets/:datasetId/health` are the app's own routes and are recorded. A record is a number, a UUID,
  a long hex or opaque id, or a route parameter.
- The FastAPI, Starlette, Flask, Django and Litestar middleware judge an answer the same way: a request's
  span is `ERROR` for an answer in `server_error_status_codes` (every 4xx and 5xx by default), with the
  exception the server answered with, also when a view raised what the framework answers with a status
  (Django's `Http404`, `PermissionDenied`, `SuspiciousOperation` and `BadRequest`, Litestar's HTTP
  exceptions). A request that raises after its response started is still an error; a cancelled request is
  recorded as before. A traced function (`trace_function`, `trace_tool`, function tracing) that raises a web
  framework's HTTP answer is judged by the status it carries, and records `http.response.status_code`.
- The span of a streamed Flask or Django response ends after its last chunk is sent, so its latency
  covers the stream.

### Added
- `server_error_status_codes`: the status codes an incoming request's span is an error for, as codes
  (`503`) and inclusive ranges (`"500-599"`), by default `["400-599"]`, as the TypeScript SDK's
  `serverErrorStatusCodes`. `["500-599"]` follows OpenTelemetry's HTTP conventions, where a 4xx is the
  caller's mistake: the span keeps `http.status_code`, stays `OK` and records no exception, so scanners'
  404s, expired tokens' 401s and bad request bodies stay out of the error rate and the Errors page.
  `[429, "500-599"]` also counts the 429s the app answers. A value that is not such a list is skipped with
  one warning and the default stays. Outgoing calls are not affected: a 4xx they get back is an error.

### Fixed
- The Litestar request span is named by the route template (`GET /orders/{order_id}`) and records
  `http.route`; it was named by the raw path.
- The Litestar middleware records the exception Litestar answered with a 5xx, with `exception.escaped`
  false; only the status reached it.
- An exception Flask hands on to the server (`PROPAGATE_EXCEPTIONS`, debug or testing mode) is recorded
  with `exception.escaped` true.

## 0.3.6 — 2026-10-26

The SDK can no longer crash, hang, slow down or grow without bound in the application it observes, and a
failed span records its exception by the convention the TypeScript SDK and the receiver share.

### Added
- `evalkit.stats()`: spans queued and dropped (by queue length, byte budget, failed export, open circuit,
  rate limit and sampling), failed exports, the last error, and the circuit breaker's state (`closed`,
  `open`, `half_open` or `disabled`). It never raises, before `init()` included.
- A circuit breaker on exporting. After 3 export cycles in a row the receiver did not take (no answer,
  5xx, 429 or a refused key), spans are dropped as they are recorded, which only counts them, for 30
  seconds, then one batch of at most 16 spans probes the receiver; each failed probe doubles the wait, up
  to 5 minutes.
- `flush(timeout=5.0)` on the module and the client: the flush waits at most `timeout` seconds.
- A kill switch: `EVALKIT_DISABLED=true`, or OpenTelemetry's `OTEL_SDK_DISABLED=true`, makes the SDK a
  no-op that records, patches and sends nothing.
- `sample_rate`: the share of traces kept, decided from the trace id as OpenTelemetry's
  TraceIdRatioBased sampler does, so a trace is kept whole. Without it, `OTEL_TRACES_SAMPLER` and
  `OTEL_TRACES_SAMPLER_ARG` are read. Model calls, tool calls and failed spans are always kept; the default
  keeps everything.
- `max_queue_bytes` (32 MiB): a byte budget on the export queue, past which the oldest spans that are not
  model calls are dropped first. A full queue (`max_queue_size`) drops by the same rule.
- `max_value_bytes` (16 MiB): the most one value keeps. A longer prompt, completion, body or attribute
  keeps its start and its end around a `…[cut N bytes]…` marker, and `evalkit.truncated.<field>` records
  its size in UTF-8 bytes. `OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT` or `OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT`
  lowers it.
- `function_rate_limit` (100 a second for each traced function), `log_level` (`WARNING`, also
  `EVALKIT_LOG_LEVEL`) and `log_rate_limit` (50 a second).
- The exception contract. A failed span has status `ERROR` described as `"<type>: <message>"`, `error.type`
  with the fully qualified class name (`openai.RateLimitError`), one `exception` event with
  `exception.type`, `exception.message` (2 KiB), `exception.stacktrace` (16 KiB, start and end kept) and
  `exception.escaped`, and the attributes `evalkit.exception.frames` (up to 64 frames, innermost first,
  each `{file, line, col, function, module, in_app}`), `evalkit.exception.source` (the innermost frame of
  the application's own code) and `evalkit.exception.causes` (up to 5 from `__cause__` and `__context__`).
  The `exception.*` and `error.source` attributes of earlier releases are kept. No local variable or
  argument is read.
- `evalkit.set_error_fingerprint()` and `clear_error_fingerprint()`, and `error_fingerprint=` on
  `trace_function` and `trace_tool`, record `evalkit.exception.fingerprint`.
- `in_app_include` and `in_app_exclude`: module prefixes that are, or are not, the application's own code
  in an exception's frames.
- `capture_unhandled=True` records uncaught exceptions of the main thread, of other threads and of asyncio
  tasks and callbacks, and hands each to the hook installed before. Off by default.
- `ignore_routes`: incoming requests answered untraced, the way a health probe is, in the FastAPI,
  Starlette, Flask, Django and Litestar middleware: no request span, no response trace headers, and
  nothing the handler records while it answers. A pattern is an exact path (`"/metrics"`), a glob (`*` one
  segment, `**` any number), a route template (`"GET /v1/orders/:id"`; `{id}` and `<int:id>` too) matched
  against the path and the framework's route when it is known, an optional leading method, a compiled
  regex searched in the path, or a callable given a `RouteInfo` (`method`, `path`, `route`). Matching
  ignores a trailing slash and the query string; paths are case-sensitive, methods are not.
- `ignore_urls`: outgoing calls made with requests, httpx, aiohttp, or urllib3 through its OpenTelemetry
  instrumentation, recorded with no span and sent with no `traceparent`. A pattern is a host
  (`"api.stripe.com"`, `"*.internal.example.com"`), host and path, a full URL with scheme and port, a path
  on any host, a compiled regex searched in the URL, or a callable given a `UrlInfo`.
- `ignore_functions`: qualified names (`module.qualname`, `"myapp.cache.*"`) that function tracing leaves
  unwrapped. An explicit `trace_function` or `trace_tool` on a matching function leaves it unwrapped too:
  the ignore wins.
- `evalkit.untraced`: a decorator for sync, async and generator functions, and a `with` / `async with`
  block, inside which nothing is recorded (model calls, HTTP calls and their `traceparent`, database
  queries, function and log spans, other instrumentations' spans). It is per thread and per asyncio task,
  so concurrent requests are unaffected, and works before `init()`.
- `trace_health_checks=True` traces health and readiness probes, incoming and outgoing, which stay left
  out by default.
- `EVALKIT_IGNORE_ROUTES`, `EVALKIT_IGNORE_URLS` and `EVALKIT_IGNORE_FUNCTIONS`: comma-separated patterns
  added to the options. A pattern that cannot be read is skipped with one warning; `init()` never fails
  over one. Patterns are compiled once, at `init()`.

### Changed
- `max_body_bytes` defaults to 64 KiB (it was 10 MiB).
- Only log records at `WARNING` and up become `log` spans unless `log_level` says otherwise, at most 50 a
  second; a message keeps at most 8 KiB, start and end; the SDK's own `evalkit` loggers are never recorded.
- Function arguments and results are written by a serializer that stops at the 4096-character cap, so a
  large argument is never serialized whole, and secrets are masked as it goes.
- The model-call wrappers no longer cut prompts, completions, thinking or tool arguments to 32 KiB (8 KiB
  for the voice integrations), and request and response bodies are no longer cut to their first 2 MiB:
  every value goes through the one per-value limit, which keeps the newest messages at the end.
- A span over 8 MiB keeps the start and the end of its longest values instead of having them replaced
  by a note. Spans are measured and sent in UTF-8 bytes rather than as escaped ASCII.
- Failed exports log one warning when they start, then at most one every 5 minutes with the counts; a
  response body in a warning is cut to 200 characters. A refused subscription key is logged clearly at
  the first refusal and when exporting stops.
- An invalid `base_url` (not `http://` or `https://`), an empty or malformed key, or a size or rate that is
  not a positive number makes the SDK a no-op with one warning naming each problem, never the key.
- Calling `init()` again with other options warns once, naming the options that do not apply.
- An explicit `patch_*` call that cannot instrument its target logs a warning instead of raising.
- The status message of a failed model call is `"<type>: <message>"`.

### Fixed
- An outgoing call made while a health probe is answered no longer sends a `traceparent` naming a span
  that is never exported, and a span another OpenTelemetry instrumentation starts there hands no context
  on either.
- The FastAPI, Starlette and Litestar middleware no longer read the whole request before the app: the
  body is copied from the messages the app receives, up to the cap, and uploads (multipart,
  octet-stream, media) are not kept. The response body is kept as a list of chunks with a running length
  instead of growing one bytes object, and not at all for `text/event-stream` and compressed responses.
- Flask and Django read the request body only when its `Content-Length` is under the cap and it is not an
  upload, and a streamed Flask response is no longer drained to record it.
- A request's trace id no longer stays the ambient trace of the worker thread after the request, which put
  spans recorded later on a reused WSGI thread into the previous request's trace.
- The `traceparent` parser follows the W3C rules: lowercase hex only, all-zero ids and version `ff`
  rejected, and a later version with more fields joined.
- `trace_function` and `trace_tool` end their span for a cancelled task (status `ERROR`,
  `evalkit.cancelled=true`), `KeyboardInterrupt` and `SystemExit` as well, and re-raise them; such a call
  used to leave its span unrecorded. Generator and async generator functions are traced across their
  iteration, with errors raised while iterating recorded.
- The stack trace recorded is the exception's own: it was read from the exception being handled, which
  inside an `except` block can be another one.
- FastAPI, Flask and Django record the exception that failed the request; mid-stream model errors carry the
  full exception.
- An integration that raises anything on import, not only `ImportError`, is skipped without stopping
  `init()` or the integrations after it.
- `flush()` no longer raises for a `base_url` that is not a URL.
- The LangChain callback's run records and the Google ADK tool timers are capped at 10,000 entries that
  expire after an hour, so runs that never end do not keep theirs.
- Credential parameters of a URL inside an exception or log message are masked (`?api_key=***`).

## 0.3.5 — 2026-10-14

### Security
- Function and tool arguments are masked by the parameter they are bound to. `trace_function`,
  `trace_tool`, `@traced`, `trace_module` and automatic function tracing record `function.args` as
  a list of positional values, which the scrubber could only search by value shape, so a key
  passed positionally (`judge_body(model, messages, api_key)`) went out as written unless it
  looked like `sk-…` or `tk_live_…`. Each value is now matched to its parameter through
  `inspect.signature`, and one whose parameter is named like a credential (`key`, `api_key`,
  `secret`, `token`, `password`, `credentials`, `auth`, `cookie`, `dsn`, `jwt`, `bearer`, in snake or
  camel case) is recorded as `***`, positional or keyword. `max_tokens`, `token_count` and other
  counts keep their values. A function or tool whose own name reads that way (`get_api_key`,
  `presented_token`) records neither arguments nor result.
- Arguments and results are scrubbed before they are serialized and cut to 4096 characters. A
  longer value used to be cut first, and the cut JSON no longer parsed, so `"api_key": "…"`
  inside it was found only by the value's shape. An object recorded through its `str()` (a
  pydantic model, `api_key='…'`) is scrubbed the same way, and the tail of a cut value is
  dropped with the cut, so no secret is left half-recognised at the end.
- More shapes are masked everywhere the scrubber runs: PEM private keys (whole, or cut before
  their `END` line), JSON Web Tokens, Google API keys (`AIza…`), the userinfo of a URL or DSN
  written into text (`postgres://user:pass@host`), and quoted assignments to a credential name
  (`password: "…"`, `api_key='…'`). An unquoted `token=abc` in prose and `max_tokens: 5` are left
  alone.

- `capture_content=False` removes content that other OpenTelemetry instrumentation put on the
  spans the SDK forwards, whatever that instrumentation's own content switch says:
  `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`, `gen_ai.prompt`,
  `gen_ai.completion`, `gen_ai.retrieval.query.text` and `gen_ai.retrieval.documents`, flattened
  message fields such as `gen_ai.prompt.0.content` and `llm.input_messages.0.message.content`,
  `input.value` and `output.value`, keys ending in `.observation.input`, `.trace.output`,
  `.entity.input` and the like, and the `content`, `message` and `tool_calls` of `gen_ai.*` events
  (`gen_ai.user.message`, `gen_ai.choice`). Model, tokens, cost, timing, roles, finish reasons and
  tool names stay. A streamed refusal (`gen_ai.response.refusal`) is now content too.
- camelCase keys are matched like their snake_case spellings: `accessToken`, `authToken`,
  `refreshToken` and `secretKey` are masked; `maxTokens`, `inputCostPerAudioToken` and flags such as
  `hasToken` are not.

### Added
- psycopg2 is traced: each `connect()` as a `postgresql CONNECT` span (with `db.name`,
  `server.address`, `server.port`, never the DSN), and each cursor's `execute` and `executemany` as
  a `db_query` span with the statement, operation, table and row count, under the same
  `capture_content` rules as psycopg 3. Cursors keep their class (`RealDictCursor`, `DictCursor`)
  and every method; connections made through `psycopg2.pool` are traced too.
- A process forked after `init()` (gunicorn `--preload`, `multiprocessing` with `fork`, Celery
  prefork) restarts the export thread with new locks and an empty queue, so its spans are sent while
  it runs; a `multiprocessing` child also flushes as it ends.

### Fixed
- A streamed call that fails before its stream opens (a refused connection, a 400 or a 401) is
  recorded as an `ERROR` model-call span with the model, the prompt, the request settings and the
  exception's type and stack, for OpenAI and Anthropic `create(stream=True)` (sync and async,
  patched automatically or by hand), Anthropic on Vertex and LiteLLM. It used to leave no span.
- A function decorated with `trace_function` or `trace_tool` (also with `capture_io=False`) in the
  automatically traced source tree is no longer wrapped a second time, which recorded two spans per
  call and the arguments `capture_io=False` withheld. A decorator stacked on top with
  `functools.wraps` carries the same mark.
- An Anthropic reply made only of tool calls has no time to first token: the first
  `input_json_delta` no longer stops the clock. Text and thinking deltas still do, and thinking stays
  out of the completion.
- `init()` no longer changes the application's logging. It used to replace the formatter of the
  root logger's stream handlers, add a console handler and set the root level to `INFO`, and its
  handler on the root logger made a later `logging.basicConfig()` do nothing. Records that reach the
  root logger are still recorded as `log` spans and carry `trace_id` and `span_id`.
- The health-check filter matches whole path segments (`/health`, `/healthz`, `/health/live`,
  `/readyz`, `/livez`, `/actuator/health`), so `/healthcare` and `/api/health-records` are recorded.
  A skipped probe made with `requests` is also left out by urllib3's OpenTelemetry instrumentation.
- Per-token prices such as `input_cost_per_audio_token` and `output_cost_per_reasoning_token` are no
  longer masked as credentials; `access_token`, `refresh_token` and `id_token` still are.
- A key that prices or counts tokens is left unmasked by the same rule as in the TypeScript SDK,
  so it masks alike from either: `price_per_*` (`price_per_1k_token`), any number of billed kinds
  after `per` (`per_cache_read_token`, `per_cache_creation_input_token`), and the `tool`, `read`,
  `write`, `creation` and `thinking` kinds (`input_cost_per_tool_token`), as well as the
  `per_character_token` already exempt. `per_user_token` and any other per-something token stay
  masked.
- The framework integrations no longer record health and readiness probes: `EvalKitMiddleware`
  (FastAPI and Starlette), `instrument_flask`, `EvalKitDjangoMiddleware` and
  `create_litestar_middleware()` skip `/health`, `/healthz`, `/health/live`, `/readyz`, `/livez`
  and the other paths of the whole-segment rule outgoing calls are skipped by, which the
  TypeScript SDK applies as well; `/healthcare` and `/api/health-records` are recorded. Each poll was
  a trace of its own. Nothing recorded while the probe is answered is sent either: a database ping,
  a traced function, a log line, another library's span, or the server span FastAPI's own telemetry
  opened for the probe. Work the handler starts that runs on after the response, such as a
  Starlette background task or a thread, is traced as usual. The probe's response is unchanged and
  carries no `X-Trace-Id`.
- The warning for a streamed OpenAI call without a usage chunk says what happens: the trace service
  estimates the tokens and marks the span `evalkit.cost_estimated`. It names
  `stream_options={"include_usage": True}` for exact counts.
- OpenTelemetry's psycopg and psycopg2 spans are dropped only when the SDK traces that driver
  itself; before, they were dropped even when it did not.
- Instrumentation never breaks the instrumented call: a failure while starting, recording or
  exporting a span is logged at debug level and the call runs and returns as if untraced, and a
  provider's exception reaches the caller unchanged. Passing the message object of an earlier reply
  back in `messages` no longer raises inside the SDK, and `messages` given as a generator is no
  longer consumed before the request.
- Stream wrappers behave as the provider's stream: `next()`, `close()`, `isinstance(stream,
  openai.Stream)` and the raw-response helpers (`with_raw_response`, `with_streaming_response`) work,
  and a stream dropped before it ends records its span when it is collected. What a span keeps of a
  stream is capped (1 MiB of text, 64 KiB of arguments per tool call, 256 tool calls).
- The `requests` and `aiohttp` wrappers add `traceparent` to a copy of the caller's headers instead
  of the caller's own dict, and pass positional arguments through.
- `httpx2` clients are traced like `httpx` ones, so a provider client running on httpx2 (OpenAI's
  `DefaultHttpx2Client`, and newer OpenAI releases by default) records its HTTP request again.
- A worker stopped by its server no longer loses the spans of its last seconds. The ASGI
  middleware flushes on lifespan shutdown, before reporting it complete (uvicorn `--workers` and
  `--reload` end the worker right after, without running `atexit`), and `init()` chains SIGTERM and
  SIGINT: the handler already installed is called as before, and only where the signal's default
  action would end the process without `atexit` is the queue sent first.
- The flush as the process ends takes at most 3 s in all, so an unreachable receiver no longer
  holds an exit for about 20 s (or about 4 s when the connection is refused). `evalkit.flush()`
  keeps its full retry budget.
- A LlamaIndex model call and the provider request it makes are one `llm_call` span: the provider's
  record, with its exact usage, plus `evalkit.framework` and `llama_index.class`. They were two
  spans, so the call's tokens and cost were counted twice. A streamed chat is sent once its request
  ends.
- A Litestar app, and a Django app served through ASGI, no longer lose the spans of a worker's
  last seconds either; `on_shutdown=[evalkit.flush]` is no longer needed. Their middleware never
  sees the lifespan scope: Litestar answers it before any middleware runs, so
  `create_litestar_middleware()` makes the `Litestar` class flush before it reports the shutdown
  complete, after the app's own `on_shutdown` hooks. Django's ASGI handler refuses the lifespan,
  so uvicorn ran it without one; `EvalKitDjangoMiddleware` gives the handler a lifespan that starts
  at once and flushes at shutdown, and leaves alone a handler that answers the lifespan itself.
  Under `uvicorn --workers 2`, 3 requests sent just before a SIGTERM left no spans for either and
  now all arrive, as they do for Litestar under `--reload`. The flush keeps the 3 s exit budget.
  Flask and Django under gunicorn already sent them through the exit flush.
- Streamed OpenAI Responses calls on the async client are traced:
  `AsyncOpenAI().responses.create(stream=True)` and `responses.stream()` record one `llm_call`
  span, as the sync client does; they left none. Both clients now read a Responses stream through
  the tracer the chat completion streams use, so the span has the time to first token (text,
  refusal and reasoning deltas stop the clock, tool arguments do not), the assembled text, the
  tool calls (from the final response, built-in tools and citations included, or from the argument
  deltas when the stream was cut off), the usage and the response id and model. A stream that fails
  to open is an `ERROR` span with the request and the exception; one that fails part way, or whose
  response ends with `response.failed` or an `error` event, is an `ERROR` span with the text
  received so far. The caller gets the client's own stream: `isinstance(stream, openai.AsyncStream)`,
  `next()` and `anext()`, `close()`, `with` and `async with`, `get_final_response()` and
  `with_raw_response` work. A sync Responses stream used to have no time to first token, was
  recorded `OK` when it failed and was not an `openai.Stream`.
- A Responses reply, streamed or not, records its reasoning summary as `gen_ai.response.thinking`,
  its cached input tokens and `gen_ai.usage.reasoning_tokens`. Reading a Responses request for its
  span no longer fails the call: an input part whose `text` is `None` raised `TypeError` before
  the request was sent, and any other failure there now sends the request untraced.

### Changed
- `evalkit.flush()` returns whether the receiver took everything that was queued, and `False`
  before `init()`, as the TypeScript SDK's `flush()` does; it never raises. It raised
  `RuntimeError` before `init()`, so a shutdown hook or test teardown that flushes failed in a
  process that never initialised the SDK.
- The receiver of a method (`self`, `cls`) is no longer recorded as the first item of
  `function.args`; it was the object's default repr.
- `capture_secrets=True` (`EVALKIT_CAPTURE_SECRETS`) still records arguments and results as given.
- A statement SQLAlchemy already records (through its OpenTelemetry instrumentation or
  `patch_sqlalchemy_engine`) is not recorded again by the psycopg 3, psycopg2 or PyMySQL tracer
  underneath.
- `install_logging_handler()` only adds its handler; it no longer sets the logger's level or
  replaces other handlers' formatters.

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
