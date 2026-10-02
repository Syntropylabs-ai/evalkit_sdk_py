# Changelog

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
