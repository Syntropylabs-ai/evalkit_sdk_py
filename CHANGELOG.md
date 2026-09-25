# Changelog

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
  `start_trace(..., join_trace_id=...)` joins an existing trace on purpose.
- OpenTelemetry spans from other libraries keep the session, user and agent in force when they
  started (they were read on the export thread, which has none), and a span started inside an
  evalkit span joins that evalkit trace instead of exporting as a separate one.
