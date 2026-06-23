from evalkit.core.otel_provider import _readable_to_envelope


class _Ctx:
    trace_id = 0x1
    span_id = 0x2
    is_valid = True


class _Span:
    """Minimal ReadableSpan stand-in (no opentelemetry import needed)."""

    def __init__(self, attrs):
        self._a = attrs
        self.name = "gen"
        self.start_time = 1_000_000_000
        self.end_time = 2_000_000_000
        self.status = None
        self.parent = None
        self.instrumentation_scope = None

    def get_span_context(self):
        return _Ctx()

    @property
    def attributes(self):
        return self._a


def test_foreign_gen_ai_span_defers_for_cost():
    # Foreign/bridged OTel gen_ai span (e.g. Google ADK) — no evalkit marker.
    foreign = _Span({
        "gen_ai.system": "gemini",
        "gen_ai.request.model": "gemini-2.5-flash",
        "gen_ai.usage.input_tokens": 1135,
        "gen_ai.usage.output_tokens": 20,
    })
    env = _readable_to_envelope(foreign)
    assert env.span_type == "function_call"
    assert env.tokens_in is None and env.tokens_out is None


def test_native_marked_span_keeps_llm_call_and_tokens():
    native = _Span({
        "evalkit.span_type": "llm_call",
        "evalkit.sdk_version": "0.2.2",
        "gen_ai.system": "openai",
        "gen_ai.usage.input_tokens": 50,
        "gen_ai.usage.output_tokens": 10,
    })
    env = _readable_to_envelope(native)
    assert env.span_type == "llm_call"
    assert env.tokens_in == 50 and env.tokens_out == 10


def test_opt_out_keeps_foreign_llm_countable(monkeypatch):
    monkeypatch.setenv("EVALKIT_COUNT_FOREIGN_LLM", "1")
    foreign = _Span({"gen_ai.system": "gemini", "gen_ai.usage.input_tokens": 99})
    env = _readable_to_envelope(foreign)
    assert env.span_type == "llm_call"
    assert env.tokens_in == 99


def test_non_llm_foreign_span_untouched():
    db = _Span({"db.system": "postgresql", "db.statement": "SELECT 1"})
    env = _readable_to_envelope(db)
    assert env.span_type == "db_query"
