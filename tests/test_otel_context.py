import pytest

pytest.importorskip("opentelemetry")


class _StubClient:
    def __init__(self):
        self.spans = []
        self._opts = type("O", (), {"service_name": "test"})()

    def enqueue(self, e):
        self.spans.append(e)


def test_hand_rolled_span_nests_under_active_otel_span():
    from evalkit.core.otel_provider import setup_otel
    from evalkit.core.context import current_span_ctx, get_or_create_ambient_trace_id
    from opentelemetry import trace

    assert setup_otel(_StubClient())
    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("request") as span:
        sc = span.get_span_context()
        otel_trace = format(sc.trace_id, "032x")
        otel_span = format(sc.span_id, "016x")

        parent = current_span_ctx()
        assert parent is not None
        assert parent.trace_id == otel_trace
        assert parent.span_id == otel_span
        assert get_or_create_ambient_trace_id() == otel_trace


def test_no_otel_span_uses_fresh_ambient_trace():
    from evalkit.core.context import current_span_ctx, get_or_create_ambient_trace_id

    assert current_span_ctx() is None
    trace_id = get_or_create_ambient_trace_id()
    assert len(trace_id) == 32


def test_otel_span_maps_to_envelope_via_exporter():
    from evalkit.core.otel_provider import setup_otel, _readable_to_envelope
    from opentelemetry import trace

    setup_otel(_StubClient())
    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("db.query") as span:
        span.set_attribute("db.system", "postgresql")
        span.set_attribute("db.statement", "SELECT 1")
    env = _readable_to_envelope(span)
    assert env.span_type == "db_query"
    assert len(env.trace_id) == 32 and len(env.span_id) == 16
    assert env.attributes.get("db.system") == "postgresql"
