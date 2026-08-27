import pytest

pytest.importorskip("opentelemetry")


class _StubClient:
    def __init__(self):
        self.spans = []
        self._opts = type("O", (), {"service_name": "bridge-test"})()

    def enqueue(self, e):
        self.spans.append(e)


@pytest.fixture
def isolated_otel():
    from opentelemetry import trace
    from opentelemetry.util._once import Once
    from evalkit.core import otel_provider

    saved_provider = trace._TRACER_PROVIDER
    saved_once = trace._TRACER_PROVIDER_SET_ONCE
    saved_started = otel_provider._started
    saved_owned = set(otel_provider._OWNED_SCOPES)
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE = Once()
    otel_provider._started = False
    otel_provider._OWNED_SCOPES.clear()
    try:
        yield otel_provider
    finally:
        trace._TRACER_PROVIDER = saved_provider
        trace._TRACER_PROVIDER_SET_ONCE = saved_once
        otel_provider._started = saved_started
        otel_provider._OWNED_SCOPES.clear()
        otel_provider._OWNED_SCOPES.update(saved_owned)


def test_setup_otel_attaches_to_a_provider_the_app_already_installed(isolated_otel):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    host = TracerProvider()
    trace._TRACER_PROVIDER = host

    client = _StubClient()
    assert isolated_otel.setup_otel(client), (
        "set_tracer_provider only logs a warning when a provider is already global, so replacing it "
        "silently drops every span; setup_otel must attach a processor to the host provider instead"
    )
    assert isolated_otel.otel_active(), (
        "otel_active() gates whether EvalKit defers to the bridge, so it must never report success "
        "unless a span processor is genuinely attached to the global provider"
    )
    assert trace.get_tracer_provider() is host, (
        "the app's provider keeps its own exporters; hijacking the global would break their backend"
    )

    host.get_tracer("app").start_span("checkout").end()
    host.force_flush()
    assert [e.operation for e in client.spans] == ["checkout"], (
        "spans from the host provider are the entire point of attaching to it"
    )


def test_setup_otel_reports_failure_when_the_global_provider_rejects_processors(isolated_otel):
    from opentelemetry import trace

    class _Opaque(trace.TracerProvider):
        def get_tracer(self, *a, **kw):
            return trace.NoOpTracer()

    trace.set_tracer_provider(_Opaque())

    client = _StubClient()
    assert not isolated_otel.setup_otel(client), (
        "a non-SDK provider cannot take a span processor, so claiming success here is the bug that "
        "made otel_active() lie and made auto instrumentation skip patches it still needed to apply"
    )
    assert not isolated_otel.otel_active()
    assert client.spans == []


def test_skip_list_uses_the_real_contrib_entry_point_names(isolated_otel):
    skip = isolated_otel._SKIP_INSTRUMENTORS

    for name in ("requests", "httpx", "httpx2", "aiohttp-client",
                 "redis", "pymongo", "asyncpg", "psycopg", "psycopg2", "pymysql"):
        assert name in skip, (
            f"{name!r} is the entry point name published by opentelemetry-instrumentation-{name}; "
            "EvalKit instruments this library itself, so registering contrib's version too emits a "
            "second span for every call — the duplicate-span bug that trace-service2 works around "
            "by de-duplicating on (trace_id, tokens_in, tokens_out)"
        )

    assert "bedrock" not in skip, (
        "there is no 'bedrock' instrumentor: Bedrock is a botocore service extension, so this entry "
        "was a no-op that read as coverage"
    )
    assert "botocore" not in skip and "aiobotocore" not in skip, (
        "botocore contrib also covers S3/DynamoDB/SQS/Lambda, which EvalKit does not instrument; "
        "skipping it wholesale would trade one duplicate span for a total loss of AWS visibility, so "
        "the Bedrock overlap is dropped by scope at export time instead"
    )
    assert "sqlalchemy" not in skip, (
        "patch_sqlalchemy needs an Engine handed to it and is never called from auto instrumentation, "
        "so contrib's sqlalchemy instrumentor is the only automatic coverage there is"
    )


def test_owned_scopes_drop_contrib_spans_that_duplicate_evalkit_instrumentation(isolated_otel):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    host = TracerProvider()
    trace._TRACER_PROVIDER = host
    client = _StubClient()
    assert isolated_otel.setup_otel(client)

    isolated_otel.mark_owned_scope(
        "opentelemetry.instrumentation.requests",
        "opentelemetry.instrumentation.botocore.bedrock-runtime",
    )

    host.get_tracer("opentelemetry.instrumentation.requests").start_span("GET").end()
    host.get_tracer("opentelemetry.instrumentation.botocore.bedrock-runtime").start_span(
        "Bedrock Runtime.Converse"
    ).end()
    host.get_tracer("opentelemetry.instrumentation.botocore").start_span("S3.GetObject").end()
    host.get_tracer("app").start_span("handler").end()
    host.force_flush()

    assert [e.operation for e in client.spans] == ["S3.GetObject", "handler"], (
        "the skip list only stops instrumentors EvalKit registers; an app that registered contrib "
        "itself still double-reports, so owned scopes must also be dropped at export. Bedrock's "
        "scope is service-namespaced (registry.get_instrumentation_name), which is what lets the "
        "S3 span through"
    )


def _ended_span(attributes, *, events=(), error=None, kind=None):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import SpanKind
    from opentelemetry.trace.status import Status, StatusCode

    tracer = TracerProvider().get_tracer("app")
    span = tracer.start_span(
        "chat", kind=SpanKind.CLIENT if kind is None else kind, attributes=attributes
    )
    for name, event_attrs, *timestamp in events:
        span.add_event(name, event_attrs, *timestamp)
    if error is not None:
        span.set_status(Status(StatusCode.ERROR, error))
    span.end()
    return span


def test_bridge_promotes_every_column_the_dashboard_filters_on():
    from evalkit.core.otel_provider import _readable_to_envelope

    env = _readable_to_envelope(_ended_span(
        {
            "evalkit.span_type": "llm_call",
            "evalkit.sdk_version": "9.9.9",
            "evalkit.prompt": "summarise this ticket",
            "evalkit.completion": "the customer wants a refund",
            "evalkit.environment": "production",
            "evalkit.app_version": "2026.8.1",
            "evalkit.user_id": "u_42",
            "evalkit.session_id": "s_7",
            "evalkit.device_id": "d_1",
            "gen_ai.system": "anthropic",
            "gen_ai.request.model": "claude-sonnet-4-6",
            "gen_ai.usage.input_tokens": 1135,
            "gen_ai.usage.output_tokens": 20,
            "gen_ai.usage.cache_read.input_tokens": 900,
            "gen_ai.usage.cache_creation_tokens": 12,
        },
        events=[("gen_ai.tool.call", {"gen_ai.tool.name": "refund"})],
        error="upstream 503",
    ))

    assert env.span_kind == "CLIENT", (
        "spanKind is a first-class ClickHouse column that the APM views use to tell an inbound SERVER span from "
        "the CLIENT call it made; leaving it unset makes every bridged span default to INTERNAL and collapses "
        "that distinction"
    )
    assert env.status_message == "upstream 503", (
        "an OTel span's failure detail lives in Status.description, and it is the only human-readable reason a "
        "bridged span failed; dropping it leaves the customer with status=ERROR and nothing to debug"
    )
    assert env.prompt == "summarise this ticket" and env.completion == "the customer wants a refund", (
        "evalkit.prompt/evalkit.completion is the cross-SDK convention the TypeScript exporter already promotes; "
        "a bridged span that leaves them buried in the attributes blob never populates the prompt/completion "
        "columns the trace viewer renders"
    )
    assert env.cache_read_tokens == 900 and env.cache_creation_tokens == 12, (
        "the tracing dashboard's estimateSpanCost carves the cache buckets out of tokensIn and prices them at the "
        "cheaper cache rate; without these two columns every cached input token on a bridged span is billed at "
        "the full input rate, which over-charges an Anthropic cache read by roughly ten times"
    )
    assert (env.environment, env.app_version, env.user_id, env.session_id, env.device_id, env.sdk_version) == (
        "production", "2026.8.1", "u_42", "s_7", "d_1", "9.9.9"
    ), (
        "these six are promoted ClickHouse columns and the only way to filter traces per environment, release, "
        "end user or session; left in the attributes blob they are invisible to every dashboard query"
    )
    assert env.events and env.events[0].name == "gen_ai.tool.call", (
        "span events are how tool calls travel inline on an LLM span, and the client derives child tool_call "
        "spans from them; dropping them makes a bridged agent span look like it called no tools at all"
    )


def test_bridge_reads_a_numeric_span_kind():
    from evalkit.core.otel_provider import _readable_to_envelope

    span = _ended_span({"evalkit.span_type": "llm_call"})
    span._kind = 1
    assert span.kind == 1, "the point of this test is a ReadableSpan whose kind is a plain int, not a SpanKind"

    env = _readable_to_envelope(span)
    assert env.span_kind == "SERVER", (
        "OTLP-derived and hand-rolled ReadableSpans expose kind as the raw protobuf int rather than a SpanKind "
        "enum, so a name-only read yields None and the span silently files as INTERNAL; sdk-ts maps the numbers "
        "and sdk-py has to agree, or the same span changes kind depending on which SDK bridged it"
    )


def test_bridge_ignores_an_unmappable_span_kind():
    from evalkit.core.otel_provider import _readable_to_envelope

    span = _ended_span({"evalkit.span_type": "llm_call"})
    span._kind = 99
    env = _readable_to_envelope(span)
    assert env.span_kind is None, (
        "spanKind is a ClickHouse LowCardinality enum column; inventing a value for an out-of-range int would "
        "either fail the insert or store a kind no view knows how to filter, so omitting it and taking the "
        "server's INTERNAL default is the only safe reading"
    )


def test_bridge_does_not_resurrect_the_tokens_the_foreign_llm_rule_drops(monkeypatch):
    from evalkit.core.otel_provider import _readable_to_envelope

    monkeypatch.delenv("EVALKIT_COUNT_FOREIGN_LLM", raising=False)
    env = _readable_to_envelope(_ended_span({
        "gen_ai.system": "gemini",
        "gen_ai.request.model": "gemini-2.5-flash",
        "gen_ai.usage.input_tokens": 1135,
        "gen_ai.usage.output_tokens": 20,
        "gen_ai.usage.cached_tokens": 900,
        "gen_ai.usage.cache_creation.input_tokens": 12,
    }))

    assert env.span_type == "function_call"
    assert (env.tokens_in, env.tokens_out) == (None, None)
    assert (env.cache_read_tokens, env.cache_creation_tokens) == (None, None), (
        "a gen_ai span with no evalkit marker is a foreign instrumentor's copy of a call EvalKit already "
        "reported, so its tokens are nulled to stop double counting; the dashboard still prices cache buckets "
        "after clamping full-price input to zero, so leaking cache tokens here bills the customer twice for the "
        "cached half of a prompt that was already counted once"
    )
    assert not any("oken" in key for key in env.to_dict()), (
        "the nulled buckets must be absent from the wire payload, not sent as zeros, because evalkit.traces is a "
        "plain MergeTree with no dedupe and a zero row is indistinguishable from a measured zero"
    )


def test_bridge_events_carry_mutable_attribute_copies_so_redaction_can_strip_them():
    from evalkit.core.otel_provider import _readable_to_envelope
    from evalkit.core.redact import strip_content

    env = strip_content(_readable_to_envelope(_ended_span(
        {"evalkit.span_type": "llm_call", "evalkit.prompt": "card 4111 1111 1111 1111",
         "evalkit.completion": "declined", "gen_ai.usage.input_tokens": 11},
        events=[("gen_ai.tool.call", {
            "gen_ai.tool.name": "charge_card",
            "gen_ai.tool.call.arguments": '{"card":"4111 1111 1111 1111"}',
            "gen_ai.tool.call.result": '{"charged":true}',
        })],
    )))

    assert env.prompt is None and env.completion is None
    assert "evalkit.prompt" not in env.attributes and "evalkit.completion" not in env.attributes
    event_attrs = env.events[0].attributes
    assert isinstance(event_attrs, dict), (
        "OpenTelemetry hands out event attributes as an immutable BoundedAttributes, which redact._strip_attrs "
        "silently skips because it is not a dict - forwarding it unconverted would ship tool arguments and "
        "results to the server for a customer who set capture_content=False"
    )
    assert "gen_ai.tool.call.arguments" not in event_attrs
    assert "gen_ai.tool.call.result" not in event_attrs
    assert event_attrs["gen_ai.tool.name"] == "charge_card", (
        "redaction removes payloads, not identity: the tool name is what makes a redacted trace still useful"
    )
    assert env.tokens_in == 11, "tokens are billing data, not content"


def test_bridge_does_not_mutate_the_live_span_when_redacting():
    from evalkit.core.otel_provider import _readable_to_envelope
    from evalkit.core.redact import strip_content

    span = _ended_span(
        {"evalkit.span_type": "llm_call"},
        events=[("gen_ai.tool.call", {
            "gen_ai.tool.name": "charge_card",
            "gen_ai.tool.call.arguments": '{"card":"4111"}',
        })],
    )
    strip_content(_readable_to_envelope(span))

    assert span.events[0].attributes["gen_ai.tool.call.arguments"] == '{"card":"4111"}', (
        "a ReadableSpan is shared with every other exporter on the host provider, so redacting EvalKit's copy "
        "must not reach back into it - stripping in place would silently censor the customer's own Jaeger or "
        "OTLP pipeline"
    )


def test_bridge_emits_wire_safe_scalars():
    import json
    from evalkit.core.otel_provider import _readable_to_envelope

    env = _readable_to_envelope(_ended_span({
        "evalkit.span_type": "llm_call",
        "gen_ai.usage.input_tokens": 1135.0,
        "gen_ai.usage.cache_read_tokens": -1,
        "gen_ai.usage.cached_tokens": 7,
        "gen_ai.request.model": 4,
    }))

    assert env.tokens_in == 1135 and isinstance(env.tokens_in, int), (
        "trace-service2 decodes the whole batch into a Go struct whose TokensIn is an int, and encoding/json "
        "rejects 1135.0 for an int field - one float token count from one foreign instrumentor returns 400 for "
        "the entire POST, so every span batched with it is lost silently"
    )
    assert env.cache_read_tokens == 7, (
        "cache_read_tokens is a UInt32 in ClickHouse, so a negative value from a buggy instrumentor cannot be "
        "stored; skip it and fall through to the next known key rather than shipping it"
    )
    assert env.model == "4", (
        "OTel attribute values are typed and an instrumentor may hand back a non-string model; Model is a Go "
        "string and the decoder rejects a bare number for it, taking the whole batch with it"
    )
    assert json.loads(json.dumps(env.to_dict()))["cacheReadTokens"] == 7


def test_bridge_never_ships_an_unparseable_event_timestamp():
    from evalkit.core.otel_provider import _readable_to_envelope

    span = _ended_span(
        {"evalkit.span_type": "llm_call"},
        events=[("gen_ai.tool.call", {"gen_ai.tool.name": "refund"}, 0)],
    )
    env = _readable_to_envelope(span)

    assert env.events[0].timestamp == env.end_time, (
        "trace-service2 decodes event timestamps into a Go time.Time, and an empty string fails RFC3339 parsing "
        "for the whole batch; falling back to the span's own end time keeps the event and the spans batched "
        "alongside it"
    )


def test_derived_tool_spans_keep_the_user_they_belong_to():
    from evalkit.core.client import EvalKitClient
    from evalkit.core.otel_provider import _readable_to_envelope

    client = EvalKitClient.__new__(EvalKitClient)
    sent = []
    client.enqueue = lambda e: sent.append(e)

    env = _readable_to_envelope(_ended_span(
        {
            "evalkit.span_type": "llm_call",
            "evalkit.user_id": "u_42",
            "evalkit.session_id": "s_7",
        },
        events=[("gen_ai.tool.call", {
            "gen_ai.tool.name": "refund",
            "gen_ai.tool.call.result": '{"ok":true}',
        })],
    ))
    EvalKitClient._maybe_emit_tool_spans(client, env)

    assert len(sent) == 1 and sent[0].span_type == "tool_call"
    assert sent[0].user_id == "u_42", (
        "bridging events made this derivation reachable for the first time, and it copied every promoted column "
        "onto the child except user_id - so a per-user filter would return the parent LLM span and silently omit "
        "the tool calls it made, which is exactly the trace a support engineer is looking for"
    )
    assert sent[0].session_id == "s_7"
