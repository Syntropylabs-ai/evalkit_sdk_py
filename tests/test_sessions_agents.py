import asyncio
import threading

import pytest

import evalkit
from tests.helpers import fresh, reset_context


@pytest.fixture
def client():
    reset_context()
    evalkit.set_agent(None)
    captured = []
    c = fresh(captured)
    c.captured = captured
    try:
        yield c
    finally:
        evalkit.set_user()
        evalkit.set_agent(None)


def _span(c, operation):
    c.flush()
    spans = [e for e in c.captured if e.operation == operation]
    assert len(spans) == 1, operation
    return spans[0]


def test_bridge_spans_join_the_enclosing_evalkit_trace_tree(client, bridge):
    host, stub = bridge
    trace_id, end, ctx = evalkit.start_trace("turn")
    with evalkit.with_trace(ctx):
        tracer = host.get_tracer("third-party")
        with tracer.start_as_current_span("http.fetch") as outer:
            tracer.start_span("dns.lookup").end()
    end("OK")
    _flush_from_another_thread(host)
    outer_env = next(e for e in stub.spans if e.operation == "http.fetch")
    inner_env = next(e for e in stub.spans if e.operation == "dns.lookup")
    assert outer_env.trace_id == trace_id and inner_env.trace_id == trace_id
    assert outer_env.parent_span_id == ctx.span_id
    assert inner_env.parent_span_id == format(outer.get_span_context().span_id, "016x")


def test_bridge_span_outside_any_evalkit_span_keeps_its_own_trace(bridge):
    host, stub = bridge
    span = host.get_tracer("third-party").start_span("standalone")
    span.end()
    _flush_from_another_thread(host)
    envelope = next(e for e in stub.spans if e.operation == "standalone")
    assert envelope.trace_id == format(span.get_span_context().trace_id, "032x")
    assert envelope.parent_span_id is None


@pytest.mark.parametrize(
    "key", ["session.id", "conversation.id", "gen_ai.conversation.id", "session_id", "evalkit.session_id"]
)
def test_start_trace_reads_every_session_spelling_and_children_inherit(client, key):
    _, end, ctx = client.start_trace("root", {key: "s-attr"})
    with evalkit.with_trace(ctx):
        end_child, _ = client.start_span("child")
        end_child("OK")
    end("OK")
    assert ctx.session_id == "s-attr"
    assert _span(client, "root").session_id == "s-attr"
    assert _span(client, "child").session_id == "s-attr", (
        "the id has to be resolved at span start so everything opened inside the trace groups "
        "into the same session"
    )


@pytest.mark.parametrize("key", ["user.id", "enduser.id", "evalkit.user_id"])
def test_start_span_reads_every_user_spelling(client, key):
    end, ctx = client.start_span("tagged", {key: "u-attr"})
    end("OK")
    assert ctx.user_id == "u-attr"
    assert _span(client, "tagged").to_dict()["userId"] == "u-attr"


def test_precedence_explicit_then_attribute_then_parent_then_ambient(client):
    evalkit.set_user(user_id="u-ambient", session_id="s-ambient")
    _, end, ctx = client.start_trace("explicit", {"session.id": "s-attr"}, session_id="s-arg")
    end("OK")
    assert _span(client, "explicit").session_id == "s-arg"

    _, end, ctx = client.start_trace("attr", {"session.id": "s-attr"})
    with evalkit.with_trace(ctx):
        end_child, _ = client.start_span("attr-child")
        end_child("OK")
    end("OK")
    assert _span(client, "attr").session_id == "s-attr"
    assert _span(client, "attr-child").session_id == "s-attr"
    assert _span(client, "attr-child").user_id == "u-ambient"

    end_plain, _ = client.start_span("ambient-only")
    end_plain("OK")
    assert _span(client, "ambient-only").session_id == "s-ambient"


def test_instrumented_envelope_attribute_names_its_session(client):
    from evalkit.core.client import _now_iso
    from evalkit.core.types import TraceEnvelope

    evalkit.set_user(session_id="s-ambient")
    client.enqueue(TraceEnvelope(
        trace_id="t" * 32, span_id="a" * 16, created_at=_now_iso(), operation="conv",
        attributes={"gen_ai.conversation.id": "s-conv"},
    ))
    assert _span(client, "conv").session_id == "s-conv"


def test_session_scope_applies_and_restores(client):
    evalkit.set_user(user_id="u-outer", session_id="s-outer")
    with evalkit.session("s-inner", user_id="u-inner"):
        assert evalkit.current_user().session_id == "s-inner"
        end, _ = client.start_span("inside")
        end("OK")
    assert evalkit.current_user().session_id == "s-outer"
    assert evalkit.current_user().user_id == "u-outer"
    end, _ = client.start_span("outside")
    end("OK")
    inside = _span(client, "inside")
    assert (inside.session_id, inside.user_id) == ("s-inner", "u-inner")
    assert _span(client, "outside").session_id == "s-outer"


def test_session_scope_keeps_ambient_user_when_none_given(client):
    evalkit.set_user(user_id="u-keep")
    with evalkit.session("s-only"):
        end, _ = client.start_span("kept-user")
        end("OK")
    span = _span(client, "kept-user")
    assert (span.session_id, span.user_id) == ("s-only", "u-keep")


def test_session_decorator_scopes_each_call(client):
    @evalkit.session("s-sync")
    def handle():
        end, _ = client.start_span("sync-call")
        end("OK")

    @evalkit.session("s-async", user_id="u-async")
    async def handle_async(label):
        await asyncio.sleep(0)
        end, _ = client.start_span(label)
        end("OK")

    handle()

    async def run():
        await asyncio.gather(handle_async("async-a"), handle_async("async-b"))

    asyncio.run(run())
    assert evalkit.current_user().session_id is None
    assert _span(client, "sync-call").session_id == "s-sync"
    assert _span(client, "async-a").session_id == "s-async"
    assert _span(client, "async-b").user_id == "u-async"


def test_set_agent_stamps_envelope_and_both_attributes(client):
    evalkit.set_agent("support-bot")
    _, end, ctx = client.start_trace("turn")
    with evalkit.with_trace(ctx):
        end_child, _ = client.start_span("lookup")
        end_child("OK")
    end("OK")
    for op in ("turn", "lookup"):
        span = _span(client, op)
        payload = span.to_dict()
        assert payload["agentName"] == "support-bot"
        assert payload["attributes"]["evalkit.agent_name"] == "support-bot"
        assert payload["attributes"]["gen_ai.agent.name"] == "support-bot"
    evalkit.set_agent(None)
    end, _ = client.start_span("no-agent")
    end("OK")
    assert "agentName" not in _span(client, "no-agent").to_dict()


def test_start_trace_agent_beats_ambient_and_reaches_children(client):
    evalkit.set_agent("ambient-agent")
    _, end, ctx = evalkit.start_trace("planner-turn", agent="planner")
    with evalkit.with_trace(ctx):
        end_child, _ = client.start_span("plan-step")
        end_child("OK")
    end("OK")
    assert _span(client, "planner-turn").agent_name == "planner"
    assert _span(client, "plan-step").agent_name == "planner"


@pytest.mark.parametrize("key", ["evalkit.agent_name", "gen_ai.agent.name", "agent.name"])
def test_agent_name_attribute_spellings(client, key):
    end, ctx = client.start_span("sub-agent", {key: "researcher"})
    end("OK")
    span = _span(client, "sub-agent")
    assert ctx.agent_name == "researcher"
    assert span.agent_name == "researcher"
    assert span.attributes["gen_ai.agent.name"] == "researcher"


def test_agent_attribute_precedence(client):
    end, _ = client.start_span("both", {"agent.name": "loose", "gen_ai.agent.name": "semconv"})
    end("OK")
    assert _span(client, "both").agent_name == "semconv"


def test_set_agent_is_task_scoped_and_token_restores(client):
    evalkit.set_agent("main-agent")

    async def run():
        async def task(name):
            evalkit.set_agent(name)
            await asyncio.sleep(0)
            end, _ = client.start_span(f"agent-task-{name}")
            end("OK")

        await asyncio.gather(task("a"), task("b"))

    asyncio.run(run())
    assert _span(client, "agent-task-a").agent_name == "a"
    assert _span(client, "agent-task-b").agent_name == "b"
    assert evalkit.current_agent() == "main-agent"
    token = evalkit.set_agent("temporary")
    evalkit.clear_agent(token)
    assert evalkit.current_agent() == "main-agent"


# -- OpenTelemetry bridge ---------------------------------------------------------------------


class _StubClient:
    def __init__(self):
        self.spans = []
        self._opts = type("O", (), {"service_name": "bridge-identity"})()

    def enqueue(self, e):
        self.spans.append(e)


@pytest.fixture
def bridge():
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.util._once import Once
    from evalkit.core import otel_provider

    reset_context()
    evalkit.set_agent(None)
    saved = (trace._TRACER_PROVIDER, trace._TRACER_PROVIDER_SET_ONCE, otel_provider._started,
             otel_provider._processor)
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE = Once()
    otel_provider._started = False
    host = TracerProvider()
    trace._TRACER_PROVIDER = host
    stub = _StubClient()
    assert otel_provider.setup_otel(stub)
    try:
        yield host, stub
    finally:
        evalkit.set_user()
        evalkit.set_agent(None)
        (trace._TRACER_PROVIDER, trace._TRACER_PROVIDER_SET_ONCE, otel_provider._started,
         otel_provider._processor) = saved


def _flush_from_another_thread(host):
    t = threading.Thread(target=host.force_flush)
    t.start()
    t.join()


def test_bridge_span_keeps_identity_set_on_the_starting_thread(bridge):
    host, stub = bridge
    evalkit.set_user(user_id="u-bridge", session_id="s-bridge")
    evalkit.set_agent("bridge-agent")
    host.get_tracer("third-party").start_span("vector.search").end()
    evalkit.set_user()
    evalkit.set_agent(None)
    _flush_from_another_thread(host)

    spans = [e for e in stub.spans if e.operation == "vector.search"]
    assert len(spans) == 1
    envelope = spans[0]
    assert envelope.user_id == "u-bridge", (
        "the export thread has no set_user scope, so identity must be captured when the span starts"
    )
    assert envelope.session_id == "s-bridge"
    assert envelope.agent_name == "bridge-agent"


def test_bridge_span_inherits_enclosing_evalkit_trace(client, bridge):
    host, stub = bridge
    _, end, ctx = evalkit.start_trace("request", session_id="s-trace", agent="router")
    with evalkit.with_trace(ctx):
        host.get_tracer("third-party").start_span("db.call").end()
    end("OK")
    _flush_from_another_thread(host)
    envelope = next(e for e in stub.spans if e.operation == "db.call")
    assert envelope.session_id == "s-trace"
    assert envelope.agent_name == "router"


def test_bridge_span_inside_an_instrumented_call_keeps_the_callers_scope(client, bridge):
    """The HTTP span an OpenAI client opens runs while the llm_call span is current."""
    from tests.helpers import OResp
    from evalkit.instrumentation.auto import _patch_openai_class_sync

    host, stub = bridge

    class Completions:
        def create(self, *a, **k):
            host.get_tracer("httpx").start_span("POST /v1/chat/completions").end()
            return OResp("ok")

    _patch_openai_class_sync(client, Completions)
    _, end, ctx = evalkit.start_trace("turn", session_id="s-turn", user_id="u-turn", agent="support-bot")
    with evalkit.with_trace(ctx):
        end_sub, sub_ctx = evalkit.start_span("refund", {"gen_ai.agent.name": "refund-specialist"})
        with evalkit.with_trace(sub_ctx):
            Completions().create(model="gpt-4o-mini", messages=[])
        end_sub("OK")
    end("OK")
    _flush_from_another_thread(host)
    envelope = next(e for e in stub.spans if e.operation == "POST /v1/chat/completions")
    assert (envelope.session_id, envelope.user_id, envelope.agent_name) == ("s-turn", "u-turn", "refund-specialist")


@pytest.mark.parametrize(
    "key", ["session.id", "conversation.id", "gen_ai.conversation.id", "session_id", "evalkit.session_id"]
)
def test_bridge_reads_every_session_spelling_over_ambient(bridge, key):
    host, stub = bridge
    evalkit.set_user(session_id="s-ambient")
    span = host.get_tracer("third-party").start_span("tagged")
    span.set_attribute(key, "s-own")
    span.set_attribute("enduser.id", "u-own")
    span.set_attribute("gen_ai.agent.name", "own-agent")
    span.end()
    _flush_from_another_thread(host)
    envelope = next(e for e in stub.spans if e.operation == "tagged")
    assert envelope.session_id == "s-own"
    assert envelope.user_id == "u-own"
    assert envelope.agent_name == "own-agent"
