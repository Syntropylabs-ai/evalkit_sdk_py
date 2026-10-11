"""Spans an agent framework records on the GenAI conventions (Strands Agents 1.59.0), as the bridge types them.

The attribute and event shapes below are a real Strands session's, as stored in ClickHouse.
"""
import json

import pytest

pytest.importorskip("opentelemetry")

from tests.helpers import OResp, fresh, install_fake_openai_sync, reset_context  # noqa: E402

USER = "Hi, my blender from order 48213 arrived cracked. I want my money back."
REPLY = "Order 48213 (Vortex blender, $89.99) was delivered on 2 October. Shall I refund $89.99 to your card?"
STRANDS = {"gen_ai.system": "strands-agents", "session.id": "strands-1", "gen_ai.conversation.id": "strands-1"}


def _blocks(*texts):
    return json.dumps([{"text": t} for t in texts])


def _span(name, attributes, *, events=(), parent=None, tracer=None, framework=True):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    tracer = tracer or TracerProvider().get_tracer("strands.telemetry.tracer")
    context = trace.set_span_in_context(parent) if parent is not None else None
    span = tracer.start_span(name, context=context, attributes={**(STRANDS if framework else {}), **attributes})
    for event, event_attrs in events:
        span.add_event(event, event_attrs)
    span.end()
    return span


def _envelope(span):
    from evalkit.core.otel_provider import _readable_to_envelope

    return _readable_to_envelope(span)


def _agent_span(**over):
    return _span(
        over.pop("name", "invoke_agent refund-agent"),
        {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "refund-agent",
            "gen_ai.agent.tools": json.dumps(["lookup_order", "search_policy"]),
            "gen_ai.request.model": "mock-sonnet",
            "gen_ai.usage.input_tokens": 478,
            "gen_ai.usage.output_tokens": 121,
            **over.pop("attributes", {}),
        },
        events=[
            ("gen_ai.system.message", {"content": _blocks("You are the Northwind refund agent.")}),
            ("gen_ai.user.message", {"content": _blocks(USER)}),
            ("gen_ai.choice", {"finish_reason": "end_turn", "message": REPLY + "\n"}),
        ],
    )


def _chat_span(**kwargs):
    return _span(
        "chat",
        {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": "mock-sonnet",
            "gen_ai.usage.prompt_tokens": 328,
            "gen_ai.usage.input_tokens": 328,
            "gen_ai.usage.completion_tokens": 59,
            "gen_ai.usage.output_tokens": 59,
            "gen_ai.usage.total_tokens": 387,
        },
        events=[
            ("gen_ai.user.message", {"content": _blocks(USER)}),
            ("gen_ai.choice", {"finish_reason": "tool_use", "message": json.dumps(
                [{"toolUse": {"toolUseId": "call_1", "name": "lookup_order", "input": {"order_id": "48213"}}}]
            )}),
        ],
        **kwargs,
    )


def _tool_span(name="lookup_order", *, status="success", result=None, attributes=None):
    return _span(
        f"execute_tool {name}",
        {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": name,
            "gen_ai.tool.call.id": "call_1",
            "gen_ai.tool.status": status,
            **(attributes or {}),
        },
        events=[
            ("gen_ai.tool.message", {"role": "tool", "content": json.dumps({"order_id": "48213"}), "id": "call_1"}),
            ("gen_ai.choice", {"id": "call_1", "message": result if result is not None else json.dumps(
                [{"text": json.dumps({"order_id": "48213", "item": "Vortex blender", "amount": 89.99})}]
            )}),
        ],
    )


@pytest.fixture(autouse=True)
def _no_foreign_opt_out(monkeypatch):
    monkeypatch.delenv("EVALKIT_COUNT_FOREIGN_LLM", raising=False)


def test_invoke_agent_is_an_agent_row_with_its_name_turn_and_reply():
    env = _envelope(_agent_span())
    assert env.span_type == "agent"
    assert env.agent_name == "refund-agent"
    assert env.operation == "invoke_agent refund-agent"
    assert env.prompt == USER, "the user's text, not the content-block list it came in"
    assert env.completion == REPLY
    assert env.model == "mock-sonnet"
    assert (env.tokens_in, env.tokens_out) == (None, None), (
        "an invocation's usage totals the model calls inside it, which carry their own tokens"
    )


def test_an_agent_named_only_by_its_span_name_keeps_that_name():
    span = _span("invoke_agent billing", {"gen_ai.operation.name": "invoke_agent"})
    env = _envelope(span)
    assert (env.span_type, env.agent_name) == ("agent", "billing")


def test_a_multi_agent_run_is_an_agent_row():
    env = _envelope(_span("invoke_swarm", {"gen_ai.operation.name": "invoke_swarm"}))
    assert env.span_type == "agent"


def test_execute_tool_is_a_tool_call_with_its_arguments_and_result_from_the_events():
    env = _envelope(_tool_span())
    assert env.span_type == "tool_call"
    assert env.operation == "lookup_order"
    assert env.attributes["tool.name"] == "lookup_order"
    assert json.loads(env.attributes["tool.arguments"]) == {"order_id": "48213"}
    assert json.loads(env.attributes["tool.result"]) == {"order_id": "48213", "item": "Vortex blender", "amount": 89.99}, (
        "the content block's text, not the block list around it"
    )
    assert env.status == "OK"
    assert env.session_id == "strands-1"


def test_a_tool_call_prefers_the_convention_attributes_and_reads_failure_from_gen_ai_tool_status():
    env = _envelope(_tool_span(
        "issue_refund",
        status="error",
        result=json.dumps([{"json": {"error": "card declined"}}]),
        attributes={"gen_ai.tool.call.arguments": '{"order_id":"1"}'},
    ))
    assert env.span_type == "tool_call"
    assert env.attributes["tool.arguments"] == '{"order_id":"1"}'
    assert env.attributes["tool.result"] == '{"error":"card declined"}'
    assert env.status == "ERROR"


def test_a_tool_named_only_by_its_span_name():
    span = _span("execute_tool escalate_to_human", {"gen_ai.operation.name": "execute_tool"})
    env = _envelope(span)
    assert (env.span_type, env.operation, env.attributes["tool.name"]) == (
        "tool_call", "escalate_to_human", "escalate_to_human"
    )


def test_an_event_loop_cycle_is_internal_never_a_tool_or_model_call():
    span = _span(
        "execute_event_loop_cycle",
        {"gen_ai.operation.name": "execute_event_loop_cycle", "event_loop.cycle_id": "c-1",
         "gen_ai.usage.input_tokens": 21},
        events=[("gen_ai.user.message", {"content": _blocks(USER)})],
    )
    env = _envelope(span)
    assert env.span_type == "internal"
    assert env.attributes["gen_ai.operation.name"] == "execute_event_loop_cycle"
    assert (env.tokens_in, env.tokens_out) == (None, None)


def test_a_chat_span_the_sdk_did_not_record_is_the_model_call():
    env = _envelope(_chat_span())
    assert env.span_type == "llm_call"
    assert env.model == "mock-sonnet"
    assert (env.tokens_in, env.tokens_out) == (328, 59)
    assert env.prompt == f"user: {USER}"
    assert json.loads(env.completion) == {
        "toolUse": {"toolUseId": "call_1", "name": "lookup_order", "input": {"order_id": "48213"}}
    }


def test_a_chat_span_around_the_sdks_own_record_of_the_request_is_not_counted_again():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    reset_context()
    captured = []
    client = fresh(captured)
    completions = install_fake_openai_sync(client, OResp("Order 48213 was delivered.", pin=328, pout=59))
    tracer = TracerProvider().get_tracer("strands.telemetry.tracer")
    with tracer.start_as_current_span("chat", attributes={
        **STRANDS, "gen_ai.operation.name": "chat", "gen_ai.request.model": "mock-sonnet",
        "gen_ai.usage.input_tokens": 328, "gen_ai.usage.output_tokens": 59,
    }) as chat:
        completions.create(model="mock-sonnet", messages=[{"role": "user", "content": [{"type": "text", "text": USER}]}])
    client.flush()

    own = [e for e in captured if e.span_type == "llm_call"]
    assert len(own) == 1 and own[0].parent_span_id == format(chat.get_span_context().span_id, "016x")
    assert own[0].prompt == f"user: {USER}", "list content is the text it holds, never Python's repr of the list"
    env = _envelope(chat)
    assert env.span_type == "internal", "the SDK's own llm_call is this request's record"
    assert (env.tokens_in, env.tokens_out) == (None, None)


def test_the_duplicate_mark_passes_through_a_span_between_the_chat_span_and_the_sdks_call():
    from evalkit.core.otel_provider import note_sdk_model_call

    chat = _chat_span()
    between = _span("POST /v1/chat/completions", {"http.request.method": "POST"}, parent=chat, framework=False)
    note_sdk_model_call("00000000000000aa", format(between.get_span_context().span_id, "016x"))
    assert _envelope(between).span_type == "http_call"
    assert _envelope(chat).span_type == "internal"


def test_a_chat_span_that_started_inside_the_sdks_call_is_not_counted_again():
    from evalkit.core.context import SpanContext, _current_span
    from evalkit.core.otel_provider import capture_start_scope, note_sdk_model_call
    from opentelemetry.sdk.trace import TracerProvider

    sdk_call = SpanContext(trace_id="ab" * 16, span_id="00000000000000bb")
    token = _current_span.set(sdk_call)
    try:
        span = TracerProvider().get_tracer("app").start_span("chat", attributes={
            **STRANDS, "gen_ai.operation.name": "chat", "gen_ai.usage.input_tokens": 5,
        })
        capture_start_scope(span)
    finally:
        _current_span.reset(token)
    span.end()
    note_sdk_model_call(sdk_call.span_id, None)
    assert _envelope(span).span_type == "internal"


def test_the_opt_out_still_counts_a_duplicate_chat_span(monkeypatch):
    from evalkit.core.otel_provider import note_sdk_model_call

    monkeypatch.setenv("EVALKIT_COUNT_FOREIGN_LLM", "1")
    chat = _chat_span()
    note_sdk_model_call("00000000000000cc", format(chat.get_span_context().span_id, "016x"))
    env = _envelope(chat)
    assert env.span_type == "llm_call" and env.tokens_in == 328


def test_a_gen_ai_span_that_names_no_operation_keeps_the_foreign_llm_rule():
    env = _envelope(_span("generate", {"gen_ai.request.model": "gemini-2.5-flash", "gen_ai.usage.input_tokens": 9}))
    assert env.span_type == "function_call"
    assert env.tokens_in is None


def test_the_sdks_own_spans_keep_their_type():
    env = _envelope(_span("chat", {"evalkit.span_type": "llm_call", "evalkit.sdk_version": "0.3.2",
                                   "gen_ai.operation.name": "chat", "gen_ai.usage.input_tokens": 7}))
    assert env.span_type == "llm_call" and env.tokens_in == 7


def test_list_content_is_stored_as_text_and_json_never_as_a_python_repr():
    from evalkit.core.content import message_content_text

    assert message_content_text([{"text": "a", "type": "text"}, {"text": "b"}]) == "a b"
    assert message_content_text([{"type": "tool_use", "id": "t1", "name": "lookup", "input": {"q": "é"}}]) == (
        '{"type":"tool_use","id":"t1","name":"lookup","input":{"q":"é"}}'
    )
    assert message_content_text([{"type": "image_url", "image_url": {"url": "data:..."}}]) == "[image_url]"
    assert message_content_text("plain") == "plain"
    assert "'" not in message_content_text([{"toolResult": {"content": [{"text": "x"}]}}])
