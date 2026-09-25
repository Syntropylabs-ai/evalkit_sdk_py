"""The request settings a model-call span records so the trace service can tell agent versions apart."""

import asyncio

import pytest

from tests.helpers import (
    AResp, OResp, OStreamChunk, ORespResult,
    fresh, install_fake_anthropic, install_fake_openai_async, install_fake_openai_responses_sync,
    install_fake_openai_sync, reset_context,
)

CHAT_TOOLS = [
    {"type": "function", "function": {"name": "lookup_order", "parameters": {}}},
    {"type": "function", "function": {"name": "issue_refund", "parameters": {}}},
]


class NotGiven:
    """Stands in for the provider clients' NOT_GIVEN sentinel."""

    def __bool__(self):
        return False


@pytest.fixture
def client():
    reset_context()
    captured = []
    c = fresh(captured)
    c.captured = captured
    return c


def _llm_span(c):
    c.flush()
    spans = [e for e in c.captured if e.span_type == "llm_call"]
    assert len(spans) == 1
    return spans[0]


def test_openai_chat_records_tools_and_sampling(client):
    install_fake_openai_sync(client, OResp("ok")).create(
        model="gpt-4o-mini", messages=[{"role": "system", "content": "Be terse."}],
        tools=CHAT_TOOLS, temperature=0.2, top_p=1, max_completion_tokens=256,
    )
    attrs = _llm_span(client).attributes
    assert attrs["gen_ai.request.tools"] == ["lookup_order", "issue_refund"]
    assert (attrs["gen_ai.request.temperature"], attrs["gen_ai.request.top_p"]) == (0.2, 1)
    assert attrs["gen_ai.request.max_tokens"] == 256


def test_openai_stream_and_async_record_them_too(client):
    stream = install_fake_openai_sync(client, [OStreamChunk("ok")])
    list(stream.create(model="gpt-4o-mini", messages=[], tools=CHAT_TOOLS, temperature=0, stream=True))
    assert _llm_span(client).attributes["gen_ai.request.temperature"] == 0

    client.captured.clear()
    call = install_fake_openai_async(client, OResp("ok")).create(model="gpt-4o-mini", messages=[], tools=CHAT_TOOLS)
    asyncio.run(call)
    assert _llm_span(client).attributes["gen_ai.request.tools"] == ["lookup_order", "issue_refund"]


def test_unset_settings_are_left_out(client):
    install_fake_openai_sync(client, OResp("ok")).create(
        model="gpt-4o-mini", messages=[], tools=NotGiven(), temperature=NotGiven(), top_p=True, max_tokens=None,
    )
    attrs = _llm_span(client).attributes
    assert not {k for k in attrs if k in (
        "gen_ai.request.tools", "gen_ai.request.temperature", "gen_ai.request.top_p", "gen_ai.request.max_tokens",
    )}


def test_a_failed_call_still_names_its_configuration(client):
    from evalkit.instrumentation.auto import _patch_openai_class_sync

    class Failing:
        def create(self, *a, **k):
            raise RuntimeError("down")

    _patch_openai_class_sync(client, Failing)
    with pytest.raises(RuntimeError):
        Failing().create(model="gpt-4o-mini", messages=[], tools=CHAT_TOOLS)
    span = _llm_span(client)
    assert span.status == "ERROR"
    assert span.attributes["gen_ai.request.tools"] == ["lookup_order", "issue_refund"]


def test_anthropic_records_tools_and_renders_a_block_system_prompt_as_text(client):
    install_fake_anthropic(client, AResp(text="ok")).create(
        model="claude-sonnet-4-5", max_tokens=512, temperature=0.5,
        system=[{"type": "text", "text": "You are the support agent.", "cache_control": {"type": "ephemeral"}}],
        tools=[{"name": "lookup_order", "input_schema": {}}, {"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": "where is my order?"}],
    )
    span = _llm_span(client)
    assert span.attributes["gen_ai.request.tools"] == ["lookup_order", "web_search"]
    assert span.attributes["gen_ai.request.max_tokens"] == 512
    assert span.prompt.startswith("system: You are the support agent.\nuser: where is my order?")


def test_responses_names_function_and_builtin_tools(client):
    install_fake_openai_responses_sync(client, ORespResult("ok")).create(
        model="gpt-4o", instructions="Be terse.", input="hi", max_output_tokens=64,
        tools=[{"type": "function", "name": "lookup_order", "parameters": {}}, {"type": "web_search"}],
    )
    attrs = _llm_span(client).attributes
    assert attrs["gen_ai.request.tools"] == ["lookup_order", "web_search"]
    assert attrs["gen_ai.request.max_tokens"] == 64
