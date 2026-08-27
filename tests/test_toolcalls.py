import json
import types
import unittest

from evalkit.core.types import TraceEnvelope
from evalkit.instrumentation import _toolcalls as tc
from tests.helpers import (
    AResp, GenAIResp, ORespResult, OResp, fresh, reset_context,
)

TS = "2026-06-07T00:00:00Z"


class _Annotation:
    def __init__(self, title, url, type="url_citation"):
        self.type = type
        self.title = title
        self.url = url


class _MsgBlock:
    def __init__(self, annotations):
        self.annotations = annotations


class _WebSearchItem:
    def __init__(self, query):
        self.type = "web_search_call"
        self.id = "ws_1"
        self.action = types.SimpleNamespace(query=query)


class _MessageItem:
    def __init__(self, annotations):
        self.type = "message"
        self.content = [_MsgBlock(annotations)]


class _LCToolCallObj:
    def __init__(self, name, args, id):
        self.name = name
        self.args = args
        self.id = id


class _LCMessage:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


class TestOpenAIChat(unittest.TestCase):
    def test_extracts_tool_call(self):
        events = tc.from_openai_chat(OResp(tool=("search", '{"q":"x"}')), TS)
        self.assertEqual(len(events), 1)
        a = events[0].attributes
        self.assertEqual(a["gen_ai.tool.name"], "search")
        self.assertEqual(a["gen_ai.tool.call.id"], "tc_1")
        self.assertEqual(a["gen_ai.tool.call.arguments"], '{"q":"x"}')

    def test_no_tool(self):
        self.assertEqual(tc.from_openai_chat(OResp(), TS), [])


class TestAnthropicBlocks(unittest.TestCase):
    def test_object_blocks_json_args(self):
        events = tc.from_anthropic_blocks(AResp(tool=("get_weather", {"city": "london"})).content, TS)
        self.assertEqual(len(events), 1)
        a = events[0].attributes
        self.assertEqual(a["gen_ai.tool.name"], "get_weather")
        self.assertEqual(a["gen_ai.tool.call.id"], "t1")
        self.assertEqual(json.loads(a["gen_ai.tool.call.arguments"]), {"city": "london"})

    def test_dict_blocks(self):
        blocks = [{"type": "tool_use", "name": "f", "id": "x", "input": {"a": 1}}]
        events = tc.from_anthropic_blocks(blocks, TS)
        self.assertEqual(events[0].attributes["gen_ai.tool.name"], "f")
        self.assertEqual(json.loads(events[0].attributes["gen_ai.tool.call.arguments"]), {"a": 1})

    def test_text_only(self):
        self.assertEqual(tc.from_anthropic_blocks(AResp(text="hi").content, TS), [])

    def test_server_tool_use_blocks_are_tool_calls_too(self):
        blocks = [{"type": "server_tool_use", "name": "web_search", "id": "srvtoolu_1",
                   "input": {"query": "weather"}}]
        events = tc.from_anthropic_blocks(blocks, TS)
        self.assertEqual(
            len(events), 1,
            "web_search, web_fetch and code_execution come back as server_tool_use rather than tool_use; skipping "
            "that type meant a Claude call that really did search the web recorded no tool activity at all",
        )
        self.assertEqual(events[0].attributes["gen_ai.tool.name"], "web_search")
        self.assertEqual(json.loads(events[0].attributes["gen_ai.tool.call.arguments"]), {"query": "weather"})

    def test_other_block_types_are_still_skipped(self):
        blocks = [{"type": "text", "text": "hi"}, {"type": "thinking", "thinking": "hmm"},
                  {"type": "tool_result", "content": "42"}]
        self.assertEqual(
            tc.from_anthropic_blocks(blocks, TS), [],
            "widening the accepted block types must not turn text or thinking blocks into phantom tool calls",
        )


class TestGoogleParts(unittest.TestCase):
    def test_function_call(self):
        events = tc.from_google_parts(GenAIResp(tool=("get_weather", {"city": "paris"})), TS)
        self.assertEqual(len(events), 1)
        a = events[0].attributes
        self.assertEqual(a["gen_ai.tool.name"], "get_weather")
        self.assertTrue(a["gen_ai.tool.call.id"].startswith("call_"))
        self.assertEqual(json.loads(a["gen_ai.tool.call.arguments"]), {"city": "paris"})

    def test_text_only(self):
        self.assertEqual(tc.from_google_parts(GenAIResp(text="hello"), TS), [])


class TestOpenAIResponses(unittest.TestCase):
    def test_function_call(self):
        events = tc.from_openai_responses(ORespResult(tool=("get_weather", '{"city":"Paris"}')).output, TS)
        self.assertEqual(len(events), 1)
        a = events[0].attributes
        self.assertEqual(a["gen_ai.tool.name"], "get_weather")
        self.assertEqual(a["gen_ai.tool.call.id"], "call_1")

    def test_web_search_with_citations(self):
        items = [
            _WebSearchItem("current AI news"),
            _MessageItem([_Annotation("Apple News", "https://example.com/aapl")]),
        ]
        events = tc.from_openai_responses(items, TS)
        self.assertEqual(len(events), 1)
        a = events[0].attributes
        self.assertEqual(a["gen_ai.tool.name"], "web_search")
        self.assertEqual(a["gen_ai.tool.call.arguments"], "current AI news")
        self.assertIn("https://example.com/aapl", a["gen_ai.tool.call.result"])
        self.assertIn("Apple News", a["gen_ai.tool.call.result"])

    def test_empty(self):
        self.assertEqual(tc.from_openai_responses([], TS), [])


class TestLangChain(unittest.TestCase):
    def test_dict_form(self):
        msg = _LCMessage([{"name": "t", "args": {"a": 1}, "id": "x"}])
        events = tc.from_langchain_message(msg, TS)
        self.assertEqual(events[0].attributes["gen_ai.tool.name"], "t")
        self.assertEqual(events[0].attributes["gen_ai.tool.call.id"], "x")
        self.assertEqual(json.loads(events[0].attributes["gen_ai.tool.call.arguments"]), {"a": 1})

    def test_object_form(self):
        msg = _LCMessage([_LCToolCallObj("t", {"a": 1}, "x")])
        events = tc.from_langchain_message(msg, TS)
        self.assertEqual(events[0].attributes["gen_ai.tool.name"], "t")

    def test_result_walks_generations(self):
        gen = types.SimpleNamespace(message=_LCMessage([{"name": "t", "args": {}, "id": "x"}]))
        response = types.SimpleNamespace(generations=[[gen]])
        events = tc.from_langchain_result(response, TS)
        self.assertEqual(len(events), 1)


class TestEdgeCases(unittest.TestCase):
    def test_nameless_skipped(self):
        self.assertIsNone(tc.tool_event(name="", arguments="{}", ts=TS))
        self.assertIsNone(tc.tool_event(name=None, arguments="{}", ts=TS))

    def test_none_never_raises(self):
        self.assertEqual(tc.from_openai_chat(None, TS), [])
        self.assertEqual(tc.from_anthropic_blocks(None, TS), [])
        self.assertEqual(tc.from_google_parts(None, TS), [])
        self.assertEqual(tc.from_openai_responses(None, TS), [])
        self.assertEqual(tc.from_langchain_message(None, TS), [])

    def test_result_only_when_present(self):
        ev = tc.tool_event(name="t", arguments="{}", ts=TS)
        self.assertNotIn("gen_ai.tool.call.result", ev.attributes)
        ev2 = tc.tool_event(name="t", arguments="{}", result="r", ts=TS)
        self.assertEqual(ev2.attributes["gen_ai.tool.call.result"], "r")

    def test_dispatcher_autodetects(self):
        self.assertEqual(len(tc.extract_tool_events(OResp(tool=("s", "{}")), TS)), 1)
        self.assertEqual(len(tc.extract_tool_events(GenAIResp(tool=("g", {})), TS)), 1)
        self.assertEqual(len(tc.extract_tool_events(ORespResult(tool=("r", "{}")), TS)), 1)


class TestPromotionToSpans(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_web_search_event_becomes_child_span_with_result(self):
        items = [
            _WebSearchItem("ai news"),
            _MessageItem([_Annotation("AP", "https://ap.example/x")]),
        ]
        events = tc.from_openai_responses(items, TS)
        envelope = TraceEnvelope(
            trace_id="t" * 32, span_id="s" * 16, parent_span_id=None,
            span_type="llm_call", status="OK", operation="responses.create",
            service_name="test", events=events, start_time=TS, end_time=TS, created_at=TS,
        )
        self.client._maybe_emit_tool_spans(envelope)
        self.client.flush()
        tool_spans = [e for e in self.captured if e.span_type == "tool_call"]
        self.assertEqual(len(tool_spans), 1)
        sp = tool_spans[0]
        self.assertEqual(sp.parent_span_id, "s" * 16)
        self.assertEqual(sp.attributes["tool.name"], "web_search")
        self.assertEqual(sp.attributes["tool.arguments"], "ai news")
        self.assertIn("https://ap.example/x", sp.attributes["tool.result"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
