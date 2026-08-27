from __future__ import annotations

import time
import types
import unittest

from tests.helpers import fresh, reset_context

from evalkit.instrumentation.llm.vertex_ai import patch_anthropic_vertex, patch_vertex_generative_model

TTFT = "gen_ai.server.time_to_first_token_ms"


class _VPart:
    def __init__(self, text=None, function_call=None, thought=False):
        self._text = text
        self.function_call = function_call
        self.thought = thought

    @property
    def text(self):
        if self._text is None:
            raise AttributeError("Response candidate content part has no text.")
        return self._text


class _VContent:
    def __init__(self, parts, role=None):
        self.parts = parts
        self.role = role

    @property
    def text(self):
        if len(self.parts) > 1:
            raise ValueError("Multiple content parts are not supported.")
        if not self.parts:
            raise ValueError("Response candidate content has no parts (and thus no text).")
        return self.parts[0].text


class _VCandidate:
    def __init__(self, parts, finish_reason=None):
        self.content = _VContent(parts)
        self.finish_reason = finish_reason

    @property
    def text(self):
        return self.content.text


class _VUsage:
    def __init__(self, pin=0, pout=0, total=0):
        self.prompt_token_count = pin
        self.candidates_token_count = pout
        self.total_token_count = total


class VGenResp:
    def __init__(self, parts, pin=0, pout=0, total=0, finish_reason=None):
        self.candidates = [_VCandidate(parts, finish_reason)]
        self.usage_metadata = _VUsage(pin, pout, total)

    @property
    def text(self):
        if len(self.candidates) > 1:
            raise ValueError("The response has multiple candidates.")
        if not self.candidates:
            raise ValueError("Response has no candidates (and thus no text).")
        return self.candidates[0].text


def _fake_vertex_model(script=None, response=None, pace=0.0):
    def generate_content(contents, **kwargs):
        if kwargs.get("stream"):
            def gen():
                for step in script or []:
                    if pace:
                        time.sleep(pace)
                    if isinstance(step, BaseException):
                        raise step
                    yield step
            return gen()
        return response

    return types.SimpleNamespace(_model_name="gemini-2.5-flash", generate_content=generate_content)


def _a_message_start(input_tokens=11):
    return types.SimpleNamespace(
        type="message_start",
        message=types.SimpleNamespace(
            usage=types.SimpleNamespace(
                input_tokens=input_tokens,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            )
        ),
    )


def _a_text_delta(text):
    return types.SimpleNamespace(
        type="content_block_delta",
        delta=types.SimpleNamespace(type="text_delta", text=text),
    )


def _a_message_delta(output_tokens=7):
    return types.SimpleNamespace(
        type="message_delta",
        usage=types.SimpleNamespace(output_tokens=output_tokens),
    )


def _fake_anthropic_vertex(script, pace=0.0):
    def create(*args, **kwargs):
        if kwargs.get("stream"):
            def gen():
                for step in script:
                    if pace:
                        time.sleep(pace)
                    if isinstance(step, BaseException):
                        raise step
                    yield step
            return gen()
        return script

    return types.SimpleNamespace(messages=types.SimpleNamespace(create=create))


class TestAnthropicVertexStreaming(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def _spans(self):
        self.client.flush()
        return [s for s in self.captured if s.span_type == "llm_call"]

    def test_streaming_call_produces_a_span(self):
        vertex = _fake_anthropic_vertex([
            _a_message_start(11),
            _a_text_delta("Hel"),
            _a_text_delta("lo"),
            _a_message_delta(7),
        ])
        patch_anthropic_vertex(self.client, vertex)

        chunks = list(vertex.messages.create(model="claude-sonnet-4-5", messages=[{"role": "user", "content": "hi"}], stream=True))

        self.assertEqual(len(chunks), 4)
        spans = self._spans()
        self.assertEqual(len(spans), 1,
                         "a streaming Anthropic-on-Vertex call emitted no span, so its tokens and cost were never "
                         "billed to the project — the explicit patcher bailed out on stream=True")
        span = spans[0]
        self.assertEqual(span.provider, "anthropic-vertex")
        self.assertEqual(span.completion, "Hello")
        self.assertEqual(span.tokens_in, 11)
        self.assertEqual(span.tokens_out, 7)
        self.assertEqual(span.status, "OK")

    def test_streaming_call_records_time_to_first_token(self):
        vertex = _fake_anthropic_vertex([
            _a_message_start(4),
            _a_text_delta("a"),
            _a_text_delta("b"),
            _a_message_delta(2),
        ], pace=0.03)
        patch_anthropic_vertex(self.client, vertex)

        list(vertex.messages.create(model="claude-sonnet-4-5", messages=[], stream=True))

        span = self._spans()[0]
        self.assertIn(TTFT, span.attributes,
                      "streaming latency is the number a user feels; a Vertex-hosted Claude stream must report it "
                      "like every other provider or the dashboard column is empty for this provider alone")
        self.assertLess(span.attributes[TTFT], span.latency_ms + 1,
                        "TTFT must be bounded by the span's own duration; a larger value means the clock started "
                        "in the wrong place")

    def test_failed_stream_is_recorded_as_an_error(self):
        vertex = _fake_anthropic_vertex([
            _a_message_start(4),
            _a_text_delta("partial"),
            RuntimeError("vertex stream reset"),
        ])
        patch_anthropic_vertex(self.client, vertex)

        with self.assertRaises(RuntimeError):
            list(vertex.messages.create(model="claude-sonnet-4-5", messages=[], stream=True))

        span = self._spans()[0]
        self.assertEqual(span.status, "ERROR",
                         "a stream that died mid-flight was recorded as a success, so failed streaming calls never "
                         "show up in the dashboard's error rate")
        self.assertIn("vertex stream reset", span.status_message or "")
        self.assertEqual(span.completion, "partial",
                         "the partial text is usually the only clue as to why the stream stopped")
        self.assertNotIn(TTFT, span.attributes,
                         "a truncated stream must stay out of the TTFT percentiles, otherwise a provider outage that "
                         "fails fast reads as a latency improvement")

    def test_non_streaming_call_still_works(self):
        result = types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="sync reply")],
            usage=types.SimpleNamespace(input_tokens=5, output_tokens=9,
                                        cache_read_input_tokens=0, cache_creation_input_tokens=0),
            model="claude-sonnet-4-5",
            id="msg_1",
            stop_reason="end_turn",
        )
        vertex = types.SimpleNamespace(
            messages=types.SimpleNamespace(create=lambda *a, **k: result)
        )
        patch_anthropic_vertex(self.client, vertex)

        vertex.messages.create(model="claude-sonnet-4-5", messages=[{"role": "user", "content": "hi"}])

        span = self._spans()[0]
        self.assertEqual(span.completion, "sync reply")
        self.assertEqual(span.tokens_in, 5)
        self.assertEqual(span.tokens_out, 9)


class TestVertexGenerativeModelStreaming(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def _spans(self):
        self.client.flush()
        return [s for s in self.captured if s.span_type == "llm_call"]

    def test_streaming_captures_completion_and_usage(self):
        model = _fake_vertex_model(script=[
            VGenResp([_VPart("Hel")]),
            VGenResp([_VPart("lo")], pin=6, pout=3, total=9),
        ])
        patch_vertex_generative_model(self.client, model)

        chunks = list(model.generate_content("hi", stream=True))

        self.assertEqual(len(chunks), 2)
        spans = self._spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertEqual(span.completion, "Hello",
                         "a streaming generate_content span had an empty completion and zero tokens, which reads in "
                         "the dashboard as a model that answered nothing for free")
        self.assertEqual(span.tokens_in, 6)
        self.assertEqual(span.tokens_out, 3)
        self.assertEqual(span.status, "OK")

    def test_streaming_records_time_to_first_token(self):
        model = _fake_vertex_model(script=[
            VGenResp([_VPart("a")]),
            VGenResp([_VPart("b")], pin=1, pout=2, total=3),
        ], pace=0.03)
        patch_vertex_generative_model(self.client, model)

        list(model.generate_content("hi", stream=True))

        span = self._spans()[0]
        self.assertIn(TTFT, span.attributes)
        self.assertLess(span.attributes[TTFT], span.latency_ms + 1)

    def test_failed_stream_is_recorded_as_an_error_without_ttft(self):
        model = _fake_vertex_model(script=[
            VGenResp([_VPart("partial")]),
            RuntimeError("deadline exceeded"),
        ])
        patch_vertex_generative_model(self.client, model)

        with self.assertRaises(RuntimeError):
            list(model.generate_content("hi", stream=True))

        span = self._spans()[0]
        self.assertEqual(span.status, "ERROR")
        self.assertIn("deadline exceeded", span.status_message or "")
        self.assertEqual(span.completion, "partial")
        self.assertNotIn(TTFT, span.attributes)

    def test_thinking_text_is_not_mixed_into_the_completion(self):
        model = _fake_vertex_model(script=[
            VGenResp([_VPart("weighing options", thought=True)]),
            VGenResp([_VPart("the answer")], pin=2, pout=4, total=6),
        ])
        patch_vertex_generative_model(self.client, model)

        list(model.generate_content("hi", stream=True))

        span = self._spans()[0]
        self.assertEqual(span.completion, "the answer",
                         "reasoning text billed as output must not be shown as the model's answer")
        self.assertEqual(span.attributes.get("gen_ai.response.thinking"), "weighing options")

    def test_tool_call_chunk_becomes_a_tool_event(self):
        fc = types.SimpleNamespace(name="lookup_order", args={"id": "A-1"})
        model = _fake_vertex_model(script=[VGenResp([_VPart(function_call=fc)], pin=3, pout=5, total=8)])
        patch_vertex_generative_model(self.client, model)

        list(model.generate_content("hi", stream=True))

        span = self._spans()[0]
        self.assertTrue(span.events, "a streamed function call produced no tool event, so tool metrics undercount")
        self.assertEqual(span.events[0].name, "gen_ai.tool.call")

    def test_multi_part_prompt_does_not_break_the_caller(self):
        response = VGenResp([_VPart("ok")], pin=1, pout=1, total=2)
        model = _fake_vertex_model(response=response)
        patch_vertex_generative_model(self.client, model)

        contents = [_VContent([_VPart("describe this"), _VPart(function_call=None, text=None)], role="user")]
        result = model.generate_content(contents)

        self.assertIs(result, response,
                      "a multimodal Content raises ValueError from its .text property, and reading it while building "
                      "the prompt made instrumentation break the user's own call")
        span = self._spans()[0]
        self.assertIn("describe this", span.prompt)

    def test_multi_part_response_keeps_the_whole_completion(self):
        response = VGenResp([_VPart("first "), _VPart("second")], pin=1, pout=1, total=2)
        model = _fake_vertex_model(response=response)
        patch_vertex_generative_model(self.client, model)

        model.generate_content("hi")

        span = self._spans()[0]
        self.assertEqual(span.completion, "first second",
                         "response.text raises on a multi-part candidate, so the completion was silently dropped for "
                         "exactly the responses that carry the most content")


if __name__ == "__main__":
    unittest.main()
