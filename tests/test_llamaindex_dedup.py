"""A LlamaIndex model call and the provider request inside it are one llm_call span.

The tracker reads duck-typed events, so these run without LlamaIndex installed.
"""
import unittest

from tests.helpers import OResp, OStreamChunk, fresh, install_fake_openai_sync, reset_context


class _Event:
    def __init__(self, name, span_id, **fields):
        self._name = name
        self.span_id = span_id
        self.__dict__.update(fields)

    def class_name(self):
        return self._name


class _Message:
    def __init__(self, role, content):
        self.role, self.content = role, content


class OpenAI:
    """Stands in for a LlamaIndex LLM class; the tracker reads its name."""


class TestLlamaIndexProviderDedup(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        from evalkit.instrumentation.llm.llamaindex import _Tracker

        self.tracker = _Tracker(self.client)
        self.completions = install_fake_openai_sync(self.client, OResp("Order 1001 shipped.", pin=12, pout=5))

    def _llm(self):
        self.client.flush()
        return [e for e in self.captured if e.span_type == "llm_call"]

    def _start(self, span_id):
        self.tracker.on_span_enter(span_id, None, instance=OpenAI())
        self.tracker.on_event(_Event("LLMChatStartEvent", span_id, model_dict={"model": "gpt-4o-mini"},
                                     messages=[_Message("user", "Where is order 1001?")]))

    def _end(self, span_id):
        response = _Event("ChatResponse", span_id, message=_Message("assistant", "Order 1001 shipped."), raw={})
        self.tracker.on_event(_Event("LLMChatEndEvent", span_id, response=response))

    def test_a_chat_and_its_provider_request_are_one_span(self):
        self._start("s1")
        self.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "Where is order 1001?"}])
        self._end("s1")
        self.tracker.on_span_exit("s1")
        spans = self._llm()
        self.assertEqual(len(spans), 1, "the same request recorded twice is counted, and priced, twice")
        span = spans[0]
        self.assertEqual(span.operation, "chat.completions.create")
        self.assertEqual((span.tokens_in, span.tokens_out), (12, 5), "the provider's exact usage is kept")
        self.assertEqual(span.attributes["evalkit.framework"], "llama_index")
        self.assertEqual(span.attributes["llama_index.class"], "OpenAI")

    def test_a_streamed_chat_waits_for_its_request(self):
        stream_completions = install_fake_openai_sync(self.client, [OStreamChunk("Order "), OStreamChunk("shipped.")])
        self._start("s2")
        self.tracker.on_span_exit("s2")
        self.assertEqual(self._llm(), [], "the span waits for the request the stream makes")
        list(stream_completions.create(model="gpt-4o-mini", messages=[], stream=True))
        self._end("s2")
        spans = self._llm()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].completion, "Order shipped.")
        self.assertEqual(spans[0].attributes["evalkit.framework"], "llama_index")

    def test_a_chat_with_no_instrumented_provider_is_still_recorded(self):
        self._start("s3")
        self._end("s3")
        self.tracker.on_span_exit("s3")
        spans = self._llm()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].operation, "chat")
        self.assertIn("Where is order 1001?", spans[0].prompt)

    def test_provider_calls_after_the_chat_are_recorded_on_their_own(self):
        self._start("s4")
        self._end("s4")
        self.tracker.on_span_exit("s4")
        self.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "later"}])
        self.assertEqual(len(self._llm()), 2)


if __name__ == "__main__":
    unittest.main()
