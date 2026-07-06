import inspect
import unittest

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, CompletionResponse, MessageRole
from llama_index.core.base.response.schema import Response
from llama_index.core.instrumentation.events.embedding import EmbeddingEndEvent, EmbeddingStartEvent
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatStartEvent,
    LLMCompletionEndEvent,
    LLMCompletionStartEvent,
)
from llama_index.core.instrumentation.events.query import QueryEndEvent, QueryStartEvent
from llama_index.core.instrumentation.events.rerank import ReRankEndEvent, ReRankStartEvent
from llama_index.core.instrumentation.events.retrieval import RetrievalEndEvent, RetrievalStartEvent
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from tests.helpers import fresh, reset_context

_BOUND = inspect.signature(lambda: None).bind()


class FakeLLM:
    pass


class TestLlamaIndexTracker(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def _tracker(self):
        from evalkit.instrumentation.llm.llamaindex import _Tracker
        return _Tracker(self.client)

    def test_llm_chat_emits_llm_call_span(self):
        t = self._tracker()
        t.on_span_enter("s1", None, instance=FakeLLM())
        t.on_event(LLMChatStartEvent(
            span_id="s1",
            messages=[ChatMessage(role=MessageRole.USER, content="hi there")],
            additional_kwargs={},
            model_dict={"model": "gpt-4o"},
        ))
        t.on_event(LLMChatEndEvent(
            span_id="s1",
            messages=[ChatMessage(role=MessageRole.USER, content="hi there")],
            response=ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content="hello!"),
                raw={"usage": {"prompt_tokens": 5, "completion_tokens": 3}},
            ),
        ))
        t.on_span_exit("s1")
        self.client.flush()

        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].model, "gpt-4o")
        self.assertIn("hi there", llm[0].prompt)
        self.assertEqual(llm[0].completion, "hello!")
        self.assertEqual(llm[0].tokens_in, 5)
        self.assertEqual(llm[0].tokens_out, 3)
        self.assertEqual(llm[0].attributes["evalkit.framework"], "llama_index")

    def test_llm_completion_emits_llm_call_span(self):
        t = self._tracker()
        t.on_span_enter("s2", None, instance=FakeLLM())
        t.on_event(LLMCompletionStartEvent(
            span_id="s2", prompt="say hi", additional_kwargs={}, model_dict={"model": "text-davinci"},
        ))
        t.on_event(LLMCompletionEndEvent(
            span_id="s2", prompt="say hi",
            response=CompletionResponse(text="hi!", raw={"usage": {"prompt_tokens": 2, "completion_tokens": 1}}),
        ))
        t.on_span_exit("s2")
        self.client.flush()

        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].operation, "complete")
        self.assertEqual(llm[0].completion, "hi!")
        self.assertEqual(llm[0].tokens_in, 2)

    def test_embedding_never_captures_raw_vectors(self):
        t = self._tracker()
        t.on_span_enter("s3", None, instance=FakeLLM())
        t.on_event(EmbeddingStartEvent(span_id="s3", model_dict={"model": "text-embedding-3-small"}))
        t.on_event(EmbeddingEndEvent(span_id="s3", chunks=["chunk one", "chunk two"], embeddings=[[0.1, 0.2], [0.3, 0.4]]))
        t.on_span_exit("s3")
        self.client.flush()

        emb = [e for e in self.captured if e.span_type == "llm_call" and e.operation == "embedding"]
        self.assertEqual(len(emb), 1)
        self.assertEqual(emb[0].model, "text-embedding-3-small")
        self.assertEqual(emb[0].attributes["evalkit.embedding.chunk_count"], 2)
        self.assertEqual(emb[0].attributes["evalkit.embedding.dimensions"], 2)
        for v in emb[0].attributes.values():
            self.assertNotIn("0.1", str(v))

    def test_retrieval_emits_db_query_span(self):
        t = self._tracker()
        t.on_span_enter("s4", None, instance=FakeLLM())
        t.on_event(RetrievalStartEvent(span_id="s4", str_or_query_bundle=QueryBundle(query_str="what is evalkit?")))
        nodes = [
            NodeWithScore(node=TextNode(text="doc a", id_="a"), score=0.9),
            NodeWithScore(node=TextNode(text="doc b", id_="b"), score=0.7),
        ]
        t.on_event(RetrievalEndEvent(span_id="s4", str_or_query_bundle=QueryBundle(query_str="what is evalkit?"), nodes=nodes))
        t.on_span_exit("s4")
        self.client.flush()

        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertEqual(len(db), 1)
        self.assertEqual(db[0].prompt, "what is evalkit?")
        self.assertEqual(db[0].attributes["db.vector.match_count"], 2)

    def test_query_emits_function_call_span(self):
        t = self._tracker()
        t.on_span_enter("s5", None, instance=FakeLLM())
        t.on_event(QueryStartEvent(span_id="s5", query="what is evalkit?"))
        resp = Response(response="evalkit is an eval SDK", source_nodes=[NodeWithScore(node=TextNode(text="doc a", id_="a"), score=0.9)])
        t.on_event(QueryEndEvent(span_id="s5", query="what is evalkit?", response=resp))
        t.on_span_exit("s5")
        self.client.flush()

        fc = [e for e in self.captured if e.span_type == "function_call"]
        self.assertEqual(len(fc), 1)
        self.assertEqual(fc[0].completion, "evalkit is an eval SDK")
        self.assertEqual(fc[0].attributes["rag.source_node_count"], 1)

    def test_rerank_emits_function_call_span(self):
        t = self._tracker()
        t.on_span_enter("s6", None, instance=FakeLLM())
        nodes_in = [NodeWithScore(node=TextNode(text="a", id_="a"), score=0.5), NodeWithScore(node=TextNode(text="b", id_="b"), score=0.4)]
        t.on_event(ReRankStartEvent(span_id="s6", query="q", nodes=nodes_in, top_n=1, model_name="rerank-v2"))
        t.on_event(ReRankEndEvent(span_id="s6", nodes=[nodes_in[0]]))
        t.on_span_exit("s6")
        self.client.flush()

        fc = [e for e in self.captured if e.span_type == "function_call" and e.operation == "rerank"]
        self.assertEqual(len(fc), 1)
        self.assertEqual(fc[0].model, "rerank-v2")
        self.assertEqual(fc[0].attributes["rerank.input_count"], 2)
        self.assertEqual(fc[0].attributes["rerank.output_count"], 1)

    def test_span_without_event_is_not_emitted(self):
        """Framework-internal spans (no *StartEvent inside) must be filtered out."""
        t = self._tracker()
        t.on_span_enter("s7", None, instance=FakeLLM())
        t.on_span_exit("s7")
        self.client.flush()
        self.assertEqual(len(self.captured), 0)

    def test_span_drop_emits_error_status(self):
        t = self._tracker()
        t.on_span_enter("s8", None, instance=FakeLLM())
        t.on_event(QueryStartEvent(span_id="s8", query="boom?"))
        t.on_span_drop("s8", RuntimeError("index unavailable"))
        self.client.flush()

        fc = [e for e in self.captured if e.span_type == "function_call"]
        self.assertEqual(len(fc), 1)
        self.assertEqual(fc[0].status, "ERROR")
        self.assertIn("index unavailable", fc[0].status_message)

    def test_nested_recognized_span_gets_correct_parent(self):
        """query -> retrieve nesting: retrieve's parent_span_id must resolve to
        query's evalkit span_id, even though retrieve's llama-index parent_id
        is query's llama-index id_ (a different id space)."""
        t = self._tracker()
        t.on_span_enter("query-1", None, instance=FakeLLM())
        t.on_event(QueryStartEvent(span_id="query-1", query="q"))

        t.on_span_enter("retrieve-1", "query-1", instance=FakeLLM())
        t.on_event(RetrievalStartEvent(span_id="retrieve-1", str_or_query_bundle="q"))
        t.on_event(RetrievalEndEvent(span_id="retrieve-1", str_or_query_bundle="q", nodes=[]))
        query_evalkit_span_id = t._spans["query-1"]["span_id"]
        t.on_span_exit("retrieve-1")

        t.on_event(QueryEndEvent(span_id="query-1", query="q", response=Response(response="a", source_nodes=[])))
        t.on_span_exit("query-1")
        self.client.flush()

        retrieve_span = [e for e in self.captured if e.span_type == "db_query"][0]
        self.assertEqual(retrieve_span.parent_span_id, query_evalkit_span_id)

    def test_nested_span_walks_past_uninteresting_ancestor(self):
        """query -> (uninteresting wrapper) -> retrieve: retrieve should still
        parent to query's evalkit span_id, skipping the unrecognized middle span."""
        t = self._tracker()
        t.on_span_enter("query-2", None, instance=FakeLLM())
        t.on_event(QueryStartEvent(span_id="query-2", query="q"))

        t.on_span_enter("wrapper-2", "query-2", instance=FakeLLM())  # no event -> uninteresting

        t.on_span_enter("retrieve-2", "wrapper-2", instance=FakeLLM())
        t.on_event(RetrievalStartEvent(span_id="retrieve-2", str_or_query_bundle="q"))
        t.on_event(RetrievalEndEvent(span_id="retrieve-2", str_or_query_bundle="q", nodes=[]))
        query_evalkit_span_id = t._spans["query-2"]["span_id"]
        t.on_span_exit("retrieve-2")
        t.on_span_exit("wrapper-2")

        t.on_event(QueryEndEvent(span_id="query-2", query="q", response=Response(response="a", source_nodes=[])))
        t.on_span_exit("query-2")
        self.client.flush()

        retrieve_span = [e for e in self.captured if e.span_type == "db_query"][0]
        self.assertEqual(retrieve_span.parent_span_id, query_evalkit_span_id)


class TestLlamaIndexPatchIntegration(unittest.TestCase):
    """End-to-end through the real dispatcher, proving auto_instrument() wiring works.

    Deliberately does NOT call patch_llamaindex() itself: llama-index-core is
    installed in this environment, so evalkit.init() (via fresh(), in setUp of the
    TestLlamaIndexTracker class above, which always runs first in this process)
    already triggers auto_instrument() -> _try_patch_llamaindex() exactly once
    against the shared singleton client. Calling patch_llamaindex() again here
    would register a *second* handler on the process-wide dispatcher (handlers are
    additive, never replaced) and double-emit — that's not a bug, just not how a
    real process (which calls init() once) ever exercises this path.
    """

    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_auto_instrument_wires_llamaindex_into_real_dispatcher(self):
        from llama_index.core.instrumentation import get_dispatcher

        dispatcher = get_dispatcher()
        dispatcher.span_enter(id_="int-1", bound_args=_BOUND, instance=FakeLLM(), parent_id=None)
        dispatcher.event(QueryStartEvent(span_id="int-1", query="hello"))
        dispatcher.event(QueryEndEvent(span_id="int-1", query="hello", response=Response(response="world", source_nodes=[])))
        dispatcher.span_exit(id_="int-1", bound_args=_BOUND, instance=FakeLLM(), result=None)
        self.client.flush()

        fc = [e for e in self.captured if e.span_type == "function_call" and e.completion == "world"]
        self.assertEqual(len(fc), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
