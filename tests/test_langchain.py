import sys
import types
import unittest

import evalkit
from tests.helpers import fresh, reset_context


def _setup_fake_langchain_core():
    base = types.ModuleType("langchain_core.callbacks.base")

    class BaseCallbackHandler:
        pass

    class BaseCallbackManager:
        def __init__(self, handlers=None, inheritable_handlers=None, **kwargs):
            self.handlers = list(handlers or [])
            self.inheritable_handlers = list(inheritable_handlers or [])

        def add_handler(self, handler, inherit=True):
            self.handlers.append(handler)
            if inherit:
                self.inheritable_handlers.append(handler)

    base.BaseCallbackHandler = BaseCallbackHandler
    base.BaseCallbackManager = BaseCallbackManager
    manager = types.ModuleType("langchain_core.callbacks.manager")
    callbacks = types.ModuleType("langchain_core.callbacks")
    root = types.ModuleType("langchain_core")

    mods = {
        "langchain_core": root,
        "langchain_core.callbacks": callbacks,
        "langchain_core.callbacks.base": base,
        "langchain_core.callbacks.manager": manager,
    }
    sys.modules.update(mods)
    return mods


class TestLangChain(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self._mods = _setup_fake_langchain_core()

        import evalkit.instrumentation.llm.langchain as lc
        lc._registered = False

    def tearDown(self):
        for name in self._mods:
            sys.modules.pop(name, None)
        import evalkit.instrumentation.llm.langchain as lc
        lc._registered = False

    def _make_human_msg(self, text):
        return types.SimpleNamespace(type="human", content=text)

    def _make_llm_output(self, text, pin=4, pout=6):
        return types.SimpleNamespace(
            generations=[[types.SimpleNamespace(text=text, message=None)]],
            llm_output={"token_usage": {"prompt_tokens": pin, "completion_tokens": pout}},
        )

    def test_full_span_tree(self):
        handler = evalkit.patch_langchain()
        self.assertIsNotNone(handler)

        handler.on_chain_start({"name": "graph"}, {}, run_id="A", parent_run_id=None)
        handler.on_chat_model_start(
            {"id": ["langchain", "chat_models", "openai", "ChatOpenAI"],
             "kwargs": {"model": "gpt-4o"}},
            [[self._make_human_msg("hi")]],
            run_id="B", parent_run_id="A",
        )
        handler.on_llm_end(self._make_llm_output("hello", 4, 6), run_id="B")
        handler.on_tool_start({"name": "search"}, "query", run_id="C", parent_run_id="A")
        handler.on_tool_end("result", run_id="C")
        handler.on_chain_end({}, run_id="A")
        self.client.flush()

        by_type = {}
        for e in self.captured:
            by_type.setdefault(e.span_type, []).append(e)

        chain = by_type["function_call"][0]
        llm = by_type["llm_call"][0]
        tool = by_type["tool_call"][0]

        self.assertEqual(chain.trace_id, llm.trace_id)
        self.assertEqual(chain.trace_id, tool.trace_id)
        self.assertEqual(llm.parent_span_id, chain.span_id)
        self.assertEqual(tool.parent_span_id, chain.span_id)
        self.assertEqual(llm.model, "gpt-4o")
        self.assertEqual(llm.provider, "openai")
        self.assertEqual(llm.completion, "hello")
        self.assertEqual(llm.tokens_in, 4)
        self.assertEqual(llm.tokens_out, 6)
        self.assertEqual(tool.attributes.get("tool.name"), "search")

    def test_langgraph_metadata_places_every_span_in_the_graph_run(self):
        handler = evalkit.patch_langchain()
        agent = {
            "langgraph_step": 3,
            "langgraph_node": "agent",
            "langgraph_triggers": ("branch:to:agent",),
            "langgraph_path": ("__pregel_pull", "agent"),
            "langgraph_checkpoint_ns": "agent:6f0e2b7c",
            "checkpoint_ns": "agent:6f0e2b7c",
        }
        tools = {**agent, "langgraph_step": 4, "langgraph_node": "tools",
                 "langgraph_triggers": ["branch:to:tools", "agent"], "langgraph_checkpoint_ns": "tools:91aa"}

        handler.on_chain_start({"name": "LangGraph"}, {}, run_id="G", parent_run_id=None)
        handler.on_chain_start({"name": "agent"}, {}, run_id="A", parent_run_id="G",
                               tags=["graph:step:3"], metadata=agent)
        handler.on_chat_model_start({"id": ["langchain", "chat_models", "openai", "ChatOpenAI"]},
                                    [[self._make_human_msg("hi")]], run_id="B", parent_run_id="A",
                                    tags=["seq:step:1"], metadata=agent)
        handler.on_llm_end(self._make_llm_output("calling search"), run_id="B")
        handler.on_llm_start({"id": ["langchain", "llms", "openai", "OpenAI"]}, ["hi"],
                             run_id="L", parent_run_id="A", metadata=agent)
        handler.on_llm_end(self._make_llm_output("done"), run_id="L")
        handler.on_chain_end({}, run_id="A")
        handler.on_tool_start({"name": "search"}, "query", run_id="C", parent_run_id="G", metadata=tools)
        handler.on_tool_end("result", run_id="C")
        handler.on_chain_end({}, run_id="G")
        self.client.flush()

        spans = {e.operation: e.attributes for e in self.captured}
        self.assertFalse(
            [k for k in spans["LangGraph"] if k.startswith("langgraph.")],
            "the graph's own run sits outside every node and carries no node metadata",
        )
        for operation in ("agent", "chat.generate", "llm.generate"):
            attrs = spans[operation]
            self.assertEqual(attrs.get("langgraph.node"), "agent", operation)
            self.assertEqual(attrs.get("langgraph.step"), 3, operation)
            self.assertIs(type(attrs.get("langgraph.step")), int, operation)
            self.assertEqual(attrs.get("langgraph.triggers"), '["branch:to:agent"]', operation)
            self.assertEqual(attrs.get("langgraph.checkpoint_ns"), "agent:6f0e2b7c", operation)
        tool = spans["search"]
        self.assertEqual(tool.get("langgraph.node"), "tools")
        self.assertEqual(tool.get("langgraph.step"), 4)
        self.assertEqual(tool.get("langgraph.triggers"), '["branch:to:tools","agent"]',
                         "the same compact JSON array the TypeScript SDK writes")
        self.assertEqual(tool.get("langgraph.checkpoint_ns"), "tools:91aa")
        self.assertEqual(spans["agent"].get("langchain.tags"), ["graph:step:3"], "tags keep their own attribute")
        self.assertNotIn("langgraph.path", spans["agent"])

    def test_metadata_without_langgraph_keys_adds_nothing(self):
        handler = evalkit.patch_langchain()
        handler.on_chain_start({"name": "chain"}, {}, run_id="A", parent_run_id=None,
                               metadata={"user": "u1", "langgraph_step": True, "langgraph_triggers": []})
        handler.on_chain_end({}, run_id="A")
        handler.on_tool_start({"name": "t"}, "x", run_id="T", parent_run_id=None, metadata="not a mapping")
        handler.on_tool_end("y", run_id="T")
        self.client.flush()
        for envelope in self.captured:
            self.assertFalse([k for k in envelope.attributes if k.startswith("langgraph.")], envelope.operation)

    def test_handler_actually_gets_attached_to_new_callback_managers(self):
        handler = evalkit.patch_langchain()
        self.assertIsNotNone(handler)

        from langchain_core.callbacks.base import BaseCallbackManager

        mgr = BaseCallbackManager(handlers=[])
        self.assertTrue(
            any(isinstance(h, type(handler)) for h in mgr.inheritable_handlers),
            "evalkit handler was not attached to a newly constructed BaseCallbackManager",
        )

        mgr2 = BaseCallbackManager(handlers=[])
        self.assertTrue(any(isinstance(h, type(handler)) for h in mgr2.inheritable_handlers))

    def test_noop_when_no_langchain(self):
        from evalkit.instrumentation.auto import _try_patch_langchain
        _try_patch_langchain(self.client)

    def test_patch_langchain_export(self):
        result = evalkit.patch_langchain()
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
