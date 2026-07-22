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
        # Mirrors the real class closely enough to exercise the patch: takes
        # a `handlers` list, tracks `inheritable_handlers`, and supports
        # `add_handler(handler, inherit)` the same way langchain-core does.
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

    def test_handler_actually_gets_attached_to_new_callback_managers(self):
        """Guards against the class of bug that shipped silently in 0.2.9:
        register_configure_hook was removed in langchain-core 1.x, so the old
        registration path built a handler but never actually wired it up —
        zero spans were ever emitted against any current LangChain install.
        This constructs a callback manager the way LangChain itself does
        internally and asserts our handler is actually present on it,
        rather than only exercising the handler's methods directly.
        """
        handler = evalkit.patch_langchain()
        self.assertIsNotNone(handler)

        from langchain_core.callbacks.base import BaseCallbackManager

        mgr = BaseCallbackManager(handlers=[])
        self.assertTrue(
            any(isinstance(h, type(handler)) for h in mgr.inheritable_handlers),
            "evalkit handler was not attached to a newly constructed BaseCallbackManager",
        )

        # A second manager must also get it (every run constructs its own).
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
