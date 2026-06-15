import asyncio
import sys
import types
import unittest

from tests.helpers import fresh, reset_context
from evalkit.instrumentation.mcp import patch_mcp


class _FakeResult:
    def __init__(self):
        self.content = [types.SimpleNamespace(text="42")]
        self.isError = False


class _FakeClientSession:
    async def call_tool(self, name, arguments=None):
        return _FakeResult()


class TestMcp(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_call_tool_emits_tool_call_span(self):
        fake_mcp = types.ModuleType("mcp")
        fake_mcp.ClientSession = _FakeClientSession
        sys.modules["mcp"] = fake_mcp
        try:
            patch_mcp(self.client)
            asyncio.run(_FakeClientSession().call_tool("add", {"a": 1, "b": 2}))
            self.client.flush()
        finally:
            sys.modules.pop("mcp", None)

        spans = [e for e in self.captured if e.span_type == "tool_call"]
        self.assertTrue(spans, "expected a tool_call span from mcp.call_tool")
        s = spans[0]
        self.assertEqual(s.operation, "add")
        attrs = s.attributes or {}
        self.assertEqual(attrs.get("gen_ai.tool.name"), "add")
        self.assertIn("a", attrs.get("gen_ai.tool.call.arguments", ""))
        self.assertIn("42", attrs.get("gen_ai.tool.call.result", ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
