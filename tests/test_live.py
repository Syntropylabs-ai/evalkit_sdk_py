from __future__ import annotations

import asyncio
import os
import time
import types
import unittest

_LIVE = os.environ.get("EVALKIT_LIVE") == "1"
_KEY = os.environ.get("EVALKIT_KEY", "tk_live_7f90ff7e2f89e2dccd4964ceb1c4de75acf32193f5bc8fd0")
_BASE = os.environ.get("EVALKIT_BASE_URL", "https://api.syntropylabs.ai")

skip_unless_live = unittest.skipUnless(_LIVE, "set EVALKIT_LIVE=1 to run live tests")


def _live_client():
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import evalkit
    return evalkit.init(
        subscription_key=_KEY,
        base_url=_BASE,
        service_name="evalkit-test-suite",
        app_version="0.1.19",
    )


@skip_unless_live
class TestLiveIngest(unittest.TestCase):
    def setUp(self):
        self.client = _live_client()

    def test_minimal_span_reaches_api(self):
        tid, end, ctx = self.client.start_trace("live-smoke-test")
        end("OK")
        ok = self.client.flush()
        self.assertTrue(ok, "flush must return True on successful ingest")

    def test_http_call_span(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import evalkit
        tid, end, ctx = self.client.start_http_trace(
            "GET /healthz",
            method="GET",
            url=f"{_BASE}/healthz",
        )
        end("OK", status_code=200)
        ok = self.client.flush()
        self.assertTrue(ok)

    def test_llm_call_span_via_fake(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from tests.helpers import install_fake_openai_sync, OResp, reset_context
        from evalkit.core.context import _ambient_trace_id, _current_span

        _ambient_trace_id.set(None)
        _current_span.set(None)

        fc = install_fake_openai_sync(self.client, OResp("live test completion", pin=10, pout=15))
        fc.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hello from live test"}])
        ok = self.client.flush()
        self.assertTrue(ok)

    def test_anthropic_span_with_tool_event(self):
        from tests.helpers import install_fake_anthropic, AResp, reset_context
        from evalkit.core.context import _ambient_trace_id, _current_span

        _ambient_trace_id.set(None)
        _current_span.set(None)

        m = install_fake_anthropic(
            self.client,
            AResp(text="I'll look that up", tool=("search_knowledge_base", {"query": "refund policy"})),
        )
        m.create(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "What is the refund policy?"}],
        )
        ok = self.client.flush()
        self.assertTrue(ok)

    def test_litellm_span_with_cost(self):
        import sys
        sys.modules.pop("litellm", None)
        litellm_mod = types.ModuleType("litellm")
        litellm_mod.get_llm_provider = lambda m: (m, "openai", None, None)
        litellm_mod.completion_cost = lambda completion_response=None: 0.0

        from tests.helpers import LLMResult

        def fake_completion(model="gpt-4o", messages=None, **k):
            return LLMResult(content="live litellm response", cost=0.00123)

        litellm_mod.completion = fake_completion
        sys.modules["litellm"] = litellm_mod

        from evalkit.instrumentation.llm.litellm import patch_litellm
        from evalkit.core.context import _ambient_trace_id, _current_span
        _ambient_trace_id.set(None)
        _current_span.set(None)

        patch_litellm(self.client)
        litellm_mod.completion(model="gemini/gemini-2.0-flash", messages=[{"role": "user", "content": "live cost test"}])
        ok = self.client.flush()
        self.assertTrue(ok)
        sys.modules.pop("litellm", None)

    def test_evaluate_result_span(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import evalkit

        from evalkit.core.context import _ambient_trace_id, _current_span
        _ambient_trace_id.set(None)
        _current_span.set(None)

        evalkit.evaluate(
            output="The refund was processed via the issue_refund tool.",
            expected_tools=["issue_refund"],
            tool_calls=[{"name": "issue_refund"}],
            constraints={"required_terms": ["refund"]},
        )
        ok = self.client.flush()
        self.assertTrue(ok)

    def test_simulate_user_tagging(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import evalkit
        from tests.helpers import install_fake_anthropic, AResp, reset_context
        from evalkit.core.context import _ambient_trace_id, _current_span

        _ambient_trace_id.set(None)
        _current_span.set(None)

        fresh_client = _live_client()
        captured = []
        fresh_client._exporter.export = lambda batch: (captured.extend(batch), True)[1]

        m = install_fake_anthropic(fresh_client, AResp(text="Order found", tool=("lookup_order", {"id": "ORD-1"})))

        def agent(ctx):
            m.create(model="claude-3-5-sonnet", messages=[{"role": "user", "content": ctx.message}])
            return "ok"

        report = evalkit.simulate_user(
            agent,
            [{"name": "order_lookup", "starting_prompt": "Where is ORD-1?", "expected_tools": ["lookup_order"]}],
        )
        self.assertEqual(report["results"][0]["status"], "OK")
        self.assertIn("simulation_id", report)
        ok = fresh_client.flush()
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
