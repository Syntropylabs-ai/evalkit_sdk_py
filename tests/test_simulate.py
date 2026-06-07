import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import evalkit
from tests.helpers import GenAIResp, fresh, reset_context, install_fake_genai


class TestSimulateUser(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_sync_tool_capture_and_tagging(self):
        m = install_fake_genai(self.client, response=GenAIResp(tool=("get_weather", {"city": "london"})))

        def agent(ctx):
            m.generate_content(model="g", contents=ctx.message)
            return "done"

        scenarios = [
            {"name": "s1", "starting_prompt": "weather in london?", "expected_tools": ["get_weather"]},
            {"name": "s2", "starting_prompt": "and tokyo?", "expected_tools": ["get_weather"]},
        ]
        report = evalkit.simulate_user(agent, scenarios)
        self.assertEqual(len(report["results"]), 2)
        for r in report["results"]:
            self.assertEqual(r["status"], "OK")
            self.assertEqual(r["scores"].get("tool_correctness"), 1.0)
        sim_spans = [e for e in self.captured if e.is_simulation]
        self.assertTrue(sim_spans)
        self.assertTrue(all(e.simulation_id == report["simulation_id"] for e in sim_spans))
        scenario_roots = [e for e in self.captured if (e.operation or "").startswith("scenario:")]
        # All scenarios in the same simulate_user() call share one ambient trace.
        self.assertEqual(len({e.trace_id for e in scenario_roots}), 1)

    def test_async_entrypoint(self):
        m = install_fake_genai(self.client, response=GenAIResp(text="ok"))

        async def agent(ctx):
            m.generate_content(model="g", contents=ctx.message)
            return "async-done"

        scenarios = [{"name": "a1", "starting_prompt": "hi"}]
        report = evalkit.simulate_user(agent, scenarios)
        self.assertEqual(report["results"][0]["status"], "OK")
        self.assertEqual(report["results"][0]["turns"], 1)

    def test_thread_spawned_by_entrypoint_is_tagged(self):
        m = install_fake_genai(self.client, response=GenAIResp(tool=("get_weather", {"city": "x"})))

        def agent(ctx):
            def work():
                m.generate_content(model="g", contents=ctx.message)
            t = threading.Thread(target=work)
            t.start()
            t.join()
            return "ok"

        report = evalkit.simulate_user(
            agent, [{"name": "s", "starting_prompt": "hi", "expected_tools": ["get_weather"]}]
        )
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertTrue(llm)
        self.assertTrue(all(e.is_simulation for e in llm))
        root = [e for e in self.captured if (e.operation or "").startswith("scenario:")][0]
        self.assertTrue(all(e.trace_id == root.trace_id for e in llm))
        self.assertEqual(report["results"][0]["scores"].get("tool_correctness"), 1.0)

    def test_threadpool_spawned_by_entrypoint_is_tagged(self):
        m = install_fake_genai(self.client, response=GenAIResp(text="ok"))
        pool = ThreadPoolExecutor(max_workers=2)

        def agent(ctx):
            futs = [pool.submit(m.generate_content, model="g", contents=ctx.message) for _ in range(3)]
            for f in futs:
                f.result()
            return "ok"

        report = evalkit.simulate_user(agent, [{"name": "s", "starting_prompt": "hi"}])
        pool.shutdown()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 3)
        self.assertTrue(all(e.is_simulation for e in llm))
        root = [e for e in self.captured if (e.operation or "").startswith("scenario:")][0]
        self.assertEqual(len({e.trace_id for e in llm}), 1)
        self.assertTrue(all(e.trace_id == root.trace_id for e in llm))

    def test_entrypoint_exception_is_isolated(self):
        def agent(ctx):
            raise RuntimeError("boom")

        report = evalkit.simulate_user(agent, [{"name": "x", "starting_prompt": "hi"}])
        self.assertEqual(report["results"][0]["status"], "ERROR")
        self.assertIn("boom", report["results"][0]["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
