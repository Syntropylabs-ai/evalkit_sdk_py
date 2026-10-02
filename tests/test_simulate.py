import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import evalkit
from evalkit.core.client import MAX_STATUS_MESSAGE
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
        self.assertEqual(len({e.trace_id for e in scenario_roots}), 1)
        self.assertEqual(
            sorted(e.operation for e in scenario_roots),
            ["scenario:s1", "scenario:s2"],
            "each scenario must produce exactly one root span; opening an OpenTelemetry span alongside "
            "client.start_trace() emits the root twice, and ingest is append-only so both rows stay",
        )
        self.assertTrue(
            all(e.parent_span_id is None for e in scenario_roots),
            "a scenario root whose parent_span_id points at a span that was never exported leaves the "
            "waterfall dangling",
        )
        roots_by_id = {e.span_id for e in scenario_roots}
        children = [e for e in self.captured if e.parent_span_id is not None]
        self.assertTrue(children)
        self.assertTrue(
            all(e.parent_span_id in roots_by_id for e in children),
            "spans emitted inside a scenario must be parented to that scenario's root",
        )

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


class TestScenarioRootSpan(unittest.TestCase):
    """The scenario's root span is the row the platform lists for a scenario
    execution, so it has to say on its own what the scenario asked, what the
    agent answered last and why it failed, under the agent and in the session
    of that one execution."""

    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def _roots(self):
        return [e for e in self.captured if (e.operation or "").startswith("scenario:")]

    def test_root_carries_the_opening_and_the_last_reply_under_the_agent_and_its_session(self):
        m = install_fake_genai(self.client, response=GenAIResp(text="ok"))
        replies = iter(["It shipped yesterday.", "The refund is on its way."])

        def agent(ctx):
            m.generate_content(model="g", contents=ctx.message)
            return next(replies)

        report = evalkit.simulate_user(
            agent,
            [{"name": "refund", "scenario_id": "scn_refund", "turns": ["Where is my order?", "And my refund?"]}],
            agent=" Support bot ",
        )

        [root] = self._roots()
        self.assertEqual(root.prompt, "Where is my order?")
        self.assertEqual(root.completion, "The refund is on its way.")
        spans = [e for e in self.captured if e.trace_id == root.trace_id]
        self.assertTrue(any(e.span_type == "llm_call" for e in spans))
        session = f"{report['simulation_id']}:{report['run_id']}:scn_refund"
        self.assertEqual({e.session_id for e in spans}, {session})
        self.assertEqual({e.agent_name for e in spans}, {"Support bot"})
        for span in spans:
            self.assertEqual(span.attributes.get("gen_ai.agent.name"), "Support bot", span.operation)
            self.assertEqual(span.attributes.get("evalkit.agent_name"), "Support bot", span.operation)

    def test_each_scenario_is_its_own_session(self):
        report = evalkit.simulate_user(
            lambda ctx: "ok",
            [{"name": "a", "scenario_id": "scn_a", "starting_prompt": "hi"},
             {"name": "b", "scenario_id": "scn_b", "starting_prompt": "hello"}],
        )
        prefix = f"{report['simulation_id']}:{report['run_id']}:"
        self.assertEqual(sorted(e.session_id for e in self._roots()), [prefix + "scn_a", prefix + "scn_b"])

    def test_a_failed_scenario_says_why_on_its_root(self):
        def agent(ctx):
            raise TimeoutError("model call timed out after 30s")

        report = evalkit.simulate_user(agent, [{"name": "x", "starting_prompt": "hi"}])

        [root] = self._roots()
        self.assertEqual(report["results"][0]["status"], "ERROR")
        self.assertEqual(root.status, "ERROR")
        self.assertEqual(root.status_message, "model call timed out after 30s")
        self.assertEqual(root.attributes["error.type"], "TimeoutError")
        self.assertEqual(root.attributes["error.message"], "model call timed out after 30s")
        self.assertEqual(root.prompt, "hi")
        self.assertIsNone(root.completion)

    def test_an_error_without_a_message_is_named_by_its_type_and_a_long_one_is_bounded(self):
        def silent(ctx):
            raise RuntimeError()

        def verbose(ctx):
            raise ValueError("y" * (MAX_STATUS_MESSAGE * 2))

        evalkit.simulate_user(silent, [{"name": "silent", "starting_prompt": "hi"}])
        evalkit.simulate_user(verbose, [{"name": "verbose", "starting_prompt": "hi"}])

        roots = {e.operation: e for e in self._roots()}
        self.assertEqual(roots["scenario:silent"].status_message, "RuntimeError")
        self.assertEqual(roots["scenario:verbose"].status_message, "y" * MAX_STATUS_MESSAGE)
        self.assertEqual(roots["scenario:verbose"].attributes["error.message"], "y" * MAX_STATUS_MESSAGE)

    def test_the_async_path_ends_its_root_the_same_way(self):
        async def answers(ctx):
            return f"echo: {ctx.message}"

        async def fails(ctx):
            raise ConnectionError("upstream refused")

        evalkit.simulate_user(answers, [{"name": "ok", "turns": ["one", "two"]}], agent="Async bot")
        evalkit.simulate_user(fails, [{"name": "bad", "starting_prompt": "hi"}], agent="Async bot")

        roots = {e.operation: e for e in self._roots()}
        self.assertEqual((roots["scenario:ok"].prompt, roots["scenario:ok"].completion), ("one", "echo: two"))
        self.assertEqual(roots["scenario:bad"].status_message, "upstream refused")
        self.assertEqual(roots["scenario:bad"].attributes["error.type"], "ConnectionError")
        self.assertEqual({e.agent_name for e in roots.values()}, {"Async bot"})

    def test_with_content_capture_off_the_conversation_stays_off_the_root(self):
        saved = self.client._opts.capture_content
        self.client._opts.capture_content = False
        try:
            def agent(ctx):
                raise RuntimeError("card declined")

            evalkit.simulate_user(agent, [{"name": "s", "starting_prompt": "my card is 4242 4242 4242 4242"}])
        finally:
            self.client._opts.capture_content = saved

        [root] = self._roots()
        self.assertIsNone(root.prompt)
        self.assertIsNone(root.completion)
        self.assertEqual(root.status_message, "card declined")


if __name__ == "__main__":
    unittest.main(verbosity=2)
