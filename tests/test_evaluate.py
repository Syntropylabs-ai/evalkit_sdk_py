import unittest

import evalkit
from tests.helpers import fresh, reset_context


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_all_tools_correct(self):
        scores = evalkit.evaluate(
            output="The answer covers AWS and Azure.",
            expected_tools=["search", "render"],
            tool_calls=[{"name": "search"}, {"name": "render"}],
            constraints={"required_terms": ["AWS", "Azure"]},
        )
        self.assertEqual(scores["tool_f1"], 1.0)
        self.assertEqual(scores["response_match"], 1.0)

    def test_eval_result_span_pushed(self):
        evalkit.evaluate(output="x", expected_tools=["a"], tool_calls=[{"name": "a"}])
        self.client.flush()
        evs = [e for e in self.captured if e.span_type == "eval_result"]
        self.assertTrue(evs)

    def test_partial_tool_overlap(self):
        scores = evalkit.evaluate(
            output="result",
            expected_tools=["search", "summarize", "cite"],
            tool_calls=[{"name": "search"}],
        )
        self.assertLess(scores["tool_f1"], 1.0)
        self.assertGreater(scores["tool_f1"], 0.0)
        self.assertLess(scores.get("tool_correctness", 0.0), 1.0)

    def test_no_expected_tools_no_trajectory_score(self):
        scores = evalkit.evaluate(
            output="some result",
            tool_calls=[{"name": "tool_a"}],
        )
        self.assertNotIn("tool_trajectory", scores)
        self.assertNotIn("tool_correctness", scores)
        self.assertEqual(scores.get("tool_f1", 0.0), 0.0)

    def test_contextual_precision_and_recall(self):
        scores = evalkit.evaluate(
            output="result",
            retrieved_context=["A", "B", "C"],
            expected_context=["A", "B", "D"],
        )
        self.assertIn("contextual_precision", scores)
        self.assertIn("contextual_recall", scores)
        self.assertAlmostEqual(scores["contextual_precision"], 2 / 3, places=5)
        self.assertAlmostEqual(scores["contextual_recall"], 2 / 3, places=5)

    def test_constraint_compliance_citations_present(self):
        scores = evalkit.evaluate(
            output="See https://example.com for details.",
            constraints={"has_citations": True},
        )
        self.assertIn("constraint_compliance", scores)
        self.assertEqual(scores["constraint_compliance"], 1.0)

    def test_constraint_compliance_no_citations_expected(self):
        scores = evalkit.evaluate(
            output="Simple answer with no links.",
            constraints={"has_citations": False},
        )
        self.assertEqual(scores["constraint_compliance"], 1.0)

    def test_requested_metrics_subset(self):
        scores = evalkit.evaluate(
            output="AWS and Azure are covered.",
            expected_tools=["search"],
            tool_calls=[{"name": "search"}],
            constraints={"required_terms": ["AWS", "Azure"]},
            metrics=["tool_f1"],
        )
        self.assertIn("tool_f1", scores)
        self.assertNotIn("response_match", scores)
        self.assertNotIn("tool_trajectory", scores)

    def test_response_match_case_insensitive(self):
        scores = evalkit.evaluate(
            output="The quick brown fox",
            constraints={"required_terms": ["QUICK", "fox"]},
        )
        self.assertEqual(scores["response_match"], 1.0)

    def test_rag_perfect_overlap(self):
        scores = evalkit.evaluate(
            output="result",
            retrieved_context=["A", "B"],
            expected_context=["A", "B"],
        )
        self.assertEqual(scores["contextual_precision"], 1.0)
        self.assertEqual(scores["contextual_recall"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
