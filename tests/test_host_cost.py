"""What tracing costs the application: bounded work per traced call and per log record, and bounded
memory for state kept between the start and the end of an operation."""
import json
import logging
import unittest

from evalkit.core.bounded import BoundedTTLMap, RateLimiter
from evalkit.core.client import _safe_io
from evalkit.core.serialize import bounded_json
from tests.helpers import fresh, reset_context


class _CountingList(list):
    """A list that counts how many of its items were read."""

    def __init__(self, items):
        super().__init__(items)
        self.read = 0

    def __iter__(self):
        for item in super().__iter__():
            self.read += 1
            yield item


class BoundedSerializerTest(unittest.TestCase):
    def test_a_value_that_fits_is_written_as_json_dumps_writes_it(self):
        for value in (
            [1, 2.5, "x", None, True, False],
            {"a": {"b": [1, {"c": "d"}]}, "e": "é ünïcode"},
            {3: "int key", None: "null key", True: "bool key", 2.5: "float key"},
            ("tuple", 1),
            "plain",
            float("inf"),
        ):
            with self.subTest(value=value):
                self.assertEqual(bounded_json(value, 10_000, scrub=False), json.dumps(value))

    def test_a_large_value_is_cut_without_reading_all_of_it(self):
        big = _CountingList(range(1_000_000))
        out = bounded_json(big, 4096, scrub=False)
        self.assertEqual(len(out), 4097)
        self.assertTrue(out.endswith("…"))
        self.assertTrue(json.dumps(list(range(2000))).startswith(out[:-1]))
        self.assertLess(big.read, 5000, "the writer stops at the cap instead of serializing a million items")

    def test_secrets_are_masked_as_the_value_is_walked(self):
        out = bounded_json({"api_key": "plain-value", "nested": [{"password": "p"}], "note": "sk-" + "a" * 30}, 4096)
        self.assertNotIn("plain-value", out)
        self.assertNotIn('"p"', out)
        self.assertNotIn("a" * 30, out)
        self.assertIn('"api_key": "***"', out)

    def test_cycles_depth_bytes_and_unreadable_values_never_raise(self):
        cyclic = []
        cyclic.append(cyclic)
        self.assertIn("[Circular]", bounded_json(cyclic, 100))
        deep = current = []
        for _ in range(500):
            nxt = []
            current.append(nxt)
            current = nxt
        self.assertIn("[Object]", bounded_json(deep, 10_000))
        self.assertTrue(bounded_json(b"\x00" * 10_000_000, 64).startswith('"b\''))

        class Broken:
            def __str__(self):
                raise RuntimeError("no str")

        self.assertIn("[Unserializable]", bounded_json({"x": Broken()}, 100))

    def test_function_io_keeps_its_4096_character_shape(self):
        out = _safe_io(["x" * 10_000])
        self.assertEqual(len(out), 4097)
        self.assertTrue(out.startswith('["xxx'))


class FunctionRateLimitTest(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_a_function_in_a_hot_loop_records_about_its_rate_per_second(self):
        @self.client.trace_function("hot")
        def hot(i):
            return i

        for i in range(1000):
            self.assertEqual(hot(i), i)
        self.client.flush()
        recorded = [e for e in self.captured if e.operation == "hot"]
        self.assertLessEqual(len(recorded), 110)
        self.assertGreaterEqual(len(recorded), 90)
        self.assertGreaterEqual(self.client.stats()["dropped_rate_limited"], 890)

    def test_tool_calls_are_never_left_out_by_rate(self):
        @self.client.trace_tool("lookup")
        def lookup(i):
            return i

        for i in range(300):
            lookup(i)
        self.client.flush()
        self.assertEqual(len([e for e in self.captured if e.operation == "lookup"]), 300)

    def test_the_rate_limiter(self):
        now = [0.0]
        limiter = RateLimiter(10, clock=lambda: now[0])
        self.assertEqual(sum(limiter.allow() for _ in range(50)), 10)
        now[0] += 0.5
        self.assertEqual(sum(limiter.allow() for _ in range(50)), 5)
        self.assertTrue(all(RateLimiter(0).allow() for _ in range(1000)), "0 is no limit")


class LogCaptureLimitsTest(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        from evalkit.instrumentation.runtime.logging import install_log_capture

        install_log_capture(self.client)

    def _logs(self, text):
        self.client.flush()
        return [e for e in self.captured if e.span_type == "log" and text in (e.attributes or {}).get("log.message", "")]

    def test_only_warnings_and_up_become_log_spans_by_default(self):
        logger = logging.getLogger("app.levels")
        logger.setLevel(logging.DEBUG)
        logger.info("an info line")
        logger.warning("a warning line")
        self.assertEqual(self._logs("an info line"), [])
        self.assertEqual(len(self._logs("a warning line")), 1)
        self.assertEqual(self.client._opts.log_level, logging.WARNING)

    def test_the_sdks_own_loggers_are_never_recorded(self):
        logging.getLogger("evalkit").warning("sdk internal line")
        logging.getLogger("evalkit.core.exporter").warning("sdk internal line too")
        self.assertEqual(self._logs("sdk internal line"), [])

    def test_a_logging_storm_is_rate_limited(self):
        logger = logging.getLogger("app.storm")
        for i in range(500):
            logger.warning("storm line %d", i)
        recorded = self._logs("storm line")
        self.assertLessEqual(len(recorded), 60)
        self.assertGreaterEqual(len(recorded), 40)

    def test_a_huge_message_is_cut_before_it_is_queued(self):
        logging.getLogger("app.huge").warning("HEAD" + "m" * 100_000 + "TAIL")
        [span] = self._logs("HEAD")
        message = span.attributes["log.message"]
        self.assertLessEqual(len(message.encode()), 8 * 1024)
        self.assertTrue(message.endswith("TAIL"))
        self.assertEqual(span.attributes["evalkit.truncated.log.message"], 100_008)


class BoundedMapTest(unittest.TestCase):
    def test_the_map_is_capped_oldest_first(self):
        m = BoundedTTLMap(max_entries=3)
        for i in range(5):
            m[i] = i
        self.assertEqual(sorted(m), [2, 3, 4])

    def test_entries_expire(self):
        now = [0.0]
        m = BoundedTTLMap(ttl_s=10, clock=lambda: now[0])
        m["a"] = 1
        now[0] = 5
        m["b"] = 2
        self.assertEqual(m.get("a"), 1)
        now[0] = 11
        self.assertIsNone(m.get("a"))
        self.assertEqual(m.pop("b"), 2)
        self.assertEqual(len(m), 0)

    def test_the_langchain_handler_and_adk_timers_are_bounded(self):
        from evalkit.instrumentation.llm import google_adk

        self.assertIsInstance(google_adk._tool_call_starts, BoundedTTLMap)
        try:
            from evalkit.instrumentation.llm.langchain import _build_handler
        except Exception:
            self.skipTest("langchain not importable")
        reset_context()
        handler_cls = _build_handler(fresh([]))
        if handler_cls is None:
            self.skipTest("langchain_core not installed")
        handler = handler_cls() if isinstance(handler_cls, type) else handler_cls
        runs = getattr(handler, "_runs", None)
        self.assertIsInstance(runs, BoundedTTLMap)
        self.assertEqual(runs._max, 10_000)
        self.assertEqual(runs._ttl, 3600.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
