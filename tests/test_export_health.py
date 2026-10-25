"""A receiver that is down, slow or refusing must cost the application nothing: bounded memory, no
warning per failed request, no export thread retrying into it, and a flush that never raises."""
import logging
import time
import unittest
from unittest import mock

import evalkit
from evalkit.core import health as health_mod
from evalkit.core.client import EvalKitClient
from evalkit.core.health import ExportHealth
from evalkit.core.types import EvalKitOptions, TraceEnvelope


def _envelope(i, span_type="function_call", payload=0):
    attrs = {"blob": "x" * payload} if payload else None
    return TraceEnvelope(
        trace_id=f"{i:032x}", span_id=f"{i:016x}", created_at="2026-10-26T00:00:00Z",
        span_type=span_type, operation=f"op-{i}", attributes=attrs,
    )


class _Client:
    """An EvalKitClient whose export thread never wakes, with exports answered by `answer`."""

    def __init__(self, test, answer=lambda batch: True, **opts):
        self.client = EvalKitClient(EvalKitOptions(
            subscription_key="tk_test", base_url="http://127.0.0.1:1", scheduled_delay_ms=3_600_000, **opts,
        ))
        test.addCleanup(self.client._stop_event.set)
        self.batches = []

        def export(batch):
            self.batches.append(list(batch))
            return answer(batch)

        self.client._exporter.export = export


class QueueBudgetTest(unittest.TestCase):
    def test_over_the_byte_budget_the_oldest_spans_that_are_not_model_calls_go_first(self):
        holder = _Client(self, max_queue_bytes=60_000)
        client = holder.client
        client.enqueue(_envelope(0, "llm_call", payload=20_000))
        for i in range(1, 4):
            client.enqueue(_envelope(i, "function_call", payload=20_000))
        queued = [e.span_type for e in client._queue]
        self.assertIn("llm_call", queued, "the model call outlives the function spans queued after it")
        self.assertLessEqual(client._queue_bytes, 60_000)
        self.assertGreaterEqual(client.stats()["dropped_bytes_budget"], 1)

    def test_a_model_call_takes_the_place_of_queued_spans_that_are_not(self):
        holder = _Client(self, max_queue_bytes=50_000)
        client = holder.client
        for i in range(2):
            client.enqueue(_envelope(i, "log", payload=20_000))
        client.enqueue(_envelope(9, "llm_call", payload=20_000))
        self.assertEqual([e.span_id for e in client._queue if e.span_type == "llm_call"], [f"{9:016x}"])

    def test_a_full_queue_counts_what_it_drops(self):
        holder = _Client(self, max_queue_size=5)
        for i in range(8):
            holder.client.enqueue(_envelope(i))
        stats = holder.client.stats()
        self.assertEqual(stats["queued"], 5)
        self.assertEqual(stats["dropped_queue_full"], 3)

    def test_the_default_byte_budget_is_32_mib(self):
        self.assertEqual(EvalKitOptions(subscription_key="k").max_queue_bytes, 32 * 1024 * 1024)


class StatsTest(unittest.TestCase):
    def test_stats_has_every_documented_key(self):
        holder = _Client(self)
        stats = holder.client.stats()
        for key in ("queued", "dropped_queue_full", "dropped_bytes_budget", "dropped_export_failed",
                    "export_failures", "last_error", "circuit_state"):
            self.assertIn(key, stats)
        self.assertEqual(stats["circuit_state"], "closed")

    def test_module_stats_never_raises(self):
        stats = evalkit.stats()
        self.assertIn("circuit_state", stats)
        with mock.patch.object(evalkit, "_instance", None):
            self.assertEqual(evalkit.stats()["queued"], 0)


class CircuitBreakerTest(unittest.TestCase):
    def setUp(self):
        self.now = [1000.0]
        clock = lambda: self.now[0]  # noqa: E731
        self.holder = _Client(self, answer=lambda batch: False)
        client = self.holder.client
        client._health = ExportHealth(clock=clock)
        client._exporter._halted = True
        # A real exporter reports a halted export; this one is told to.
        original = client._exporter.export

        def export(batch):
            result = original(batch)
            client._exporter._halted = not result
            return result

        client._exporter.export = export
        self.client = client

    def _cycle(self, spans=3):
        for i in range(spans):
            self.client.enqueue(_envelope(i))
        return self.client._drain()

    def test_three_failed_cycles_open_the_circuit_and_spans_are_then_only_counted(self):
        for _ in range(3):
            self.assertFalse(self._cycle())
        self.assertEqual(self.client.stats()["circuit_state"], "open")
        sent = len(self.holder.batches)
        self._cycle()
        self.assertEqual(len(self.holder.batches), sent, "nothing is sent while the circuit is open")
        self.assertEqual(self.client.stats()["queued"], 0)
        self.assertEqual(self.client.stats()["dropped_circuit_open"], 3)

    def test_after_the_cooldown_one_small_batch_probes_and_success_closes_it(self):
        for _ in range(3):
            self._cycle()
        self.now[0] += health_mod.BASE_COOLDOWN_S + 1
        answers = iter([True, True, True])
        original_batches = self.holder.batches

        def export(batch):
            original_batches.append(list(batch))
            ok = next(answers)
            self.client._exporter._halted = not ok
            return ok

        self.client._exporter.export = export
        for i in range(40):
            self.client.enqueue(_envelope(i))
        self.assertEqual(self.client.stats()["circuit_state"], "half_open")
        self.assertTrue(self.client._drain())
        self.assertEqual(len(original_batches[-2]), health_mod.PROBE_SPANS, "the probe is one small batch")
        self.assertEqual(self.client.stats()["circuit_state"], "closed")

    def test_a_failed_probe_opens_it_again_for_twice_as_long_up_to_five_minutes(self):
        for _ in range(3):
            self._cycle()
        circuit = self.client._health.circuit
        self.assertEqual(circuit.cooldown, 30.0)
        expected = [60.0, 120.0, 240.0, 300.0, 300.0]
        for want in expected:
            self.now[0] = circuit.open_until + 1
            self._cycle()
            self.assertEqual(circuit.state, "open")
            self.assertEqual(circuit.cooldown, want)

    def test_a_refused_key_disables_exporting_and_says_so(self):
        holder = _Client(self)
        exporter = holder.client._exporter
        exporter._disabled = True
        holder.client.enqueue(_envelope(1))
        self.assertEqual(holder.client.stats()["circuit_state"], "disabled")
        self.assertEqual(holder.client.stats()["queued"], 0)


class WarningThrottleTest(unittest.TestCase):
    def test_one_warning_when_exporting_starts_failing_then_one_summary_per_interval(self):
        now = [0.0]
        health = ExportHealth(clock=lambda: now[0])
        with self.assertLogs("evalkit", level="WARNING") as logs:
            for _ in range(50):
                health.export_failed("HTTP 503 after retrying")
                health.count("dropped_export_failed", 10)
            now[0] += 301
            health.export_failed("HTTP 503 after retrying")
        self.assertEqual(len(logs.output), 2, logs.output)
        self.assertIn("failed: HTTP 503", logs.output[0])
        self.assertIn("still failing", logs.output[1])
        self.assertIn("50 failed export(s)", logs.output[1])
        self.assertIn("500 dropped span(s)", logs.output[1])
        self.assertEqual(health.export_failures, 51)

    def test_logged_response_bodies_are_cut_to_200_characters(self):
        self.assertEqual(len(health_mod.clip("y" * 5000)), 201)

    def test_a_rejected_export_logs_at_most_200_characters_of_the_body(self):
        from evalkit.core.exporter import _Exporter

        exporter = _Exporter(EvalKitOptions(subscription_key="k", base_url="http://127.0.0.1:1", export_gzip=False))
        exporter._post = lambda payload, **kw: (400, "E" * 10_000, None)
        with self.assertLogs("evalkit", level="WARNING") as logs:
            exporter.export([_envelope(1)])
        self.assertLess(len(logs.output[0]), 800)
        self.assertIn("E" * 200, logs.output[0])
        self.assertNotIn("E" * 201, logs.output[0])

    def test_the_first_refused_key_is_logged_clearly(self):
        from evalkit.core.exporter import _Exporter

        exporter = _Exporter(EvalKitOptions(subscription_key="k", base_url="http://127.0.0.1:1"))
        exporter._post = lambda payload, **kw: (401, '{"error":"bad key"}', None)
        with self.assertLogs("evalkit", level="WARNING") as logs:
            exporter.export([_envelope(1)])
        self.assertIn("refused the subscription key (HTTP 401", logs.output[0])
        self.assertIn("after 3 refusals", logs.output[0])


class FlushTest(unittest.TestCase):
    def test_flush_with_a_base_url_that_is_not_a_url_never_raises(self):
        client = EvalKitClient(EvalKitOptions(subscription_key="k", base_url="not a url", scheduled_delay_ms=3_600_000))
        self.addCleanup(client._stop_event.set)
        client.enqueue(_envelope(1))
        self.assertFalse(client.flush())
        self.assertIn("ValueError", client.stats()["last_error"])

    def test_flush_returns_within_its_timeout_behind_a_receiver_that_never_answers(self):
        import socket

        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(16)
        self.addCleanup(server.close)
        port = server.getsockname()[1]
        client = EvalKitClient(EvalKitOptions(
            subscription_key="k", base_url=f"http://127.0.0.1:{port}", scheduled_delay_ms=3_600_000,
        ))
        self.addCleanup(client._stop_event.set)
        client.enqueue(_envelope(1))
        start = time.monotonic()
        with self.assertLogs("evalkit", level="WARNING"):
            self.assertFalse(client.flush(timeout=0.5))
        self.assertLess(time.monotonic() - start, 3.0)

    def test_module_flush_takes_a_timeout(self):
        import inspect

        self.assertEqual(inspect.signature(evalkit.flush).parameters["timeout"].default, 5.0)
        self.assertEqual(inspect.signature(EvalKitClient.flush).parameters["timeout"].default, 5.0)


if __name__ == "__main__":
    logging.basicConfig()
    unittest.main(verbosity=2)
