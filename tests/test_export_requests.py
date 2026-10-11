import gzip
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from evalkit.core.client import EvalKitClient
from evalkit.core.exporter import MAX_REQUEST_BYTES, _Exporter, resolve_export_gzip
from evalkit.core.types import EvalKitOptions, TraceEnvelope


class _Receiver:
    """A stand-in for /v1/ingest that records every request and answers with `respond(request)`."""

    def __init__(self, respond=None):
        self.requests = []
        self.respond = respond or (lambda request: 202)
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                encoding = self.headers.get("Content-Encoding")
                request = {"encoding": encoding, "sent_bytes": len(raw), "events": None, "json_bytes": None}
                try:
                    body = gzip.decompress(raw) if encoding == "gzip" else raw
                    request["events"] = json.loads(body)["events"]
                    request["json_bytes"] = len(body)
                except Exception:
                    pass
                status = receiver.respond(request)
                request["status"] = status
                receiver.requests.append(request)
                if status < 300:
                    payload = {"accepted": len(request["events"] or []), "status": "queued"}
                else:
                    payload = {"error": "refused"}
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def close(self):
        self._server.shutdown()
        self._server.server_close()

    def accepted_ids(self):
        return [e["spanId"] for r in self.requests if r["status"] < 300 for e in r["events"]]


def _envelope(i, payload=0):
    return TraceEnvelope(
        trace_id="a" * 32,
        span_id=f"{i:016x}",
        created_at="2026-10-09T00:00:00.000Z",
        span_type="function_call",
        operation=f"op-{i}",
        attributes={"test.payload": "x" * payload} if payload else None,
    )


def _ids(envelopes):
    return [e.span_id for e in envelopes]


def _exporter(url, **opts):
    return _Exporter(EvalKitOptions(subscription_key="tk_test", base_url=url, **opts))


class _ReceiverTest(unittest.TestCase):
    def setUp(self):
        self.receiver = _Receiver()

    def tearDown(self):
        self.receiver.close()


class TestCompression(_ReceiverTest):
    def test_requests_are_gzipped_by_default(self):
        batch = [_envelope(i) for i in range(3)]
        self.assertTrue(_exporter(self.receiver.url).export(batch))
        [request] = self.receiver.requests
        self.assertEqual(request["encoding"], "gzip", "compression is on unless it is turned off")
        self.assertEqual(self.receiver.accepted_ids(), _ids(batch), "the body inflates to the same envelope")
        self.assertLess(request["sent_bytes"], request["json_bytes"])

    def test_export_gzip_false_sends_plain_json(self):
        batch = [_envelope(0)]
        self.assertTrue(_exporter(self.receiver.url, export_gzip=False).export(batch))
        [request] = self.receiver.requests
        self.assertIsNone(request["encoding"])
        self.assertEqual(self.receiver.accepted_ids(), _ids(batch))

    def test_the_environment_variable_turns_compression_off_and_an_argument_wins(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("EVALKIT_EXPORT_GZIP", None)
            self.assertTrue(resolve_export_gzip(None))
        with mock.patch.dict(os.environ, {"EVALKIT_EXPORT_GZIP": "false"}):
            self.assertFalse(resolve_export_gzip(None))
            self.assertTrue(resolve_export_gzip(True))
        self.assertTrue(EvalKitOptions(subscription_key="k").export_gzip)


class TestReceiverWithoutGzip(_ReceiverTest):
    def _old_receiver(self, gzip_status):
        self.receiver.respond = lambda r: gzip_status if r["encoding"] == "gzip" else 202

    def _falls_back_once(self, gzip_status):
        self._old_receiver(gzip_status)
        exporter = _exporter(self.receiver.url)
        with self.assertLogs("evalkit", level="WARNING") as logs:
            self.assertTrue(exporter.export([_envelope(0)]), "the uncompressed resend is what the receiver keeps")
            self.assertTrue(exporter.export([_envelope(1)]))
        self.assertEqual(
            [(r["encoding"], r["status"]) for r in self.receiver.requests],
            [("gzip", gzip_status), (None, 202), (None, 202)],
            "after one refusal of a compressed body the exporter stops compressing for good",
        )
        self.assertEqual(self.receiver.accepted_ids(), _ids([_envelope(0), _envelope(1)]))
        notices = [m for m in logs.output if "does not accept gzip" in m]
        self.assertEqual(len(notices), 1, logs.output)
        self.assertIn(str(gzip_status), notices[0])

    def test_a_415_switches_to_uncompressed(self):
        self._falls_back_once(415)

    def test_a_400_from_a_receiver_that_parses_the_gzip_bytes_as_json_switches_to_uncompressed(self):
        self._falls_back_once(400)

    def test_a_400_for_the_spans_themselves_keeps_compression(self):
        self.receiver.respond = lambda r: 400
        exporter = _exporter(self.receiver.url)
        with self.assertLogs("evalkit", level="WARNING") as logs:
            self.assertFalse(exporter.export([_envelope(0)]))
            self.assertFalse(exporter.export([_envelope(1)]))
        self.assertEqual(
            [r["encoding"] for r in self.receiver.requests],
            ["gzip", None, "gzip", None],
            "a receiver that refuses the plain body too has said nothing about gzip",
        )
        self.assertFalse(any("does not accept gzip" in m for m in logs.output), logs.output)


class TestRequestSize(_ReceiverTest):
    def test_a_batch_is_split_into_requests_under_the_byte_cap(self):
        exporter = _exporter(self.receiver.url)
        exporter._max_request_bytes = 50_000
        batch = [_envelope(i, payload=12_000) for i in range(10)]
        self.assertTrue(exporter.export(batch))
        sizes = [r["json_bytes"] for r in self.receiver.requests]
        self.assertGreaterEqual(len(sizes), 3, sizes)
        self.assertTrue(all(size <= 50_000 for size in sizes), sizes)
        self.assertEqual(self.receiver.accepted_ids(), _ids(batch), "every span arrives once, in order")

    def test_a_span_larger_than_the_cap_travels_alone(self):
        exporter = _exporter(self.receiver.url)
        exporter._max_request_bytes = 50_000
        batch = [_envelope(0, 100), _envelope(1, 80_000), _envelope(2, 100)]
        self.assertTrue(exporter.export(batch))
        self.assertEqual([len(r["events"]) for r in self.receiver.requests], [1, 1, 1])
        self.assertEqual(self.receiver.accepted_ids(), _ids(batch))

    def test_the_default_cap_is_4_mib(self):
        self.assertEqual(MAX_REQUEST_BYTES, 4 * 1024 * 1024)
        batch = [_envelope(i, payload=1024 * 1024) for i in range(6)]
        self.assertTrue(_exporter(self.receiver.url).export(batch))
        sizes = [r["json_bytes"] for r in self.receiver.requests]
        self.assertEqual(len(sizes), 2, sizes)
        self.assertTrue(all(size <= MAX_REQUEST_BYTES for size in sizes), sizes)

    def test_no_request_carries_more_than_max_export_batch_size_spans(self):
        batch = [_envelope(i) for i in range(10)]
        self.assertTrue(_exporter(self.receiver.url, max_export_batch_size=4).export(batch))
        self.assertEqual([len(r["events"]) for r in self.receiver.requests], [4, 4, 2])


class TestPayloadTooLarge(_ReceiverTest):
    def test_a_413_halves_the_request_until_each_part_is_accepted(self):
        self.receiver.respond = lambda r: 413 if len(r["events"]) > 2 else 202
        batch = [_envelope(i) for i in range(8)]
        self.assertTrue(_exporter(self.receiver.url).export(batch))
        self.assertEqual(
            [len(r["events"]) for r in self.receiver.requests],
            [8, 4, 2, 2, 4, 2, 2],
        )
        self.assertEqual(self.receiver.accepted_ids(), _ids(batch), "each span is accepted exactly once")

    def test_a_single_span_that_still_gets_413_is_dropped_with_a_warning_naming_its_size(self):
        self.receiver.respond = lambda r: 413 if any("attributes" in e for e in r["events"]) else 202
        big = _envelope(1, payload=20_000)
        batch = [_envelope(0), big, _envelope(2)]
        with self.assertLogs("evalkit", level="WARNING") as logs:
            self.assertFalse(_exporter(self.receiver.url).export(batch))
        self.assertEqual(self.receiver.accepted_ids(), _ids([batch[0], batch[2]]))
        size = len(json.dumps(big.to_dict()))
        [warning] = [m for m in logs.output if "413" in m]
        self.assertIn("op-1", warning)
        self.assertIn(f"{size} bytes", warning)


class TestRetries(_ReceiverTest):
    def test_a_receiver_that_stays_down_halts_the_export_after_one_round_of_retries(self):
        self.receiver.respond = lambda r: 503
        exporter = _exporter(self.receiver.url, max_export_batch_size=2)
        with mock.patch("evalkit.core.exporter.time.sleep") as sleep, self.assertLogs("evalkit", level="WARNING"):
            self.assertFalse(exporter.export([_envelope(i) for i in range(6)]))
        self.assertEqual(len(self.receiver.requests), 4, "a 5xx is retried 3 times, and the other requests are not sent")
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [0.2, 0.6, 1.8])
        self.assertTrue(exporter.halted)

    def test_a_5xx_then_success_delivers_the_spans(self):
        answers = iter([503, 202])
        self.receiver.respond = lambda r: next(answers)
        with mock.patch("evalkit.core.exporter.time.sleep"):
            self.assertTrue(_exporter(self.receiver.url).export([_envelope(0)]))
        self.assertEqual([r["status"] for r in self.receiver.requests], [503, 202])

    def test_a_refused_key_counts_once_per_export(self):
        self.receiver.respond = lambda r: 401
        exporter = _exporter(self.receiver.url, max_export_batch_size=1)
        self.assertFalse(exporter.export([_envelope(i) for i in range(3)]))
        self.assertEqual(len(self.receiver.requests), 1, "the requests after a refused key would be refused too")
        self.assertEqual(exporter._auth_failures, 1)
        with self.assertLogs("evalkit", level="WARNING"):
            exporter.export([_envelope(3)])
            exporter.export([_envelope(4)])
        self.assertTrue(exporter._disabled, "three refused exports in a row still turn tracing off")


class TestDrain(_ReceiverTest):
    def _client(self, **opts):
        client = EvalKitClient(EvalKitOptions(
            subscription_key="tk_test", base_url=self.receiver.url, scheduled_delay_ms=3_600_000, **opts,
        ))
        self.addCleanup(client.destroy)
        return client

    def _queue(self, client, envelopes):
        with client._lock:
            client._queue.extend(envelopes)

    def test_the_queue_is_sent_in_requests_of_at_most_max_export_batch_size(self):
        client = self._client(max_export_batch_size=5)
        batch = [_envelope(i) for i in range(12)]
        self._queue(client, batch)
        self.assertTrue(client._drain())
        self.assertEqual([len(r["events"]) for r in self.receiver.requests], [5, 5, 2])
        self.assertEqual(self.receiver.accepted_ids(), _ids(batch))
        self.assertEqual(client._queue, [])

    def test_a_full_default_queue_is_not_one_request(self):
        client = self._client()
        self._queue(client, [_envelope(i) for i in range(4096)])
        self.assertTrue(client._drain())
        self.assertEqual([len(r["events"]) for r in self.receiver.requests], [512] * 8)

    def test_draining_stops_at_a_batch_the_receiver_never_answers(self):
        self.receiver.respond = lambda r: 503
        client = self._client(max_export_batch_size=5)
        self._queue(client, [_envelope(i) for i in range(12)])
        with mock.patch("evalkit.core.exporter.time.sleep"), self.assertLogs("evalkit", level="WARNING") as logs:
            self.assertFalse(client._drain())
        self.assertEqual(len(self.receiver.requests), 4, "the later batches would wait out the same retries")
        self.assertTrue(any("dropping the 7 queued span(s)" in m for m in logs.output), logs.output)

    def test_draining_continues_past_a_batch_the_receiver_refuses(self):
        self.receiver.respond = lambda r: 400 if r["events"][0]["spanId"] == f"{0:016x}" else 202
        client = self._client(max_export_batch_size=5, export_gzip=False)
        batch = [_envelope(i) for i in range(12)]
        self._queue(client, batch)
        with self.assertLogs("evalkit", level="WARNING"):
            self.assertFalse(client._drain())
        self.assertEqual(self.receiver.accepted_ids(), _ids(batch[5:]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
