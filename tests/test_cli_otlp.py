import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from evalkit_cli import otlp


def receiver(answers):
    """Answers each POST with the next of `answers`: a status, or (status, Retry-After). The last one repeats."""
    statuses = []
    queue = list(answers)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            answer = queue.pop(0) if len(queue) > 1 else queue[0]
            status, retry_after = answer if isinstance(answer, tuple) else (answer, None)
            statuses.append(status)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if retry_after is not None:
                self.send_header("Retry-After", retry_after)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return server, statuses


class TestSend(unittest.TestCase):
    def _send(self, answers, deadline_in=None):
        server, statuses = receiver(answers)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        deadline = time.monotonic() + deadline_in if deadline_in is not None else None
        with mock.patch("evalkit_cli.otlp.time.sleep") as sleep:
            status, _ = otlp.send_logs(endpoint, "k", {"resourceLogs": []}, deadline=deadline)
        return status, statuses, [c.args[0] for c in sleep.call_args_list]

    def test_without_a_deadline_a_record_is_tried_once(self):
        status, statuses, waits = self._send([503, 200])
        self.assertEqual((status, statuses, waits), (503, [503], []))

    def test_with_a_deadline_429_and_5xx_are_retried(self):
        for first in (429, 502, 503):
            with self.subTest(first=first):
                status, statuses, _ = self._send([first, 200], deadline_in=4)
                self.assertEqual((status, statuses), (200, [first, 200]))

    def test_retry_after_sets_the_wait(self):
        status, statuses, waits = self._send([(503, "1"), 200], deadline_in=4)
        self.assertEqual((status, statuses, waits), (200, [503, 200], [1.0]))

    def test_a_wait_that_does_not_fit_before_the_deadline_is_not_taken(self):
        status, statuses, waits = self._send([(429, "3"), 200], deadline_in=2)
        self.assertEqual((status, statuses, waits), (429, [429], []))

    def test_retries_stop_after_three(self):
        status, statuses, waits = self._send([503], deadline_in=60)
        self.assertEqual((status, len(statuses), len(waits)), (503, 4, 3))

    def test_other_4xx_answers_are_not_retried(self):
        for code in (400, 401, 404):
            with self.subTest(code=code):
                status, statuses, waits = self._send([(code, "1"), 200], deadline_in=4)
                self.assertEqual((status, statuses, waits), (code, [code], []))


class TestRetryAfter(unittest.TestCase):
    def test_delta_seconds_and_http_dates_are_read(self):
        now = 1767225600.0  # 2026-01-01T00:00:00Z
        self.assertEqual(otlp._retry_after("5", now), 5.0)
        self.assertEqual(otlp._retry_after("Thu, 01 Jan 2026 00:00:07 GMT", now), 7.0)
        for value in (None, "", "soon", "1.5"):
            self.assertIsNone(otlp._retry_after(value, now))

    def test_a_long_retry_after_is_capped(self):
        self.assertEqual(otlp._retry_wait(0, "3600"), otlp._MAX_RETRY_WAIT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
