"""A request that carries a credential follows a redirect only within its own origin: the key never
reaches a server a redirect names elsewhere."""

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

from evalkit.core.api import ApiError, get_json
from evalkit.core.exporter import _Exporter
from evalkit.core.types import EvalKitOptions
from evalkit.voicetest import net
from evalkit_cli import otlp
from evalkit_cli.redirects import carries_credential, same_origin

KEY = "tk_live_dummy_redirect_test"


class _Server:
    """A server that answers every request with `answer(handler) -> (status, headers, body)` and
    records the credentials each request carried."""

    def __init__(self, answer):
        self.seen = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def _any(self):
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                outer.seen.append({
                    "method": self.command,
                    "path": self.path,
                    "key": self.headers.get("X-Subscription-Key"),
                    "authorization": self.headers.get("Authorization"),
                    "run_token": self.headers.get("X-Voice-Run-Token"),
                })
                status, headers, body = answer(self)
                raw = json.dumps(body).encode() if body is not None else b""
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            do_GET = do_POST = do_PUT = _any

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


class RedirectTestBase(unittest.TestCase):
    def setUp(self):
        self.elsewhere = _Server(lambda h: (200, {"Content-Type": "application/json"}, {"stolen": True}))
        self.addCleanup(self.elsewhere.close)

    def redirecting(self, status=302, same=False):
        def answer(h):
            if h.path.startswith("/moved"):
                return 200, {"Content-Type": "application/json"}, {"ok": True, "path": h.path}
            target = f"{server.url}/moved{h.path}" if same else f"{self.elsewhere.url}/elsewhere{h.path}"
            return status, {"Location": target}, None

        server = _Server(answer)
        self.addCleanup(server.close)
        return server


class SameOriginRuleTest(unittest.TestCase):
    def test_same_scheme_host_and_port(self):
        self.assertTrue(same_origin("https://api.example.com/a", "https://API.example.com:443/b?x=1"))
        self.assertTrue(same_origin("http://127.0.0.1:8085/a", "http://127.0.0.1:8085/b"))

    def test_http_to_https_on_the_same_host(self):
        self.assertTrue(same_origin("http://api.example.com/a", "https://api.example.com/a"))
        self.assertFalse(same_origin("https://api.example.com/a", "http://api.example.com/a"))
        self.assertFalse(same_origin("http://api.example.com:8080/a", "https://api.example.com/a"))

    def test_another_host_or_port_is_another_origin(self):
        self.assertFalse(same_origin("https://api.example.com/a", "https://evil.example/a"))
        self.assertFalse(same_origin("https://api.example.com/a", "https://api.example.com.evil.example/a"))
        self.assertFalse(same_origin("http://127.0.0.1:8085/a", "http://127.0.0.1:8086/a"))
        self.assertFalse(same_origin("https://api.example.com/a", "https://user@evil.example/a"))
        self.assertFalse(same_origin("https://api.example.com/a", "https://api.example.com:notaport/a"))

    def test_credential_headers(self):
        self.assertTrue(carries_credential(["X-subscription-key"]))
        self.assertTrue(carries_credential(["Authorization", "Accept"]))
        self.assertTrue(carries_credential(["X-Voice-Run-Token"]))
        self.assertFalse(carries_credential(["Accept", "Content-type"]))


class ApiRedirectTest(RedirectTestBase):
    def client(self):
        return SimpleNamespace(_opts=SimpleNamespace(subscription_key=KEY))

    def test_a_redirect_elsewhere_is_refused_and_the_key_stays(self):
        server = self.redirecting(302)
        with self.assertRaises(ApiError) as err:
            get_json(self.client(), server.url + "/v1/prompts/x", timeout=5, span_name="GET /v1/prompts/x")
        self.assertEqual(err.exception.code, 302)
        self.assertIn("refused a redirect to http://127.0.0.1:", str(err.exception))
        self.assertEqual(self.elsewhere.seen, [])

    def test_a_redirect_on_the_same_origin_is_followed(self):
        server = self.redirecting(301, same=True)
        body = get_json(self.client(), server.url + "/v1/prompts/x", timeout=5, span_name="GET /v1/prompts/x")
        self.assertEqual(body, {"ok": True, "path": "/moved/v1/prompts/x"})
        self.assertEqual([s["key"] for s in server.seen], [KEY, KEY])
        self.assertEqual(self.elsewhere.seen, [])


class ExporterRedirectTest(RedirectTestBase):
    def test_spans_are_not_posted_where_a_redirect_points(self):
        for status in (301, 302, 307, 308):
            with self.subTest(status=status):
                server = self.redirecting(status)
                exporter = _Exporter(EvalKitOptions(subscription_key=KEY, base_url=server.url, export_gzip=False))
                code, text, error = exporter._post(b'{"spans":[]}', compressed=False, deadline=time.monotonic() + 5)
                self.assertEqual(code, status)
                self.assertIsNone(error)
                self.assertEqual(self.elsewhere.seen, [])


class VoiceNetRedirectTest(RedirectTestBase):
    def test_a_run_token_never_follows_a_redirect_elsewhere(self):
        server = self.redirecting(307)
        with self.assertRaises(net.HttpError) as err:
            net.json_request("POST", server.url + "/v1/voice-tests/runs/r/calls", headers={"X-Voice-Run-Token": "dummy-run-token"},
                             body={"a": 1}, timeout=5)
        self.assertEqual(err.exception.status, 307)
        self.assertIn("refused a redirect", err.exception.message)
        self.assertEqual(self.elsewhere.seen, [])

    def test_a_signed_url_with_no_credential_still_follows_its_redirect(self):
        server = self.redirecting(302)
        status, raw = net.request("GET", server.url + "/audio/line.wav", timeout=5)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw), {"stolen": True})
        self.assertEqual(self.elsewhere.seen[0]["key"], None)


class CliOtlpRedirectTest(RedirectTestBase):
    def test_the_hook_sender_keeps_its_key(self):
        server = self.redirecting(307)
        status, error = otlp.send(server.url, KEY, "traces", {"resourceSpans": []})
        self.assertEqual(status, 307)
        self.assertEqual(self.elsewhere.seen, [])

    def test_the_doctor_read_back_keeps_its_key(self):
        server = self.redirecting(302)
        status, body, error = otlp.get_json(server.url + "/v1/traces/me/x", KEY)
        self.assertEqual(status, 302)
        self.assertIsNone(body)
        self.assertEqual(self.elsewhere.seen, [])


if __name__ == "__main__":
    unittest.main()
