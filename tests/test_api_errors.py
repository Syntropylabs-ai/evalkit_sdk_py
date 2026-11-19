"""A refused call to the SyntropyLabs API raises `evalkit.ApiError`: an ``HTTPError`` whose message is the
call, the status and the server's own message, with the server's Retry-After, and never a key."""

import json
import threading
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import evalkit
from tests.helpers import fresh, reset_context

API_KEY = "AIzaSyTestProviderKey0123456789abcdefgh"


class _Server:
    """Answers each path with the status, body and headers registered for it."""

    def __init__(self, routes):
        self.routes = routes
        self.requests = []
        server = self

        class H(BaseHTTPRequestHandler):
            def _answer(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length) if length else b""
                server.requests.append((self.command, self.path, body))
                status, payload, headers = server.routes[self.path.split("?")[0]]
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = _answer

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


class TestApiErrors(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_generate_scenarios_raises_the_servers_message_and_status(self):
        no_model = ("No model to generate scenarios with. Either add a text model under Models in the console, "
                    "or pass model, provider and api_key from the SDK.")
        with _Server({"/scenarios/generate": (400, {"statusCode": 400, "message": no_model,
                                                    "error": "Bad Request"}, None)}) as srv:
            with self.assertRaises(evalkit.ApiError) as cm:
                evalkit.generate_scenarios("billing agent", api_url=srv.url)
        err = cm.exception
        self.assertIsInstance(err, urllib.error.HTTPError)
        self.assertEqual((err.code, err.status), (400, 400))
        self.assertEqual(str(err), f"POST /scenarios/generate: HTTP 400: {no_model}")
        self.assertEqual(err.server_message, no_model)
        self.assertIn('"statusCode": 400', err.response_text)
        self.assertIsNone(err.retry_after)
        self.assertEqual(json.loads(err.read())["message"], no_model)

    def test_validation_lists_are_joined_and_retry_after_is_read(self):
        routes = {
            "/scenarios/generate": (429, {"statusCode": 429, "message": ["the provider is rate limiting", "slow down"]},
                                    {"Retry-After": "12"}),
        }
        with _Server(routes) as srv:
            with self.assertRaises(evalkit.ApiError) as cm:
                evalkit.generate_scenarios("billing agent", api_url=srv.url)
        self.assertEqual(str(cm.exception),
                         "POST /scenarios/generate: HTTP 429: the provider is rate limiting; slow down")
        self.assertEqual(cm.exception.retry_after, 12.0)

    def test_fastapi_detail_and_plain_text_bodies(self):
        routes = {
            "/scenarios/generate": (503, {"detail": {"error": "The model is overloaded", "type": "llm_error"}},
                                    {"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}),
            "/v1/feedback": (502, b"<html>Bad Gateway</html>", None),
        }
        with _Server(routes) as srv:
            with self.assertRaises(evalkit.ApiError) as cm:
                evalkit.generate_scenarios("billing agent", api_url=srv.url)
            self.assertEqual(str(cm.exception), "POST /scenarios/generate: HTTP 503: The model is overloaded")
            self.assertEqual(cm.exception.retry_after, 0.0)  # a date already past: no wait
            with self.assertRaises(evalkit.ApiError) as cm:
                evalkit.feedback("trace-1", "thumbs", True, api_url=srv.url)
            self.assertEqual(str(cm.exception), "POST /v1/feedback: HTTP 502: <html>Bad Gateway</html>")

    def test_no_key_reaches_the_message_even_when_the_server_echoes_one(self):
        echo = f"The provider refused the key {API_KEY} (and tk_test)"
        with _Server({"/scenarios/generate": (400, {"message": echo}, None)}) as srv:
            with self.assertRaises(evalkit.ApiError) as cm:
                evalkit.generate_scenarios("billing agent", provider="gemini", model="gemini-2.5-flash",
                                           api_key=API_KEY, api_url=srv.url)
            sent = json.loads(srv.requests[0][2])
        self.assertEqual(sent["apiKey"], API_KEY)  # sent once, to the API, for this request
        text = str(cm.exception) + cm.exception.response_text + repr(cm.exception)
        self.assertNotIn(API_KEY, text)
        self.assertNotIn("tk_test", text)
        self.assertIn("The provider refused the key ***", str(cm.exception))
        self.client.flush()
        spans = [e for e in self.captured if (e.operation or "") == "POST /scenarios/generate"]
        self.assertTrue(spans)
        self.assertNotIn(API_KEY, json.dumps(spans[-1].attributes, default=str))
        self.assertEqual(spans[-1].attributes.get("http.status_code"), 400)

    def test_a_get_names_its_path(self):
        with _Server({"/v1/prompts/greeting": (404, {"statusCode": 404, "message": "Prompt greeting not found"},
                                               None)}) as srv:
            with self.assertRaises(evalkit.ApiError) as cm:
                evalkit.prompts.get("greeting", api_url=srv.url, cache_ttl=0)
        self.assertEqual(str(cm.exception), "GET /v1/prompts/greeting: HTTP 404: Prompt greeting not found")

    def test_the_cli_otlp_reads_carry_the_receivers_message(self):
        from evalkit_cli.otlp import get_json, send_traces

        routes = {"/v1/traces/t/x": (403, {"message": "this key belongs to another tenant"}, None),
                  "/v1/traces": (400, {"message": "resourceSpans is required"}, None)}
        with _Server(routes) as srv:
            self.assertEqual(get_json(srv.url + "/v1/traces/t/x", "k"), (403, None, "this key belongs to another tenant"))
            self.assertEqual(send_traces(srv.url, "k", {}), (400, "resourceSpans is required"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
