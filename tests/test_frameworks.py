import asyncio
import unittest

import evalkit
from tests.helpers import fresh, reset_context


EXPECTED_EXPORTS = [
    "EvalKitMiddleware", "EvalKitStarletteMiddleware", "EvalKitDjangoMiddleware",
    "instrument_flask", "create_litestar_middleware",
    "patch_sqlalchemy_engine", "patch_async_sqlalchemy_engine",
    "patch_openai_client", "patch_anthropic_client",
    "patch_bedrock_client", "patch_cohere_client", "patch_google_ai_model",
    "patch_google_genai", "patch_mistral_client", "patch_langchain", "patch_litellm",
    "patch_claude_agent_sdk",
    "start_trace", "start_span", "start_http_trace", "with_trace", "trace_function",
    "flush", "current_trace_id",
    "install_logging_handler", "EvalKitLoggingHandler", "EvalKitLogFilter",
    "SimContext", "AgentTurnResult", "with_simulation",
]


class TestFrameworkExports(unittest.TestCase):
    def test_all_exports_present(self):
        for name in EXPECTED_EXPORTS:
            self.assertTrue(hasattr(evalkit, name), f"missing export: {name}")


class TestASGIMiddleware(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def _run_mw(self, app, *, include_traceparent=True):
        mw = evalkit.EvalKitMiddleware(app, self.client)
        headers = [(b"content-type", b"application/json")]
        if include_traceparent:
            tp = b"00-" + b"a" * 32 + b"-" + b"b" * 16 + b"-01"
            headers.append((b"traceparent", tp))
        scope = {
            "type": "http", "method": "POST", "path": "/api/chat",
            "query_string": b"", "headers": headers,
            "scheme": "http", "server": ("localhost", 80),
        }

        async def receive():
            return {"type": "http.request", "body": b'{"q":"hi"}', "more_body": False}

        sent = []

        async def send(m):
            sent.append(m)

        asyncio.run(mw(scope, receive, send))
        return sent

    def test_success_joins_upstream_trace(self):
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": b'{"ok":true}'})

        sent = self._run_mw(app)
        self.client.flush()
        http = [e for e in self.captured if e.span_type == "http_call"]
        self.assertTrue(http)
        e = http[0]
        self.assertEqual(e.status, "OK")
        self.assertEqual(e.trace_id, "a" * 32, "must join upstream W3C traceparent")
        self.assertEqual(e.attributes["http.status_code"], 200)
        start = [m for m in sent if m["type"] == "http.response.start"][0]
        hdr_keys = {k for k, _ in start["headers"]}
        self.assertIn(b"x-trace-id", hdr_keys)
        self.assertIn(b"traceparent", hdr_keys)

    def test_error_path_records_error_span(self):
        async def app(scope, receive, send):
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            self._run_mw(app)
        self.client.flush()
        http = [e for e in self.captured if e.span_type == "http_call"]
        self.assertTrue(http)
        self.assertEqual(http[0].status, "ERROR")

    def test_no_traceparent_creates_fresh_trace(self):
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        self._run_mw(app, include_traceparent=False)
        self.client.flush()
        http = [e for e in self.captured if e.span_type == "http_call"]
        self.assertTrue(http)
        self.assertNotEqual(http[0].trace_id, "a" * 32)

    def test_websocket_scope_passes_through(self):
        called = []

        async def app(scope, receive, send):
            called.append(scope["type"])

        mw = evalkit.EvalKitMiddleware(app, self.client)
        scope = {"type": "websocket", "path": "/ws", "headers": []}

        async def receive():
            return {}

        async def send(m):
            pass

        asyncio.run(mw(scope, receive, send))
        self.assertEqual(called, ["websocket"])
        self.client.flush()
        http = [e for e in self.captured if e.span_type == "http_call"]
        self.assertEqual(len(http), 0, "websocket must not emit http_call span")

    def test_chunked_body_fully_captured(self):
        """ASGI multi-chunk bodies (more_body=True then False) must be fully captured."""
        chunks = [b'{"key":', b'"value"}']

        async def chunked_receive(_iter=[iter(chunks)]):
            chunk = next(_iter[0], None)
            if chunk is not None:
                return {"type": "http.request", "body": chunk, "more_body": True}
            return {"type": "http.request", "body": b"", "more_body": False}

        # app must be able to read the body via receive (replay must work)
        app_body = []

        async def app(scope, receive, send):
            msg = await receive()
            while True:
                app_body.append(msg.get("body", b""))
                if not msg.get("more_body", False):
                    break
                msg = await receive()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        mw = evalkit.EvalKitMiddleware(app, self.client)
        scope = {
            "type": "http", "method": "POST", "path": "/upload",
            "query_string": b"", "headers": [(b"content-type", b"application/json")],
            "scheme": "http", "server": ("localhost", 80),
        }

        sent = []

        async def send_capture(m):
            sent.append(m)

        asyncio.run(mw(scope, chunked_receive, send_capture))

        self.client.flush()
        http = [e for e in self.captured if e.span_type == "http_call"]
        self.assertTrue(http)
        # span must see the assembled body
        body_attr = http[0].attributes.get("http.request.body", "")
        self.assertIn("key", body_attr, "assembled body must appear in span attributes")
        # app must have received all chunks via replay
        reassembled = b"".join(app_body)
        self.assertEqual(reassembled, b'{"key":"value"}', "app must receive full body via replay")


if __name__ == "__main__":
    unittest.main(verbosity=2)
