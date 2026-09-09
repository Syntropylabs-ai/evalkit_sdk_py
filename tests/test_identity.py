import asyncio
import unittest

import evalkit
from evalkit.frameworks._identity import user_id_of
from tests.helpers import fresh, reset_context


class _FakeUser:
    def __init__(self, authenticated=True, **fields):
        self.is_authenticated = authenticated
        for key, value in fields.items():
            setattr(self, key, value)


class TestIdentity(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def tearDown(self):
        evalkit.set_user()

    def _span(self, operation):
        self.client.flush()
        spans = [e for e in self.captured if e.operation == operation]
        self.assertEqual(len(spans), 1, operation)
        return spans[0]

    def test_start_trace_accepts_identity(self):
        _, end, ctx = self.client.start_trace(
            "checkout", user_id="u-1", session_id="s-1", device_id="d-1"
        )
        end("OK")
        span = self._span("checkout")
        self.assertEqual((span.user_id, span.session_id, span.device_id), ("u-1", "s-1", "d-1"))
        self.assertEqual((ctx.user_id, ctx.session_id, ctx.device_id), ("u-1", "s-1", "d-1"))

    def test_identity_lands_on_the_wire_envelope(self):
        _, end, _ = self.client.start_trace("wire", user_id="u-2", session_id="s-2", device_id="d-2")
        end("OK")
        payload = self._span("wire").to_dict()
        self.assertEqual(payload["userId"], "u-2")
        self.assertEqual(payload["sessionId"], "s-2")
        self.assertEqual(payload["deviceId"], "d-2")

    def test_child_spans_inherit_trace_identity(self):
        _, end, ctx = self.client.start_trace("root", user_id="u-3", session_id="s-3")
        with evalkit.with_trace(ctx):
            end_child, _ = self.client.start_span("child")
            end_child("OK")
        end("OK")
        child = self._span("child")
        self.assertEqual(child.user_id, "u-3")
        self.assertEqual(child.session_id, "s-3")

    def test_set_user_applies_to_every_span_until_cleared(self):
        evalkit.set_user(user_id="u-4", session_id="s-4", device_id="d-4")
        end_a, _ = self.client.start_span("before")
        end_a("OK")
        before = self._span("before")
        self.assertEqual((before.user_id, before.session_id, before.device_id), ("u-4", "s-4", "d-4"))

        evalkit.set_user()
        end_b, _ = self.client.start_span("after")
        end_b("OK")
        after = self._span("after")
        self.assertIsNone(after.user_id)
        self.assertIsNone(after.session_id)

    def test_set_user_merges_fields_and_reports_current(self):
        evalkit.set_user(session_id="s-5")
        evalkit.set_user(user_id="u-5")
        identity = evalkit.current_user()
        self.assertEqual((identity.user_id, identity.session_id), ("u-5", "s-5"))
        end, _ = self.client.start_span("merged")
        end("OK")
        span = self._span("merged")
        self.assertEqual((span.user_id, span.session_id), ("u-5", "s-5"))

    def test_clear_user_restores_previous_scope(self):
        token = evalkit.set_user(user_id="u-6")
        evalkit.clear_user(token)
        self.assertIsNone(evalkit.current_user().user_id)

    def test_explicit_start_trace_identity_beats_ambient(self):
        evalkit.set_user(user_id="ambient", session_id="ambient-session")
        _, end, _ = self.client.start_trace("explicit", user_id="explicit")
        end("OK")
        span = self._span("explicit")
        self.assertEqual(span.user_id, "explicit")
        self.assertEqual(span.session_id, "ambient-session")

    def test_identity_reaches_instrumented_spans(self):
        evalkit.set_user(user_id="u-7", session_id="s-7")

        @self.client.trace_tool("search")
        def search(query):
            return [query]

        search("weather")
        span = self._span("search")
        self.assertEqual(span.user_id, "u-7")
        self.assertEqual(span.session_id, "s-7")

    def test_identity_does_not_leak_across_tasks(self):
        async def scenario():
            async def request(user):
                evalkit.set_user(user_id=user)
                end, _ = self.client.start_span(f"task-{user}")
                await asyncio.sleep(0)
                end("OK")

            await asyncio.gather(request("a"), request("b"))

        asyncio.run(scenario())
        self.assertEqual(self._span("task-a").user_id, "a")
        self.assertEqual(self._span("task-b").user_id, "b")
        self.assertIsNone(evalkit.current_user().user_id)

    def test_device_id_falls_back_to_client_option(self):
        end, _ = self.client.start_span("device-default")
        end("OK")
        self.assertIsNone(self._span("device-default").device_id)

    def test_module_level_start_trace_forwards_identity(self):
        _, end, ctx = evalkit.start_trace("module-level", user_id="u-8", session_id="s-8")
        end("OK")
        span = self._span("module-level")
        self.assertEqual((span.user_id, span.session_id), ("u-8", "s-8"))
        self.assertEqual(ctx.user_id, "u-8")

    def test_start_trace_keeps_positional_attributes(self):
        _, end, _ = evalkit.start_trace("positional", {"custom.key": "v"})
        end("OK")
        span = self._span("positional")
        self.assertEqual((span.attributes or {}).get("custom.key"), "v")

    def test_user_id_of_reads_framework_users(self):
        self.assertEqual(user_id_of(_FakeUser(pk=17)), "17")
        self.assertEqual(user_id_of(_FakeUser(identity="alice@example.com")), "alice@example.com")
        self.assertEqual(user_id_of(_FakeUser(username="bob")), "bob")
        self.assertIsNone(user_id_of(_FakeUser(authenticated=False, pk=17)))
        self.assertIsNone(user_id_of(object()))
        self.assertIsNone(user_id_of(None))

    def test_asgi_middleware_passes_scope_user(self):
        from evalkit.frameworks.fastapi import EvalKitMiddleware

        seen = {}

        async def app(scope, receive, send):
            end, _ = self.client.start_span("handler")
            end("OK")
            seen["identity"] = evalkit.current_user().user_id
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"{}"})

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            return None

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/orders",
            "headers": [],
            "server": ("testserver", 80),
            "scheme": "http",
            "user": _FakeUser(pk="u-9"),
        }
        asyncio.run(EvalKitMiddleware(app, self.client)(scope, receive, send))

        self.assertEqual(seen["identity"], "u-9")
        self.assertEqual(self._span("handler").user_id, "u-9")
        self.assertEqual(self._span("GET /orders").user_id, "u-9")
        self.assertIsNone(evalkit.current_user().user_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
