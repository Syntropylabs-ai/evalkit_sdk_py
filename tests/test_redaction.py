import json
import os
import unittest

from evalkit.core.redact import (
    CAPTURE_CONTENT_ENV_VARS,
    CONTENT_STRIPPED_ATTRIBUTE,
    resolve_capture_content,
)
from evalkit.core.types import SpanEvent, TraceEnvelope
from tests.helpers import fresh, reset_context


def _llm_envelope():
    return TraceEnvelope(
        trace_id="0" * 32,
        span_id="1" * 16,
        created_at="2026-08-26T00:00:00Z",
        span_type="llm_call",
        operation="InvokeModel",
        model="anthropic.claude-3-haiku-20240307-v1:0",
        provider="bedrock",
        prompt="user: my card is 4111 1111 1111 1111",
        completion="I will not repeat that",
        tokens_in=11,
        tokens_out=4,
        cache_read_tokens=3,
        latency_ms=42,
        attributes={
            "gen_ai.request.body": '{"messages":[{"role":"user","content":"my card is 4111"}]}',
            "gen_ai.response.body": '{"content":[{"text":"I will not repeat that"}]}',
            "gen_ai.response.thinking": "the user pasted a card number",
            "gen_ai.request.model": "anthropic.claude-3-haiku-20240307-v1:0",
            "gen_ai.usage.input_tokens": 11,
            "gen_ai.response.finish_reasons": ["end_turn"],
            "llm.cost_usd": 0.0001,
        },
        events=[SpanEvent(
            name="gen_ai.tool.call",
            timestamp="2026-08-26T00:00:00Z",
            attributes={
                "gen_ai.tool.name": "charge_card",
                "gen_ai.tool.call.id": "call_1",
                "gen_ai.tool.call.arguments": '{"card":"4111 1111 1111 1111"}',
                "gen_ai.tool.call.result": '{"charged":true}',
            },
        )],
    )


class CaptureContentEnvTest(unittest.TestCase):
    def setUp(self):
        self._saved = {name: os.environ.get(name) for name in CAPTURE_CONTENT_ENV_VARS}
        for name in CAPTURE_CONTENT_ENV_VARS:
            os.environ.pop(name, None)

    def tearDown(self):
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_defaults_to_capturing(self):
        self.assertTrue(
            resolve_capture_content(None),
            "content capture is the whole point of the product, so it stays on unless someone opts out",
        )

    def test_explicit_argument_beats_every_env_var(self):
        os.environ["EVALKIT_CAPTURE_CONTENT"] = "true"
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"
        self.assertFalse(resolve_capture_content(False))
        self.assertTrue(resolve_capture_content(True))

    def test_evalkit_var_beats_the_otel_var(self):
        os.environ["EVALKIT_CAPTURE_CONTENT"] = "false"
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"
        self.assertFalse(
            resolve_capture_content(None),
            "the EvalKit-specific var is the more specific signal, so it wins",
        )

    def test_otel_standard_var_is_honoured(self):
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "false"
        self.assertFalse(
            resolve_capture_content(None),
            "an app already configured for OpenTelemetry must not need an EvalKit-specific var to stay private",
        )

    def test_accepts_the_usual_boolean_spellings(self):
        for raw in ("0", "false", "FALSE", "no", "off", " Off "):
            os.environ["EVALKIT_CAPTURE_CONTENT"] = raw
            self.assertFalse(resolve_capture_content(None), raw)
        for raw in ("1", "true", "YES", "on"):
            os.environ["EVALKIT_CAPTURE_CONTENT"] = raw
            self.assertTrue(resolve_capture_content(None), raw)

    def test_unparseable_value_falls_through_rather_than_guessing(self):
        os.environ["EVALKIT_CAPTURE_CONTENT"] = "maybe"
        self.assertTrue(
            resolve_capture_content(None),
            "guessing 'off' from a typo would silently blind a customer's tracing; guessing 'on' matches the default",
        )


class RedactionTest(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self._saved = (self.client._opts.capture_content, self.client._opts.mask)

    def tearDown(self):
        self.client._opts.capture_content, self.client._opts.mask = self._saved

    def _emit(self, envelope):
        self.client.enqueue(envelope)
        self.client.flush()
        return self.captured

    def test_content_is_kept_by_default(self):
        span = next(e for e in self._emit(_llm_envelope()) if e.span_type == "llm_call")
        self.assertEqual(span.prompt, "user: my card is 4111 1111 1111 1111")
        self.assertIn("gen_ai.request.body", span.attributes)
        self.assertNotIn(CONTENT_STRIPPED_ATTRIBUTE, span.attributes)

    def test_capture_content_false_drops_payloads_and_keeps_metrics(self):
        self.client._opts.capture_content = False
        spans = self._emit(_llm_envelope())
        span = next(e for e in spans if e.span_type == "llm_call")
        self.assertIsNone(span.prompt)
        self.assertIsNone(span.completion)
        for key in ("gen_ai.request.body", "gen_ai.response.body", "gen_ai.response.thinking"):
            self.assertNotIn(key, span.attributes, key)
        self.assertEqual(span.tokens_in, 11, "tokens are what the bill is computed from and are not content")
        self.assertEqual(span.tokens_out, 4)
        self.assertEqual(span.cache_read_tokens, 3)
        self.assertEqual(span.latency_ms, 42)
        self.assertEqual(span.model, "anthropic.claude-3-haiku-20240307-v1:0")
        self.assertEqual(span.attributes["gen_ai.response.finish_reasons"], ["end_turn"])
        self.assertEqual(span.attributes["llm.cost_usd"], 0.0001)
        self.assertIs(
            span.attributes[CONTENT_STRIPPED_ATTRIBUTE], False,
            "the dashboard needs to tell 'the customer turned capture off' apart from 'the SDK failed to read the body'",
        )

    def test_capture_content_false_keeps_tool_identity_and_drops_tool_payloads(self):
        self.client._opts.capture_content = False
        spans = self._emit(_llm_envelope())
        parent = next(e for e in spans if e.span_type == "llm_call")
        event = parent.events[0]
        self.assertEqual(event.attributes["gen_ai.tool.name"], "charge_card")
        self.assertEqual(event.attributes["gen_ai.tool.call.id"], "call_1")
        self.assertNotIn("gen_ai.tool.call.arguments", event.attributes)
        self.assertNotIn("gen_ai.tool.call.result", event.attributes)

    def test_derived_tool_span_survives_redaction(self):
        self.client._opts.capture_content = False
        spans = self._emit(_llm_envelope())
        tools = [e for e in spans if e.span_type == "tool_call"]
        self.assertEqual(
            len(tools), 1,
            "the child tool_call span is derived from the parent's tool arguments and result, so redaction has to "
            "run after that derivation or turning capture off silently deletes the tool spans too",
        )
        self.assertEqual(tools[0].attributes["tool.name"], "charge_card")
        self.assertNotIn("tool.arguments", tools[0].attributes)
        self.assertNotIn("tool.result", tools[0].attributes)

    def test_function_tracing_io_is_redacted(self):
        self.client._opts.capture_content = False

        @self.client.trace_function("io-fn", capture_io=True)
        def compute(a, b):
            return a + b

        compute(3, 4)
        self.client.flush()
        span = next(e for e in self.captured if e.operation == "io-fn")
        self.assertNotIn(
            "function.args", span.attributes,
            "APM function tracing captures every argument, which is the largest content surface in the SDK",
        )
        self.assertNotIn("function.kwargs", span.attributes)
        self.assertNotIn("function.result", span.attributes)
        self.assertEqual(span.status, "OK")

    def test_trace_tool_payloads_are_redacted(self):
        self.client._opts.capture_content = False

        @self.client.trace_tool("search_web")
        def search_web(query):
            return {"results": [query]}

        search_web("weather")
        self.client.flush()
        span = next(e for e in self.captured if e.span_type == "tool_call")
        self.assertEqual(span.attributes["tool.name"], "search_web")
        self.assertNotIn("tool.arguments", span.attributes)
        self.assertNotIn("tool.result", span.attributes)

    def test_db_statement_and_params_are_redacted(self):
        self.client._opts.capture_content = False
        span = self._emit(TraceEnvelope(
            trace_id="0" * 32, span_id="2" * 16, created_at="2026-08-26T00:00:00Z",
            span_type="db_query", operation="SELECT",
            attributes={
                "db.system": "postgresql",
                "db.operation": "SELECT",
                "db.statement": "SELECT * FROM users WHERE ssn = '123-45-6789'",
                "db.params": ["123-45-6789"],
                "db.rows_affected": 1,
            },
        ))[0]
        self.assertNotIn("db.statement", span.attributes, "SQL text carries literal values")
        self.assertNotIn("db.params", span.attributes, "bound parameters are user data by definition")
        self.assertEqual(span.attributes["db.system"], "postgresql")
        self.assertEqual(span.attributes["db.operation"], "SELECT")
        self.assertEqual(span.attributes["db.rows_affected"], 1)

    def test_http_bodies_and_headers_are_redacted(self):
        self.client._opts.capture_content = False
        span = self._emit(TraceEnvelope(
            trace_id="0" * 32, span_id="4" * 16, created_at="2026-08-26T00:00:00Z",
            span_type="http_call", operation="POST /charge",
            attributes={
                "http.request.method": "POST",
                "url.full": "https://api.example.com/charge",
                "http.response.status_code": 200,
                "http.request.body": '{"card":"4111 1111 1111 1111"}',
                "http.response.body": '{"receipt":"r_1"}',
                "http.request.headers": '{"authorization":"Bearer sk-live-..."}',
                "http.response.headers": '{"set-cookie":"session=..."}',
            },
        ))[0]
        for key in ("http.request.body", "http.response.body", "http.request.headers", "http.response.headers"):
            self.assertNotIn(key, span.attributes, key)
        self.assertEqual(
            span.attributes["url.full"], "https://api.example.com/charge",
            "the URL is how an HTTP span is identified; only the payload and headers are content",
        )
        self.assertEqual(span.attributes["http.response.status_code"], 200)

    def test_envelope_with_no_attributes_still_gets_the_marker(self):
        self.client._opts.capture_content = False
        span = self._emit(TraceEnvelope(
            trace_id="0" * 32, span_id="3" * 16, created_at="2026-08-26T00:00:00Z",
            span_type="function_call", operation="bare", prompt="secret",
        ))[0]
        self.assertIsNone(span.prompt)
        self.assertIs(span.attributes[CONTENT_STRIPPED_ATTRIBUTE], False)

    def test_mask_can_rewrite_content(self):
        def mask(envelope):
            if envelope.prompt:
                envelope.prompt = "<redacted>"
            return envelope

        self.client._opts.mask = mask
        span = next(e for e in self._emit(_llm_envelope()) if e.span_type == "llm_call")
        self.assertEqual(span.prompt, "<redacted>")
        self.assertEqual(span.tokens_in, 11)

    def test_mask_sees_the_already_stripped_envelope(self):
        seen = []

        def mask(envelope):
            seen.append(envelope.prompt)
            return envelope

        self.client._opts.capture_content = False
        self.client._opts.mask = mask
        self._emit(_llm_envelope())
        self.assertIn(
            None, seen,
            "capture_content is the coarse switch and mask is the fine one, so mask must run on the stripped "
            "envelope rather than being handed content the customer already said not to capture",
        )

    def test_mask_returning_none_drops_the_span(self):
        self.client._opts.mask = lambda envelope: None
        self.assertEqual(
            self._emit(_llm_envelope()), [],
            "returning None is the documented way to drop a span entirely",
        )

    def test_mask_that_raises_drops_the_span_rather_than_leaking(self):
        def mask(envelope):
            raise ValueError("scrubber misconfigured")

        self.client._opts.mask = mask
        self.assertEqual(
            self._emit(_llm_envelope()), [],
            "a mask hook exists to stop content leaving the process, so a broken hook must fail closed even though "
            "the SDK's general rule is to never let instrumentation affect the app",
        )

    def test_mask_returning_the_wrong_type_drops_the_span(self):
        self.client._opts.mask = lambda envelope: {"prompt": "not an envelope"}
        self.assertEqual(self._emit(_llm_envelope()), [])

    def test_mask_failure_does_not_raise_into_the_caller(self):
        self.client._opts.mask = lambda envelope: 1 / 0
        self.client.enqueue(_llm_envelope())
        self.client.flush()
        self.assertEqual(self.captured, [])


def _foreign_genai_span():
    """A chat span another OpenTelemetry GenAI instrumentation wrote, with its own content on it."""
    from opentelemetry.sdk.trace import TracerProvider

    tracer = TracerProvider().get_tracer("some.other.genai.library")
    span = tracer.start_span("chat gpt-4o", attributes={
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": "gpt-4o",
        "gen_ai.usage.input_tokens": 21,
        "gen_ai.usage.output_tokens": 5,
        "gen_ai.input.messages": '[{"role":"user","parts":[{"type":"text","content":"my card is 4242 4242 4242 4242"}]}]',
        "gen_ai.output.messages": '[{"role":"assistant","parts":[{"type":"text","content":"noted your card"}]}]',
        "gen_ai.system_instructions": "You are a billing agent",
        "gen_ai.prompt": "legacy prompt text",
        "gen_ai.completion": "legacy completion text",
        "gen_ai.prompt.0.role": "user",
        "gen_ai.prompt.0.content": "indexed prompt text",
        "gen_ai.completion.0.content": "indexed completion text",
        "gen_ai.completion.0.finish_reason": "tool_calls",
        "gen_ai.completion.0.tool_calls.0.name": "charge_card",
        "gen_ai.completion.0.tool_calls.0.arguments": '{"card":"4242"}',
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.content": "flattened prompt text",
        "input.value": "plain input value",
        "output.value": "plain output value",
        "acme.observation.input": "prefixed observation input",
        "acme.trace.output": "prefixed trace output",
        "acme.entity.input": "prefixed entity input",
        "gen_ai.tool.name": "charge_card",
        "gen_ai.request.tools": ["charge_card"],
        "gen_ai.prompt.name": "billing-v2",
    })
    span.add_event("gen_ai.system.message", {"content": "event-borne system text"})
    span.add_event("gen_ai.user.message", {"content": "event-borne user text"})
    span.add_event("gen_ai.assistant.message", {"content": "", "tool_calls": '[{"function":{"name":"charge_card","arguments":"{}"}}]'})
    span.add_event("gen_ai.tool.message", {"content": "charged", "id": "call_1"})
    span.add_event("gen_ai.choice", {"index": 0, "finish_reason": "stop", "message": '{"content":"noted your card"}'})
    span.add_event("gen_ai.client.inference.operation.details", {
        "gen_ai.input.messages": '[{"role":"user","parts":[{"type":"text","content":"details text"}]}]',
        "gen_ai.output.messages": '[{"role":"assistant","parts":[{"type":"text","content":"details reply"}]}]',
        "gen_ai.response.finish_reasons": ["stop"],
    })
    span.add_event("exception", {"exception.type": "ValueError"})
    span.end()
    return span


class ForeignContentRedactionTest(unittest.TestCase):
    """capture_content=False holds for content another instrumentation put on its spans, too."""

    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self._saved = self.client._opts.capture_content

    def tearDown(self):
        self.client._opts.capture_content = self._saved

    def _bridged(self):
        from evalkit.core.otel_provider import _readable_to_envelope

        self.client.enqueue(_readable_to_envelope(_foreign_genai_span()))
        self.client.flush()
        spans = [e for e in self.captured if e.operation == "chat gpt-4o"]
        self.assertEqual(len(spans), 1)
        return spans[0]

    def test_foreign_message_attributes_are_dropped(self):
        self.client._opts.capture_content = False
        span = self._bridged()
        self.assertIsNone(span.prompt)
        self.assertIsNone(span.completion)
        for key in (
            "gen_ai.input.messages", "gen_ai.output.messages", "gen_ai.system_instructions",
            "gen_ai.prompt", "gen_ai.completion", "gen_ai.prompt.0.content", "gen_ai.completion.0.content",
            "gen_ai.completion.0.tool_calls.0.arguments", "llm.input_messages.0.message.content",
            "input.value", "output.value", "acme.observation.input", "acme.trace.output", "acme.entity.input",
        ):
            self.assertNotIn(key, span.attributes, key)
        flat = json.dumps(span.to_dict())
        for text in ("4242", "noted your card", "billing agent", "indexed prompt", "flattened prompt", "plain input",
                     "plain output", "prefixed observation", "prefixed trace", "prefixed entity",
                     "event-borne", "details text", "details reply", "charged", "legacy prompt"):
            self.assertNotIn(text, flat, text)

    def test_foreign_metadata_and_tool_names_are_kept(self):
        self.client._opts.capture_content = False
        span = self._bridged()
        self.assertEqual(span.model, "gpt-4o")
        for key, value in (
            ("gen_ai.request.model", "gpt-4o"), ("gen_ai.usage.input_tokens", 21),
            ("gen_ai.usage.output_tokens", 5), ("gen_ai.tool.name", "charge_card"),
            ("gen_ai.prompt.0.role", "user"), ("gen_ai.completion.0.finish_reason", "tool_calls"),
            ("gen_ai.completion.0.tool_calls.0.name", "charge_card"), ("gen_ai.prompt.name", "billing-v2"),
            ("llm.input_messages.0.message.role", "user"),
        ):
            self.assertEqual(span.attributes.get(key), value, key)
        self.assertEqual(list(span.attributes["gen_ai.request.tools"]), ["charge_card"])
        self.assertIs(span.attributes[CONTENT_STRIPPED_ATTRIBUTE], False)
        events = {e.name: (e.attributes or {}) for e in span.events}
        self.assertEqual(events["gen_ai.choice"].get("finish_reason"), "stop")
        self.assertEqual(events["gen_ai.tool.message"].get("id"), "call_1")
        self.assertEqual(list(events["gen_ai.client.inference.operation.details"]["gen_ai.response.finish_reasons"]), ["stop"])
        self.assertEqual(events["exception"].get("exception.type"), "ValueError")

    def test_foreign_content_is_kept_when_capture_is_on(self):
        span = self._bridged()
        self.assertIn("gen_ai.input.messages", span.attributes)
        self.assertIn("event-borne user text", span.prompt or "")
        events = {e.name: (e.attributes or {}) for e in span.events}
        self.assertEqual(events["gen_ai.user.message"]["content"], "event-borne user text")

    def test_streamed_refusal_text_is_content(self):
        self.client._opts.capture_content = False
        envelope = _llm_envelope()
        envelope.attributes["gen_ai.response.refusal"] = "I can't share card numbers"
        span = next(e for e in self._emit_one(envelope) if e.span_type == "llm_call")
        self.assertNotIn("gen_ai.response.refusal", span.attributes)

    def _emit_one(self, envelope):
        self.client.enqueue(envelope)
        self.client.flush()
        return self.captured


class RedactionSerializationTest(unittest.TestCase):
    def test_stripped_envelope_still_serializes(self):
        from evalkit.core.redact import strip_content

        payload = strip_content(_llm_envelope()).to_dict()
        self.assertNotIn("prompt", payload)
        self.assertNotIn("completion", payload)
        self.assertEqual(payload["tokensIn"] if "tokensIn" in payload else payload.get("tokens_in"), 11)
        json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
