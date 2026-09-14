import json
import unittest

from evalkit.core.secrets import is_secret_key, scrub_secrets, scrub_string, scrub_value
from evalkit.core.types import SpanEvent, TraceEnvelope
from tests.helpers import fresh, reset_context

LIVE_KEY = "tk_live_9f2b7c41ade05e3b"

KEY_CASES = (
    (
        "the camel-case key the orchestrator leaked",
        {"subscriptionKey": LIVE_KEY, "tenantId": "acme"},
        {"subscriptionKey": "***", "tenantId": "acme"},
    ),
    (
        "snake case and dashed spellings of the same thing",
        {"subscription_key": LIVE_KEY, "x-subscription-key": LIVE_KEY},
        {"subscription_key": "***", "x-subscription-key": "***"},
    ),
    (
        "a credentials object nested inside function args",
        {"args": [{"credentials": {"apiKey": "abc", "password": "hunter2"}}]},
        {"args": [{"credentials": "***"}]},
    ),
    (
        "secrets nested several objects deep",
        {"a": {"b": {"c": {"client_secret": "abc"}}}},
        {"a": {"b": {"c": {"client_secret": "***"}}}},
    ),
    (
        "arrays of objects, each with its own secret",
        {"users": [{"email": "a@b.c", "token": "t1"}, {"email": "d@e.f", "token": "t2"}]},
        {"users": [{"email": "a@b.c", "token": "***"}, {"email": "d@e.f", "token": "***"}]},
    ),
    (
        "a dot-delimited segment inside an otherwise innocent key",
        {"http.request.api_key": "abc", "http.request.method": "POST"},
        {"http.request.api_key": "***", "http.request.method": "POST"},
    ),
    (
        "booleans and numbers under a secret key keep the masked shape",
        {"hasToken": True, "password": 1234, "apiKey": None},
        {"hasToken": True, "password": "***", "apiKey": "***"},
    ),
    (
        "every spelling on the list",
        {
            "api_key": "1", "api-key": "1", "apikey": "1", "secret": "1", "token": "1",
            "password": "1", "passwd": "1", "authorization": "1", "credential": "1",
            "credentials": "1", "cookie": "1", "set-cookie": "1", "private_key": "1",
            "access_key": "1", "client_secret": "1",
        },
        {
            "api_key": "***", "api-key": "***", "apikey": "***", "secret": "***", "token": "***",
            "password": "***", "passwd": "***", "authorization": "***", "credential": "***",
            "credentials": "***", "cookie": "***", "set-cookie": "***", "private_key": "***",
            "access_key": "***", "client_secret": "***",
        },
    ),
    (
        "identity and metrics are not secrets and must survive",
        {
            "email": "ada@example.com", "user_id": "u_1", "userId": "u_1",
            "gen_ai.usage.input_tokens": 11, "tokensIn": 11, "tokenizer": "cl100k",
            "secretariat": "un", "db.statement": "SELECT 1",
        },
        {
            "email": "ada@example.com", "user_id": "u_1", "userId": "u_1",
            "gen_ai.usage.input_tokens": 11, "tokensIn": 11, "tokenizer": "cl100k",
            "secretariat": "un", "db.statement": "SELECT 1",
        },
    ),
)

PATTERN_CASES = (
    ("live subscription key", LIVE_KEY, "tk_live_***"),
    ("test subscription key", "tk_test_0123456789abcdef", "tk_test_***"),
    ("openai style key", "sk-proj-Abcdefghij0123456789_-", "sk-***"),
    ("bearer token", "Bearer eyJhbGciOiJIUzI1NiJ9.abc", "Bearer ***"),
    ("lower case bearer keeps its prefix", "bearer eyJhbGciOiJIUzI1NiJ9", "bearer ***"),
    ("aws access key id", "AKIAIOSFODNN7EXAMPLE", "AKIA***"),
    ("github personal access token", "ghp_" + "a" * 20, "ghp_***"),
    ("a key inside a longer sentence", f"auth failed for {LIVE_KEY} on tenant acme",
     "auth failed for tk_live_*** on tenant acme"),
    ("a header line", f"X-Subscription-Key: {LIVE_KEY}", "X-Subscription-Key: tk_live_***"),
    ("too short to be a key", "tk_live_1", "tk_live_1"),
    ("not hex", "tk_live_zzzzzzzzzz", "tk_live_zzzzzzzzzz"),
    ("short bearer word", "Bearer x", "Bearer x"),
    ("plain prose mentioning tokens", "we spent 40 tokens on this", "we spent 40 tokens on this"),
    ("scrubbing twice is idempotent", "tk_live_***", "tk_live_***"),
)

UNTOUCHED_STRINGS = (
    "SELECT * FROM users WHERE ssn = '123-45-6789'",
    '{"not really json',
    "[1,2,3]",
    '{"charged":true}',
    "",
    "https://api.example.com/charge?id=7",
)


def _leaky_envelope():
    return TraceEnvelope(
        trace_id="0" * 32,
        span_id="1" * 16,
        created_at="2026-09-09T00:00:00Z",
        span_type="function_call",
        operation="TracingService.findById",
        status="ERROR",
        status_message=f"401 rejected for {LIVE_KEY}",
        prompt=f"the environment key is {LIVE_KEY}",
        completion="done",
        attributes={
            "function.result": json.dumps({
                "id": "env_1",
                "name": "production",
                "subscriptionKey": LIVE_KEY,
                "ownerEmail": "ada@example.com",
            }),
            "function.args": json.dumps([
                {"credentials": {"subscriptionKey": LIVE_KEY, "password": "hunter2"}},
                {"email": "ada@example.com"},
            ]),
            "http.request.headers": json.dumps({
                "X-Subscription-Key": LIVE_KEY,
                "authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.abc",
                "content-type": "application/json",
            }),
            "http.response.headers": json.dumps({"set-cookie": "session=abc123"}),
            "gen_ai.usage.input_tokens": 11,
            "url.full": "https://api.example.com/charge",
        },
        events=[SpanEvent(
            name="log",
            timestamp="2026-09-09T00:00:00Z",
            attributes={
                "log.message": f"calling with key {LIVE_KEY}",
                "subscriptionKey": LIVE_KEY,
                "user.email": "ada@example.com",
            },
        )],
    )


class SecretKeyTest(unittest.TestCase):
    def test_key_matching_is_a_whole_key_or_a_delimited_segment(self):
        for name, raw, expected in KEY_CASES:
            with self.subTest(name):
                self.assertEqual(scrub_value(raw), expected)

    def test_matching_ignores_case(self):
        for key in ("APIKEY", "Api_Key", "SubscriptionKey", "Set-Cookie", "AWS_SECRET_ACCESS_KEY",
                    "X-Auth-Token", "HTTP_PROXY_AUTHORIZATION"):
            with self.subTest(key):
                self.assertTrue(is_secret_key(key))

    def test_telemetry_vocabulary_is_not_mistaken_for_a_secret(self):
        for key in ("gen_ai.usage.input_tokens", "gen_ai.server.time_to_first_token_ms", "token_count",
                    "prompt_token_count", "x-amzn-bedrock-input-token-count", "token_usage",
                    "input_token_details", "tokenizer"):
            with self.subTest(key):
                self.assertFalse(
                    is_secret_key(key),
                    "token counts are how cost and TTFT are computed; masking them would blank out a dashboard "
                    "column to protect nothing, which reads as a broken product",
                )
        self.assertTrue(
            is_secret_key("access_token"),
            "'token' still has to be caught as the whole key or the last segment, which is where a credential "
            "actually sits",
        )


class PatternTest(unittest.TestCase):
    def test_each_pattern_keeps_its_prefix_and_drops_the_rest(self):
        for name, raw, expected in PATTERN_CASES:
            with self.subTest(name):
                self.assertEqual(scrub_string(raw), expected)

    def test_strings_that_are_not_secrets_are_left_byte_for_byte(self):
        for raw in UNTOUCHED_STRINGS:
            with self.subTest(raw):
                self.assertEqual(
                    scrub_value(raw), raw,
                    "a scrubber that reformats innocent payloads makes every diff and every dashboard string suspect",
                )


class EnvelopeScrubTest(unittest.TestCase):
    def test_json_string_attributes_are_parsed_scrubbed_and_reserialized(self):
        env = scrub_secrets(_leaky_envelope())
        result = json.loads(env.attributes["function.result"])
        self.assertEqual(result["subscriptionKey"], "***")
        self.assertEqual(
            result["ownerEmail"], "ada@example.com",
            "identity is what the Users view is for; masking the email would be masking the product",
        )
        args = json.loads(env.attributes["function.args"])
        self.assertEqual(args[0]["credentials"], "***")
        self.assertEqual(args[1]["email"], "ada@example.com")

    def test_header_maps_are_masked_through_the_header_list(self):
        env = scrub_secrets(_leaky_envelope())
        req = json.loads(env.attributes["http.request.headers"])
        self.assertEqual(req["X-Subscription-Key"], "***")
        self.assertEqual(req["authorization"], "***")
        self.assertEqual(req["content-type"], "application/json")
        self.assertEqual(json.loads(env.attributes["http.response.headers"])["set-cookie"], "***")

    def test_prompt_completion_and_status_message_are_pattern_scrubbed(self):
        env = scrub_secrets(_leaky_envelope())
        self.assertEqual(env.prompt, "the environment key is tk_live_***")
        self.assertEqual(env.completion, "done")
        self.assertEqual(env.status_message, "401 rejected for tk_live_***")

    def test_event_attributes_are_scrubbed_too(self):
        event = scrub_secrets(_leaky_envelope()).events[0]
        self.assertEqual(event.attributes["subscriptionKey"], "***")
        self.assertEqual(event.attributes["log.message"], "calling with key tk_live_***")
        self.assertEqual(event.attributes["user.email"], "ada@example.com")

    def test_metrics_and_identifiers_survive(self):
        env = scrub_secrets(_leaky_envelope())
        self.assertEqual(env.attributes["gen_ai.usage.input_tokens"], 11)
        self.assertEqual(env.attributes["url.full"], "https://api.example.com/charge")

    def test_no_live_key_survives_anywhere_in_the_payload(self):
        payload = json.dumps(scrub_secrets(_leaky_envelope()).to_dict())
        self.assertNotIn(
            LIVE_KEY, payload,
            "one clear-text key anywhere in the envelope is the whole bug; the assertion is deliberately on the "
            "serialized payload rather than on the fields we remembered to check",
        )
        self.assertIn("ada@example.com", payload)


class ExporterPathTest(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self._saved = (
            self.client._opts.capture_content,
            self.client._opts.capture_secrets,
            self.client._opts.mask,
        )

    def tearDown(self):
        (
            self.client._opts.capture_content,
            self.client._opts.capture_secrets,
            self.client._opts.mask,
        ) = self._saved

    def _emit(self, envelope):
        self.client.enqueue(envelope)
        self.client.flush()
        return self.captured

    def test_an_exported_envelope_is_scrubbed_by_default(self):
        span = self._emit(_leaky_envelope())[0]
        self.assertNotIn(LIVE_KEY, json.dumps(span.to_dict()))
        self.assertEqual(json.loads(span.attributes["function.result"])["subscriptionKey"], "***")

    def test_scrubbing_runs_even_though_content_capture_is_on(self):
        self.assertTrue(self.client._opts.capture_content)
        span = self._emit(_leaky_envelope())[0]
        self.assertIn(
            "function.result", span.attributes,
            "the scrubber is not a content switch: the payload stays, only the credential inside it goes",
        )
        self.assertNotIn(LIVE_KEY, json.dumps(span.to_dict()))

    def test_the_opt_out_ships_the_secret_untouched(self):
        self.client._opts.capture_secrets = True
        span = self._emit(_leaky_envelope())[0]
        self.assertEqual(
            json.loads(span.attributes["function.result"])["subscriptionKey"], LIVE_KEY,
            "the opt-out exists for people debugging their own auth, so it has to be a real bypass and not a softer "
            "mask",
        )

    def test_the_mask_hook_sees_the_already_scrubbed_envelope(self):
        seen = []
        self.client._opts.mask = lambda envelope: (seen.append(envelope.prompt), envelope)[1]
        self._emit(_leaky_envelope())
        self.assertEqual(
            seen, ["the environment key is tk_live_***"],
            "a customer mask hook is user code; handing it the clear-text key would leak the credential into "
            "whatever that hook logs",
        )

    def test_stripping_content_and_scrubbing_compose(self):
        self.client._opts.capture_content = False
        span = self._emit(_leaky_envelope())[0]
        self.assertNotIn("function.result", span.attributes)
        self.assertNotIn(LIVE_KEY, json.dumps(span.to_dict()))


if __name__ == "__main__":
    unittest.main()
