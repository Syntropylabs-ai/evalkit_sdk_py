import json
import unittest

from evalkit.core.secrets import is_secret_key, scrub_secrets, scrub_string, scrub_url, scrub_value
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

URL_CASES = (
    (
        "api_key in the query of http.url is masked and page is kept",
        "http.url",
        "https://api.example.com/v1/x?api_key=AKIAIOSFODNN7EXAMPLE&page=2",
        "https://api.example.com/v1/x?api_key=***&page=2",
    ),
    (
        "userinfo is masked",
        "url.full",
        "https://user:pass@api.example.com/v1/x",
        "https://***@api.example.com/v1/x",
    ),
    (
        "a key used as the username",
        "url.full",
        f"https://{LIVE_KEY}@api.example.com/v1/x",
        "https://***@api.example.com/v1/x",
    ),
    (
        "an unrelated query is untouched",
        "http.url",
        "https://api.example.com/search?q=hello&page=2",
        "https://api.example.com/search?q=hello&page=2",
    ),
    (
        "http.target is path and query without a scheme",
        "http.target",
        "/v1/x?access_token=abc&limit=5",
        "/v1/x?access_token=***&limit=5",
    ),
    (
        "url.query is a bare query string",
        "url.query",
        "sig=abc&se=2026-01-01",
        "sig=***&se=2026-01-01",
    ),
    (
        "server.address is a plain host",
        "server.address",
        "api.example.com",
        "api.example.com",
    ),
    (
        "a scheme-prefixed string under any key is a url",
        "tool.args",
        "wss://h/socket?token=abc",
        "wss://h/socket?token=***",
    ),
    (
        "query keys match case-insensitively and after url-decoding",
        "http.url",
        "https://h/p?api%5Fkey=a&API_KEY=b&Signature=c&sig=d&stream_key=e&key_id=keep&keyword=keep",
        "https://h/p?api%5Fkey=***&API_KEY=***&Signature=***&sig=***&stream_key=***&key_id=keep&keyword=keep",
    ),
    (
        "fragment parameters are masked like query parameters",
        "http.url",
        "https://app.example.com/cb#access_token=abc&token_type=bearer",
        "https://app.example.com/cb#access_token=***&token_type=bearer",
    ),
    (
        "flags, empty params and repeated equals are kept as they were",
        "http.url",
        "https://h/p?flag&token=abc&&x=1=2",
        "https://h/p?flag&token=***&&x=1=2",
    ),
    (
        "a url inside a log or exception message loses its credential parameters",
        "log.message",
        "See https://example.com/x?token=abc&page=2 for details",
        "See https://example.com/x?token=***&page=2 for details",
    ),
    (
        "prose around a url in a message still gets pattern scrubbing",
        "log.message",
        f"See https://example.com/x?key={LIVE_KEY} for details",
        "See https://example.com/x?key=*** for details",
    ),
    (
        "a url in an exception message loses its credential parameters",
        "exception.message",
        "GET https://api.example.com/v1/q?api_key=abc123&page=2 failed: 500",
        "GET https://api.example.com/v1/q?api_key=***&page=2 failed: 500",
    ),
)

NESTED_URL_CASES = (
    (
        "a url inside a json request body",
        '{"url": "https://h/p?token=abc"}',
        {"url": "https://h/p?token=***"},
    ),
    (
        "a url inside a json array of objects",
        '[{"endpoint": "https://h/p?api_key=abc"}, {"endpoint": "https://h/q?page=2"}]',
        [{"endpoint": "https://h/p?api_key=***"}, {"endpoint": "https://h/q?page=2"}],
    ),
)

OPERATION_CASES = (
    (
        "a subscription key in the operation query is pattern scrubbed",
        f"GET /v1/traces?key={LIVE_KEY}",
        "GET /v1/traces?key=tk_live_***",
    ),
    (
        "the operation is free text, so a plain query value is left alone",
        "GET /v1/traces?key=abc123",
        "GET /v1/traces?key=abc123",
    ),
    (
        "an ordinary operation name is untouched",
        "TracingService.findById",
        "TracingService.findById",
    ),
)

PROSE_URL = "See https://example.com/x?token=abc for details"


def _nested_url_json(levels):
    leaf = {"url": "https://h/p?token=abc"}
    for _ in range(levels):
        leaf = {"a": leaf}
    return json.dumps(leaf)


def _envelope(**fields):
    return TraceEnvelope(
        trace_id="0" * 32,
        span_id="1" * 16,
        created_at="2026-09-09T00:00:00Z",
        span_type="function_call",
        **fields,
    )


def _scrub_attr(key, value):
    return scrub_secrets(_envelope(attributes={key: value})).attributes[key]


def _leaky_envelope():
    return TraceEnvelope(
        trace_id="0" * 32,
        span_id="1" * 16,
        created_at="2026-09-09T00:00:00Z",
        span_type="function_call",
        operation=f"GET /v1/traces?key={LIVE_KEY}",
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
            "http.url": f"https://api.example.com/v1/x?api_key={LIVE_KEY}&page=2",
        },
        events=[SpanEvent(
            name="log",
            timestamp="2026-09-09T00:00:00Z",
            attributes={
                "log.message": f"calling with key {LIVE_KEY}",
                "subscriptionKey": LIVE_KEY,
                "user.email": "ada@example.com",
                "http.url": f"https://api.example.com/v1/x?api_key={LIVE_KEY}",
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
                    "input_token_details", "tokenizer",
                    "gen_ai.server.time_to_first_token", "gen_ai.server.time_per_output_token",
                    "GEN_AI.SERVER.TIME_TO_FIRST_TOKEN", "time-to-first-token", "input_cost_per_token",
                    "output_cost_per_token", "cost_per_token", "gen_ai.usage.output_tokens",
                    "llm.usage.total_tokens", "llm.token_count.prompt", "gen_ai.request.max_tokens", "max_tokens",
                    "max_completion_tokens", "maxTokens"):
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


    def test_credentials_named_token_stay_masked(self):
        for key in ("token", "access_token", "refresh_token", "id_token", "auth_token", "csrf_token",
                    "session_token", "bearer_token", "api_token", "github_token", "oauth.token", "x-access-token",
                    "X-Auth-Token", "first_token", "per_user_token"):
            with self.subTest(key):
                self.assertTrue(
                    is_secret_key(key),
                    "the time-to-first-token and per-token exemption is a vocabulary, not a loophole: any other "
                    "key ending in token can still hold a credential",
                )

    def test_per_token_prices_of_every_kind_are_not_credentials(self):
        for key in ("input_cost_per_audio_token", "output_cost_per_audio_token", "output_cost_per_reasoning_token",
                    "input_cost_per_image_token", "input_cost_per_video_token", "cache_read_input_cost_per_token",
                    "input_cost_per_cached_token", "price_per_audio_token", "litellm.input_cost_per_audio_token",
                    "cost_per_1k_token", "inputCostPerAudioToken"):
            with self.subTest(key):
                self.assertFalse(
                    is_secret_key(key),
                    "a price per token is a number on the cost sheet; masking it blanks the cost the dashboard "
                    "shows and protects nothing",
                )
        for key in ("access_token", "refresh_token", "id_token", "per_user_token", "audio_token", "image_token"):
            with self.subTest(key):
                self.assertTrue(is_secret_key(key), "only prices are exempt; other *_token keys stay masked")

    def test_per_token_exemptions_are_the_same_in_both_sdks(self):
        # Exempt in TypeScript and in Python: prices named by price_per_*, any number of billed kinds
        # after "per", and the tool, read, write, creation, thinking and character kinds.
        for key in ("price_per_token", "price_per_1k_token", "output_price_per_audio_token",
                    "input_cost_per_tool_token", "cost_per_cache_read_token", "input_cost_per_cache_read_token",
                    "cache_write_cost_per_token", "per_cache_read_token", "per_cache_creation_input_token",
                    "per_input_audio_token", "per_tool_token", "per_read_token", "per_write_token",
                    "per_creation_token", "per_thinking_token", "per_character_token",
                    "input_cost_per_character_token", "time_per_output_token", "priceReadPerToken",
                    "outputCostPerThinkingToken", "costPerCacheReadToken"):
            with self.subTest(key):
                self.assertFalse(is_secret_key(key), "a price or a count per token is not a credential")
        for key in ("access_token", "refresh_token", "id_token", "per_user_token", "per_session_token",
                    "per_tenant_token", "accessToken", "perUserToken", "price_token", "tool_token"):
            with self.subTest(key):
                self.assertTrue(is_secret_key(key), "a per-something token outside the billed kinds can hold a credential")

    def test_camel_case_credential_keys_are_masked(self):
        for key in ("accessToken", "authToken", "refreshToken", "idToken", "secretKey", "clientSecret",
                    "privateKey", "sessionToken", "xApiKey"):
            with self.subTest(key):
                self.assertTrue(is_secret_key(key))
        for key in ("maxTokens", "totalTokens", "inputCostPerAudioToken", "timeToFirstToken", "promptTokenCount",
                    "cachedTokens", "publicKey", "keyId"):
            with self.subTest(key):
                self.assertFalse(is_secret_key(key))
        self.assertEqual(scrub_value({"accessToken": "ya29.x", "maxTokens": 64}), {"accessToken": "***", "maxTokens": 64})

    def test_a_price_sheet_survives_the_scrubber(self):
        sheet = {"model": "gpt-4o-audio", "input_cost_per_token": 2.5e-06, "input_cost_per_audio_token": 4e-05,
                 "output_cost_per_audio_token": 8e-05, "output_cost_per_reasoning_token": 1e-05,
                 "access_token": "ya29.secret", "refresh_token": "1//secret", "id_token": "eyJ-not-really"}
        self.assertEqual(scrub_value(sheet), {**sheet, "access_token": "***", "refresh_token": "***", "id_token": "***"})

    def test_a_bridged_span_keeps_its_timing_and_usage_numbers(self):
        envelope = TraceEnvelope(
            trace_id="t", span_id="s", span_type="llm_call", created_at="2026-10-11T00:00:00Z",
            attributes={
                "gen_ai.server.time_to_first_token": 0.42,
                "gen_ai.server.time_per_output_token": 0.012,
                "gen_ai.request.max_tokens": 256,
                "gen_ai.usage.input_tokens": 120,
                "gen_ai.usage.output_tokens": 30,
                "auth_token": "abc123",
            },
        )
        attrs = scrub_secrets(envelope).attributes
        self.assertEqual(
            attrs,
            {
                "gen_ai.server.time_to_first_token": 0.42,
                "gen_ai.server.time_per_output_token": 0.012,
                "gen_ai.request.max_tokens": 256,
                "gen_ai.usage.input_tokens": 120,
                "gen_ai.usage.output_tokens": 30,
                "auth_token": "***",
            },
            "OpenTelemetry instrumentations report time-to-first-token as gen_ai.server.time_to_first_token; "
            "masking it to *** drops the span's TTFT from every latency view",
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


class UrlScrubTest(unittest.TestCase):
    def test_url_attributes_mask_credentials_and_keep_the_rest(self):
        for name, key, raw, expected in URL_CASES:
            with self.subTest(name):
                self.assertEqual(_scrub_attr(key, raw), expected)

    def test_scrub_url_is_idempotent_and_keeps_innocent_urls_byte_for_byte(self):
        for name, key, raw, expected in URL_CASES:
            if key in ("log.message",):
                continue
            with self.subTest(name):
                self.assertEqual(_scrub_attr(key, expected), expected)

    def test_scrub_url_handles_a_url_without_a_query_or_userinfo(self):
        self.assertEqual(scrub_url("https://api.example.com/charge"), "https://api.example.com/charge")
        self.assertEqual(scrub_url("https://@h/p"), "https://@h/p")

    def test_urls_nested_in_json_attributes_are_masked(self):
        for name, raw, expected in NESTED_URL_CASES:
            with self.subTest(name):
                self.assertEqual(json.loads(_scrub_attr("request.body", raw)), expected)

    def test_the_depth_limit_still_applies_to_nested_urls(self):
        shallow = _nested_url_json(3)
        self.assertIn("token=***", _scrub_attr("request.body", shallow))
        self.assertNotIn("token=abc", _scrub_attr("request.body", shallow))
        deep = _nested_url_json(20)
        self.assertEqual(_scrub_attr("request.body", deep), deep)

    def test_a_url_in_event_attributes_is_masked(self):
        event = scrub_secrets(_leaky_envelope()).events[0]
        self.assertEqual(event.attributes["http.url"], "https://api.example.com/v1/x?api_key=***")


class EnvelopeScrubTest(unittest.TestCase):
    def test_the_operation_is_pattern_scrubbed_only(self):
        for name, raw, expected in OPERATION_CASES:
            with self.subTest(name):
                self.assertEqual(scrub_secrets(_envelope(operation=raw)).operation, expected)

    def test_a_url_inside_prompt_or_completion_prose_is_left_as_free_text(self):
        env = scrub_secrets(_envelope(
            prompt=PROSE_URL,
            completion=f"See https://example.com/x?key={LIVE_KEY} for details",
        ))
        self.assertEqual(env.prompt, PROSE_URL)
        self.assertEqual(env.completion, "See https://example.com/x?key=tk_live_*** for details")

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
        self.assertEqual(env.attributes["http.url"], "https://api.example.com/v1/x?api_key=***&page=2")
        self.assertEqual(env.operation, "GET /v1/traces?key=tk_live_***")

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
