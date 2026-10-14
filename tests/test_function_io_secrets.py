import asyncio
import json
import unittest

from evalkit.core.secrets import is_secret_name, scrub_string
from tests.helpers import fresh, reset_context

PLAIN_KEY = "FAKEPLAINKEY0123456789zyxwvu"
LIVE_KEY = "tk_live_" + "fa4e" * 12
LONG = "Background paragraph. " * 300


class _Request:
    """Stands in for a pydantic model, whose str() is `field='value' ...`."""

    def __str__(self):
        return f"model='gpt-4o-mini' api_key='{PLAIN_KEY}' max_tokens=20"


class FunctionIOSecretsTest(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self._saved = (self.client._opts.capture_content, self.client._opts.capture_secrets)
        self.client._opts.capture_content = True
        self.client._opts.capture_secrets = False

    def tearDown(self):
        self.client._opts.capture_content, self.client._opts.capture_secrets = self._saved

    def _span(self, name):
        self.client.flush()
        return next(e for e in self.captured if e.operation == name)

    def _payload(self, name):
        return json.dumps(self._span(name).to_dict())

    def test_a_key_passed_positionally_is_masked_by_its_parameter_name(self):
        @self.client.trace_function("judge")
        def judge(model, messages, api_key, temperature):
            return "ok"

        judge("gpt-4o-mini", [{"role": "user", "content": "hi"}], PLAIN_KEY, 0.3)
        span = self._span("judge")
        self.assertEqual(
            json.loads(span.attributes["function.args"]),
            ["gpt-4o-mini", [{"role": "user", "content": "hi"}], "***", 0.3],
            "a key with no recognisable shape can only be found by the parameter it was bound to",
        )
        self.assertNotIn(PLAIN_KEY, self._payload("judge"))

    def test_a_key_passed_by_keyword_is_masked(self):
        @self.client.trace_function("judge-kw")
        def judge(model, api_key=None, **extra):
            return "ok"

        judge("m", api_key=PLAIN_KEY, auth=PLAIN_KEY, note="kept")
        kwargs = json.loads(self._span("judge-kw").attributes["function.kwargs"])
        self.assertEqual(kwargs, {"api_key": "***", "auth": "***", "note": "kept"})

    def test_every_credential_like_parameter_name_is_withheld(self):
        for name in ("key", "api_key", "apiKey", "secret", "client_secret", "token", "x_service_token",
                     "accessToken", "password", "passwd", "credentials", "vertex_credentials", "auth",
                     "authorization", "cookie", "dsn", "jwt", "bearer"):
            with self.subTest(name):
                self.assertTrue(is_secret_name(name))
        for name in ("model", "messages", "max_tokens", "token_count", "tokens_in", "author", "monkey",
                     "request", "keyboard", "temperature"):
            with self.subTest(name):
                self.assertFalse(is_secret_name(name), "a count or an ordinary argument keeps its value")

    def test_a_long_argument_is_masked_before_it_is_cut(self):
        @self.client.trace_function("send")
        def send(body):
            return None

        send({"api_key": PLAIN_KEY, "note": LONG})
        span = self._span("send")
        self.assertTrue(span.attributes["function.args"].endswith("…"), "the argument passed the 4096 cap")
        self.assertNotIn(
            PLAIN_KEY, self._payload("send"),
            "cut JSON does not parse, so masking after the cut found the key only by its shape",
        )

    def test_shaped_secrets_in_a_long_argument_are_masked_too(self):
        @self.client.trace_function("prose")
        def prose(text):
            return None

        prose(
            f"subscription {LIVE_KEY}, db postgres://admin:FAKEDSNPASS@db.test/app, "
            "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJGQUtFIn0.FAKEJWTSIGNATURE0123, "
            "google AIzaFAKEKEY0123456789abcdefghijklmnopq "
            "-----BEGIN PRIVATE KEY-----\nFAKEPEMKEYDATA\n-----END PRIVATE KEY-----\n" + LONG
        )
        payload = self._payload("prose")
        for value in (LIVE_KEY, "FAKEDSNPASS", "FAKEJWTSIGNATURE", "AIzaFAKEKEY", "FAKEPEMKEYDATA"):
            with self.subTest(value):
                self.assertNotIn(value, payload)

    def test_an_object_recorded_by_its_str_is_masked(self):
        @self.client.trace_function("complete")
        def complete(request):
            return request

        complete(_Request())
        span = self._span("complete")
        self.assertIn("model='gpt-4o-mini'", span.attributes["function.args"])
        self.assertNotIn(PLAIN_KEY, self._payload("complete"))

    def test_a_result_holding_a_key_is_masked(self):
        @self.client.trace_function("body")
        def body(model):
            return {"model": model, "api_key": PLAIN_KEY, "messages": [LONG]}

        body("m")
        self.assertNotIn(PLAIN_KEY, self._payload("body"))
        self.assertIn('"model": "m"', self._span("body").attributes["function.result"])

    def test_a_function_named_for_a_credential_records_no_arguments_or_result(self):
        @self.client.trace_function("auth.presented_token")
        def presented_token(request):
            return PLAIN_KEY

        presented_token({"headers": {}})
        span = self._span("auth.presented_token")
        self.assertNotIn("function.args", span.attributes)
        self.assertNotIn(
            "function.result", span.attributes,
            "a function that returns a credential has nothing to record that is not the credential",
        )
        self.assertEqual(span.status, "OK")

    def test_the_receiver_is_left_out_of_the_arguments(self):
        class Service:
            def run(self, prompt):
                return prompt

        Service.run = self.client.trace_function("Service.run")(Service.run)
        Service().run("hello")
        self.assertEqual(json.loads(self._span("Service.run").attributes["function.args"]), ["hello"])

    def test_async_functions_mask_the_same_way(self):
        @self.client.trace_function("async-judge")
        async def judge(model, api_key):
            return {"api_key": api_key}

        asyncio.run(judge("m", PLAIN_KEY))
        span = self._span("async-judge")
        self.assertEqual(json.loads(span.attributes["function.args"]), ["m", "***"])
        self.assertNotIn(PLAIN_KEY, self._payload("async-judge"))

    def test_tool_arguments_are_masked_by_parameter_name(self):
        @self.client.trace_tool("lookup")
        def lookup(order_id, api_key):
            return "found"

        lookup("o-1", PLAIN_KEY)
        self.client.flush()
        span = next(e for e in self.captured if e.span_type == "tool_call")
        self.assertEqual(json.loads(span.attributes["tool.arguments"]), ["o-1", "***"])

    def test_the_capture_secrets_opt_out_records_the_arguments_as_given(self):
        self.client._opts.capture_secrets = True

        @self.client.trace_function("raw")
        def raw(api_key):
            return None

        raw(PLAIN_KEY)
        self.assertEqual(json.loads(self._span("raw").attributes["function.args"]), [PLAIN_KEY])


class SecretShapeTest(unittest.TestCase):
    CASES = (
        ("a quoted assignment", "model='x' api_key='abc123' n=1", "model='x' api_key='***' n=1"),
        ("a JSON member in cut text", '{"password": "hunter2", "page": 2', '{"password": "***", "page": 2'),
        ("a camel-case key in a repr", "Creds(subscriptionKey='abc')", "Creds(subscriptionKey='***')"),
        ("a DSN in prose", "connect to postgres://u:p@db.test:5432/app now",
         "connect to postgres://***@db.test:5432/app now"),
        ("a JWT", "id eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJlMTIz end", "id *** end"),
        ("a whole PEM key", "-----BEGIN RSA PRIVATE KEY-----\nAAAA\n-----END RSA PRIVATE KEY-----\nafter",
         "***\nafter"),
        ("a PEM key cut before its end", "-----BEGIN PRIVATE KEY-----\nAAAA", "***"),
        ("a Google API key", "key AIzaSyA0123456789abcdefghijklmnopqrstu", "key AIza***"),
    )
    UNTOUCHED = (
        "max_tokens: 5, tokens='12'",
        "hasToken: 'yes'",
        "token=abc",
        "monkey='banana'",
        "write to ada@example.com",
        "https://example.com/a@b",
    )

    def test_each_shape_is_masked(self):
        for name, raw, expected in self.CASES:
            with self.subTest(name):
                self.assertEqual(scrub_string(raw), expected)
                self.assertEqual(scrub_string(expected), expected, "scrubbing twice changes nothing")

    def test_text_that_only_looks_close_is_left_alone(self):
        for raw in self.UNTOUCHED:
            with self.subTest(raw):
                self.assertEqual(scrub_string(raw), raw)


if __name__ == "__main__":
    unittest.main()
