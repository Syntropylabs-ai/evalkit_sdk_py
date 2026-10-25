"""Big payloads: values keep their start and end, sizes are UTF-8 bytes, and no wrapper cuts a prompt."""
import json
import os
import types
import unittest
from unittest import mock

from evalkit.core.exporter import _MAX_SPAN_BYTES, _encode, _fit_event
from evalkit.core.limits import (
    DEFAULT_VALUE_LIMIT_BYTES,
    cap_envelope_values,
    cut_end,
    cut_middle,
    resolve_value_limit,
    utf8_len,
)
from evalkit.core.types import EvalKitOptions, SpanEvent, TraceEnvelope
from evalkit.instrumentation.llm._bodies import _safe_body
from tests.helpers import fresh, reset_context


def _envelope(**fields):
    return TraceEnvelope(trace_id="a" * 32, span_id="b" * 16, created_at="2026-10-26T00:00:00Z", **fields)


class CutMiddleTest(unittest.TestCase):
    def test_a_value_within_the_limit_is_returned_as_it_is(self):
        self.assertEqual(cut_middle("hello", 5), ("hello", None))

    def test_a_long_value_keeps_its_start_and_its_end_around_a_marker(self):
        text = "START" + "x" * 10_000 + "END"
        cut, original = cut_middle(text, 200)
        self.assertEqual(original, len(text))
        self.assertTrue(cut.startswith("START"))
        self.assertTrue(cut.endswith("END"), "the newest messages of a conversation are at the end")
        self.assertRegex(cut, r"…\[cut \d+ bytes\]…")
        self.assertLessEqual(utf8_len(cut), 200)

    def test_the_marker_names_exactly_the_bytes_removed(self):
        text = "a" * 1000
        cut, original = cut_middle(text, 100)
        removed = int(cut.split("[cut ")[1].split(" bytes]")[0])
        kept = utf8_len(cut) - utf8_len(f"…[cut {removed} bytes]…")
        self.assertEqual(kept + removed, original)

    def test_sizes_are_utf8_bytes_and_a_character_is_never_split(self):
        text = "é" * 1000  # two bytes each
        cut, original = cut_middle(text, 101)
        self.assertEqual(original, 2000)
        self.assertLessEqual(utf8_len(cut), 101)
        cut.encode("utf-8")  # no half character
        self.assertEqual(cut_middle("日本語" * 10, 1000), ("日本語" * 10, None))

    def test_cut_end_keeps_the_start(self):
        self.assertEqual(cut_end("abcdef", 10), "abcdef")
        out = cut_end("abcdefghijklmnop", 8)
        self.assertTrue(out.startswith("abcde") and out.endswith("…"))
        self.assertLessEqual(utf8_len(out), 8)


class ValueLimitTest(unittest.TestCase):
    def test_the_default_is_16_mib(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for name in ("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT", "OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT"):
                os.environ.pop(name, None)
            self.assertEqual(resolve_value_limit(None), 16 * 1024 * 1024)
            self.assertEqual(DEFAULT_VALUE_LIMIT_BYTES, 16 * 1024 * 1024)
            self.assertEqual(resolve_value_limit(1000), 1000)

    def test_the_opentelemetry_attribute_limit_caps_it(self):
        with mock.patch.dict(os.environ, {"OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT": "4096"}):
            self.assertEqual(resolve_value_limit(None), 4096)
            self.assertEqual(resolve_value_limit(10_000), 4096)
            self.assertEqual(resolve_value_limit(100), 100)
        with mock.patch.dict(os.environ, {"OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT": "nonsense"}):
            self.assertEqual(resolve_value_limit(None), DEFAULT_VALUE_LIMIT_BYTES)

    def test_every_value_of_a_span_is_capped_and_its_size_recorded(self):
        env = _envelope(
            prompt="p" * 5000, completion="c" * 300,
            attributes={"gen_ai.request.body": "b" * 9000, "small": "ok", "n": 3},
            events=[SpanEvent(name="tool", timestamp="t", attributes={"gen_ai.tool.call.result": "r" * 7000})],
        )
        cap_envelope_values(env, 1000)
        self.assertLessEqual(utf8_len(env.prompt), 1000)
        self.assertEqual(env.completion, "c" * 300)
        self.assertEqual(env.attributes["evalkit.truncated.prompt"], 5000)
        self.assertEqual(env.attributes["evalkit.truncated.gen_ai.request.body"], 9000)
        self.assertNotIn("evalkit.truncated.completion", env.attributes)
        self.assertEqual(env.attributes["small"], "ok")
        event_attrs = env.events[0].attributes
        self.assertLessEqual(utf8_len(event_attrs["gen_ai.tool.call.result"]), 1000)
        self.assertEqual(event_attrs["evalkit.truncated.gen_ai.tool.call.result"], 7000)


class EncodeTest(unittest.TestCase):
    def test_a_span_is_measured_and_sent_in_utf8_bytes(self):
        env = _envelope(prompt="日本語のテキスト", span_type="llm_call")
        data = _encode(env.to_dict())
        self.assertIn("日本語のテキスト".encode("utf-8"), data, "UTF-8 on the wire, not six-byte escapes")
        self.assertEqual(json.loads(data)["prompt"], "日本語のテキスト")

    def test_a_lone_surrogate_does_not_stop_the_span(self):
        data = _encode(_envelope(prompt="bad \udcff byte").to_dict())
        self.assertIsNotNone(data)
        json.loads(data)

    def test_an_oversized_span_keeps_the_start_and_end_of_its_longest_values(self):
        body = '{"messages":[{"role":"system","content":"RULES"},' + '"x",' * 3_000_000 + '{"content":"NEWEST"}]}'
        d = _envelope(span_type="llm_call", prompt="hi", attributes={"gen_ai.request.body": body}).to_dict()
        fitted = _fit_event(d)
        self.assertIsNotNone(fitted)
        kept = fitted["attributes"]["gen_ai.request.body"]
        self.assertIn("RULES", kept)
        self.assertIn("NEWEST", kept)
        self.assertEqual(fitted["attributes"]["evalkit.truncated.gen_ai.request.body"], len(body))
        self.assertLessEqual(len(json.dumps(fitted, ensure_ascii=False).encode()), _MAX_SPAN_BYTES)
        self.assertEqual(fitted["prompt"], "hi", "a short value is not cut to make room")


class NoSilentCapsTest(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_request_bodies_are_not_cut_to_their_first_two_mib(self):
        payload = {"messages": [{"role": "user", "content": "x" * (3 * 1024 * 1024)}, {"role": "user", "content": "LAST"}]}
        body = _safe_body(payload)
        self.assertIn("LAST", body)
        self.assertEqual(len(body), len(json.dumps(payload)))

    def test_the_openai_wrapper_keeps_a_prompt_longer_than_32_kib(self):
        from evalkit.instrumentation.llm.openai import patch_openai

        message = types.SimpleNamespace(content="answer")
        result = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message, finish_reason="stop")],
            usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            model="gpt-x", id="r1",
        )
        completions = types.SimpleNamespace(create=lambda **kw: result)
        fake = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
        patch_openai(self.client, fake)
        long_prompt = "q" * 100_000 + "TAIL"
        fake.chat.completions.create(model="gpt-x", messages=[{"role": "user", "content": long_prompt}])
        self.client.flush()
        [span] = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertTrue(span.prompt.endswith("TAIL"), "the 32 KiB prompt cap is gone")
        self.assertGreater(len(span.prompt), 100_000)

    def test_a_value_over_the_configured_limit_is_cut_at_enqueue(self):
        saved = self.client._opts.max_value_bytes
        self.client._opts.max_value_bytes = 2048
        try:
            self.client.enqueue(_envelope(span_type="llm_call", prompt="A" * 10_000 + "Z", sdk_version="t"))
            self.client.flush()
        finally:
            self.client._opts.max_value_bytes = saved
        [span] = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertLessEqual(utf8_len(span.prompt), 2048)
        self.assertTrue(span.prompt.endswith("Z"))
        self.assertEqual(span.attributes["evalkit.truncated.prompt"], 10_001)

    def test_the_default_max_body_bytes_is_64_kib(self):
        self.assertEqual(EvalKitOptions(subscription_key="k").max_body_bytes, 64 * 1024)
        import inspect

        import evalkit

        self.assertEqual(inspect.signature(evalkit.init).parameters["max_body_bytes"].default, 64 * 1024)


if __name__ == "__main__":
    unittest.main(verbosity=2)
