import sys
import types
import unittest

from tests.helpers import fresh, reset_context


def _make_fake_botocore(*, fail=False, content=None, invoke_body=None):
    botocore_mod = types.ModuleType("botocore")
    client_mod = types.ModuleType("botocore.client")

    class _ServiceModel:
        service_name = "bedrock-runtime"

    class _Meta:
        service_model = _ServiceModel()

    class FakeBaseClient:
        meta = _Meta()

        def _make_api_call(self, operation_name, api_params):
            if operation_name == "Converse":
                if fail:
                    raise RuntimeError("throttled")
                return {
                    "output": {"message": {"content": content or [{"text": "Bedrock says hello"}]}},
                    "usage": {"inputTokens": 8, "outputTokens": 15},
                }
            if operation_name == "InvokeModel" and invoke_body is not None:
                return {"body": invoke_body}
            return {}

    client_mod.BaseClient = FakeBaseClient
    botocore_mod.client = client_mod
    return botocore_mod, client_mod, FakeBaseClient


class TestBedrock(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self._injected = []

    def tearDown(self):
        for n in self._injected:
            sys.modules.pop(n, None)

    def _install(self, *, fail=False, content=None, invoke_body=None):
        botocore_mod, client_mod, FakeBC = _make_fake_botocore(
            fail=fail, content=content, invoke_body=invoke_body)
        sys.modules["botocore"] = botocore_mod
        sys.modules["botocore.client"] = client_mod
        self._injected += ["botocore", "botocore.client"]
        from evalkit.instrumentation.auto import _patch_bedrock_class
        _patch_bedrock_class(self.client, FakeBC)
        return FakeBC

    def test_converse_emits_llm_span(self):
        FakeBC = self._install()
        FakeBC()._make_api_call("Converse", {
            "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "messages": [{"role": "user", "content": [{"text": "hello"}]}],
        })
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].provider, "bedrock")
        self.assertEqual(llm[0].completion, "Bedrock says hello")
        self.assertEqual(llm[0].tokens_in, 8)
        self.assertEqual(llm[0].tokens_out, 15)
        self.assertEqual(llm[0].status, "OK")

    def test_reasoning_content_captured(self):
        FakeBC = self._install(content=[
            {"reasoningContent": {"reasoningText": {"text": "Thinking step by step..."}}},
            {"text": "Bedrock says hello"},
        ])
        FakeBC()._make_api_call("Converse", {
            "modelId": "anthropic.claude-3-7-sonnet-20250219-v1:0",
            "messages": [{"role": "user", "content": [{"text": "hello"}]}],
        })
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(llm[0].attributes.get("gen_ai.response.thinking"), "Thinking step by step...")
        self.assertNotIn("gen_ai.response.has_redacted_thinking", llm[0].attributes)

    def test_redacted_reasoning_flagged(self):
        FakeBC = self._install(content=[
            {"reasoningContent": {"redactedContent": "opaque-blob"}},
            {"text": "Bedrock says hello"},
        ])
        FakeBC()._make_api_call("Converse", {
            "modelId": "anthropic.claude-3-7-sonnet-20250219-v1:0",
            "messages": [{"role": "user", "content": [{"text": "hello"}]}],
        })
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertTrue(llm[0].attributes.get("gen_ai.response.has_redacted_thinking"))
        self.assertNotIn("gen_ai.response.thinking", llm[0].attributes)

    def test_invoke_model_keeps_the_bedrock_model_id_as_the_priced_model(self):
        FakeBC = self._install(invoke_body=(
            '{"id":"msg_1","model":"claude-3-5-sonnet-20241022",'
            '"content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn",'
            '"usage":{"input_tokens":8,"output_tokens":15}}'
        ))
        FakeBC()._make_api_call("InvokeModel", {
            "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "body": '{"messages":[{"role":"user","content":"hello"}]}',
        })
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(
            llm[0].model, "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "the envelope model is the only field the platform prices from, and Bedrock rates are keyed "
            "on the Bedrock model id; the Anthropic body reports a bare name that matches no Bedrock "
            "price row, so letting it win would silently report every InvokeModel call as free",
        )
        self.assertEqual(llm[0].provider, "bedrock")
        self.assertEqual(llm[0].tokens_in, 8)
        self.assertEqual(llm[0].tokens_out, 15)

    def test_non_converse_no_span(self):
        FakeBC = self._install()
        FakeBC()._make_api_call("DescribeInstances", {})
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 0, "non-Converse calls must not emit spans")

    def test_error_emits_error_span(self):
        FakeBC = self._install(fail=True)
        with self.assertRaises(RuntimeError):
            FakeBC()._make_api_call("Converse", {"modelId": "m", "messages": []})
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].status, "ERROR")


if __name__ == "__main__":
    unittest.main(verbosity=2)
