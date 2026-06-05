import sys
import types
import unittest

from tests.helpers import fresh, reset_context


def _make_fake_botocore(*, fail=False):
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
                    "output": {"message": {"content": [{"text": "Bedrock says hello"}]}},
                    "usage": {"inputTokens": 8, "outputTokens": 15},
                }
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

    def _install(self, *, fail=False):
        botocore_mod, client_mod, FakeBC = _make_fake_botocore(fail=fail)
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
