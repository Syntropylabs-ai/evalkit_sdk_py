import io
import json
import sys
import types
import unittest

from tests.helpers import fresh, reset_context


class _FakeStreamingBody:
    def __init__(self, raw_stream, content_length):
        self._raw = raw_stream
        self.content_length = content_length

    def read(self, amt=None):
        return self._raw.read() if amt is None else self._raw.read(amt)


def _body(payload):
    raw = json.dumps(payload).encode()
    return _FakeStreamingBody(io.BytesIO(raw), len(raw))


class _FakeEventStream:
    def __init__(self, events):
        self._events = list(events)
        self.closed = False

    def __iter__(self):
        for event in self._events:
            if isinstance(event, Exception):
                raise event
            yield event

    def close(self):
        self.closed = True


def _chunk(payload):
    return {"chunk": {"bytes": json.dumps(payload).encode()}}


def _install_botocore():
    botocore_mod = types.ModuleType("botocore")
    client_mod = types.ModuleType("botocore.client")
    response_mod = types.ModuleType("botocore.response")
    response_mod.StreamingBody = _FakeStreamingBody
    botocore_mod.client = client_mod
    botocore_mod.response = response_mod
    sys.modules["botocore"] = botocore_mod
    sys.modules["botocore.client"] = client_mod
    sys.modules["botocore.response"] = response_mod
    return client_mod


class BedrockOpsTest(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        _install_botocore()

    def tearDown(self):
        for name in ("botocore", "botocore.client", "botocore.response"):
            sys.modules.pop(name, None)

    def _client(self, responses, service_name="bedrock-runtime"):
        from evalkit.instrumentation.auto import _patch_bedrock_class

        class _ServiceModel:
            pass

        _ServiceModel.service_name = service_name

        class _Meta:
            service_model = _ServiceModel()

        class FakeBaseClient:
            meta = _Meta()

            def _make_api_call(self, operation_name, api_params):
                entry = responses[operation_name]
                if isinstance(entry, Exception):
                    raise entry
                return entry() if callable(entry) else entry

        _patch_bedrock_class(self.client, FakeBaseClient)
        return FakeBaseClient()

    def _spans(self, span_type="llm_call"):
        self.client.flush()
        return [e for e in self.captured if e.span_type == span_type]

    def test_invoke_model_anthropic_family(self):
        payload = {
            "content": [{"type": "text", "text": "hi there"}],
            "usage": {"input_tokens": 11, "output_tokens": 4},
            "stop_reason": "end_turn",
            "model": "claude-3-5-sonnet-20241022",
        }
        client = self._client({"InvokeModel": lambda: {"body": _body(payload)}})
        result = client._make_api_call("InvokeModel", {
            "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "body": json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 256,
                "system": "be brief",
                "messages": [{"role": "user", "content": "hello"}],
            }),
        })
        spans = self._spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertEqual(span.provider, "bedrock")
        self.assertEqual(span.operation, "InvokeModel")
        self.assertEqual(span.completion, "hi there")
        self.assertEqual(span.tokens_in, 11)
        self.assertEqual(span.tokens_out, 4)
        self.assertIn("user: hello", span.prompt)
        self.assertIn("system: be brief", span.prompt)
        self.assertEqual(span.attributes["gen_ai.operation.name"], "chat")
        self.assertEqual(span.attributes["gen_ai.request.max_tokens"], 256)
        self.assertEqual(span.attributes["gen_ai.response.finish_reasons"], ["end_turn"])
        self.assertEqual(
            span.attributes["gen_ai.response.model"],
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "an Anthropic-on-Bedrock body reports the bare Anthropic name, which identifies no Bedrock model and "
            "is not a key the Bedrock price catalog carries; only a promptRouter invokedModelId may override the "
            "requested model",
        )
        self.assertEqual(json.loads(result["body"].read().decode()), payload)

    def test_invoke_model_titan_family(self):
        client = self._client({"InvokeModel": lambda: {"body": _body({
            "inputTextTokenCount": 5,
            "results": [{"tokenCount": 9, "outputText": "because rayleigh", "completionReason": "FINISHED"}],
        })}})
        client._make_api_call("InvokeModel", {
            "modelId": "amazon.titan-text-express-v1",
            "body": json.dumps({
                "inputText": "why is the sky blue",
                "textGenerationConfig": {"maxTokenCount": 128, "temperature": 0.4},
            }),
        })
        span = self._spans()[0]
        self.assertEqual(span.completion, "because rayleigh")
        self.assertEqual(span.tokens_in, 5)
        self.assertEqual(span.tokens_out, 9)
        self.assertEqual(span.prompt, "why is the sky blue")
        self.assertEqual(span.attributes["gen_ai.operation.name"], "text_completion")
        self.assertEqual(span.attributes["gen_ai.request.max_tokens"], 128)
        self.assertEqual(span.attributes["gen_ai.request.temperature"], 0.4)
        self.assertEqual(span.attributes["gen_ai.response.finish_reasons"], ["FINISHED"])

    def test_invoke_model_llama_family_with_inference_profile_prefix(self):
        client = self._client({"InvokeModel": lambda: {"body": _body({
            "generation": "42",
            "prompt_token_count": 17,
            "generation_token_count": 2,
            "stop_reason": "stop",
        })}})
        client._make_api_call("InvokeModel", {
            "modelId": "us.meta.llama3-1-70b-instruct-v1:0",
            "body": json.dumps({"prompt": "the answer is", "max_gen_len": 64}),
        })
        span = self._spans()[0]
        self.assertEqual(span.completion, "42")
        self.assertEqual(span.tokens_in, 17)
        self.assertEqual(span.tokens_out, 2)
        self.assertEqual(span.model, "us.meta.llama3-1-70b-instruct-v1:0")
        self.assertEqual(span.attributes["gen_ai.request.max_tokens"], 64)

    def test_invoke_model_family_from_foundation_model_arn(self):
        client = self._client({"InvokeModel": lambda: {"body": _body({
            "content": [{"type": "text", "text": "arn ok"}],
            "usage": {"input_tokens": 3, "output_tokens": 2},
        })}})
        client._make_api_call("InvokeModel", {
            "modelId": "arn:aws:bedrock:us-east-1:123456789012:foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
            "body": json.dumps({"max_tokens": 32, "messages": [{"role": "user", "content": "hi"}]}),
        })
        span = self._spans()[0]
        self.assertEqual(span.completion, "arn ok")
        self.assertEqual(span.tokens_in, 3)
        self.assertEqual(span.tokens_out, 2)

    def test_invoke_model_unknown_family_does_not_raise(self):
        client = self._client({"InvokeModel": lambda: {"body": _body({"nope": {"deeply": [1, 2]}})}})
        result = client._make_api_call("InvokeModel", {
            "modelId": "somevendor.private-model-v9",
            "body": json.dumps({"prompt": "still captured", "unexpected": [1, 2, 3]}),
        })
        span = self._spans()[0]
        self.assertEqual(span.status, "OK")
        self.assertEqual(span.completion, "")
        self.assertEqual(span.tokens_in, 0)
        self.assertEqual(span.tokens_out, 0)
        self.assertEqual(span.prompt, "still captured")
        self.assertIn("gen_ai.response.body", span.attributes)
        self.assertEqual(json.loads(result["body"].read().decode()), {"nope": {"deeply": [1, 2]}})

    def test_invoke_model_malformed_bodies_do_not_raise(self):
        raw = b"<<<not json"
        client = self._client({"InvokeModel": lambda: {"body": _FakeStreamingBody(io.BytesIO(raw), len(raw))}})
        result = client._make_api_call("InvokeModel", {
            "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "body": b"not json either {{{",
        })
        span = self._spans()[0]
        self.assertEqual(span.status, "OK")
        self.assertEqual(span.completion, "")
        self.assertEqual(span.prompt, "")
        self.assertEqual(result["body"].read(), raw)

    def test_invoke_model_missing_body_key_does_not_raise(self):
        client = self._client({"InvokeModel": lambda: {"ResponseMetadata": {"HTTPStatusCode": 200}}})
        client._make_api_call("InvokeModel", {"modelId": "amazon.titan-text-express-v1"})
        span = self._spans()[0]
        self.assertEqual(span.status, "OK")
        self.assertEqual(span.tokens_in, 0)

    def test_invoke_model_nova_folds_cache_tokens_into_input(self):
        client = self._client({"InvokeModel": lambda: {"body": _body({
            "output": {"message": {"content": [{"text": "cached nova"}]}},
            "usage": {
                "inputTokens": 12, "outputTokens": 7, "totalTokens": 19,
                "cacheReadInputTokens": 1024, "cacheWriteInputTokens": 2048,
            },
            "stopReason": "end_turn",
        })}})
        client._make_api_call("InvokeModel", {
            "modelId": "amazon.nova-pro-v1:0",
            "body": json.dumps({"messages": [{"role": "user", "content": [{"text": "hi"}]}]}),
        })
        span = self._spans()[0]
        self.assertEqual(span.completion, "cached nova")
        self.assertEqual(
            span.tokens_in, 3084,
            "Bedrock excludes cache tokens from inputTokens, so reading inputTokens alone bills only the uncached remainder",
        )
        self.assertEqual(span.tokens_out, 7)
        self.assertEqual(
            span.cache_read_tokens, 1024,
            "Nova prompt caching is automatic for text prompts, so dropping the breakdown loses the discounted rate on every call",
        )
        self.assertEqual(span.cache_creation_tokens, 2048)

    def test_invoke_model_falls_back_to_token_count_headers(self):
        client = self._client({"InvokeModel": lambda: {
            "ResponseMetadata": {"HTTPHeaders": {
                "x-amzn-bedrock-input-token-count": "41",
                "x-amzn-bedrock-output-token-count": "13",
            }},
            "body": _body({"choices": [{"message": {"content": "bonjour"}, "stop_reason": "stop"}]}),
        }})
        client._make_api_call("InvokeModel", {
            "modelId": "mistral.mistral-large-2407-v1:0",
            "body": json.dumps({"messages": [{"role": "user", "content": "salut"}]}),
        })
        span = self._spans()[0]
        self.assertEqual(span.completion, "bonjour")
        self.assertEqual(
            span.tokens_in, 41,
            "Mistral, Cohere command, AI21 J2 and DeepSeek InvokeModel bodies carry no usage object, so without the header fallback these spans report zero tokens and therefore zero cost",
        )
        self.assertEqual(span.tokens_out, 13)

    def test_invoke_model_token_count_headers_survive_unparseable_body(self):
        raw = b"<<<not json"
        client = self._client({"InvokeModel": lambda: {
            "ResponseMetadata": {"HTTPHeaders": {
                "X-Amzn-Bedrock-Input-Token-Count": "7",
                "X-Amzn-Bedrock-Output-Token-Count": "2",
            }},
            "body": _FakeStreamingBody(io.BytesIO(raw), len(raw)),
        }})
        client._make_api_call("InvokeModel", {
            "modelId": "cohere.command-text-v14",
            "body": json.dumps({"prompt": "hi"}),
        })
        span = self._spans()[0]
        self.assertEqual(span.status, "OK")
        self.assertEqual(
            span.tokens_in, 7,
            "botocore lower-cases header keys but the fallback must not depend on that, and a body it cannot parse still has billable usage in the headers",
        )
        self.assertEqual(span.tokens_out, 2)

    def test_invoke_model_body_usage_wins_over_token_count_headers(self):
        client = self._client({"InvokeModel": lambda: {
            "ResponseMetadata": {"HTTPHeaders": {
                "x-amzn-bedrock-input-token-count": "999",
                "x-amzn-bedrock-output-token-count": "999",
            }},
            "body": _body({
                "content": [{"type": "text", "text": "hi"}],
                "usage": {"input_tokens": 15, "output_tokens": 25},
            }),
        }})
        client._make_api_call("InvokeModel", {
            "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "body": json.dumps({"max_tokens": 32, "messages": [{"role": "user", "content": "hi"}]}),
        })
        span = self._spans()[0]
        self.assertEqual(
            span.tokens_in, 15,
            "the headers are a fallback only, since a body that reports usage also reports the cache breakdown the headers omit",
        )
        self.assertEqual(span.tokens_out, 25)

    def test_converse_stream_tokens_and_deferred_span(self):
        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hel"}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "lo"}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {
                "usage": {
                    "inputTokens": 12, "outputTokens": 3, "totalTokens": 19,
                    "cacheReadInputTokens": 4,
                },
                "metrics": {"latencyMs": 210},
                "trace": {"promptRouter": {"invokedModelId": "anthropic.claude-3-haiku-20240307-v1:0"}},
            }},
        ]
        client = self._client({"ConverseStream": lambda: {"stream": _FakeEventStream(events)}})
        result = client._make_api_call("ConverseStream", {
            "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
        })
        self.assertEqual(len(self._spans()), 0, "stream span must not close before consumption")
        seen = list(result["stream"])
        self.assertEqual(len(seen), len(events))
        span = self._spans()[0]
        self.assertEqual(span.completion, "Hello")
        self.assertEqual(span.tokens_in, 16)
        self.assertEqual(span.tokens_out, 3)
        self.assertEqual(span.cache_read_tokens, 4)
        self.assertEqual(span.operation, "ConverseStream")
        self.assertTrue(span.attributes["gen_ai.streaming"])
        self.assertEqual(span.attributes["gen_ai.response.finish_reasons"], ["end_turn"])
        self.assertEqual(
            span.attributes["gen_ai.response.model"],
            "anthropic.claude-3-haiku-20240307-v1:0",
        )

    def test_converse_stream_closes_span_when_abandoned(self):
        events = [
            {"contentBlockDelta": {"delta": {"text": "partial"}}},
            {"metadata": {"usage": {"inputTokens": 9, "outputTokens": 1}}},
        ]
        stream = _FakeEventStream(events)
        client = self._client({"ConverseStream": lambda: {"stream": stream}})
        result = client._make_api_call("ConverseStream", {"modelId": "amazon.nova-pro-v1:0", "messages": []})
        wrapper = result["stream"]
        for _ in wrapper:
            break
        wrapper.close()
        spans = self._spans()
        self.assertEqual(len(spans), 1, "abandoned stream must emit exactly one span")
        self.assertEqual(spans[0].completion, "partial")
        self.assertEqual(spans[0].status, "OK")
        self.assertTrue(stream.closed)

    def test_converse_stream_error_event_marks_error(self):
        events = [
            {"contentBlockDelta": {"delta": {"text": "boom"}}},
            {"throttlingException": {"message": "slow down"}},
        ]
        client = self._client({"ConverseStream": lambda: {"stream": _FakeEventStream(events)}})
        result = client._make_api_call("ConverseStream", {"modelId": "amazon.nova-lite-v1:0", "messages": []})
        list(result["stream"])
        span = self._spans()[0]
        self.assertEqual(span.status, "ERROR")
        self.assertEqual(span.status_message, "throttlingException")

    def test_converse_stream_exception_during_iteration(self):
        events = [{"contentBlockDelta": {"delta": {"text": "x"}}}, RuntimeError("stream broke")]
        client = self._client({"ConverseStream": lambda: {"stream": _FakeEventStream(events)}})
        result = client._make_api_call("ConverseStream", {"modelId": "amazon.nova-lite-v1:0", "messages": []})
        with self.assertRaises(RuntimeError):
            list(result["stream"])
        spans = self._spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status, "ERROR")
        self.assertEqual(spans[0].completion, "x")

    def test_invoke_model_with_response_stream_anthropic(self):
        events = [
            _chunk({"type": "message_start", "message": {"usage": {"input_tokens": 7, "output_tokens": 0}}}),
            _chunk({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}),
            _chunk({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "!"}}),
            _chunk({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}),
            _chunk({"type": "message_stop"}),
        ]
        client = self._client({"InvokeModelWithResponseStream": lambda: {"body": _FakeEventStream(events)}})
        result = client._make_api_call("InvokeModelWithResponseStream", {
            "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "body": json.dumps({"max_tokens": 64, "messages": [{"role": "user", "content": "yo"}]}),
        })
        self.assertEqual(len(self._spans()), 0)
        self.assertEqual(len(list(result["body"])), len(events))
        span = self._spans()[0]
        self.assertEqual(span.completion, "Hi!")
        self.assertEqual(span.tokens_in, 7)
        self.assertEqual(span.tokens_out, 5)
        self.assertEqual(span.operation, "InvokeModelWithResponseStream")
        self.assertTrue(span.attributes["gen_ai.streaming"])
        self.assertEqual(span.attributes["gen_ai.response.finish_reasons"], ["end_turn"])

    def test_invoke_model_with_response_stream_titan(self):
        events = [
            _chunk({"index": 0, "inputTextTokenCount": 6, "outputText": "one ", "totalOutputTextTokenCount": 1}),
            _chunk({
                "index": 1, "inputTextTokenCount": 6, "outputText": "two",
                "totalOutputTextTokenCount": 2, "completionReason": "FINISHED",
            }),
        ]
        client = self._client({"InvokeModelWithResponseStream": lambda: {"body": _FakeEventStream(events)}})
        result = client._make_api_call("InvokeModelWithResponseStream", {
            "modelId": "amazon.titan-text-lite-v1",
            "body": json.dumps({"inputText": "count"}),
        })
        list(result["body"])
        span = self._spans()[0]
        self.assertEqual(span.completion, "one two")
        self.assertEqual(span.tokens_in, 6)
        self.assertEqual(span.tokens_out, 2)
        self.assertEqual(span.attributes["gen_ai.response.finish_reasons"], ["FINISHED"])

    def test_invoke_model_stream_invocation_metrics_fallback(self):
        events = [
            _chunk({"generation": "ignored-shape"}),
            _chunk({"amazon-bedrock-invocationMetrics": {
                "inputTokenCount": 31, "outputTokenCount": 8,
                "invocationLatency": 900, "firstByteLatency": 120,
            }}),
        ]
        client = self._client({"InvokeModelWithResponseStream": lambda: {"body": _FakeEventStream(events)}})
        result = client._make_api_call("InvokeModelWithResponseStream", {
            "modelId": "meta.llama3-1-8b-instruct-v1:0",
            "body": json.dumps({"prompt": "hello"}),
        })
        list(result["body"])
        span = self._spans()[0]
        self.assertEqual(span.tokens_in, 31)
        self.assertEqual(span.tokens_out, 8)

    def test_invoke_model_stream_invocation_metrics_cache_tokens(self):
        events = [
            _chunk({"contentBlockDelta": {"delta": {"text": "pong"}}}),
            _chunk({"amazon-bedrock-invocationMetrics": {
                "inputTokenCount": 12, "outputTokenCount": 7,
                "cacheReadInputTokenCount": 1024, "cacheWriteInputTokenCount": 2048,
            }}),
        ]
        client = self._client({"InvokeModelWithResponseStream": lambda: {"body": _FakeEventStream(events)}})
        result = client._make_api_call("InvokeModelWithResponseStream", {
            "modelId": "amazon.nova-lite-v1:0",
            "body": json.dumps({"messages": [{"role": "user", "content": [{"text": "ping"}]}]}),
        })
        list(result["body"])
        span = self._spans()[0]
        self.assertEqual(span.completion, "pong")
        self.assertEqual(
            span.tokens_in, 3084,
            "invocationMetrics reports cache counts alongside inputTokenCount and Bedrock excludes them from it, so ignoring them under-bills the prompt",
        )
        self.assertEqual(span.tokens_out, 7)
        self.assertEqual(span.cache_read_tokens, 1024)
        self.assertEqual(span.cache_creation_tokens, 2048)

    def test_invoke_model_stream_metrics_do_not_overwrite_family_tokens(self):
        events = [
            _chunk({"type": "message_start", "message": {"usage": {
                "input_tokens": 10, "output_tokens": 0,
                "cache_read_input_tokens": 100, "cache_creation_input_tokens": 5,
            }}}),
            _chunk({"type": "content_block_delta", "index": 0,
                    "delta": {"type": "text_delta", "text": "hi"}}),
            _chunk({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 7}}),
            _chunk({"type": "message_stop", "amazon-bedrock-invocationMetrics": {
                "inputTokenCount": 115, "outputTokenCount": 7,
            }}),
        ]
        client = self._client({"InvokeModelWithResponseStream": lambda: {"body": _FakeEventStream(events)}})
        result = client._make_api_call("InvokeModelWithResponseStream", {
            "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "body": json.dumps({"max_tokens": 64, "messages": [{"role": "user", "content": "yo"}]}),
        })
        list(result["body"])
        span = self._spans()[0]
        self.assertEqual(span.tokens_in, 115)
        self.assertEqual(span.tokens_out, 7)
        self.assertEqual(span.cache_read_tokens, 100)
        self.assertEqual(span.cache_creation_tokens, 5)

    def test_invoke_agent_prefers_completion_stream_key(self):
        client = self._client(
            {"InvokeAgent": lambda: {
                "sideChannel": _FakeEventStream([{"chunk": {"bytes": b"ignore me"}}]),
                "completion": _FakeEventStream([{"chunk": {"bytes": b"real answer"}}]),
            }},
            service_name="bedrock-agent-runtime",
        )
        result = client._make_api_call("InvokeAgent", {"sessionId": "s9", "inputText": "go"})
        list(result["completion"])
        spans = self._spans("function_call")
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].completion, "real answer")

    def test_invoke_model_stream_malformed_chunk_does_not_raise(self):
        events = [{"chunk": {"bytes": b"}}not json"}}, {"unexpected": {"shape": 1}}]
        client = self._client({"InvokeModelWithResponseStream": lambda: {"body": _FakeEventStream(events)}})
        result = client._make_api_call("InvokeModelWithResponseStream", {
            "modelId": "cohere.command-r-plus-v1:0",
            "body": json.dumps({"message": "hi"}),
        })
        self.assertEqual(len(list(result["body"])), 2)
        span = self._spans()[0]
        self.assertEqual(span.status, "OK")
        self.assertEqual(span.completion, "")

    def test_converse_tool_use_emits_tool_event(self):
        client = self._client({"Converse": lambda: {
            "output": {"message": {"role": "assistant", "content": [
                {"toolUse": {"toolUseId": "tu_1", "name": "get_weather", "input": {"city": "Paris"}}},
            ]}},
            "stopReason": "tool_use",
            "usage": {"inputTokens": 20, "outputTokens": 6},
        }})
        client._make_api_call("Converse", {
            "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "messages": [{"role": "user", "content": [{"text": "weather?"}]}],
            "toolConfig": {"tools": [{"toolSpec": {"name": "get_weather", "inputSchema": {"json": {}}}}]},
            "inferenceConfig": {"maxTokens": 512, "temperature": 0.2, "topP": 0.9},
        })
        span = self._spans()[0]
        self.assertEqual(span.attributes["gen_ai.response.finish_reasons"], ["tool_use"])
        self.assertEqual(span.attributes["gen_ai.request.max_tokens"], 512)
        self.assertEqual(span.attributes["gen_ai.request.top_p"], 0.9)
        self.assertIn("get_weather", span.attributes["gen_ai.tool.definitions"])
        self.assertEqual(len(span.events), 1)
        self.assertEqual(span.events[0].name, "gen_ai.tool.call")
        self.assertEqual(span.events[0].attributes["gen_ai.tool.name"], "get_weather")

    def test_converse_cache_tokens_added_to_input(self):
        client = self._client({"Converse": lambda: {
            "output": {"message": {"content": [{"text": "cached"}]}},
            "usage": {
                "inputTokens": 10, "outputTokens": 2,
                "cacheReadInputTokens": 100, "cacheWriteInputTokens": 5,
            },
        }})
        client._make_api_call("Converse", {"modelId": "amazon.nova-pro-v1:0", "messages": []})
        span = self._spans()[0]
        self.assertEqual(span.tokens_in, 115)
        self.assertEqual(span.cache_read_tokens, 100)
        self.assertEqual(span.cache_creation_tokens, 5)

    def test_invoke_agent_stream_emits_function_call_span(self):
        events = [
            {"chunk": {"bytes": b"Agent "}},
            {"trace": {"agentId": "AG1", "sessionId": "s1"}},
            {"chunk": {"bytes": b"reply"}},
        ]
        client = self._client(
            {"InvokeAgent": lambda: {"completion": _FakeEventStream(events)}},
            service_name="bedrock-agent-runtime",
        )
        result = client._make_api_call("InvokeAgent", {
            "agentId": "AG1", "agentAliasId": "AL1", "sessionId": "s1",
            "inputText": "do the thing", "enableTrace": True,
        })
        self.assertEqual(len(self._spans("function_call")), 0)
        list(result["completion"])
        spans = self._spans("function_call")
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].provider, "bedrock")
        self.assertEqual(spans[0].operation, "InvokeAgent")
        self.assertEqual(spans[0].completion, "Agent reply")
        self.assertEqual(spans[0].prompt, "do the thing")
        self.assertEqual(spans[0].attributes["gen_ai.operation.name"], "invoke_agent")
        self.assertEqual(spans[0].attributes["gen_ai.conversation.id"], "s1")
        self.assertEqual(spans[0].session_id, "s1")
        self.assertEqual(spans[0].to_dict()["sessionId"], "s1")

    def test_retrieve_emits_db_span(self):
        client = self._client(
            {"Retrieve": lambda: {"retrievalResults": [{"content": {"text": "a"}}, {"content": {"text": "b"}}]}},
            service_name="bedrock-agent-runtime",
        )
        client._make_api_call("Retrieve", {
            "knowledgeBaseId": "KB1",
            "retrievalQuery": {"text": "what is bedrock"},
        })
        spans = self._spans("db_query")
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].attributes["db.system"], "bedrock")
        self.assertEqual(spans[0].attributes["db.vector.match_count"], 2)
        self.assertEqual(spans[0].attributes["aws.bedrock.knowledge_base.id"], "KB1")
        self.assertEqual(spans[0].attributes["db.statement"], "what is bedrock")

    def test_retrieve_and_generate_emits_llm_span(self):
        client = self._client(
            {"RetrieveAndGenerate": lambda: {"output": {"text": "grounded answer"}, "sessionId": "s2"}},
            service_name="bedrock-agent-runtime",
        )
        client._make_api_call("RetrieveAndGenerate", {
            "sessionId": "s2",
            "retrieveAndGenerateConfiguration": {
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": "KB2",
                    "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
                },
            },
        })
        span = self._spans()[0]
        self.assertEqual(span.completion, "grounded answer")
        self.assertEqual(span.attributes["aws.bedrock.knowledge_base.id"], "KB2")
        self.assertEqual(span.attributes["gen_ai.operation.name"], "retrieval")
        self.assertTrue(span.model.endswith("anthropic.claude-3-haiku-20240307-v1:0"))

    def test_unhandled_agent_operation_passes_through(self):
        client = self._client({"ListFlowExecutions": {"flowExecutionSummaries": []}},
                              service_name="bedrock-agent-runtime")
        client._make_api_call("ListFlowExecutions", {})
        self.client.flush()
        self.assertEqual(len(self.captured), 0)

    def test_invoke_model_error_emits_error_span(self):
        client = self._client({"InvokeModel": RuntimeError("throttled")})
        with self.assertRaises(RuntimeError):
            client._make_api_call("InvokeModel", {
                "modelId": "amazon.titan-text-express-v1",
                "body": json.dumps({"inputText": "boom"}),
            })
        span = self._spans()[0]
        self.assertEqual(span.status, "ERROR")
        self.assertEqual(span.status_message, "RuntimeError: throttled")
        self.assertEqual(span.prompt, "boom")

    def test_explicit_patch_bedrock_uses_shared_path(self):
        from evalkit.instrumentation.llm.bedrock import patch_bedrock

        class FakeRuntime:
            def converse(self, **kwargs):
                return {
                    "output": {"message": {"content": [{"text": "explicit"}]}},
                    "usage": {"inputTokens": 4, "outputTokens": 6},
                    "stopReason": "end_turn",
                }

            def invoke_model(self, **kwargs):
                return {"body": _body({"inputTextTokenCount": 2, "results": [{"outputText": "ok", "tokenCount": 1}]})}

        runtime = FakeRuntime()
        patch_bedrock(self.client, runtime)
        runtime.converse(modelId="amazon.nova-pro-v1:0", messages=[{"role": "user", "content": [{"text": "hi"}]}])
        runtime.invoke_model(modelId="amazon.titan-text-express-v1", body=json.dumps({"inputText": "hey"}))
        spans = self._spans()
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0].operation, "Converse")
        self.assertEqual(spans[0].completion, "explicit")
        self.assertEqual(spans[1].operation, "InvokeModel")
        self.assertEqual(spans[1].completion, "ok")

    def test_explicitly_patched_client_is_not_double_traced(self):
        from evalkit.instrumentation.llm.bedrock import patch_bedrock

        client = self._client({"Converse": lambda: {
            "output": {"message": {"content": [{"text": "once"}]}},
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }})

        def converse(**kwargs):
            return client._make_api_call("Converse", kwargs)

        client.converse = converse
        patch_bedrock(self.client, client)
        client.converse(modelId="amazon.nova-pro-v1:0", messages=[])
        spans = self._spans()
        self.assertEqual(len(spans), 1, "explicit patch must not double-count the _make_api_call patch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
