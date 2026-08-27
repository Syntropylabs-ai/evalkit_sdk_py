import json
import sys
import types
import unittest

from tests.helpers import fresh, reset_context


class _FakeAioStreamingBody:
    def __init__(self, raw, backend="aiohttp"):
        self._raw = raw
        self._pos = 0
        self.content_length = len(raw)
        self.closed = False
        self._backend = backend

    async def read(self, amt=None):
        if amt is None:
            chunk = self._raw[self._pos:]
            self._pos = len(self._raw)
            return chunk
        chunk = self._raw[self._pos:self._pos + amt]
        self._pos += len(chunk)
        return chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True
        return False

    def __iter__(self):
        raise NotImplementedError("use __aiter__")

    def close(self):
        if self._backend != "aiohttp":
            raise AssertionError("httpx bodies only expose the async closers")
        self.closed = True

    async def aclose(self):
        self.closed = True


def _abody(payload, backend="aiohttp"):
    return _FakeAioStreamingBody(json.dumps(payload).encode(), backend=backend)


class _FakeAioEventStream:
    def __init__(self, events):
        self._events = list(events)
        self.closed = False

    def __iter__(self):
        raise NotImplementedError("AioEventStream is async-only")

    async def __aiter__(self):
        for event in self._events:
            if isinstance(event, Exception):
                raise event
            yield event

    def close(self):
        self.closed = True


def _chunk(payload):
    return {"chunk": {"bytes": json.dumps(payload).encode()}}


class BedrockAioTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def _client(self, responses, service_name="bedrock-runtime"):
        from evalkit.instrumentation.auto import _patch_aio_bedrock_class

        class _ServiceModel:
            pass

        _ServiceModel.service_name = service_name

        class _Meta:
            service_model = _ServiceModel()

        class FakeAioBaseClient:
            meta = _Meta()

            async def _make_api_call(self, operation_name, api_params):
                entry = responses[operation_name]
                if isinstance(entry, Exception):
                    raise entry
                return entry() if callable(entry) else entry

        _patch_aio_bedrock_class(self.client, FakeAioBaseClient)
        return FakeAioBaseClient()

    def _spans(self, span_type="llm_call"):
        self.client.flush()
        return [e for e in self.captured if e.span_type == span_type]

    async def test_invoke_model_emits_span_and_leaves_body_readable(self):
        payload = {
            "content": [{"type": "text", "text": "async hi"}],
            "usage": {"input_tokens": 11, "output_tokens": 4},
            "stop_reason": "end_turn",
        }
        client = self._client({"InvokeModel": lambda: {"body": _abody(payload)}})
        result = await client._make_api_call("InvokeModel", {
            "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "body": json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 256,
                "system": "be brief",
                "messages": [{"role": "user", "content": "hello"}],
            }),
        })
        spans = self._spans()
        self.assertEqual(
            len(spans), 1,
            "aiobotocore overrides _make_api_call with a coroutine, so the sync botocore patch never sees the call "
            "and async Bedrock traffic used to produce no llm_call span at all",
        )
        span = spans[0]
        self.assertEqual(span.provider, "bedrock")
        self.assertEqual(span.operation, "InvokeModel")
        self.assertEqual(span.completion, "async hi")
        self.assertEqual(span.tokens_in, 11)
        self.assertEqual(span.tokens_out, 4)
        self.assertIn("user: hello", span.prompt)
        self.assertIn("system: be brief", span.prompt)
        self.assertEqual(span.attributes["gen_ai.request.max_tokens"], 256)
        self.assertEqual(
            json.loads((await result["body"].read()).decode()), payload,
            "tracing consumes the stream, so the replacement body must replay the full payload to the caller",
        )

    async def test_replay_body_replays_in_chunks_and_iterates(self):
        payload = {"content": [{"type": "text", "text": "chunked"}], "usage": {"input_tokens": 1, "output_tokens": 1}}
        raw = json.dumps(payload).encode()
        client = self._client({"InvokeModel": lambda: {"body": _abody(payload)}})
        result = await client._make_api_call("InvokeModel", {
            "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
            "body": json.dumps({"max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}),
        })
        body = result["body"]
        first = await body.read(4)
        self.assertEqual(first, raw[:4], "a partial read must honour the amt argument like the real StreamingBody")
        self.assertEqual(await body.read(), raw[4:])
        self.assertEqual(body.tell(), len(raw))
        self.assertEqual(
            b"".join([c async for c in body.iter_chunks(8)]), b"",
            "iter_chunks resumes from the cursor, so a fully read body yields nothing more",
        )

        chunked = await client._make_api_call("InvokeModel", {
            "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
            "body": json.dumps({"max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}),
        })
        self.assertEqual(b"".join([c async for c in chunked["body"].iter_chunks(8)]), raw)

        iterated = await client._make_api_call("InvokeModel", {
            "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
            "body": json.dumps({"max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}),
        })
        self.assertEqual(b"".join([c async for c in iterated["body"]]), raw)

    async def test_replay_body_forwards_unknown_attributes(self):
        payload = {"content": [], "usage": {"input_tokens": 1, "output_tokens": 0}}
        upstream = _abody(payload)
        client = self._client({"InvokeModel": lambda: {"body": upstream}})
        result = await client._make_api_call("InvokeModel", {
            "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
            "body": json.dumps({"max_tokens": 8, "messages": []}),
        })
        self.assertEqual(
            result["body"].content_length, upstream.content_length,
            "attributes the proxy does not define must reach the wrapped stream rather than raise",
        )
        async with result["body"]:
            pass
        self.assertTrue(upstream.closed, "closing the proxy has to close the stream it took over")

    async def test_replay_body_closes_httpx_backend_without_sync_close(self):
        payload = {"content": [], "usage": {"input_tokens": 1, "output_tokens": 0}}
        upstream = _abody(payload, backend="httpx")
        client = self._client({"InvokeModel": lambda: {"body": upstream}})
        result = await client._make_api_call("InvokeModel", {
            "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
            "body": json.dumps({"max_tokens": 8, "messages": []}),
        })
        await result["body"].__aexit__(None, None, None)
        self.assertTrue(
            upstream.closed,
            "the httpx-backed body only has async close/aclose, so the closer must be awaited rather than called",
        )

    async def test_titan_family_unary(self):
        client = self._client({"InvokeModel": lambda: {"body": _abody({
            "inputTextTokenCount": 5,
            "results": [{"tokenCount": 9, "outputText": "because rayleigh", "completionReason": "FINISHED"}],
        })}})
        await client._make_api_call("InvokeModel", {
            "modelId": "amazon.titan-text-express-v1",
            "body": json.dumps({"inputText": "why is the sky blue", "textGenerationConfig": {"maxTokenCount": 128}}),
        })
        span = self._spans()[0]
        self.assertEqual(span.completion, "because rayleigh")
        self.assertEqual(span.tokens_in, 5)
        self.assertEqual(span.tokens_out, 9)
        self.assertEqual(span.attributes["gen_ai.operation.name"], "text_completion")

    async def test_converse_unary_is_traced_without_body_stream(self):
        client = self._client({"Converse": lambda: {
            "output": {"message": {"role": "assistant", "content": [{"text": "converse ok"}]}},
            "usage": {"inputTokens": 7, "outputTokens": 2, "cacheReadInputTokens": 3},
            "stopReason": "end_turn",
        }})
        await client._make_api_call("Converse", {
            "modelId": "amazon.nova-pro-v1:0",
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
        })
        span = self._spans()[0]
        self.assertEqual(span.completion, "converse ok")
        self.assertEqual(
            span.tokens_in, 10,
            "Converse reports inputTokens exclusive of cache reads, and the platform cost engine subtracts cache "
            "tokens back out of tokens_in, so the cache read has to be folded in here",
        )
        self.assertEqual(span.tokens_out, 2)
        self.assertEqual(span.cache_read_tokens, 3)

    async def test_converse_stream_defers_span_until_consumed(self):
        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hel"}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "lo"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {
                "usage": {"inputTokens": 12, "outputTokens": 3, "cacheReadInputTokens": 4},
                "metrics": {"latencyMs": 210},
            }},
        ]
        client = self._client({"ConverseStream": lambda: {"stream": _FakeAioEventStream(events)}})
        result = await client._make_api_call("ConverseStream", {
            "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
        })
        self.assertEqual(len(self._spans()), 0, "stream span must not close before consumption")
        seen = [event async for event in result["stream"]]
        self.assertEqual(len(seen), len(events), "the wrapper must forward every event unchanged")
        span = self._spans()[0]
        self.assertEqual(span.completion, "Hello")
        self.assertEqual(span.tokens_in, 16)
        self.assertEqual(span.tokens_out, 3)
        self.assertEqual(span.cache_read_tokens, 4)
        self.assertTrue(span.attributes["gen_ai.streaming"])
        self.assertEqual(span.attributes["gen_ai.response.finish_reasons"], ["end_turn"])

    async def test_converse_stream_closes_span_when_abandoned(self):
        events = [
            {"contentBlockDelta": {"delta": {"text": "partial"}}},
            {"metadata": {"usage": {"inputTokens": 9, "outputTokens": 1}}},
        ]
        stream = _FakeAioEventStream(events)
        client = self._client({"ConverseStream": lambda: {"stream": stream}})
        result = await client._make_api_call("ConverseStream", {"modelId": "amazon.nova-pro-v1:0", "messages": []})
        wrapper = result["stream"]
        async for _ in wrapper:
            break
        await wrapper.aclose()
        spans = self._spans()
        self.assertEqual(
            len(spans), 1,
            "an abandoned async stream must still flush a span, otherwise the call disappears from the trace",
        )
        self.assertEqual(spans[0].completion, "partial")
        self.assertTrue(stream.closed)

    async def test_converse_stream_exception_during_iteration_marks_error(self):
        events = [{"contentBlockDelta": {"delta": {"text": "x"}}}, RuntimeError("stream broke")]
        client = self._client({"ConverseStream": lambda: {"stream": _FakeAioEventStream(events)}})
        result = await client._make_api_call("ConverseStream", {"modelId": "amazon.nova-pro-v1:0", "messages": []})
        with self.assertRaises(RuntimeError):
            async for _ in result["stream"]:
                pass
        span = self._spans()[0]
        self.assertEqual(span.status, "ERROR")
        self.assertEqual(span.completion, "x")

    async def test_invoke_model_with_response_stream(self):
        events = [
            _chunk({"type": "message_start", "message": {"usage": {"input_tokens": 6, "output_tokens": 0}}}),
            _chunk({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "as"}}),
            _chunk({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "ync"}}),
            _chunk({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}}),
        ]
        client = self._client({"InvokeModelWithResponseStream": lambda: {"body": _FakeAioEventStream(events)}})
        result = await client._make_api_call("InvokeModelWithResponseStream", {
            "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
            "body": json.dumps({"max_tokens": 32, "messages": [{"role": "user", "content": "hi"}]}),
        })
        seen = [event async for event in result["body"]]
        self.assertEqual(len(seen), len(events))
        span = self._spans()[0]
        self.assertEqual(span.completion, "async")
        self.assertEqual(span.tokens_in, 6)
        self.assertEqual(span.tokens_out, 2)
        self.assertEqual(span.attributes["gen_ai.response.finish_reasons"], ["end_turn"])

    async def test_retrieve_emits_db_span(self):
        client = self._client(
            {"Retrieve": lambda: {"retrievalResults": [{"content": {"text": "a"}}, {"content": {"text": "b"}}]}},
            service_name="bedrock-agent-runtime",
        )
        await client._make_api_call("Retrieve", {
            "knowledgeBaseId": "KB1",
            "retrievalQuery": {"text": "what is bedrock"},
        })
        spans = self._spans("db_query")
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].attributes["db.system"], "bedrock")
        self.assertEqual(spans[0].attributes["db.vector.match_count"], 2)
        self.assertEqual(spans[0].attributes["aws.bedrock.knowledge_base.id"], "KB1")
        self.assertEqual(spans[0].attributes["db.statement"], "what is bedrock")

    async def test_retrieve_error_marks_span_and_propagates(self):
        client = self._client({"Retrieve": RuntimeError("kb down")}, service_name="bedrock-agent-runtime")
        with self.assertRaises(RuntimeError):
            await client._make_api_call("Retrieve", {"retrievalQuery": {"text": "q"}})
        spans = self._spans("db_query")
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status, "ERROR")

    async def test_api_error_marks_span_and_propagates(self):
        client = self._client({"InvokeModel": RuntimeError("throttled")})
        with self.assertRaises(RuntimeError):
            await client._make_api_call("InvokeModel", {
                "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
                "body": json.dumps({"max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}),
            })
        span = self._spans()[0]
        self.assertEqual(span.status, "ERROR")
        self.assertEqual(span.status_message, "throttled")
        self.assertEqual(span.attributes["exception.type"], "RuntimeError")

    async def test_non_bedrock_service_is_untouched(self):
        client = self._client({"GetObject": lambda: {"Body": _abody({"not": "bedrock"})}}, service_name="s3")
        result = await client._make_api_call("GetObject", {"Bucket": "b", "Key": "k"})
        self.assertEqual(len(self._spans()), 0)
        self.assertEqual(
            json.loads((await result["Body"].read()).decode()), {"not": "bedrock"},
            "a non-Bedrock call must not have its body consumed or replaced",
        )

    async def test_patch_is_idempotent(self):
        from evalkit.instrumentation.auto import _patch_aio_bedrock_class

        payload = {"content": [{"type": "text", "text": "once"}], "usage": {"input_tokens": 2, "output_tokens": 1}}

        class _ServiceModel:
            service_name = "bedrock-runtime"

        class _Meta:
            service_model = _ServiceModel()

        class FakeAioBaseClient:
            meta = _Meta()

            async def _make_api_call(self, operation_name, api_params):
                return {"body": _abody(payload)}

        _patch_aio_bedrock_class(self.client, FakeAioBaseClient)
        _patch_aio_bedrock_class(self.client, FakeAioBaseClient)
        await FakeAioBaseClient()._make_api_call("InvokeModel", {
            "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
            "body": json.dumps({"max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}),
        })
        self.assertEqual(
            len(self._spans()), 1,
            "double instrumentation would double-count tokens in ClickHouse, which has no dedupe on insert",
        )

    async def test_patch_bedrock_client_wraps_coroutine_methods(self):
        from evalkit.instrumentation.llm.bedrock import patch_bedrock

        payload = {"content": [{"type": "text", "text": "per client"}], "usage": {"input_tokens": 3, "output_tokens": 1}}

        class FakeAsyncRuntime:
            async def invoke_model(self, **kwargs):
                return {"body": _abody(payload)}

        runtime = FakeAsyncRuntime()
        patch_bedrock(self.client, runtime)
        result = await runtime.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({"max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}),
        )
        span = self._spans()[0]
        self.assertEqual(span.completion, "per client")
        self.assertEqual(span.tokens_in, 3)
        self.assertEqual(
            json.loads((await result["body"].read()).decode()), payload,
            "the per-client helper has to await the underlying coroutine rather than trace the coroutine object",
        )

    def test_try_patch_aiobotocore_wires_the_real_entry_point(self):
        from evalkit.instrumentation.auto import _try_patch_aiobotocore

        class AioBaseClient:
            async def _make_api_call(self, operation_name, api_params):
                return {}

        module = types.ModuleType("aiobotocore")
        client_mod = types.ModuleType("aiobotocore.client")
        client_mod.AioBaseClient = AioBaseClient
        module.client = client_mod
        sys.modules["aiobotocore"] = module
        sys.modules["aiobotocore.client"] = client_mod
        try:
            _try_patch_aiobotocore(self.client)
        finally:
            sys.modules.pop("aiobotocore", None)
            sys.modules.pop("aiobotocore.client", None)
        self.assertTrue(
            getattr(AioBaseClient._make_api_call, "_evalkit_bedrock", False),
            "auto instrumentation must patch AioBaseClient, the class aiobotocore actually dispatches through",
        )

    def test_try_patch_aiobotocore_is_a_noop_without_the_package(self):
        from evalkit.instrumentation.auto import _try_patch_aiobotocore

        for name in ("aiobotocore", "aiobotocore.client"):
            self.assertNotIn(name, sys.modules)
        _try_patch_aiobotocore(self.client)


if __name__ == "__main__":
    unittest.main()
