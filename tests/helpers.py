from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import evalkit
from evalkit.core.context import _ambient_trace_id, _current_span

_CLIENT = None


def fresh(capture: list):
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = evalkit.init(
            subscription_key="tk_test",
            base_url="http://127.0.0.1:1",
            service_name="test",
            function_tracing=False,
        )
    _CLIENT._exporter.export = lambda batch: (capture.extend(batch), True)[1]
    with _CLIENT._lock:
        _CLIENT._queue.clear()
    return _CLIENT


def reset_context():
    _ambient_trace_id.set(None)
    _current_span.set(None)


class _FC:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class _Part:
    def __init__(self, text=None, function_call=None, thought=False):
        self.text = text
        self.function_call = function_call
        self.thought = thought


class _Content:
    def __init__(self, parts):
        self.parts = parts


class _Candidate:
    def __init__(self, parts):
        self.content = _Content(parts)


class _GenAIUsage:
    def __init__(self, pin, pout, thoughts=0, cached=0, tool_use=0, total=0):
        self.prompt_token_count = pin
        self.candidates_token_count = pout
        self.thoughts_token_count = thoughts
        self.cached_content_token_count = cached
        self.tool_use_prompt_token_count = tool_use
        self.total_token_count = total


class GenAIResp:
    def __init__(self, text="", tool=None, pin=3, pout=5, thinking=None, thoughts_tokens=0,
                 cached_tokens=0, tool_use_tokens=0, total_tokens=0):
        parts = []
        if thinking:
            parts.append(_Part(text=thinking, thought=True))
        if text:
            parts.append(_Part(text=text))
        if tool:
            parts.append(_Part(function_call=_FC(tool[0], tool[1])))
        self.candidates = [_Candidate(parts)]
        self.usage_metadata = _GenAIUsage(pin, pout, thoughts=thoughts_tokens, cached=cached_tokens,
                                          tool_use=tool_use_tokens, total=total_tokens)
        self.text = text


def install_fake_genai(client, *, response=None, stream_chunks=None):
    from evalkit.instrumentation.llm import google_genai as gg

    resp = response or GenAIResp(text="hello")

    class FakeModels:
        def generate_content(self, *, model, contents, config=None, **k):
            return resp

        def generate_content_stream(self, *, model, contents, config=None, **k):
            return iter(stream_chunks or [])

    fake_mod = types.SimpleNamespace(Models=FakeModels, AsyncModels=None)
    gg._patch_sync_models(client, fake_mod)
    return FakeModels()


class ABlock:

    def __init__(self, type, text=None, name=None, id=None, input=None, thinking=None):
        self.type = type
        if type == "text":
            self.text = text
        if type == "tool_use":
            self.name = name
            self.id = id
            self.input = input
        if type == "thinking":
            self.thinking = thinking


class AUsage:
    def __init__(self, i, o):
        self.input_tokens = i
        self.output_tokens = o


class AResp:
    def __init__(self, text="hi", tool=None, thinking=None, redacted_thinking=False):
        self.content = []
        if thinking:
            self.content.append(ABlock("thinking", thinking=thinking))
        if redacted_thinking:
            self.content.append(ABlock("redacted_thinking"))
        if text:
            self.content.append(ABlock("text", text=text))
        if tool:
            self.content.append(ABlock("tool_use", name=tool[0], id="t1", input=tool[1]))
        self.usage = AUsage(3, 5)
        self.model = "claude-3-5-sonnet"


def install_fake_anthropic(client, resp):
    from evalkit.instrumentation import auto as _auto

    class FakeMessages:
        def create(self, *a, **k):
            return resp

    _auto._patch_anthropic_class_sync(client, FakeMessages)
    return FakeMessages()


class ACacheUsage(AUsage):
    def __init__(self, i, o, cache_read=0, cache_creation=0):
        super().__init__(i, o)
        self.cache_read_input_tokens = cache_read
        self.cache_creation_input_tokens = cache_creation


class FakeMessageStream:
    def __init__(self, snapshot, snapshot_error=None):
        self._snapshot = snapshot
        self._snapshot_error = snapshot_error
        self.forced_drain = False
        self.closed = False

    @property
    def current_message_snapshot(self):
        if self._snapshot_error is not None:
            raise self._snapshot_error
        return self._snapshot

    def __iter__(self):
        return iter(())

    def get_final_message(self):
        self.forced_drain = True
        return self._snapshot

    def close(self):
        self.closed = True


class FakeAsyncMessageStream(FakeMessageStream):
    def __aiter__(self):
        async def _gen():
            if False:
                yield None

        return _gen()

    async def get_final_message(self):
        self.forced_drain = True
        return self._snapshot


class FakeStreamManager:
    def __init__(self, stream, enter_error=None):
        self.stream_obj = stream
        self.enter_error = enter_error
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        if self.enter_error is not None:
            raise self.enter_error
        return self.stream_obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exited = True
        return False


class FakeAsyncStreamManager(FakeStreamManager):
    async def __aenter__(self):
        self.entered = True
        if self.enter_error is not None:
            raise self.enter_error
        return self.stream_obj

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.exited = True
        return False


class FakeStreamMessages:
    def __init__(self, manager, resp=None):
        self._manager = manager
        self._resp = resp
        self.stream_kwargs = None

    def create(self, *a, **k):
        return self._resp

    def stream(self, *a, **k):
        self.stream_kwargs = k
        return self._manager


class FakeAsyncStreamMessages(FakeStreamMessages):
    async def create(self, *a, **k):
        return self._resp


def install_fake_anthropic_streaming(client, manager, resp=None, flavor="sync"):
    from evalkit.instrumentation.llm.anthropic import patch_anthropic
    from evalkit.instrumentation.llm.anthropic_async import patch_async_anthropic
    from evalkit.instrumentation.llm.vertex_ai import patch_anthropic_vertex

    messages = (FakeAsyncStreamMessages if flavor == "async" else FakeStreamMessages)(
        manager, resp if resp is not None else AResp()
    )
    fake_client = types.SimpleNamespace(messages=messages)
    {"sync": patch_anthropic, "async": patch_async_anthropic, "vertex": patch_anthropic_vertex}[flavor](
        client, fake_client
    )
    return messages


def install_fake_anthropic_class_stream(client, manager):
    from evalkit.instrumentation import auto as _auto

    class FakeMessages:
        def stream(self, *a, **k):
            self.stream_kwargs = k
            return manager

    _auto._patch_anthropic_stream_sync(client, FakeMessages)
    return FakeMessages()


def stream_usage(**fields):
    return types.SimpleNamespace(**fields)


class OStreamToolCall:
    def __init__(self, index, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.type = "function"
        self.function = (
            types.SimpleNamespace(name=name, arguments=arguments)
            if (name is not None or arguments is not None) else None
        )


class OStreamToolDelta:
    def __init__(self, content=None, tool_calls=None, refusal=None, function_call=None):
        self.content = content
        self.tool_calls = tool_calls
        self.refusal = refusal
        self.function_call = function_call


class OStreamToolChoice:
    def __init__(self, content=None, tool_calls=None, refusal=None,
                 finish_reason=None, function_call=None):
        self.delta = OStreamToolDelta(content, tool_calls, refusal, function_call)
        self.finish_reason = finish_reason
        self.index = 0


class OStreamToolChunk:
    def __init__(self, content=None, tool_calls=None, refusal=None, finish_reason=None,
                 usage=None, id="chatcmpl-stream", model="gpt-4o-mini", function_call=None):
        self.choices = [OStreamToolChoice(content, tool_calls, refusal, finish_reason, function_call)]
        self.usage = usage
        self.id = id
        self.model = model


class AStreamMessageStart:
    def __init__(self, input_tokens=0, output_tokens=0, cache_read=None, cache_creation=None,
                 id="msg_stream", model="claude-sonnet-4-5"):
        usage_fields = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        if cache_read is not None:
            usage_fields["cache_read_input_tokens"] = cache_read
        if cache_creation is not None:
            usage_fields["cache_creation_input_tokens"] = cache_creation
        self.type = "message_start"
        self.message = types.SimpleNamespace(
            id=id, model=model, usage=types.SimpleNamespace(**usage_fields),
        )


class AStreamBlockStart:
    def __init__(self, index, type="tool_use", name=None, id=None, text=None):
        self.type = "content_block_start"
        self.index = index
        fields = {"type": type}
        if name is not None:
            fields["name"] = name
        if id is not None:
            fields["id"] = id
        if text is not None:
            fields["text"] = text
        self.content_block = types.SimpleNamespace(**fields)


class AStreamTextDelta:
    def __init__(self, text, index=0):
        self.type = "content_block_delta"
        self.index = index
        self.delta = types.SimpleNamespace(type="text_delta", text=text)


class AStreamJsonDelta:
    def __init__(self, index, partial_json):
        self.type = "content_block_delta"
        self.index = index
        self.delta = types.SimpleNamespace(type="input_json_delta", partial_json=partial_json)


class AStreamMessageDelta:
    def __init__(self, output_tokens=0, stop_reason=None, input_tokens=None,
                 cache_read=None, cache_creation=None):
        usage_fields = {"output_tokens": output_tokens}
        if input_tokens is not None:
            usage_fields["input_tokens"] = input_tokens
        if cache_read is not None:
            usage_fields["cache_read_input_tokens"] = cache_read
        if cache_creation is not None:
            usage_fields["cache_creation_input_tokens"] = cache_creation
        self.type = "message_delta"
        self.usage = types.SimpleNamespace(**usage_fields)
        self.delta = types.SimpleNamespace(stop_reason=stop_reason)


class OCompletionTokensDetails:
    def __init__(self, reasoning_tokens):
        self.reasoning_tokens = reasoning_tokens


class OUsage:
    def __init__(self, pin, pout, reasoning_tokens=0):
        self.prompt_tokens = pin
        self.completion_tokens = pout
        self.total_tokens = pin + pout
        self.completion_tokens_details = (
            OCompletionTokensDetails(reasoning_tokens) if reasoning_tokens else None
        )


class OFunction:
    def __init__(self, name, args="{}"):
        self.name = name
        self.arguments = args


class OToolCall:
    def __init__(self, name, args="{}", tid="tc_1"):
        self.id = tid
        self.function = OFunction(name, args)


class OMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
        self.role = "assistant"


class OChoice:
    def __init__(self, content="", tool_calls=None):
        self.message = OMessage(content, tool_calls)
        self.finish_reason = "stop"


class OResp:
    def __init__(self, content="The answer", tool=None, pin=5, pout=8, reasoning_tokens=0):
        tc = [OToolCall(tool[0], tool[1])] if tool else None
        self.choices = [OChoice(content, tc)]
        self.usage = OUsage(pin, pout, reasoning_tokens=reasoning_tokens)
        self.model = "gpt-4o-mini"
        self.id = "chatcmpl-test"


class ODelta:
    def __init__(self, content):
        self.content = content


class OStreamChoice:
    def __init__(self, content):
        self.delta = ODelta(content)


class OStreamChunk:
    def __init__(self, content="", usage=None):
        self.choices = [OStreamChoice(content)]
        self.usage = usage


def install_fake_openai_sync(client, resp):
    from evalkit.instrumentation.auto import _patch_openai_class_sync

    class FakeCompletions:
        def create(self, *a, **k):
            if k.get("stream"):
                return iter(resp if isinstance(resp, list) else [resp])
            return resp

    _patch_openai_class_sync(client, FakeCompletions)
    return FakeCompletions()


def install_fake_openai_async(client, resp):
    from evalkit.instrumentation.auto import _patch_openai_class_async

    class FakeAsyncCompletions:
        async def create(self, *a, **k):
            return resp

    _patch_openai_class_async(client, FakeAsyncCompletions)
    return FakeAsyncCompletions()


def install_fake_openai_responses_sync(client, resp):
    from evalkit.instrumentation.auto import _patch_openai_responses_sync

    class FakeResponses:
        def create(self, *a, **k):
            return resp

    _patch_openai_responses_sync(client, FakeResponses)
    return FakeResponses()


class ORespUsage:
    def __init__(self, pin=5, pout=8):
        self.input_tokens = pin
        self.output_tokens = pout
        self.total_tokens = pin + pout


class ORespOutput:
    def __init__(self, name=None, args="{}"):
        self.type = "function_call" if name else "text"
        self.name = name or ""
        self.arguments = args
        self.call_id = "call_1"
        self.id = "call_1"


class ORespResult:
    def __init__(self, text="Response text", tool=None, pin=5, pout=8):
        self.output_text = text
        self.output = []
        if tool:
            self.output.append(ORespOutput(name=tool[0], args=tool[1]))
        self.usage = ORespUsage(pin, pout)
        self.model = "gpt-4o-mini"
        self.id = "resp_test"


class LLMUsage:
    def __init__(self, pin=7, pout=12):
        self.prompt_tokens = pin
        self.completion_tokens = pout
        self.total_tokens = pin + pout


class LLMMessage:
    def __init__(self, content="LiteLLM response"):
        self.content = content
        self.tool_calls = None


class LLMChoice:
    def __init__(self, content="LiteLLM response"):
        self.message = LLMMessage(content)
        self.finish_reason = "stop"


class LLMResult:
    def __init__(self, content="LiteLLM response", pin=7, pout=12, cost=0.00042):
        self.choices = [LLMChoice(content)]
        self.usage = LLMUsage(pin, pout)
        self.model = "gpt-4o-mini"
        self._hidden_params = {"response_cost": cost}


class SDKTextBlock:
    def __init__(self, text):
        self.text = text


SDKTextBlock.__name__ = "TextBlock"
SDKTextBlock.__qualname__ = "TextBlock"


class SDKToolUseBlock:
    def __init__(self, name, input_data, id="tu_1"):
        self.name = name
        self.input = input_data
        self.id = id


SDKToolUseBlock.__name__ = "ToolUseBlock"
SDKToolUseBlock.__qualname__ = "ToolUseBlock"


class SDKAssistantMessage:
    def __init__(self, text="Agent reply", tool=None, model="claude-sonnet-4-6"):
        self.content = [SDKTextBlock(text)] if text else []
        if tool:
            self.content.append(SDKToolUseBlock(tool[0], tool[1]))
        self.model = model


SDKAssistantMessage.__name__ = "AssistantMessage"
SDKAssistantMessage.__qualname__ = "AssistantMessage"


class SDKUsage:
    def __init__(self, inp=10, out=20):
        self.input_tokens = inp
        self.output_tokens = out


class SDKResultMessage:
    def __init__(self, inp=10, out=20, cost=0.0006, stop="end_turn", turns=1):
        self.usage = SDKUsage(inp, out)
        self.total_cost_usd = cost
        self.stop_reason = stop
        self.num_turns = turns
        self.duration_api_ms = 480


SDKResultMessage.__name__ = "ResultMessage"
SDKResultMessage.__qualname__ = "ResultMessage"
