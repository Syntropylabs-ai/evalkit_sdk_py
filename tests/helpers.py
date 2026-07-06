"""Shared test helpers: fake objects, client setup, context reset."""
from __future__ import annotations

import os
import sys
import types

# Ensure the SDK is importable from any working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import evalkit
from evalkit.core.context import _ambient_trace_id, _current_span

# ── Singleton client ──────────────────────────────────────────────────────────
_CLIENT = None


def fresh(capture: list):
    """Return the shared EvalKitClient with exporter replaced by in-memory capture."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = evalkit.init(
            subscription_key="tk_test",
            base_url="http://127.0.0.1:1",
            service_name="test",
            function_tracing=False,  # don't install the import hook during tests
        )
    _CLIENT._exporter.export = lambda batch: (capture.extend(batch), True)[1]
    with _CLIENT._lock:
        _CLIENT._queue.clear()
    return _CLIENT


def reset_context():
    _ambient_trace_id.set(None)
    _current_span.set(None)


# ── Google GenAI fakes ────────────────────────────────────────────────────────

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
    def __init__(self, pin, pout, thoughts=0):
        self.prompt_token_count = pin
        self.candidates_token_count = pout
        self.thoughts_token_count = thoughts


class GenAIResp:
    def __init__(self, text="", tool=None, pin=3, pout=5, thinking=None, thoughts_tokens=0):
        parts = []
        if thinking:
            parts.append(_Part(text=thinking, thought=True))
        if text:
            parts.append(_Part(text=text))
        if tool:
            parts.append(_Part(function_call=_FC(tool[0], tool[1])))
        self.candidates = [_Candidate(parts)]
        self.usage_metadata = _GenAIUsage(pin, pout, thoughts=thoughts_tokens)
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


# ── Anthropic fakes ───────────────────────────────────────────────────────────

class ABlock:
    """Mirrors the real SDK: each block type only carries its own fields
    (a ThinkingBlock has no `.text`, a TextBlock has no `.thinking`, etc.) —
    tests that rely on `hasattr(block, "text")` to distinguish block types
    need that to hold here too."""

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


# ── OpenAI fakes ──────────────────────────────────────────────────────────────

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


# ── OpenAI Responses API fakes ────────────────────────────────────────────────

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


# ── LiteLLM fakes ─────────────────────────────────────────────────────────────

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


# ── Claude Agent SDK fakes ────────────────────────────────────────────────────

class SDKTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class SDKToolUseBlock:
    def __init__(self, name, input_data, id="tu_1"):
        self.type = "tool_use"
        self.name = name
        self.input = input_data
        self.id = id


class SDKAssistantMessage:
    # __name__ must be "AssistantMessage" — the instrumentation checks type(msg).__name__
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
    # __name__ must be "ResultMessage" — the instrumentation checks type(msg).__name__
    def __init__(self, inp=10, out=20, cost=0.0006, stop="end_turn", turns=1):
        self.usage = SDKUsage(inp, out)
        self.total_cost_usd = cost
        self.stop_reason = stop
        self.num_turns = turns
        self.duration_api_ms = 480


SDKResultMessage.__name__ = "ResultMessage"
SDKResultMessage.__qualname__ = "ResultMessage"
