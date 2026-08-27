from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from .anthropic import _extract_thinking

_PROFILE_PREFIXES = ("us.", "eu.", "jp.", "apac.", "global.")

HEADER_INPUT_TOKENS = "x-amzn-bedrock-input-token-count"
HEADER_OUTPUT_TOKENS = "x-amzn-bedrock-output-token-count"

CONVERSE_STREAM_ERRORS = (
    "internalServerException",
    "modelStreamErrorException",
    "serviceUnavailableException",
    "throttlingException",
    "validationException",
)

INVOKE_STREAM_ERRORS = (
    "internalServerException",
    "modelStreamErrorException",
    "modelTimeoutException",
    "serviceUnavailableException",
    "throttlingException",
    "validationException",
)

AGENT_STREAM_ERRORS = (
    "accessDeniedException",
    "badGatewayException",
    "conflictException",
    "dependencyFailedException",
    "internalServerException",
    "resourceNotFoundException",
    "serviceQuotaExceededException",
    "throttlingException",
    "validationException",
)


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _as_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _parsed_int(value: Any) -> int:
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return 0
        return parsed if parsed > 0 else 0
    return _as_int(value)


def _dig(obj: Any, *path: Any) -> Any:
    cur = obj
    for key in path:
        if isinstance(key, int):
            if not isinstance(cur, (list, tuple)) or len(cur) <= key:
                return None
            cur = cur[key]
        else:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        if cur is None:
            return None
    return cur


def base_model_id(model_id: Any) -> str:
    mid = model_id if isinstance(model_id, str) else ""
    if mid.startswith("arn:"):
        parts = mid.split(":")
        segment = parts[5].split("/")[-1] if len(parts) > 5 else ""
        mid = segment if "." in segment else ""
    for prefix in _PROFILE_PREFIXES:
        if mid.startswith(prefix):
            return mid[len(prefix):]
    return mid


def load_json(raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = bytes(raw).decode("utf-8", errors="replace")
        except Exception:
            return None
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


class BedrockAcc:
    __slots__ = (
        "text_parts", "thinking_parts", "tokens_in", "tokens_out",
        "cache_read", "cache_write", "finish_reasons", "tools", "tool_buf",
        "has_redacted_thinking", "response_model", "error", "start_ms", "ttft_ms",
    )

    def __init__(self, start_ms: Optional[float] = None) -> None:
        self.text_parts: List[str] = []
        self.thinking_parts: List[str] = []
        self.tokens_in = 0
        self.tokens_out = 0
        self.cache_read = 0
        self.cache_write = 0
        self.finish_reasons: List[str] = []
        self.tools: List[Dict[str, Any]] = []
        self.tool_buf: Dict[Any, Dict[str, Any]] = {}
        self.has_redacted_thinking = False
        self.response_model: Optional[str] = None
        self.error: Optional[str] = None
        self.start_ms = start_ms
        self.ttft_ms: Optional[int] = None

    def mark_first(self) -> None:
        if self.ttft_ms is None and self.start_ms is not None:
            self.ttft_ms = int(time.time() * 1000 - self.start_ms)

    def add_text(self, value: Any) -> None:
        text = _as_text(value)
        if text:
            self.mark_first()
            self.text_parts.append(text)

    def add_thinking(self, value: Any) -> None:
        text = _as_text(value)
        if text:
            self.mark_first()
            self.thinking_parts.append(text)

    def add_finish(self, value: Any) -> None:
        reason = _as_text(value)
        if reason and reason not in self.finish_reasons:
            self.finish_reasons.append(reason)

    def add_tool(self, name: Any, call_id: Any, arguments: Any) -> None:
        if not name:
            return
        self.tools.append({"name": name, "id": call_id or "", "arguments": arguments})

    def set_tokens_in(self, value: Any) -> None:
        count = _as_int(value)
        if count:
            self.tokens_in = count

    def set_tokens_out(self, value: Any) -> None:
        count = _as_int(value)
        if count:
            self.tokens_out = count

    def completion(self) -> str:
        return "".join(self.text_parts)

    def thinking(self) -> str:
        return "".join(self.thinking_parts)


def _usage_converse(usage: Any, acc: BedrockAcc) -> None:
    u = _as_dict(usage)
    acc.set_tokens_in(u.get("inputTokens"))
    acc.set_tokens_out(u.get("outputTokens"))
    read = _as_int(u.get("cacheReadInputTokens"))
    write = _as_int(u.get("cacheWriteInputTokens"))
    if read:
        acc.cache_read = read
    if write:
        acc.cache_write = write


def _content_blocks_text(content: Any) -> str:
    parts: List[str] = []
    for block in _as_list(content):
        text = _as_text(_as_dict(block).get("text"))
        if text:
            parts.append(text)
    return "".join(parts)


def converse_prompt(params: Dict[str, Any]) -> str:
    parts: List[str] = []
    for entry in _as_list(params.get("system")):
        text = _as_text(_as_dict(entry).get("text"))
        if text:
            parts.append("system: " + text)
    for message in _as_list(params.get("messages")):
        m = _as_dict(message)
        parts.append("%s: %s" % (_as_text(m.get("role")), _content_blocks_text(m.get("content"))))
    return "\n".join(parts)


def converse_params(params: Dict[str, Any]) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    config = _as_dict(params.get("inferenceConfig"))
    for key, attr in (
        ("maxTokens", "gen_ai.request.max_tokens"),
        ("temperature", "gen_ai.request.temperature"),
        ("topP", "gen_ai.request.top_p"),
        ("stopSequences", "gen_ai.request.stop_sequences"),
    ):
        value = config.get(key)
        if value is not None:
            attrs[attr] = value
    return attrs


def converse_tool_specs(params: Dict[str, Any]) -> List[Any]:
    specs: List[Any] = []
    for tool in _as_list(_dig(params, "toolConfig", "tools")):
        spec = _as_dict(tool).get("toolSpec")
        if spec is not None:
            specs.append(spec)
    return specs


def converse_response(result: Any, acc: BedrockAcc) -> None:
    body = _as_dict(result)
    content = _dig(body, "output", "message", "content")
    for block in _as_list(content):
        b = _as_dict(block)
        acc.add_text(b.get("text"))
        tool_use = _as_dict(b.get("toolUse"))
        if tool_use:
            acc.add_tool(tool_use.get("name"), tool_use.get("toolUseId"), tool_use.get("input"))
    reasoning, redacted = extract_reasoning(_as_list(content))
    if reasoning:
        acc.add_thinking(reasoning)
    if redacted:
        acc.has_redacted_thinking = True
    _usage_converse(body.get("usage"), acc)
    acc.add_finish(body.get("stopReason"))
    invoked = _dig(body, "trace", "promptRouter", "invokedModelId")
    if invoked:
        acc.response_model = str(invoked)


def converse_stream_event(event: Any, acc: BedrockAcc) -> None:
    chunk = _as_dict(event)
    for member in CONVERSE_STREAM_ERRORS:
        if member in chunk:
            acc.error = member
            return
    delta = _as_dict(_dig(chunk, "contentBlockDelta", "delta"))
    if delta:
        acc.add_text(delta.get("text"))
    if "messageStop" in chunk:
        acc.add_finish(_dig(chunk, "messageStop", "stopReason"))
    metadata = _as_dict(chunk.get("metadata"))
    if metadata:
        _usage_converse(metadata.get("usage"), acc)
        invoked = _dig(metadata, "trace", "promptRouter", "invokedModelId")
        if invoked:
            acc.response_model = str(invoked)


def agent_stream_event(event: Any, acc: BedrockAcc) -> None:
    member = _as_dict(event)
    for name in AGENT_STREAM_ERRORS:
        if name in member:
            acc.error = name
            return
    raw = _dig(member, "chunk", "bytes")
    if isinstance(raw, (bytes, bytearray)):
        try:
            acc.add_text(bytes(raw).decode("utf-8", errors="replace"))
        except Exception:
            pass
    elif isinstance(raw, str):
        acc.add_text(raw)
    text = _dig(member, "output", "text")
    if text:
        acc.add_text(text)
    action = _dig(member, "guardrail", "action")
    if action:
        acc.add_finish(action)


def retrieve_and_generate_response(result: Any, acc: BedrockAcc) -> None:
    body = _as_dict(result)
    acc.add_text(_dig(body, "output", "text"))
    acc.add_finish(body.get("guardrailAction"))


def retrieve_and_generate_model(params: Dict[str, Any]) -> Optional[str]:
    config = _as_dict(params.get("retrieveAndGenerateConfiguration"))
    for section in ("knowledgeBaseConfiguration", "externalSourcesConfiguration"):
        arn = _dig(config, section, "modelArn")
        if arn:
            return str(arn)
    return None


def retrieve_and_generate_kb(params: Dict[str, Any]) -> Optional[str]:
    kb = _dig(params, "retrieveAndGenerateConfiguration", "knowledgeBaseConfiguration", "knowledgeBaseId")
    return str(kb) if kb else None


def extract_reasoning(content: Any) -> "tuple[str, bool]":
    parts: List[str] = []
    has_redacted = False
    for block in _as_list(content):
        if not isinstance(block, dict):
            continue
        reasoning = block.get("reasoningContent")
        if not isinstance(reasoning, dict):
            continue
        reasoning_text = reasoning.get("reasoningText")
        text = _as_text(_as_dict(reasoning_text).get("text")) if isinstance(reasoning_text, dict) else ""
        if text:
            parts.append(text)
        if "redactedContent" in reasoning:
            has_redacted = True
    return "".join(parts), has_redacted


def _pick(body: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    for key, attr in mapping.items():
        cur: Any = body
        for part in key.split("."):
            cur = _as_dict(cur).get(part)
            if cur is None:
                break
        if cur is not None:
            attrs[attr] = cur
    return attrs


def _anthropic_prompt(body: Dict[str, Any]) -> str:
    if "messages" not in body:
        return _as_text(body.get("prompt"))
    parts: List[str] = []
    system = body.get("system")
    if isinstance(system, str) and system:
        parts.append("system: " + system)
    elif isinstance(system, list):
        text = "".join(_as_text(_as_dict(b).get("text")) for b in system)
        if text:
            parts.append("system: " + text)
    for message in _as_list(body.get("messages")):
        m = _as_dict(message)
        content = m.get("content")
        if isinstance(content, str):
            text = content
        else:
            text = "".join(
                _as_text(_as_dict(b).get("text"))
                for b in _as_list(content)
                if _as_dict(b).get("type") == "text"
            )
        parts.append("%s: %s" % (_as_text(m.get("role")), text))
    return "\n".join(parts)


def _anthropic_params(body: Dict[str, Any]) -> Dict[str, Any]:
    return _pick(body, {
        "max_tokens": "gen_ai.request.max_tokens",
        "max_tokens_to_sample": "gen_ai.request.max_tokens",
        "stop_sequences": "gen_ai.request.stop_sequences",
    })


def _anthropic_operation(body: Dict[str, Any]) -> str:
    return "chat" if "messages" in body else "text_completion"


def _anthropic_response(body: Dict[str, Any], acc: BedrockAcc) -> None:
    acc.add_text(body.get("completion"))
    content = _as_list(body.get("content"))
    for block in content:
        b = _as_dict(block)
        btype = b.get("type")
        if btype == "text":
            acc.add_text(b.get("text"))
        elif btype == "tool_use":
            acc.add_tool(b.get("name"), b.get("id"), b.get("input"))
    thinking, redacted = _extract_thinking(content)
    if thinking:
        acc.add_thinking(thinking)
    if redacted:
        acc.has_redacted_thinking = True
    usage = _as_dict(body.get("usage"))
    read = _as_int(usage.get("cache_read_input_tokens"))
    write = _as_int(usage.get("cache_creation_input_tokens"))
    if read:
        acc.cache_read = read
    if write:
        acc.cache_write = write
    acc.set_tokens_in(usage.get("input_tokens"))
    acc.set_tokens_out(usage.get("output_tokens"))
    acc.add_finish(body.get("stop_reason"))


def _anthropic_delta(chunk: Dict[str, Any], acc: BedrockAcc) -> None:
    etype = chunk.get("type")
    if etype == "content_block_delta":
        delta = _as_dict(chunk.get("delta"))
        dtype = delta.get("type")
        if dtype == "text_delta":
            acc.add_text(delta.get("text"))
        elif dtype == "thinking_delta":
            acc.add_thinking(delta.get("thinking"))
        elif dtype == "input_json_delta":
            index = chunk.get("index")
            buf = acc.tool_buf.setdefault(index, {"name": None, "id": "", "parts": []})
            buf["parts"].append(_as_text(delta.get("partial_json")))
    elif etype == "content_block_start":
        block = _as_dict(chunk.get("content_block"))
        btype = block.get("type")
        if btype == "redacted_thinking":
            acc.has_redacted_thinking = True
        elif btype == "tool_use":
            acc.tool_buf[chunk.get("index")] = {
                "name": block.get("name"), "id": block.get("id") or "", "parts": [],
            }
    elif etype == "content_block_stop":
        buf = acc.tool_buf.pop(chunk.get("index"), None)
        if buf:
            acc.add_tool(buf.get("name"), buf.get("id"), "".join(buf.get("parts") or []))
    elif etype == "message_start":
        usage = _as_dict(_dig(chunk, "message", "usage"))
        read = _as_int(usage.get("cache_read_input_tokens"))
        write = _as_int(usage.get("cache_creation_input_tokens"))
        if read:
            acc.cache_read = read
        if write:
            acc.cache_write = write
        acc.set_tokens_in(usage.get("input_tokens"))
        acc.set_tokens_out(usage.get("output_tokens"))
    elif etype == "message_delta":
        acc.add_finish(_dig(chunk, "delta", "stop_reason"))
        acc.set_tokens_out(_dig(chunk, "usage", "output_tokens"))


def _nova_prompt(body: Dict[str, Any]) -> str:
    parts: List[str] = []
    system = "".join(_as_text(_as_dict(b).get("text")) for b in _as_list(body.get("system")))
    if system:
        parts.append("system: " + system)
    for message in _as_list(body.get("messages")):
        m = _as_dict(message)
        parts.append("%s: %s" % (_as_text(m.get("role")), _content_blocks_text(m.get("content"))))
    return "\n".join(parts)


def _nova_params(body: Dict[str, Any]) -> Dict[str, Any]:
    return _pick(body, {
        "inferenceConfig.maxTokens": "gen_ai.request.max_tokens",
        "inferenceConfig.temperature": "gen_ai.request.temperature",
        "inferenceConfig.topP": "gen_ai.request.top_p",
        "inferenceConfig.topK": "gen_ai.request.top_k",
        "inferenceConfig.stopSequences": "gen_ai.request.stop_sequences",
    })


def _nova_response(body: Dict[str, Any], acc: BedrockAcc) -> None:
    acc.add_text(_content_blocks_text(_dig(body, "output", "message", "content")))
    usage = _as_dict(body.get("usage"))
    acc.set_tokens_in(usage.get("inputTokens"))
    acc.set_tokens_out(usage.get("outputTokens"))
    read = _as_int(usage.get("cacheReadInputTokenCount")) or _as_int(usage.get("cacheReadInputTokens"))
    write = _as_int(usage.get("cacheWriteInputTokenCount")) or _as_int(usage.get("cacheWriteInputTokens"))
    if read:
        acc.cache_read = read
    if write:
        acc.cache_write = write
    acc.add_finish(body.get("stopReason"))


def _nova_delta(chunk: Dict[str, Any], acc: BedrockAcc) -> None:
    acc.add_text(_dig(chunk, "contentBlockDelta", "delta", "text"))


def _titan_text_prompt(body: Dict[str, Any]) -> str:
    return _as_text(body.get("inputText"))


def _titan_text_params(body: Dict[str, Any]) -> Dict[str, Any]:
    return _pick(body, {
        "textGenerationConfig.maxTokenCount": "gen_ai.request.max_tokens",
        "textGenerationConfig.temperature": "gen_ai.request.temperature",
        "textGenerationConfig.topP": "gen_ai.request.top_p",
        "textGenerationConfig.stopSequences": "gen_ai.request.stop_sequences",
    })


def _titan_text_response(body: Dict[str, Any], acc: BedrockAcc) -> None:
    acc.set_tokens_in(body.get("inputTextTokenCount"))
    total_out = 0
    for result in _as_list(body.get("results")):
        r = _as_dict(result)
        acc.add_text(r.get("outputText"))
        total_out += _as_int(r.get("tokenCount"))
        acc.add_finish(r.get("completionReason"))
    acc.set_tokens_out(total_out)


def _titan_text_delta(chunk: Dict[str, Any], acc: BedrockAcc) -> None:
    acc.add_text(chunk.get("outputText"))
    acc.set_tokens_in(chunk.get("inputTextTokenCount"))
    acc.set_tokens_out(chunk.get("totalOutputTextTokenCount"))
    acc.add_finish(chunk.get("completionReason"))


def _titan_embed_response(body: Dict[str, Any], acc: BedrockAcc) -> None:
    acc.set_tokens_in(body.get("inputTextTokenCount"))


def _llama_prompt(body: Dict[str, Any]) -> str:
    return _as_text(body.get("prompt"))


def _llama_params(body: Dict[str, Any]) -> Dict[str, Any]:
    return _pick(body, {
        "max_gen_len": "gen_ai.request.max_tokens",
        "temperature": "gen_ai.request.temperature",
        "top_p": "gen_ai.request.top_p",
    })


def _llama_response(body: Dict[str, Any], acc: BedrockAcc) -> None:
    acc.add_text(body.get("generation"))
    acc.set_tokens_in(body.get("prompt_token_count"))
    acc.set_tokens_out(body.get("generation_token_count"))
    acc.add_finish(body.get("stop_reason"))


def _mistral_text_params(body: Dict[str, Any]) -> Dict[str, Any]:
    return _pick(body, {
        "max_tokens": "gen_ai.request.max_tokens",
        "stop": "gen_ai.request.stop_sequences",
        "top_k": "gen_ai.request.top_k",
    })


def _mistral_text_response(body: Dict[str, Any], acc: BedrockAcc) -> None:
    for output in _as_list(body.get("outputs")):
        o = _as_dict(output)
        acc.add_text(o.get("text"))
        acc.add_finish(o.get("stop_reason"))


def _openai_style_prompt(body: Dict[str, Any]) -> str:
    parts: List[str] = []
    for message in _as_list(body.get("messages")):
        m = _as_dict(message)
        content = m.get("content")
        text = content if isinstance(content, str) else _content_blocks_text(content)
        parts.append("%s: %s" % (_as_text(m.get("role")), text))
    return "\n".join(parts)


def _mistral_chat_response(body: Dict[str, Any], acc: BedrockAcc) -> None:
    for choice in _as_list(body.get("choices")):
        c = _as_dict(choice)
        acc.add_text(_dig(c, "message", "content"))
        for call in _as_list(_dig(c, "message", "tool_calls")):
            entry = _as_dict(call)
            fn = _as_dict(entry.get("function"))
            acc.add_tool(fn.get("name"), entry.get("tool_call_id") or "", fn.get("arguments"))
        acc.add_finish(c.get("stop_reason"))


def _cohere_command_prompt(body: Dict[str, Any]) -> str:
    return _as_text(body.get("prompt"))


def _cohere_command_params(body: Dict[str, Any]) -> Dict[str, Any]:
    return _pick(body, {
        "max_tokens": "gen_ai.request.max_tokens",
        "stop_sequences": "gen_ai.request.stop_sequences",
    })


def _cohere_command_response(body: Dict[str, Any], acc: BedrockAcc) -> None:
    for generation in _as_list(body.get("generations")):
        g = _as_dict(generation)
        acc.add_text(g.get("text"))
        acc.add_finish(g.get("finish_reason"))


def _cohere_command_delta(chunk: Dict[str, Any], acc: BedrockAcc) -> None:
    generations = _as_list(chunk.get("generations"))
    if generations:
        for generation in generations:
            g = _as_dict(generation)
            acc.add_text(g.get("text"))
            acc.add_finish(g.get("finish_reason"))
        return
    acc.add_text(chunk.get("text"))
    acc.add_finish(chunk.get("finish_reason"))


def _cohere_chat_prompt(body: Dict[str, Any]) -> str:
    parts: List[str] = []
    preamble = _as_text(body.get("preamble"))
    if preamble:
        parts.append("preamble: " + preamble)
    for entry in _as_list(body.get("chat_history")):
        e = _as_dict(entry)
        parts.append("%s: %s" % (_as_text(e.get("role")), _as_text(e.get("message"))))
    message = _as_text(body.get("message"))
    if message:
        parts.append("USER: " + message)
    return "\n".join(parts)


def _cohere_chat_usage(body: Dict[str, Any], acc: BedrockAcc) -> None:
    billed = _as_dict(_dig(body, "meta", "billed_units"))
    acc.set_tokens_in(billed.get("input_tokens"))
    acc.set_tokens_out(billed.get("output_tokens"))


def _cohere_chat_response(body: Dict[str, Any], acc: BedrockAcc) -> None:
    acc.add_text(body.get("text"))
    _cohere_chat_usage(body, acc)
    acc.add_finish(body.get("finish_reason"))


def _cohere_chat_delta(chunk: Dict[str, Any], acc: BedrockAcc) -> None:
    acc.add_text(chunk.get("text"))
    _cohere_chat_usage(chunk, acc)
    acc.add_finish(chunk.get("finish_reason"))


def _cohere_embed_prompt(body: Dict[str, Any]) -> str:
    return "\n".join(_as_text(t) for t in _as_list(body.get("texts")))


def _jamba_response(body: Dict[str, Any], acc: BedrockAcc) -> None:
    for choice in _as_list(body.get("choices")):
        c = _as_dict(choice)
        acc.add_text(_dig(c, "message", "content"))
        acc.add_finish(c.get("finish_reason"))
    usage = _as_dict(body.get("usage"))
    acc.set_tokens_in(usage.get("prompt_tokens"))
    acc.set_tokens_out(usage.get("completion_tokens"))


def _jurassic_params(body: Dict[str, Any]) -> Dict[str, Any]:
    return _pick(body, {
        "maxTokens": "gen_ai.request.max_tokens",
        "stopSequences": "gen_ai.request.stop_sequences",
    })


def _jurassic_response(body: Dict[str, Any], acc: BedrockAcc) -> None:
    for completion in _as_list(body.get("completions")):
        acc.add_text(_dig(_as_dict(completion), "data", "text"))


def _deepseek_params(body: Dict[str, Any]) -> Dict[str, Any]:
    return _pick(body, {
        "max_tokens": "gen_ai.request.max_tokens",
        "stop": "gen_ai.request.stop_sequences",
    })


def _deepseek_response(body: Dict[str, Any], acc: BedrockAcc) -> None:
    for choice in _as_list(body.get("choices")):
        c = _as_dict(choice)
        acc.add_text(c.get("text"))
        acc.add_finish(c.get("stop_reason"))


def _max_tokens_only(body: Dict[str, Any]) -> Dict[str, Any]:
    return _pick(body, {"max_tokens": "gen_ai.request.max_tokens"})


_ANTHROPIC = {
    "operation": _anthropic_operation,
    "prompt": _anthropic_prompt,
    "params": _anthropic_params,
    "response": _anthropic_response,
    "delta": _anthropic_delta,
}

_NOVA = {
    "operation": "chat",
    "prompt": _nova_prompt,
    "params": _nova_params,
    "response": _nova_response,
    "delta": _nova_delta,
}

_TITAN_TEXT = {
    "operation": "text_completion",
    "prompt": _titan_text_prompt,
    "params": _titan_text_params,
    "response": _titan_text_response,
    "delta": _titan_text_delta,
}

_TITAN_EMBED = {
    "operation": "embeddings",
    "prompt": _titan_text_prompt,
    "params": None,
    "response": _titan_embed_response,
    "delta": None,
}

_LLAMA = {
    "operation": "text_completion",
    "prompt": _llama_prompt,
    "params": _llama_params,
    "response": _llama_response,
    "delta": None,
}

_MISTRAL_CHAT = {
    "operation": "chat",
    "prompt": _openai_style_prompt,
    "params": _max_tokens_only,
    "response": _mistral_chat_response,
    "delta": None,
}

_MISTRAL_TEXT = {
    "operation": "text_completion",
    "prompt": _llama_prompt,
    "params": _mistral_text_params,
    "response": _mistral_text_response,
    "delta": None,
}

_COMMAND_R = {
    "operation": "chat",
    "prompt": _cohere_chat_prompt,
    "params": _cohere_command_params,
    "response": _cohere_chat_response,
    "delta": _cohere_chat_delta,
}

_COMMAND = {
    "operation": "text_completion",
    "prompt": _cohere_command_prompt,
    "params": _cohere_command_params,
    "response": _cohere_command_response,
    "delta": _cohere_command_delta,
}

_COHERE_EMBED = {
    "operation": "embeddings",
    "prompt": _cohere_embed_prompt,
    "params": None,
    "response": None,
    "delta": None,
}

_JAMBA = {
    "operation": "chat",
    "prompt": _openai_style_prompt,
    "params": _max_tokens_only,
    "response": _jamba_response,
    "delta": None,
}

_JURASSIC = {
    "operation": "text_completion",
    "prompt": _llama_prompt,
    "params": _jurassic_params,
    "response": _jurassic_response,
    "delta": None,
}

_DEEPSEEK = {
    "operation": "text_completion",
    "prompt": _llama_prompt,
    "params": _deepseek_params,
    "response": _deepseek_response,
    "delta": None,
}

FAMILIES = (
    ("anthropic.claude", _ANTHROPIC),
    ("amazon.nova", _NOVA),
    ("amazon.titan-text", _TITAN_TEXT),
    ("amazon.titan-embed", _TITAN_EMBED),
    ("meta.llama", _LLAMA),
    ("mistral.mistral-large", _MISTRAL_CHAT),
    ("mistral.", _MISTRAL_TEXT),
    ("cohere.command-r", _COMMAND_R),
    ("cohere.command-light", _COMMAND),
    ("cohere.command-text", _COMMAND),
    ("cohere.embed", _COHERE_EMBED),
    ("ai21.jamba", _JAMBA),
    ("ai21.j2", _JURASSIC),
    ("deepseek.", _DEEPSEEK),
)


def family_for(model_id: Any) -> Optional[Dict[str, Any]]:
    base = base_model_id(model_id)
    if not base:
        return None
    for prefix, family in FAMILIES:
        if base.startswith(prefix):
            return family
    return None


def family_operation(family: Optional[Dict[str, Any]], body: Dict[str, Any]) -> str:
    if not family:
        return "chat"
    operation = family.get("operation")
    if callable(operation):
        return operation(body)
    return operation or "chat"


def family_prompt(family: Optional[Dict[str, Any]], body: Dict[str, Any]) -> str:
    if family and family.get("prompt"):
        return family["prompt"](body)
    return generic_prompt(body)


def generic_prompt(body: Dict[str, Any]) -> str:
    for key in ("prompt", "inputText", "message"):
        text = _as_text(body.get(key))
        if text:
            return text
    if "messages" in body:
        return _openai_style_prompt(body)
    return ""


def family_params(family: Optional[Dict[str, Any]], body: Dict[str, Any]) -> Dict[str, Any]:
    if family and family.get("params"):
        return family["params"](body)
    return {}


def family_response(family: Optional[Dict[str, Any]], body: Dict[str, Any], acc: BedrockAcc) -> None:
    if family and family.get("response"):
        family["response"](body, acc)


def invocation_metrics(chunk: Dict[str, Any], acc: BedrockAcc) -> None:
    metrics = _as_dict(chunk.get("amazon-bedrock-invocationMetrics"))
    if not metrics:
        return
    if not acc.cache_read:
        acc.cache_read = _as_int(metrics.get("cacheReadInputTokenCount"))
    if not acc.cache_write:
        acc.cache_write = _as_int(metrics.get("cacheWriteInputTokenCount"))
    if not acc.tokens_in:
        acc.set_tokens_in(metrics.get("inputTokenCount"))
    if not acc.tokens_out:
        acc.set_tokens_out(metrics.get("outputTokenCount"))


def header_tokens(result: Any, acc: BedrockAcc) -> None:
    headers = _as_dict(_dig(result, "ResponseMetadata", "HTTPHeaders"))
    if not headers:
        return
    lowered = {str(k).lower(): v for k, v in headers.items()}
    if not acc.tokens_in:
        acc.tokens_in = _parsed_int(lowered.get(HEADER_INPUT_TOKENS))
    if not acc.tokens_out:
        acc.tokens_out = _parsed_int(lowered.get(HEADER_OUTPUT_TOKENS))


def invoke_stream_event(event: Any, family: Optional[Dict[str, Any]], acc: BedrockAcc) -> None:
    member = _as_dict(event)
    for name in INVOKE_STREAM_ERRORS:
        if name in member:
            acc.error = name
            return
    raw = _dig(member, "chunk", "bytes")
    chunk = load_json(raw)
    if chunk is None:
        return
    delta = family.get("delta") if family else None
    if delta:
        delta(chunk, acc)
    invocation_metrics(chunk, acc)
