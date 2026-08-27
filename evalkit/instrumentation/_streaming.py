from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from ._google_usage import EMPTY_GOOGLE_USAGE, extract_google_usage, merge_google_usage

_log = logging.getLogger("evalkit")

_MISSING_STREAM_USAGE_MESSAGE = (
    "a streamed OpenAI chat completion ended without a usage chunk, so this span reports 0 input and 0 output "
    "tokens and every cost report prices the call at $0 - the OpenAI API only sends the final usage chunk when "
    "the request sets stream_options={\"include_usage\": True}; add it to your chat.completions.create(...) call "
    "to get real token counts for streamed calls (logged once per process)"
)

_stream_usage_warn_lock = threading.Lock()
_stream_usage_warned = False


def stream_usage_requested(kwargs: Dict[str, Any]) -> bool:
    options = kwargs.get("stream_options")
    if isinstance(options, dict):
        return bool(options.get("include_usage"))
    return bool(getattr(options, "include_usage", False))


def _warn_missing_stream_usage() -> None:
    global _stream_usage_warned
    with _stream_usage_warn_lock:
        if _stream_usage_warned:
            return
        _stream_usage_warned = True
    _log.warning(_MISSING_STREAM_USAGE_MESSAGE)


def with_stream_usage_warning(
    on_done: Callable[..., None], kwargs: Dict[str, Any], enabled: bool = True
) -> Callable[..., None]:
    if not enabled or stream_usage_requested(kwargs):
        return on_done

    def wrapped(
        completion: str, tokens_in: int, tokens_out: int,
        thinking: str, reasoning_tokens: int, has_redacted_thinking: bool,
        ttft_ms: Optional[int] = None,
        cache_read_tokens: int = 0, cache_creation_tokens: int = 0,
        error: Optional[BaseException] = None,
    ) -> None:
        try:
            if error is None and not tokens_in and not tokens_out:
                _warn_missing_stream_usage()
        except Exception:
            pass
        on_done(
            completion, tokens_in, tokens_out, thinking, reasoning_tokens,
            has_redacted_thinking, ttft_ms, cache_read_tokens, cache_creation_tokens, error,
        )

    return wrapped


_UNSPECIFIED_FINISH_REASONS = frozenset({"0", "FINISH_REASON_UNSPECIFIED", "UNSPECIFIED"})


def _num(value: object) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


class _StreamState:

    def __init__(self, start_ms: Optional[float] = None) -> None:
        self.chunks: list[str] = []
        self.thinking_chunks: list[str] = []
        self.refusal_chunks: list[str] = []
        self.tokens_in = 0
        self.tokens_out = 0
        self.cache_read_tokens = 0
        self.cache_creation_tokens = 0
        self.reasoning_tokens = 0
        self.has_redacted_thinking = False
        self.start_ms = start_ms
        self.ttft_ms: Optional[int] = None
        self.tool_calls: List[Dict[str, Any]] = []
        self.finish_reasons: List[str] = []
        self.response_id = ""
        self.response_model = ""
        self.google_usage = EMPTY_GOOGLE_USAGE
        self._tool_index: Dict[Any, Dict[str, Any]] = {}

    def _mark_first_token(self) -> None:
        if self.ttft_ms is None and self.start_ms is not None:
            self.ttft_ms = int(time.time() * 1000 - self.start_ms)

    def _add_text(self, text: object) -> None:
        if text:
            self._mark_first_token()
            self.chunks.append(str(text))

    def _add_thinking(self, text: object) -> None:
        if text:
            self._mark_first_token()
            self.thinking_chunks.append(str(text))

    def _add_finish_reason(self, reason: object) -> None:
        if reason is None or reason == "":
            return
        text = str(getattr(reason, "name", None) or reason)
        if not text or text.upper() in _UNSPECIFIED_FINISH_REASONS:
            return
        if text not in self.finish_reasons:
            self.finish_reasons.append(text)

    def _new_tool(self) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"id": "", "name": "", "arguments": ""}
        self.tool_calls.append(entry)
        return entry

    def _tool(self, key: Any) -> Dict[str, Any]:
        entry = self._tool_index.get(key)
        if entry is None:
            entry = self._new_tool()
            self._tool_index[key] = entry
        return entry

    def _reset_tool_calls(self) -> None:
        self.tool_calls = []
        self._tool_index = {}

    @staticmethod
    def _append_arguments(entry: Dict[str, Any], value: object) -> None:
        if value is None or value == "":
            return
        if isinstance(value, str) and isinstance(entry["arguments"], str):
            entry["arguments"] += value
        else:
            entry["arguments"] = value

    def _add_openai_tool_deltas(self, delta: object) -> None:
        for call in (getattr(delta, "tool_calls", None) or []):
            entry = self._tool(("tool", _num(getattr(call, "index", 0))))
            call_id = getattr(call, "id", None)
            if call_id:
                entry["id"] = str(call_id)
            function = getattr(call, "function", None)
            if function is not None:
                name = getattr(function, "name", None)
                if name:
                    entry["name"] = str(name)
                self._append_arguments(entry, getattr(function, "arguments", None))
        legacy = getattr(delta, "function_call", None)
        if legacy is not None:
            entry = self._tool(("function_call",))
            name = getattr(legacy, "name", None)
            if name:
                entry["name"] = str(name)
            self._append_arguments(entry, getattr(legacy, "arguments", None))

    def process_openai_chunk(self, chunk: object) -> None:
        choices = getattr(chunk, "choices", None)
        if choices:
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta:
                c = getattr(delta, "content", None)
                if c:
                    self._mark_first_token()
                    self.chunks.append(c)
                refusal = getattr(delta, "refusal", None)
                if refusal:
                    self._mark_first_token()
                    self.refusal_chunks.append(str(refusal))
                self._add_openai_tool_deltas(delta)
            self._add_finish_reason(getattr(choice, "finish_reason", None))
        self.response_id = str(getattr(chunk, "id", "") or self.response_id)
        self.response_model = str(getattr(chunk, "model", "") or self.response_model)
        usage = getattr(chunk, "usage", None)
        if usage:
            self.tokens_in = getattr(usage, "prompt_tokens", 0) or 0
            self.tokens_out = getattr(usage, "completion_tokens", 0) or 0
            details = getattr(usage, "completion_tokens_details", None)
            if details:
                self.reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details:
                self.cache_read_tokens = getattr(prompt_details, "cached_tokens", 0) or 0

    def _apply_anthropic_usage(self, usage: object) -> None:
        if usage is None:
            return
        out = getattr(usage, "output_tokens", None)
        if out is not None:
            self.tokens_out = _num(out)
        cache_read = getattr(usage, "cache_read_input_tokens", None)
        if cache_read is not None:
            self.cache_read_tokens = _num(cache_read)
        cache_creation = getattr(usage, "cache_creation_input_tokens", None)
        if cache_creation is not None:
            self.cache_creation_tokens = _num(cache_creation)
        tokens_in = getattr(usage, "input_tokens", None)
        if tokens_in is not None:
            self.tokens_in = _num(tokens_in) + self.cache_read_tokens + self.cache_creation_tokens

    def process_anthropic_chunk(self, event: object) -> None:
        etype = getattr(event, "type", None)
        if etype == "content_block_delta":
            delta = getattr(event, "delta", None)
            dtype = getattr(delta, "type", None) if delta else None
            if dtype == "text_delta":
                t = getattr(delta, "text", None)
                if t:
                    self._mark_first_token()
                    self.chunks.append(t)
            elif dtype == "thinking_delta":
                t = getattr(delta, "thinking", None)
                if t:
                    self._mark_first_token()
                    self.thinking_chunks.append(t)
            elif dtype == "input_json_delta":
                self._mark_first_token()
                entry = self._tool(("block", _num(getattr(event, "index", 0))))
                self._append_arguments(entry, getattr(delta, "partial_json", None))
        elif etype == "content_block_start":
            block = getattr(event, "content_block", None)
            btype = getattr(block, "type", None) if block else None
            if btype == "redacted_thinking":
                self.has_redacted_thinking = True
            elif btype in ("tool_use", "server_tool_use"):
                entry = self._tool(("block", _num(getattr(event, "index", 0))))
                name = getattr(block, "name", None)
                if name:
                    entry["name"] = str(name)
                block_id = getattr(block, "id", None)
                if block_id:
                    entry["id"] = str(block_id)
                if btype == "server_tool_use":
                    entry["server_side"] = True
        elif etype == "message_start":
            message = getattr(event, "message", None)
            if message:
                self.response_id = str(getattr(message, "id", "") or self.response_id)
                self.response_model = str(getattr(message, "model", "") or self.response_model)
                self._apply_anthropic_usage(getattr(message, "usage", None))
        elif etype == "message_delta":
            self._apply_anthropic_usage(getattr(event, "usage", None))
            delta = getattr(event, "delta", None)
            if delta is not None:
                self._add_finish_reason(getattr(delta, "stop_reason", None))

    def process_google_legacy_chunk(self, chunk: object) -> None:
        for candidate in (getattr(chunk, "candidates", None) or []):
            content = getattr(candidate, "content", None)
            parts = (getattr(content, "parts", None) or []) if content is not None else []
            for part in parts:
                fc = getattr(part, "function_call", None)
                fname = getattr(fc, "name", None) if fc is not None else None
                if fname:
                    raw_args = getattr(fc, "args", None) or {}
                    try:
                        args: Any = dict(raw_args)
                    except Exception:
                        args = raw_args
                    entry = self._new_tool()
                    entry["name"] = str(fname)
                    entry["arguments"] = args
                    continue
                text = getattr(part, "text", None)
                if not text:
                    continue
                if getattr(part, "thought", False):
                    self._add_thinking(text)
                else:
                    self._add_text(text)
            self._add_finish_reason(getattr(candidate, "finish_reason", None))
        self.response_model = str(getattr(chunk, "model_version", "") or self.response_model)
        self.google_usage = merge_google_usage(
            self.google_usage, extract_google_usage(getattr(chunk, "usage_metadata", None))
        )
        usage = self.google_usage
        self.tokens_in = usage.tokens_in
        self.tokens_out = usage.tokens_out
        self.reasoning_tokens = usage.reasoning_tokens
        self.cache_read_tokens = usage.cached_tokens

    def _apply_cohere_usage(self, usage: object) -> None:
        if usage is None:
            return
        units = getattr(usage, "billed_units", None) or getattr(usage, "tokens", None)
        if units is not None:
            self.tokens_in = _num(getattr(units, "input_tokens", 0)) or self.tokens_in
            self.tokens_out = _num(getattr(units, "output_tokens", 0)) or self.tokens_out
        cached = _num(getattr(usage, "cached_tokens", 0))
        if cached:
            self.cache_read_tokens = cached

    def _process_cohere_v1_event(self, event: object, etype: str) -> None:
        if etype == "text-generation":
            self._add_text(getattr(event, "text", None))
        elif etype == "tool-calls-chunk":
            delta = getattr(event, "tool_call_delta", None)
            if delta is not None:
                entry = self._tool(("v1", _num(getattr(delta, "index", 0))))
                name = getattr(delta, "name", None)
                if name:
                    entry["name"] = str(name)
                self._append_arguments(entry, getattr(delta, "parameters", None))
                self._add_thinking(getattr(delta, "text", None))
        elif etype == "tool-calls-generation":
            self._reset_tool_calls()
            for call in (getattr(event, "tool_calls", None) or []):
                entry = self._new_tool()
                entry["name"] = str(getattr(call, "name", "") or "")
                entry["arguments"] = getattr(call, "parameters", None) or {}
            if not self.thinking_chunks:
                self._add_thinking(getattr(event, "text", None))
        elif etype == "stream-end":
            self._add_finish_reason(getattr(event, "finish_reason", None))
            response = getattr(event, "response", None)
            if response is not None:
                self.response_id = str(getattr(response, "response_id", "") or self.response_id)
                self._apply_cohere_usage(getattr(response, "meta", None))

    def _process_cohere_v2_event(self, event: object, etype: object) -> None:
        delta = getattr(event, "delta", None)
        message = getattr(delta, "message", None) if delta is not None else None
        if etype == "message-start":
            self.response_id = str(getattr(event, "id", "") or self.response_id)
        elif etype == "content-delta":
            content = getattr(message, "content", None) if message is not None else None
            if content is not None:
                self._add_text(getattr(content, "text", None))
                self._add_thinking(getattr(content, "thinking", None))
        elif etype == "tool-plan-delta":
            if message is not None:
                self._add_thinking(getattr(message, "tool_plan", None))
        elif etype in ("tool-call-start", "tool-call-delta"):
            call = getattr(message, "tool_calls", None) if message is not None else None
            if call is not None:
                entry = self._tool(("v2", _num(getattr(event, "index", 0))))
                call_id = getattr(call, "id", None)
                if call_id:
                    entry["id"] = str(call_id)
                function = getattr(call, "function", None)
                if function is not None:
                    name = getattr(function, "name", None)
                    if name:
                        entry["name"] = str(name)
                    self._append_arguments(entry, getattr(function, "arguments", None))
        elif etype == "message-end":
            if delta is not None:
                self._add_finish_reason(getattr(delta, "finish_reason", None))
                self._apply_cohere_usage(getattr(delta, "usage", None))

    def process_cohere_chunk(self, event: object) -> None:
        etype = getattr(event, "event_type", None)
        if etype:
            self._process_cohere_v1_event(event, str(etype))
        else:
            self._process_cohere_v2_event(event, getattr(event, "type", None))

    def _add_mistral_content(self, content: object) -> None:
        if not content:
            return
        if isinstance(content, str):
            self._add_text(content)
            return
        if isinstance(content, (list, tuple)):
            for part in content:
                if getattr(part, "type", None) == "thinking":
                    for piece in (getattr(part, "thinking", None) or []):
                        self._add_thinking(getattr(piece, "text", None))
                else:
                    self._add_text(getattr(part, "text", None))

    def process_mistral_chunk(self, event: object) -> None:
        chunk = getattr(event, "data", None) or event
        self.response_id = str(getattr(chunk, "id", "") or self.response_id)
        self.response_model = str(getattr(chunk, "model", "") or self.response_model)
        for choice in (getattr(chunk, "choices", None) or []):
            delta = getattr(choice, "delta", None)
            if delta is not None:
                self._add_mistral_content(getattr(delta, "content", None))
                for call in (getattr(delta, "tool_calls", None) or []):
                    entry = self._tool(_num(getattr(call, "index", 0)))
                    call_id = getattr(call, "id", None)
                    if call_id and call_id != "null":
                        entry["id"] = str(call_id)
                    function = getattr(call, "function", None)
                    if function is not None:
                        name = getattr(function, "name", None)
                        if name:
                            entry["name"] = str(name)
                        self._append_arguments(entry, getattr(function, "arguments", None))
            self._add_finish_reason(getattr(choice, "finish_reason", None))
        usage = getattr(chunk, "usage", None)
        if usage:
            self.tokens_in = _num(getattr(usage, "prompt_tokens", 0)) or self.tokens_in
            self.tokens_out = _num(getattr(usage, "completion_tokens", 0)) or self.tokens_out


def stream_tool_events(state: Optional[_StreamState], ts: str) -> list:
    if state is None:
        return []
    from ._toolcalls import from_stream_tool_calls

    try:
        return from_stream_tool_calls(state.tool_calls, ts)
    except Exception:
        return []


def stream_response_attrs(state: Optional[_StreamState]) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    if state is None:
        return attrs
    try:
        if state.response_model:
            attrs["gen_ai.response.model"] = state.response_model
        if state.response_id:
            attrs["gen_ai.response.id"] = state.response_id
        if state.finish_reasons:
            attrs["gen_ai.response.finish_reasons"] = list(state.finish_reasons)
        if state.refusal_chunks:
            attrs["gen_ai.response.refusal"] = "".join(state.refusal_chunks)
        if any(call.get("server_side") for call in state.tool_calls):
            attrs["gen_ai.response.has_server_side_tool_use"] = True
    except Exception:
        pass
    return attrs


ChunkProcessor = Callable[[_StreamState, object], None]


def _resolve_processor(process: Optional[ChunkProcessor], is_anthropic: bool) -> ChunkProcessor:
    if process is not None:
        return process
    if is_anthropic:
        return _StreamState.process_anthropic_chunk
    return _StreamState.process_openai_chunk


class _SyncStreamTracer:
    def __init__(
        self, stream: object, on_done: Callable[..., None], *,
        is_anthropic: bool = False, start_ms: Optional[float] = None,
        process: Optional[ChunkProcessor] = None,
        state: Optional[_StreamState] = None,
    ) -> None:
        self._stream = stream
        self._on_done = on_done
        self._done = False
        self._state = state if state is not None else _StreamState(start_ms)
        self._process = _resolve_processor(process, is_anthropic)

    def _finish(self, error: Optional[BaseException] = None) -> None:
        if not self._done:
            self._done = True
            s = self._state
            try:
                self._on_done(
                    "".join(s.chunks), s.tokens_in, s.tokens_out,
                    "".join(s.thinking_chunks), s.reasoning_tokens, s.has_redacted_thinking,
                    None if error is not None else s.ttft_ms,
                    s.cache_read_tokens, s.cache_creation_tokens,
                    error,
                )
            except Exception:
                pass

    def __iter__(self):
        error: Optional[BaseException] = None
        try:
            for chunk in self._stream:
                try:
                    self._process(self._state, chunk)
                except Exception:
                    pass
                yield chunk
        except GeneratorExit:
            raise
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._finish(error)

    def __enter__(self) -> "_SyncStreamTracer":
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args: object) -> object:
        self._finish()
        if hasattr(self._stream, "__exit__"):
            return self._stream.__exit__(*args)
        return False

    def __getattr__(self, name: str) -> object:
        return getattr(self._stream, name)


class _AsyncStreamTracer:
    def __init__(
        self, stream: object, on_done: Callable[..., None], *,
        is_anthropic: bool = False, start_ms: Optional[float] = None,
        process: Optional[ChunkProcessor] = None,
        state: Optional[_StreamState] = None,
    ) -> None:
        self._stream = stream
        self._on_done = on_done
        self._done = False
        self._state = state if state is not None else _StreamState(start_ms)
        self._process = _resolve_processor(process, is_anthropic)

    def _finish(self, error: Optional[BaseException] = None) -> None:
        if not self._done:
            self._done = True
            s = self._state
            try:
                self._on_done(
                    "".join(s.chunks), s.tokens_in, s.tokens_out,
                    "".join(s.thinking_chunks), s.reasoning_tokens, s.has_redacted_thinking,
                    None if error is not None else s.ttft_ms,
                    s.cache_read_tokens, s.cache_creation_tokens,
                    error,
                )
            except Exception:
                pass

    async def __aiter__(self):
        error: Optional[BaseException] = None
        try:
            async for chunk in self._stream:
                try:
                    self._process(self._state, chunk)
                except Exception:
                    pass
                yield chunk
        except GeneratorExit:
            raise
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._finish(error)

    async def __aenter__(self) -> "_AsyncStreamTracer":
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args: object) -> object:
        self._finish()
        if hasattr(self._stream, "__aexit__"):
            return await self._stream.__aexit__(*args)
        return False

    def __getattr__(self, name: str) -> object:
        return getattr(self._stream, name)
