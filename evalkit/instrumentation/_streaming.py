from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from ._google_usage import EMPTY_GOOGLE_USAGE, extract_google_usage, merge_google_usage
from ._safe import hook_failed

_log = logging.getLogger("evalkit")

# How much of a stream the span keeps: its text (completion, thinking, refusal) and each tool
# call's arguments. Every chunk still reaches the caller whole; past these sizes the span keeps
# the start, so a very long reply cannot make the tracer hold an unbounded copy of it. The
# arguments limit sits above what a tool-call event records, so recorded events are unchanged.
_TEXT_KEEP = 1024 * 1024
_ARGS_KEEP = 64 * 1024
_TOOL_CALLS_KEEP = 256

_MISSING_STREAM_USAGE_MESSAGE = (
    "a streamed OpenAI chat completion ended without a usage chunk, so this span carries no token counts; the "
    "trace service estimates them from the prompt and completion text and marks the span evalkit.cost_estimated, "
    "so its tokens and cost are approximate (and stay at 0 when content capture is off, as there is no text to "
    "estimate from). The OpenAI API sends the usage chunk only when the request sets "
    "stream_options={\"include_usage\": True}; add it to your chat.completions.create(...) call for exact token "
    "counts and cost (logged once per process)"
)

_stream_usage_warn_lock = threading.Lock()
_stream_usage_warned = False


def _reset_lock_after_fork() -> None:
    global _stream_usage_warn_lock
    _stream_usage_warn_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_lock_after_fork)


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
    try:
        if not enabled or stream_usage_requested(kwargs):
            return on_done
    except Exception as exc:
        hook_failed("stream usage check", exc)
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
        self.error_attrs: Dict[str, Any] = {}
        # A Responses stream: the output items of its final response, and the error a response
        # that failed reported inside the stream, which raises nothing on the caller's side.
        self.output_items: List[Any] = []
        self.failure: Optional[str] = None
        self._tool_index: Dict[Any, Dict[str, Any]] = {}
        self._kept = {"text": 0, "thinking": 0, "refusal": 0}

    def _keep(self, kind: str, parts: List[str], text: str) -> None:
        room = _TEXT_KEEP - self._kept[kind]
        if room <= 0:
            return
        piece = text if len(text) <= room else text[:room]
        parts.append(piece)
        self._kept[kind] += len(piece)

    def _mark_first_token(self) -> None:
        if self.ttft_ms is None and self.start_ms is not None:
            self.ttft_ms = int(time.time() * 1000 - self.start_ms)

    def _add_text(self, text: object) -> None:
        if text:
            self._mark_first_token()
            self._keep("text", self.chunks, str(text))

    def _add_thinking(self, text: object) -> None:
        if text:
            self._mark_first_token()
            self._keep("thinking", self.thinking_chunks, str(text))

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
        if len(self.tool_calls) < _TOOL_CALLS_KEEP:
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
            room = _ARGS_KEEP - len(entry["arguments"])
            if room > 0:
                entry["arguments"] += value[:room]
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
                self._add_text(getattr(delta, "content", None))
                refusal = getattr(delta, "refusal", None)
                if refusal:
                    self._mark_first_token()
                    self._keep("refusal", self.refusal_chunks, str(refusal))
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
                self._add_text(getattr(delta, "text", None))
            elif dtype == "thinking_delta":
                # Thinking is generated text, so it ends the wait for a first token, but it is
                # recorded as gen_ai.response.thinking and never joins the completion.
                self._add_thinking(getattr(delta, "thinking", None))
            elif dtype == "input_json_delta":
                # Tool arguments are not generated text: a reply made only of tool calls has no
                # time to first token, as in every other provider's stream.
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


    def _apply_openai_response(self, response: object, final: bool) -> None:
        if response is None:
            return
        self.response_id = str(getattr(response, "id", "") or self.response_id)
        self.response_model = str(getattr(response, "model", "") or self.response_model)
        usage = getattr(response, "usage", None)
        if usage:
            (self.tokens_in, self.tokens_out, _total,
             self.reasoning_tokens, self.cache_read_tokens) = openai_responses_usage(usage)
        if final:
            output = getattr(response, "output", None)
            if output:
                self.output_items = list(output)[:_TOOL_CALLS_KEEP]

    def _responses_tool(self, event: object) -> Dict[str, Any]:
        return self._tool(("output", _num(getattr(event, "output_index", 0))))

    def process_openai_responses_event(self, event: object) -> None:
        etype = getattr(event, "type", None)
        if etype == "response.output_text.delta":
            self._add_text(getattr(event, "delta", None))
        elif etype in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            # Reasoning is generated text, so it ends the wait for a first token, but it is
            # recorded as gen_ai.response.thinking and never joins the completion.
            self._add_thinking(getattr(event, "delta", None))
        elif etype == "response.refusal.delta":
            refusal = getattr(event, "delta", None)
            if refusal:
                self._mark_first_token()
                self._keep("refusal", self.refusal_chunks, str(refusal))
        elif etype in ("response.output_item.added", "response.output_item.done"):
            item = getattr(event, "item", None)
            if getattr(item, "type", None) == "function_call":
                entry = self._responses_tool(event)
                call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
                if call_id:
                    entry["id"] = str(call_id)
                name = getattr(item, "name", None)
                if name:
                    entry["name"] = str(name)
                arguments = getattr(item, "arguments", None)
                if etype == "response.output_item.done" and isinstance(arguments, str) and arguments:
                    entry["arguments"] = arguments[:_ARGS_KEEP]
        elif etype == "response.function_call_arguments.delta":
            # Tool arguments are not generated text, as in the chat stream: no first token here.
            self._append_arguments(self._responses_tool(event), getattr(event, "delta", None))
        elif etype == "response.function_call_arguments.done":
            arguments = getattr(event, "arguments", None)
            if isinstance(arguments, str) and arguments:
                self._responses_tool(event)["arguments"] = arguments[:_ARGS_KEEP]
        elif etype in ("response.created", "response.in_progress"):
            self._apply_openai_response(getattr(event, "response", None), final=False)
        elif etype in ("response.completed", "response.incomplete", "response.failed"):
            response = getattr(event, "response", None)
            self._apply_openai_response(response, final=True)
            if etype == "response.failed":
                error = getattr(response, "error", None)
                self.failure = str(getattr(error, "message", None) or "the response failed")
        elif etype == "error":
            self.failure = str(getattr(event, "message", None) or "the stream reported an error")


def openai_responses_usage(usage: object) -> tuple:
    """(input, output, total, reasoning, cached) tokens of a Responses usage object.

    input_tokens already counts the cached ones, as a chat completion's prompt_tokens does.
    """
    tokens_in = _num(getattr(usage, "input_tokens", 0))
    tokens_out = _num(getattr(usage, "output_tokens", 0))
    total = _num(getattr(usage, "total_tokens", 0)) or tokens_in + tokens_out
    reasoning = _num(getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0))
    cached = _num(getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0))
    return tokens_in, tokens_out, total, reasoning, cached


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
        attrs.update(state.error_attrs)
    except Exception:
        pass
    return attrs


def record_stream_open_failure(
    on_done: Callable[..., None], state: _StreamState, error: BaseException,
) -> None:
    """Record a streamed call whose request failed before the stream opened.

    A refused connection, a 400 or a 401 raises from the create call itself, so no stream tracer
    exists to report it. The span goes out through the stream's own recorder, with the model and
    the request, status ERROR, the error message and the exception's type and stack.
    """
    try:
        from ..core.client import _exc_attrs

        state.error_attrs = _exc_attrs(error)
    except Exception:
        pass
    try:
        on_done("", 0, 0, "", 0, False, None, 0, 0, error)
    except Exception:
        pass


ChunkProcessor = Callable[[_StreamState, object], None]


def _resolve_processor(process: Optional[ChunkProcessor], is_anthropic: bool) -> ChunkProcessor:
    if process is not None:
        return process
    if is_anthropic:
        return _StreamState.process_anthropic_chunk
    return _StreamState.process_openai_chunk


class _StreamTracer:
    """Records a streamed model call while the caller reads the provider's stream through it.

    Each chunk is handed on as soon as it arrives and is never read ahead. The provider's stream
    stays usable as itself: its attributes and methods are delegated, isinstance sees its class,
    and reading to the end, an early break, next(), close(), the context manager or garbage
    collection ends the span exactly once. Failures in recording are logged at debug level and
    never reach the caller.
    """

    def __init__(
        self, stream: object, on_done: Callable[..., None], *,
        is_anthropic: bool = False, start_ms: Optional[float] = None,
        process: Optional[ChunkProcessor] = None,
        state: Optional[_StreamState] = None,
    ) -> None:
        self._done = True
        self._stream = stream
        self._on_done = on_done
        self._state = state if state is not None else _StreamState(start_ms)
        self._process = _resolve_processor(process, is_anthropic)
        self._iterator: Any = None
        self._done = False

    @property  # type: ignore[misc]
    def __class__(self) -> type:
        # isinstance(stream, openai.Stream) and the like keep holding for the wrapped stream.
        try:
            return type(object.__getattribute__(self, "_stream"))
        except AttributeError:
            return type(self)

    def __getattr__(self, name: str) -> object:
        try:
            stream = object.__getattribute__(self, "_stream")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(stream, name)

    def _observe(self, chunk: object) -> None:
        try:
            self._process(self._state, chunk)
        except Exception as exc:
            hook_failed("stream chunk", exc)

    def _finish(self, error: Optional[BaseException] = None) -> None:
        if self.__dict__.get("_done", True):
            return
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
        except Exception as exc:
            hook_failed("stream end", exc)

    def __del__(self) -> None:
        # A stream dropped before it was read to the end, closed or exited is still a call that
        # was made: it is recorded with what had arrived.
        try:
            self._finish()
        except Exception:
            pass


class _SyncStreamTracer(_StreamTracer):
    def __iter__(self):
        # A generator, so a for loop that stops early closes it at once and the span ends there.
        error: Optional[BaseException] = None
        try:
            for chunk in self._stream:
                self._observe(chunk)
                yield chunk
        except GeneratorExit:
            raise
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._finish(error)

    def __next__(self) -> object:
        if self._iterator is None:
            self._iterator = iter(self._stream)
        try:
            chunk = next(self._iterator)
        except StopIteration:
            self._finish()
            raise
        except GeneratorExit:
            raise
        except BaseException as exc:
            self._finish(exc)
            raise
        self._observe(chunk)
        return chunk

    def close(self) -> object:
        self._finish()
        return self._stream.close()

    def __enter__(self) -> "_SyncStreamTracer":
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args: object) -> object:
        self._finish()
        if hasattr(self._stream, "__exit__"):
            return self._stream.__exit__(*args)
        return False


class _AsyncStreamTracer(_StreamTracer):
    async def __aiter__(self):
        error: Optional[BaseException] = None
        try:
            async for chunk in self._stream:
                self._observe(chunk)
                yield chunk
        except GeneratorExit:
            raise
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._finish(error)

    async def __anext__(self) -> object:
        if self._iterator is None:
            self._iterator = self._stream.__aiter__()
        try:
            chunk = await self._iterator.__anext__()
        except StopAsyncIteration:
            self._finish()
            raise
        except GeneratorExit:
            raise
        except BaseException as exc:
            self._finish(exc)
            raise
        self._observe(chunk)
        return chunk

    def close(self) -> object:
        # The provider's close is a coroutine on an async stream; the caller awaits what it returns.
        self._finish()
        return self._stream.close()

    def aclose(self) -> object:
        self._finish()
        return self._stream.aclose()

    async def __aenter__(self) -> "_AsyncStreamTracer":
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args: object) -> object:
        self._finish()
        if hasattr(self._stream, "__aexit__"):
            return await self._stream.__aexit__(*args)
        return False


def trace_sync_stream(stream: object, on_done: Callable[..., None], **options: Any) -> object:
    """The provider's stream behind a tracer, or the object itself when it is not a stream (a raw
    response from with_raw_response or with_streaming_response) or no tracer could be built."""
    if not hasattr(type(stream), "__iter__"):
        return stream
    try:
        return _SyncStreamTracer(stream, on_done, **options)
    except Exception as exc:
        hook_failed("stream tracer", exc)
        return stream


def trace_async_stream(stream: object, on_done: Callable[..., None], **options: Any) -> object:
    if not hasattr(type(stream), "__aiter__"):
        return stream
    try:
        return _AsyncStreamTracer(stream, on_done, **options)
    except Exception as exc:
        hook_failed("stream tracer", exc)
        return stream
