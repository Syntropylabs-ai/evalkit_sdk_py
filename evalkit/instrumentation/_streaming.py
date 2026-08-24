from __future__ import annotations

import time
from typing import Callable, Optional


class _StreamState:

    def __init__(self, start_ms: Optional[float] = None) -> None:
        self.chunks: list[str] = []
        self.thinking_chunks: list[str] = []
        self.tokens_in = 0
        self.tokens_out = 0
        self.cache_read_tokens = 0
        self.cache_creation_tokens = 0
        self.reasoning_tokens = 0
        self.has_redacted_thinking = False
        self.start_ms = start_ms
        self.ttft_ms: Optional[int] = None

    def _mark_first_token(self) -> None:
        if self.ttft_ms is None and self.start_ms is not None:
            self.ttft_ms = int(time.time() * 1000 - self.start_ms)

    def process_openai_chunk(self, chunk: object) -> None:
        choices = getattr(chunk, "choices", None)
        if choices:
            delta = getattr(choices[0], "delta", None)
            if delta:
                c = getattr(delta, "content", None)
                if c:
                    self._mark_first_token()
                    self.chunks.append(c)
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
        elif etype == "content_block_start":
            block = getattr(event, "content_block", None)
            if block and getattr(block, "type", None) == "redacted_thinking":
                self.has_redacted_thinking = True
        elif etype == "message_start":
            message = getattr(event, "message", None)
            if message:
                usage = getattr(message, "usage", None)
                if usage:
                    self.cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
                    self.cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
                    self.tokens_in = (getattr(usage, "input_tokens", 0) or 0) + self.cache_read_tokens + self.cache_creation_tokens
        elif etype == "message_delta":
            usage = getattr(event, "usage", None)
            if usage:
                self.tokens_out = getattr(usage, "output_tokens", 0) or 0


class _SyncStreamTracer:
    def __init__(
        self, stream: object, on_done: Callable[..., None], *,
        is_anthropic: bool = False, start_ms: Optional[float] = None,
    ) -> None:
        self._stream = stream
        self._on_done = on_done
        self._done = False
        self._state = _StreamState(start_ms)
        self._is_anthropic = is_anthropic

    def _finish(self) -> None:
        if not self._done:
            self._done = True
            s = self._state
            self._on_done(
                "".join(s.chunks), s.tokens_in, s.tokens_out,
                "".join(s.thinking_chunks), s.reasoning_tokens, s.has_redacted_thinking,
                s.ttft_ms, s.cache_read_tokens, s.cache_creation_tokens,
            )

    def __iter__(self):
        try:
            for chunk in self._stream:
                if self._is_anthropic:
                    self._state.process_anthropic_chunk(chunk)
                else:
                    self._state.process_openai_chunk(chunk)
                yield chunk
        finally:
            self._finish()

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
    ) -> None:
        self._stream = stream
        self._on_done = on_done
        self._done = False
        self._state = _StreamState(start_ms)
        self._is_anthropic = is_anthropic

    def _finish(self) -> None:
        if not self._done:
            self._done = True
            s = self._state
            self._on_done(
                "".join(s.chunks), s.tokens_in, s.tokens_out,
                "".join(s.thinking_chunks), s.reasoning_tokens, s.has_redacted_thinking,
                s.ttft_ms, s.cache_read_tokens, s.cache_creation_tokens,
            )

    async def __aiter__(self):
        try:
            async for chunk in self._stream:
                if self._is_anthropic:
                    self._state.process_anthropic_chunk(chunk)
                else:
                    self._state.process_openai_chunk(chunk)
                yield chunk
        finally:
            self._finish()

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
