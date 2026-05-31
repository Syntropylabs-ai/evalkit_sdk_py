from __future__ import annotations

from typing import Callable


class _SyncStreamTracer:
    def __init__(self, stream: object, on_done: Callable[[str, int, int], None], *, is_anthropic: bool = False) -> None:
        self._stream = stream
        self._on_done = on_done
        self._done = False
        self._chunks: list[str] = []
        self._tokens_in = 0
        self._tokens_out = 0
        self._is_anthropic = is_anthropic

    def _process_openai_chunk(self, chunk: object) -> object:
        choices = getattr(chunk, "choices", None)
        if choices:
            delta = getattr(choices[0], "delta", None)
            if delta:
                c = getattr(delta, "content", None)
                if c:
                    self._chunks.append(c)
        usage = getattr(chunk, "usage", None)
        if usage:
            self._tokens_in = getattr(usage, "prompt_tokens", 0) or 0
            self._tokens_out = getattr(usage, "completion_tokens", 0) or 0
        return chunk

    def _process_anthropic_chunk(self, event: object) -> object:
        etype = getattr(event, "type", None)
        if etype == "content_block_delta":
            delta = getattr(event, "delta", None)
            if delta and getattr(delta, "type", None) == "text_delta":
                t = getattr(delta, "text", None)
                if t:
                    self._chunks.append(t)
        elif etype == "message_start":
            message = getattr(event, "message", None)
            if message:
                usage = getattr(message, "usage", None)
                if usage:
                    self._tokens_in = getattr(usage, "input_tokens", 0) or 0
        elif etype == "message_delta":
            usage = getattr(event, "usage", None)
            if usage:
                self._tokens_out = getattr(usage, "output_tokens", 0) or 0
        return event

    def _finish(self) -> None:
        if not self._done:
            self._done = True
            self._on_done("".join(self._chunks), self._tokens_in, self._tokens_out)

    def __iter__(self):  # type: ignore[return]
        try:
            for chunk in self._stream:  # type: ignore[union-attr]
                if self._is_anthropic:
                    yield self._process_anthropic_chunk(chunk)
                else:
                    yield self._process_openai_chunk(chunk)
        finally:
            self._finish()

    def __enter__(self) -> "_SyncStreamTracer":
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()  # type: ignore[union-attr]
        return self

    def __exit__(self, *args: object) -> object:
        self._finish()
        if hasattr(self._stream, "__exit__"):
            return self._stream.__exit__(*args)  # type: ignore[union-attr]
        return False

    def __getattr__(self, name: str) -> object:
        return getattr(self._stream, name)


class _AsyncStreamTracer:
    def __init__(self, stream: object, on_done: Callable[[str, int, int], None], *, is_anthropic: bool = False) -> None:
        self._stream = stream
        self._on_done = on_done
        self._done = False
        self._chunks: list[str] = []
        self._tokens_in = 0
        self._tokens_out = 0
        self._is_anthropic = is_anthropic

    def _process_openai_chunk(self, chunk: object) -> object:
        choices = getattr(chunk, "choices", None)
        if choices:
            delta = getattr(choices[0], "delta", None)
            if delta:
                c = getattr(delta, "content", None)
                if c:
                    self._chunks.append(c)
        usage = getattr(chunk, "usage", None)
        if usage:
            self._tokens_in = getattr(usage, "prompt_tokens", 0) or 0
            self._tokens_out = getattr(usage, "completion_tokens", 0) or 0
        return chunk

    def _process_anthropic_chunk(self, event: object) -> object:
        etype = getattr(event, "type", None)
        if etype == "content_block_delta":
            delta = getattr(event, "delta", None)
            if delta and getattr(delta, "type", None) == "text_delta":
                t = getattr(delta, "text", None)
                if t:
                    self._chunks.append(t)
        elif etype == "message_start":
            message = getattr(event, "message", None)
            if message:
                usage = getattr(message, "usage", None)
                if usage:
                    self._tokens_in = getattr(usage, "input_tokens", 0) or 0
        elif etype == "message_delta":
            usage = getattr(event, "usage", None)
            if usage:
                self._tokens_out = getattr(usage, "output_tokens", 0) or 0
        return event

    def _finish(self) -> None:
        if not self._done:
            self._done = True
            self._on_done("".join(self._chunks), self._tokens_in, self._tokens_out)

    async def __aiter__(self):  # type: ignore[return]
        try:
            async for chunk in self._stream:  # type: ignore[union-attr]
                if self._is_anthropic:
                    yield self._process_anthropic_chunk(chunk)
                else:
                    yield self._process_openai_chunk(chunk)
        finally:
            self._finish()

    async def __aenter__(self) -> "_AsyncStreamTracer":
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()  # type: ignore[union-attr]
        return self

    async def __aexit__(self, *args: object) -> object:
        self._finish()
        if hasattr(self._stream, "__aexit__"):
            return await self._stream.__aexit__(*args)  # type: ignore[union-attr]
        return False

    def __getattr__(self, name: str) -> object:
        return getattr(self._stream, name)
