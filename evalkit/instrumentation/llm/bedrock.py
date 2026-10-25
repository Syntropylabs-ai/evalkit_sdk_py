from __future__ import annotations

import inspect
import io
import time
from typing import Any, Callable, Dict, List, Optional

from ..._version import SDK_VERSION
from ...core.client import EvalKitClient, _new_span_id, _now_iso
from ...core.context import SpanContext, _current_span, child_span_ctx, current_span_ctx, get_or_create_ambient_trace_id
from ...core.types import SpanEvent, TraceEnvelope
from .._toolcalls import tool_event
from ..db._common import start_db
from . import _bedrock_bodies as _bb
from ._bodies import _dump, _safe_body


_RUNTIME_OPERATIONS = frozenset((
    "Converse", "ConverseStream", "InvokeModel", "InvokeModelWithResponseStream",
))

_AGENT_OPERATIONS = frozenset((
    "InvokeAgent", "InvokeInlineAgent", "Retrieve",
    "RetrieveAndGenerate", "RetrieveAndGenerateStream",
))

_SERVICE_OPERATIONS = {
    "bedrock-runtime": _RUNTIME_OPERATIONS,
    "bedrock-agent-runtime": _AGENT_OPERATIONS,
}

_RUNTIME_METHODS = (
    ("converse", "Converse"),
    ("converse_stream", "ConverseStream"),
    ("invoke_model", "InvokeModel"),
    ("invoke_model_with_response_stream", "InvokeModelWithResponseStream"),
)


def handles(service_name: Any, operation_name: Any) -> bool:
    return operation_name in _SERVICE_OPERATIONS.get(service_name, ())


def patch_bedrock(client: EvalKitClient, bedrock_runtime_client: Any) -> None:
    if getattr(bedrock_runtime_client, "_evalkit", False):
        return
    wrapped = False
    for method_name, operation_name in _RUNTIME_METHODS:
        original = getattr(bedrock_runtime_client, method_name, None)
        if original is None or getattr(original, "_evalkit", False):
            continue
        setattr(bedrock_runtime_client, method_name, _wrap_method(client, original, operation_name))
        wrapped = True
    if wrapped:
        bedrock_runtime_client._evalkit = True


def _wrap_method(ek: EvalKitClient, original: Callable[..., Any], operation_name: str) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(original):
        async def traced_async(**kwargs: Any) -> Any:
            return await trace_bedrock_call_async(
                ek, "bedrock-runtime", operation_name, kwargs, lambda: original(**kwargs)
            )

        traced_async._evalkit = True
        return traced_async

    def traced(**kwargs: Any) -> Any:
        return trace_bedrock_call(ek, "bedrock-runtime", operation_name, kwargs, lambda: original(**kwargs))

    traced._evalkit = True
    return traced


def trace_bedrock_call(
    ek: EvalKitClient,
    service_name: Any,
    operation_name: Any,
    params: Any,
    invoke: Callable[[], Any],
) -> Any:
    if not handles(service_name, operation_name):
        return invoke()
    api_params = params if isinstance(params, dict) else {}
    if operation_name == "Retrieve":
        return _trace_retrieve(ek, api_params, invoke)
    try:
        plan = _plan(operation_name, api_params)
    except Exception:
        return invoke()
    span = _BedrockSpan(ek, plan)
    token = _current_span.set(span.ctx())
    try:
        result = invoke()
    except Exception as exc:
        span.emit(status="ERROR", status_message=str(exc), exception=exc)
        raise
    finally:
        _current_span.reset(token)
    try:
        if plan.stream_kind and _install_stream(span, plan, result):
            return result
        acc = _bb.BedrockAcc(span.start_ms)
        response_dump = _collect_response(plan, result, acc)
        span.emit_acc(acc, response_dump=response_dump)
    except Exception:
        try:
            span.emit()
        except Exception:
            pass
    return result


async def trace_bedrock_call_async(
    ek: EvalKitClient,
    service_name: Any,
    operation_name: Any,
    params: Any,
    invoke: Callable[[], Any],
) -> Any:
    if not handles(service_name, operation_name):
        return await invoke()
    api_params = params if isinstance(params, dict) else {}
    if operation_name == "Retrieve":
        return await _trace_retrieve_async(ek, api_params, invoke)
    try:
        plan = _plan(operation_name, api_params)
    except Exception:
        return await invoke()
    span = _BedrockSpan(ek, plan)
    token = _current_span.set(span.ctx())
    try:
        result = await invoke()
    except Exception as exc:
        span.emit(status="ERROR", status_message=str(exc), exception=exc)
        raise
    finally:
        _current_span.reset(token)
    try:
        if plan.stream_kind and _install_stream(span, plan, result, async_stream=True):
            return result
        acc = _bb.BedrockAcc(span.start_ms)
        response_dump = await _collect_response_async(plan, result, acc)
        span.emit_acc(acc, response_dump=response_dump)
    except Exception:
        try:
            span.emit()
        except Exception:
            pass
    return result


class _Plan:
    __slots__ = (
        "operation", "span_type", "model", "prompt", "attrs", "family", "stream_kind", "stream_key", "session_id",
    )

    def __init__(self, operation: str) -> None:
        self.operation = operation
        self.span_type = "llm_call"
        self.model: Optional[str] = None
        self.prompt = ""
        self.attrs: Dict[str, Any] = {}
        self.family: Optional[Dict[str, Any]] = None
        self.stream_kind: Optional[str] = None
        self.stream_key: Optional[str] = None
        self.session_id: Optional[str] = None


def _plan(operation_name: str, params: Dict[str, Any]) -> _Plan:
    plan = _Plan(operation_name)
    request_dump: Any = params

    if operation_name in ("Converse", "ConverseStream"):
        plan.model = _model_id(params)
        plan.prompt = _bb.converse_prompt(params)
        plan.attrs.update(_bb.converse_params(params))
        plan.attrs["gen_ai.operation.name"] = "chat"
        tool_specs = _bb.converse_tool_specs(params)
        if tool_specs:
            definitions = _safe_body(_dump(tool_specs))
            if definitions:
                plan.attrs["gen_ai.tool.definitions"] = definitions
        if operation_name == "ConverseStream":
            plan.stream_kind = "converse"
            plan.stream_key = "stream"

    elif operation_name in ("InvokeModel", "InvokeModelWithResponseStream"):
        plan.model = _model_id(params)
        plan.family = _bb.family_for(plan.model)
        body = _bb.load_json(params.get("body"))
        if body is not None:
            plan.prompt = _bb.family_prompt(plan.family, body)
            plan.attrs.update(_bb.family_params(plan.family, body))
            plan.attrs["gen_ai.operation.name"] = _bb.family_operation(plan.family, body)
            request_dump = dict(params)
            request_dump["body"] = body
        if operation_name == "InvokeModelWithResponseStream":
            plan.stream_kind = "invoke"
            plan.stream_key = "body"

    elif operation_name in ("InvokeAgent", "InvokeInlineAgent"):
        plan.span_type = "function_call"
        plan.attrs["gen_ai.operation.name"] = "invoke_agent"
        parts: List[str] = []
        instruction = params.get("instruction")
        if isinstance(instruction, str) and instruction:
            parts.append("instruction: " + instruction)
        input_text = params.get("inputText")
        if isinstance(input_text, str) and input_text:
            parts.append(input_text)
        plan.prompt = "\n".join(parts)
        foundation_model = params.get("foundationModel")
        if isinstance(foundation_model, str) and foundation_model:
            plan.model = foundation_model
        plan.stream_kind = "agent"
        plan.stream_key = "completion"

    elif operation_name in ("RetrieveAndGenerate", "RetrieveAndGenerateStream"):
        plan.attrs["gen_ai.operation.name"] = "retrieval"
        plan.model = _bb.retrieve_and_generate_model(params)
        knowledge_base = _bb.retrieve_and_generate_kb(params)
        if knowledge_base:
            plan.attrs["aws.bedrock.knowledge_base.id"] = knowledge_base
        if operation_name == "RetrieveAndGenerateStream":
            plan.stream_kind = "agent"

    session_id = params.get("sessionId")
    if isinstance(session_id, str) and session_id:
        # The Bedrock session is the conversation this call belongs to, so it is the span's
        # session even when the app never called set_user.
        plan.attrs["gen_ai.conversation.id"] = session_id
        plan.session_id = session_id
    plan.prompt = plan.prompt
    request_body = _safe_body(_dump(request_dump))
    if request_body:
        plan.attrs["gen_ai.request.body"] = request_body
    return plan


def _model_id(params: Dict[str, Any]) -> Optional[str]:
    model = params.get("modelId")
    return model if isinstance(model, str) and model else None


class _BedrockSpan:
    def __init__(self, ek: EvalKitClient, plan: _Plan) -> None:
        ctx_parent = current_span_ctx()
        self.ek = ek
        self.plan = plan
        self.trace_id = ctx_parent.trace_id if ctx_parent else get_or_create_ambient_trace_id()
        self.parent_span_id = ctx_parent.span_id if ctx_parent else None
        self.span_id = _new_span_id()
        self.start_ms = time.time() * 1000
        self.ts = _now_iso()

    def ctx(self) -> SpanContext:
        return child_span_ctx(self.trace_id, self.span_id)

    def emit(
        self,
        *,
        status: str = "OK",
        status_message: Optional[str] = None,
        completion: str = "",
        tokens_in: int = 0,
        tokens_out: int = 0,
        cache_read: int = 0,
        cache_write: int = 0,
        response_model: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        events: Optional[List[SpanEvent]] = None,
        exception: Optional[BaseException] = None,
    ) -> None:
        plan = self.plan
        end_ts = _now_iso()
        attrs: Dict[str, Any] = {**self.ek._meta_attrs(), **plan.attrs}
        if plan.model:
            attrs["gen_ai.request.model"] = plan.model
        resolved_model = response_model or plan.model
        if resolved_model:
            attrs["gen_ai.response.model"] = resolved_model
        if tokens_in or tokens_out:
            attrs["gen_ai.usage.total_tokens"] = tokens_in + tokens_out
        if extra:
            attrs.update(extra)
        self.ek.enqueue(TraceEnvelope(
            trace_id=self.trace_id, span_id=self.span_id, parent_span_id=self.parent_span_id,
            span_type=plan.span_type, status=status, status_message=status_message,
            operation=plan.operation,
            service_name=self.ek._opts.service_name, provider="bedrock", model=plan.model,
            prompt=plan.prompt, completion=completion,
            tokens_in=tokens_in, tokens_out=tokens_out,
            cache_read_tokens=cache_read or None, cache_creation_tokens=cache_write or None,
            latency_ms=int(time.time() * 1000 - self.start_ms),
            start_time=self.ts, end_time=end_ts, created_at=self.ts,
            events=events or None, attributes=attrs,
            environment=self.ek._opts.environment, app_version=self.ek._opts.app_version,
            session_id=plan.session_id, sdk_version=SDK_VERSION, exception=exception,
        ))

    def emit_acc(
        self,
        acc: "_bb.BedrockAcc",
        *,
        response_dump: Any = None,
        streaming: bool = False,
        exc: Optional[BaseException] = None,
    ) -> None:
        extra: Dict[str, Any] = {}
        if streaming:
            extra["gen_ai.streaming"] = True
        if acc.finish_reasons:
            extra["gen_ai.response.finish_reasons"] = acc.finish_reasons
        thinking = acc.thinking()
        if thinking:
            extra["gen_ai.response.thinking"] = thinking
        if acc.has_redacted_thinking:
            extra["gen_ai.response.has_redacted_thinking"] = True
        if acc.ttft_ms is not None and exc is None:
            extra["gen_ai.server.time_to_first_token_ms"] = acc.ttft_ms
        if response_dump is not None:
            response_body = _safe_body(_dump(response_dump))
            if response_body:
                extra["gen_ai.response.body"] = response_body
        status = "OK"
        status_message = None
        if exc is not None:
            status = "ERROR"
            status_message = str(exc)
        elif acc.error:
            status = "ERROR"
            status_message = acc.error
        end_ts = _now_iso()
        events = [
            event for event in (
                tool_event(name=t.get("name"), arguments=t.get("arguments"), call_id=t.get("id"), ts=end_ts)
                for t in acc.tools
            ) if event is not None
        ]
        self.emit(
            status=status, status_message=status_message, completion=acc.completion(),
            tokens_in=acc.tokens_in + acc.cache_read + acc.cache_write, tokens_out=acc.tokens_out,
            cache_read=acc.cache_read, cache_write=acc.cache_write,
            response_model=acc.response_model, extra=extra, events=events, exception=exc,
        )


def _collect_response(plan: _Plan, result: Any, acc: "_bb.BedrockAcc") -> Any:
    if plan.operation == "Converse":
        _bb.converse_response(result, acc)
        return result
    if plan.operation == "InvokeModel":
        raw = _restore_body(result, "body")
        body = _bb.load_json(raw)
        if body is not None:
            _bb.family_response(plan.family, body, acc)
        _bb.header_tokens(result, acc)
        return body
    if plan.operation == "RetrieveAndGenerate":
        _bb.retrieve_and_generate_response(result, acc)
        return result
    return result if isinstance(result, dict) else None


async def _collect_response_async(plan: _Plan, result: Any, acc: "_bb.BedrockAcc") -> Any:
    if plan.operation != "InvokeModel":
        return _collect_response(plan, result, acc)
    raw = await _restore_body_async(result, "body")
    body = _bb.load_json(raw)
    if body is not None:
        _bb.family_response(plan.family, body, acc)
    _bb.header_tokens(result, acc)
    return body


def _restore_body(container: Any, key: str) -> Any:
    if not isinstance(container, dict):
        return None
    raw = container.get(key)
    if isinstance(raw, (bytes, bytearray, str)):
        return raw
    if raw is None or not hasattr(raw, "read"):
        return None
    try:
        from botocore.response import StreamingBody
    except Exception:
        return None
    try:
        content = raw.read()
    except Exception:
        return None
    try:
        container[key] = StreamingBody(io.BytesIO(content), len(content))
    except Exception:
        container[key] = io.BytesIO(content)
    return content


async def _restore_body_async(container: Any, key: str) -> Any:
    if not isinstance(container, dict):
        return None
    raw = container.get(key)
    if isinstance(raw, (bytes, bytearray, str)):
        return raw
    if raw is None or not hasattr(raw, "read"):
        return None
    try:
        content = raw.read()
        if inspect.isawaitable(content):
            content = await content
    except Exception:
        return None
    if not isinstance(content, (bytes, bytearray)):
        return content
    container[key] = _AsyncReplayBody(raw, bytes(content))
    return content


class _AsyncReplayBody:
    def __init__(self, stream: Any, content: bytes) -> None:
        self._evalkit_stream = stream
        self._evalkit_content = content
        self._evalkit_pos = 0

    async def read(self, amt: Optional[int] = None) -> bytes:
        content = self._evalkit_content
        if amt is None or amt < 0:
            data = content[self._evalkit_pos:]
            self._evalkit_pos = len(content)
            return data
        if amt == 0:
            return b""
        data = content[self._evalkit_pos:self._evalkit_pos + amt]
        self._evalkit_pos += len(data)
        return data

    async def readinto(self, b: bytearray) -> int:
        if len(b) == 0:
            return 0
        data = await self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def readable(self) -> bool:
        return self._evalkit_pos < len(self._evalkit_content)

    def tell(self) -> int:
        return self._evalkit_pos

    async def iter_chunks(self, chunk_size: int = 1024) -> Any:
        while True:
            chunk = await self.read(chunk_size)
            if not chunk:
                break
            yield chunk

    async def iter_lines(self, chunk_size: int = 1024, keepends: bool = False) -> Any:
        pending = b""
        async for chunk in self.iter_chunks(chunk_size):
            lines = (pending + chunk).splitlines(True)
            for line in lines[:-1]:
                yield line.splitlines(keepends)[0]
            pending = lines[-1]
        if pending:
            yield pending.splitlines(keepends)[0]

    async def readlines(self) -> List[bytes]:
        return [line async for line in self.iter_lines()]

    def __aiter__(self) -> Any:
        return self.iter_chunks()

    async def __anext__(self) -> bytes:
        chunk = await self.read(1024)
        if chunk:
            return chunk
        raise StopAsyncIteration

    anext = __anext__

    async def __aenter__(self) -> "_AsyncReplayBody":
        return self

    async def __aexit__(self, *args: object) -> bool:
        await _close_maybe_async(self._evalkit_stream)
        return False

    def __getattr__(self, name: str) -> Any:
        stream = self.__dict__.get("_evalkit_stream")
        if stream is None:
            raise AttributeError(name)
        return getattr(stream, name)


async def _close_maybe_async(stream: Any) -> None:
    closer = getattr(stream, "aclose", None) or getattr(stream, "close", None)
    if closer is None:
        return
    try:
        result = closer()
        if inspect.isawaitable(result):
            await result
    except Exception:
        pass


def _is_event_stream(value: Any) -> bool:
    if isinstance(value, (dict, list, tuple, str, bytes, bytearray)):
        return False
    return hasattr(value, "__iter__") and not hasattr(value, "read")


def _stream_key(result: Any, preferred: Optional[str]) -> Optional[str]:
    if not isinstance(result, dict):
        return None
    if preferred and _is_event_stream(result.get(preferred)):
        return preferred
    for key, value in result.items():
        if key != "ResponseMetadata" and _is_event_stream(value):
            return key
    return None


def _install_stream(span: _BedrockSpan, plan: _Plan, result: Any, async_stream: bool = False) -> bool:
    key = _stream_key(result, plan.stream_key)
    if key is None:
        return False
    stream = result[key]
    acc = _bb.BedrockAcc(span.start_ms)
    kind = plan.stream_kind

    def on_event(event: Any) -> None:
        if kind == "converse":
            _bb.converse_stream_event(event, acc)
        elif kind == "invoke":
            _bb.invoke_stream_event(event, plan.family, acc)
        else:
            _bb.agent_stream_event(event, acc)

    def on_done(exc: Optional[BaseException]) -> None:
        span.emit_acc(acc, streaming=True, exc=exc)

    tracer = _AsyncBedrockStreamTracer if async_stream else _BedrockStreamTracer
    result[key] = tracer(stream, on_event, on_done)
    return True


class _BedrockStreamTracer:
    def __init__(
        self, stream: Any,
        on_event: Callable[[Any], None],
        on_done: Callable[[Optional[BaseException]], None],
    ) -> None:
        self._stream = stream
        self._on_event = on_event
        self._on_done = on_done
        self._done = False

    def _finish(self, exc: Optional[BaseException] = None) -> None:
        if self._done:
            return
        self._done = True
        try:
            self._on_done(exc)
        except Exception:
            pass

    def __iter__(self) -> Any:
        try:
            for event in self._stream:
                try:
                    self._on_event(event)
                except Exception:
                    pass
                yield event
        except Exception as exc:
            self._finish(exc)
            raise
        finally:
            self._finish()

    def close(self) -> None:
        try:
            closer = getattr(self._stream, "close", None)
            if closer is not None:
                closer()
        finally:
            self._finish()

    def __enter__(self) -> "_BedrockStreamTracer":
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args: object) -> object:
        self._finish()
        if hasattr(self._stream, "__exit__"):
            return self._stream.__exit__(*args)
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _AsyncBedrockStreamTracer:
    def __init__(
        self, stream: Any,
        on_event: Callable[[Any], None],
        on_done: Callable[[Optional[BaseException]], None],
    ) -> None:
        self._stream = stream
        self._on_event = on_event
        self._on_done = on_done
        self._done = False

    def _finish(self, exc: Optional[BaseException] = None) -> None:
        if self._done:
            return
        self._done = True
        try:
            self._on_done(exc)
        except Exception:
            pass

    async def __aiter__(self) -> Any:
        try:
            async for event in self._stream:
                try:
                    self._on_event(event)
                except Exception:
                    pass
                yield event
        except Exception as exc:
            self._finish(exc)
            raise
        finally:
            self._finish()

    def close(self) -> Any:
        try:
            closer = getattr(self._stream, "close", None)
            if closer is not None:
                return closer()
        finally:
            self._finish()

    async def aclose(self) -> None:
        try:
            await _close_maybe_async(self._stream)
        finally:
            self._finish()

    async def __aenter__(self) -> "_AsyncBedrockStreamTracer":
        enter = getattr(self._stream, "__aenter__", None)
        if enter is not None:
            await enter()
        return self

    async def __aexit__(self, *args: object) -> Any:
        self._finish()
        exit_ = getattr(self._stream, "__aexit__", None)
        if exit_ is not None:
            return await exit_(*args)
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


async def _trace_retrieve_async(ek: EvalKitClient, params: Dict[str, Any], invoke: Callable[[], Any]) -> Any:
    end = _start_retrieve(ek, params)
    try:
        result = await invoke()
    except Exception as exc:
        end("ERROR", error=exc)
        raise
    return _finish_retrieve(end, params, result)


def _start_retrieve(ek: EvalKitClient, params: Dict[str, Any]) -> Callable[..., Any]:
    query = params.get("retrievalQuery")
    statement = query.get("text") if isinstance(query, dict) else None
    return start_db(ek, "bedrock", operation="Retrieve", statement=statement if isinstance(statement, str) else None)


def _trace_retrieve(ek: EvalKitClient, params: Dict[str, Any], invoke: Callable[[], Any]) -> Any:
    end = _start_retrieve(ek, params)
    try:
        result = invoke()
    except Exception as exc:
        end("ERROR", error=exc)
        raise
    return _finish_retrieve(end, params, result)


def _finish_retrieve(end: Callable[..., Any], params: Dict[str, Any], result: Any) -> Any:
    extra: Dict[str, Any] = {"gen_ai.operation.name": "retrieval"}
    rows: Any = None
    try:
        knowledge_base = params.get("knowledgeBaseId")
        if isinstance(knowledge_base, str) and knowledge_base:
            extra["aws.bedrock.knowledge_base.id"] = knowledge_base
        results = result.get("retrievalResults") if isinstance(result, dict) else None
        if isinstance(results, list):
            rows = len(results)
            extra["db.vector.match_count"] = rows
        request_body = _safe_body(_dump(params))
        if request_body:
            extra["gen_ai.request.body"] = request_body
        response_body = _safe_body(_dump(result))
        if response_body:
            extra["gen_ai.response.body"] = response_body
    except Exception:
        pass
    end("OK", extra=extra, rows=rows)
    return result
