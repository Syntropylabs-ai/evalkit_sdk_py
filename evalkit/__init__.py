from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from ._version import SDK_VERSION
from .core.client import EvalKitClient
from .core.context import (
    Identity,
    SpanContext,
    _current_span,
    current_agent,
    current_identity,
    get_or_create_ambient_trace_id,
    reset_agent,
    reset_identity,
    set_agent_name,
    set_identity,
)
from .core.scopes import session
from .core.types import EvalKitOptions, SpanEvent, TraceEnvelope
from .evaluate import evaluate
from .eval import Eval, EvalResult
from .feedback import feedback, reward
from . import prompts
from .prompts import Prompt
from .simulation import AgentTurnResult, SimContext, evaluate_simulation, generate_scenarios, simulate_user, with_simulation

from .instrumentation.llm.openai import patch_openai
from .instrumentation.llm.openai_async import patch_async_openai
from .instrumentation.llm.anthropic import patch_anthropic
from .instrumentation.llm.anthropic_async import patch_async_anthropic
from .instrumentation.llm.bedrock import patch_bedrock
from .instrumentation.llm.vertex_ai import patch_anthropic_vertex, patch_vertex_generative_model
from .instrumentation.llm.cohere import patch_cohere
from .instrumentation.llm.google_ai import patch_google_ai
from .instrumentation.llm.google_genai import patch_google_genai as _patch_google_genai
from .instrumentation.llm.mistral import patch_mistral, patch_async_mistral
from .instrumentation.llm.langchain import patch_langchain as _patch_langchain
from .instrumentation.llm.litellm import patch_litellm as _patch_litellm
from .instrumentation.llm.ollama import patch_ollama as _patch_ollama
from .instrumentation.vectordb.pinecone import patch_pinecone as _patch_pinecone
from .instrumentation.vectordb.qdrant import patch_qdrant as _patch_qdrant
from .instrumentation.vectordb.chroma import patch_chroma as _patch_chroma
from .instrumentation.vectordb.weaviate import patch_weaviate as _patch_weaviate
from .instrumentation.llm.llamaindex import patch_llamaindex as _patch_llamaindex
from .instrumentation.llm.openai_audio import patch_openai_audio as _patch_openai_audio
from .instrumentation.llm.elevenlabs import patch_elevenlabs as _patch_elevenlabs
from .instrumentation.llm.assemblyai import patch_assemblyai as _patch_assemblyai
from .instrumentation.llm.claude_agent_sdk import patch_claude_agent_sdk as _patch_claude_agent_sdk
from .instrumentation.llm.google_live import (
    trace_google_live as _trace_google_live,
    patch_google_genai_live as _patch_google_genai_live,
)
from .instrumentation.llm.openai_realtime import (
    trace_openai_realtime as _trace_openai_realtime,
    patch_openai_realtime as _patch_openai_realtime,
)

from .instrumentation.http.requests import patch_requests
from .instrumentation.http.httpx import patch_httpx
from .instrumentation.http.aiohttp import patch_aiohttp as _patch_aiohttp

from .instrumentation.db.sqlalchemy import patch_sqlalchemy, patch_sqlalchemy_async

from .instrumentation.runtime.logging import EvalKitLoggingHandler, EvalKitLogFilter, install_logging_handler

from .frameworks.fastapi import EvalKitMiddleware
from .frameworks.django import EvalKitDjangoMiddleware
from .frameworks.flask import instrument_flask as _instrument_flask
from .frameworks.starlette import EvalKitMiddleware as EvalKitStarletteMiddleware
from .frameworks.litestar import create_litestar_middleware

__version__ = SDK_VERSION

__all__ = [
    "init",
    "auto_instrument",
    "flush",
    "current_trace_id",
    "set_user",
    "clear_user",
    "current_user",
    "session",
    "set_agent",
    "clear_agent",
    "current_agent",
    "Identity",
    "evaluate",
    "Eval",
    "EvalResult",
    "feedback",
    "reward",
    "prompts",
    "Prompt",
    "generate_scenarios",
    "simulate_user",
    "evaluate_simulation",
    "with_simulation",
    "SimContext",
    "AgentTurnResult",
    "instrument_flask",
    "EvalKitMiddleware",
    "EvalKitStarletteMiddleware",
    "EvalKitDjangoMiddleware",
    "create_litestar_middleware",
    "patch_sqlalchemy_engine",
    "patch_async_sqlalchemy_engine",
    "patch_openai_client",
    "patch_anthropic_client",
    "patch_bedrock_client",
    "patch_anthropic_vertex_client",
    "patch_vertex_generative_model",
    "patch_cohere_client",
    "patch_google_ai_model",
    "patch_google_genai",
    "patch_langchain",
    "patch_litellm",
    "patch_ollama",
    "patch_pinecone",
    "patch_qdrant",
    "patch_chroma",
    "patch_weaviate",
    "patch_llamaindex",
    "patch_openai_audio",
    "patch_elevenlabs",
    "patch_assemblyai",
    "patch_claude_agent_sdk",
    "trace_google_live",
    "trace_openai_realtime",
    "patch_google_genai_live",
    "patch_openai_realtime",
    "patch_mistral_client",
    "patch_async_openai_client",
    "patch_async_anthropic_client",
    "patch_async_mistral_client",
    "start_trace",
    "start_http_trace",
    "start_span",
    "with_trace",
    "trace_function",
    "trace_tool",
    "traced",
    "trace_module",
    "trace_package",
    "install_logging_handler",
    "EvalKitLoggingHandler",
    "EvalKitLogFilter",
    "EvalKitClient",
    "EvalKitOptions",
    "SpanContext",
    "TraceEnvelope",
    "SpanEvent",
]

_instance: Optional[EvalKitClient] = None


def _get_instance() -> EvalKitClient:
    if _instance is None:
        raise RuntimeError("[evalkit] call evalkit.init() before using the SDK")
    return _instance


def init(
    subscription_key: str,
    base_url: str = "https://api.syntropylabs.ai",
    api_url: str = "https://api.syntropylabs.ai",
    service_name: str = "unknown",
    debug: bool = False,
    max_export_batch_size: int = 512,
    scheduled_delay_ms: int = 5000,
    max_queue_size: int = 4096,
    max_body_bytes: int = 10 * 1024 * 1024,
    environment: Optional[str] = None,
    app_version: Optional[str] = None,
    device_id: Optional[str] = None,
    function_tracing: bool = True,
    trace_packages: Optional[List[str]] = None,
    capture_content: Optional[bool] = None,
    capture_secrets: Optional[bool] = None,
    capture_sensitive_headers: Optional[bool] = None,
    capture_stream_bodies: Optional[bool] = None,
    mask: Optional[Callable[[TraceEnvelope], Optional[TraceEnvelope]]] = None,
    export_gzip: Optional[bool] = None,
) -> EvalKitClient:
    global _instance
    if _instance is not None:
        return _instance
    from .core.exporter import resolve_export_gzip
    from .core.redact import (
        resolve_capture_content,
        resolve_capture_secrets,
        resolve_capture_stream_bodies,
    )
    from .utils.headers import set_capture_sensitive_headers

    secrets_explicit = capture_secrets if capture_secrets is not None else capture_sensitive_headers
    opts = EvalKitOptions(
        subscription_key=subscription_key,
        base_url=base_url,
        api_url=api_url,
        service_name=service_name,
        debug=debug,
        max_export_batch_size=max_export_batch_size,
        scheduled_delay_ms=scheduled_delay_ms,
        max_queue_size=max_queue_size,
        max_body_bytes=max_body_bytes,
        environment=environment,
        app_version=app_version,
        device_id=device_id,
        capture_content=resolve_capture_content(capture_content),
        capture_secrets=resolve_capture_secrets(secrets_explicit),
        capture_stream_bodies=resolve_capture_stream_bodies(capture_stream_bodies),
        mask=mask,
        export_gzip=resolve_export_gzip(export_gzip),
    )
    set_capture_sensitive_headers(opts.capture_secrets)
    _instance = EvalKitClient(opts)
    try:
        from .core.client import flush_on_termination_signals
        flush_on_termination_signals(_instance)
    except Exception:
        pass

    try:
        from .core.otel_provider import setup_otel
        setup_otel(_instance)
    except Exception:
        pass

    from .instrumentation.auto import setup_auto_instrumentation
    setup_auto_instrumentation(_instance)


    if function_tracing and os.environ.get("EVALKIT_FUNCTION_TRACE", "true") != "false":
        try:
            _install_function_tracing(_instance, trace_packages)
        except Exception:
            pass
    return _instance


def _install_function_tracing(client: EvalKitClient, trace_packages: Optional[List[str]]) -> None:
    import inspect

    from .instrumentation.functrace import install_function_tracer

    root_dir: Optional[str] = None
    for frame_info in inspect.stack()[1:]:
        mod = inspect.getmodule(frame_info.frame)
        name = getattr(mod, "__name__", "") or ""
        if name.startswith("evalkit"):
            continue
        f = frame_info.filename
        if f and os.path.isfile(f):
            root_dir = os.path.dirname(os.path.realpath(f))
            break
    if not root_dir:
        root_dir = os.getcwd()
    install_function_tracer(client, root_dir, extra_packages=trace_packages or ())


def auto_instrument(
    logging: bool = True,
    requests: bool = True,
    httpx: bool = True,
) -> None:
    from .instrumentation.auto import setup_auto_instrumentation
    setup_auto_instrumentation(_get_instance(), logging=logging, requests=requests, httpx=httpx)


def flush() -> bool:
    """Send what is queued now; whether the receiver took all of it.

    Shutdown hooks, test teardown and exit paths call this whether or not `init()` ran, so before it
    there is nothing to send and the answer is False; it never raises.
    """
    try:
        instance = _instance
        return bool(instance.flush()) if instance is not None else False
    except Exception:
        return False


def current_trace_id() -> str:
    ctx = _current_span.get()
    return ctx.trace_id if ctx else get_or_create_ambient_trace_id()


def set_user(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    device_id: Optional[str] = None,
) -> Any:
    return set_identity(user_id=user_id, session_id=session_id, device_id=device_id)


def clear_user(token: Optional[Any] = None) -> None:
    if token is None:
        set_identity()
        return
    reset_identity(token)


def current_user() -> Identity:
    return current_identity()


def set_agent(name: Optional[str]) -> Any:
    """Name the agent every span from here on belongs to (``None`` clears it).

    Scoped like ``set_user``: per thread and per asyncio task. Returns a token for
    ``clear_agent`` to restore the previous name.
    """
    return set_agent_name(name)


def clear_agent(token: Optional[Any] = None) -> None:
    if token is None:
        set_agent_name(None)
        return
    reset_agent(token)


def instrument_flask(app: Any) -> None:
    _instrument_flask(app, _get_instance())


def patch_sqlalchemy_engine(engine: Any) -> None:
    patch_sqlalchemy(_get_instance(), engine)


def patch_async_sqlalchemy_engine(async_engine: Any) -> None:
    patch_sqlalchemy_async(_get_instance(), async_engine)


def patch_openai_client(
    openai_client: Any,
    parent_ctx: Optional[SpanContext] = None,
) -> None:
    patch_openai(_get_instance(), openai_client, parent_ctx)


def patch_anthropic_client(
    anthropic_client: Any,
    parent_ctx: Optional[SpanContext] = None,
) -> None:
    patch_anthropic(_get_instance(), anthropic_client, parent_ctx)


def patch_bedrock_client(bedrock_runtime_client: Any) -> None:
    patch_bedrock(_get_instance(), bedrock_runtime_client)


def patch_anthropic_vertex_client(
    anthropic_vertex_client: Any,
    parent_ctx: Optional[SpanContext] = None,
) -> None:
    patch_anthropic_vertex(_get_instance(), anthropic_vertex_client, parent_ctx)


def patch_vertex_generative_model(model: Any) -> None:
    from .instrumentation.llm.vertex_ai import patch_vertex_generative_model as _patch_vertex
    _patch_vertex(_get_instance(), model)


def patch_cohere_client(cohere_client: Any) -> None:
    patch_cohere(_get_instance(), cohere_client)


def patch_google_ai_model(model: Any) -> None:
    patch_google_ai(_get_instance(), model)


def patch_google_genai() -> None:
    _patch_google_genai(_get_instance())


def patch_langchain() -> Any:
    return _patch_langchain(_get_instance())


def patch_litellm() -> None:
    _patch_litellm(_get_instance())


def patch_ollama() -> None:
    _patch_ollama(_get_instance())


def patch_pinecone() -> None:
    _patch_pinecone(_get_instance())


def patch_qdrant() -> None:
    _patch_qdrant(_get_instance())


def patch_chroma() -> None:
    _patch_chroma(_get_instance())


def patch_weaviate() -> None:
    _patch_weaviate(_get_instance())


def patch_llamaindex() -> None:
    _patch_llamaindex(_get_instance())


def patch_openai_audio() -> None:
    _patch_openai_audio(_get_instance())


def patch_elevenlabs() -> None:
    _patch_elevenlabs(_get_instance())


def patch_assemblyai() -> None:
    _patch_assemblyai(_get_instance())


def patch_claude_agent_sdk() -> None:
    _patch_claude_agent_sdk(_get_instance())


def patch_mistral_client(
    mistral_client: Any,
    parent_ctx: Optional[SpanContext] = None,
) -> None:
    patch_mistral(_get_instance(), mistral_client, parent_ctx)


def trace_google_live(
    genai_client: Any,
    *,
    model: Optional[str] = None,
    config: Any = None,
) -> Any:
    return _trace_google_live(_get_instance(), genai_client, model=model, config=config)


def trace_openai_realtime(
    openai_client: Any,
    *,
    model: Optional[str] = None,
) -> Any:
    return _trace_openai_realtime(_get_instance(), openai_client, model=model)


def patch_google_genai_live() -> None:
    _patch_google_genai_live(_get_instance())


def patch_openai_realtime() -> None:
    _patch_openai_realtime(_get_instance())


def patch_async_openai_client(
    async_openai_client: Any,
    parent_ctx: Optional[SpanContext] = None,
) -> None:
    patch_async_openai(_get_instance(), async_openai_client, parent_ctx)


def patch_async_anthropic_client(
    async_anthropic_client: Any,
    parent_ctx: Optional[SpanContext] = None,
) -> None:
    patch_async_anthropic(_get_instance(), async_anthropic_client, parent_ctx)


def patch_async_mistral_client(
    async_mistral_client: Any,
    parent_ctx: Optional[SpanContext] = None,
) -> None:
    patch_async_mistral(_get_instance(), async_mistral_client, parent_ctx)


def start_trace(
    name: str,
    attributes: Optional[Dict[str, str]] = None,
    session_id: Optional[str] = None,
    device_id: Optional[str] = None,
    user_id: Optional[str] = None,
    agent: Optional[str] = None,
) -> Tuple[str, Callable[..., None], SpanContext]:
    return _get_instance().start_trace(
        name,
        attributes,
        session_id=session_id,
        device_id=device_id,
        user_id=user_id,
        agent=agent,
    )


def start_http_trace(
    name: str,
    *,
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    request_body: Any = None,
    attributes: Optional[Dict[str, str]] = None,
) -> Tuple[str, Callable[..., None], SpanContext]:
    return _get_instance().start_http_trace(
        name, method=method, url=url, headers=headers,
        request_body=request_body, attributes=attributes,
    )


def start_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    parent_ctx: Optional[SpanContext] = None,
) -> Tuple[Callable[[str], None], SpanContext]:
    return _get_instance().start_span(name, attributes, parent_ctx)


@contextmanager
def with_trace(ctx: SpanContext) -> Generator[SpanContext, None, None]:
    token = _current_span.set(ctx)
    try:
        yield ctx
    finally:
        _current_span.reset(token)


def trace_function(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    capture_io: bool = True,
) -> Callable:
    return _get_instance().trace_function(name, attributes, capture_io=capture_io)


def trace_tool(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    capture_io: bool = True,
) -> Callable:
    return _get_instance().trace_tool(name, attributes, capture_io=capture_io)


def traced(cls: Any = None, *, prefix: Optional[str] = None, capture_io: bool = True) -> Any:
    def wrap(klass: Any) -> Any:
        label = prefix or klass.__name__
        for attr_name, member in list(vars(klass).items()):
            if attr_name.startswith("__"):
                continue
            span_name = f"{label}.{attr_name}"
            if isinstance(member, staticmethod):
                fn = member.__func__
                setattr(klass, attr_name, staticmethod(
                    _get_instance().trace_function(span_name, capture_io=capture_io)(fn)))
            elif isinstance(member, classmethod):
                fn = member.__func__
                setattr(klass, attr_name, classmethod(
                    _get_instance().trace_function(span_name, capture_io=capture_io)(fn)))
            elif callable(member):
                setattr(klass, attr_name,
                        _get_instance().trace_function(span_name, capture_io=capture_io)(member))
        return klass
    return wrap(cls) if cls is not None else wrap


def trace_module(
    module: Any,
    *,
    prefix: Optional[str] = None,
    capture_io: bool = True,
    include_private: bool = False,
) -> int:
    import types as _types
    inst = _get_instance()
    mod_name = getattr(module, "__name__", "")
    label = prefix or mod_name.split(".")[-1]
    count = 0
    for attr_name, member in list(vars(module).items()):
        if not isinstance(member, _types.FunctionType):
            continue
        if getattr(member, "__module__", None) != mod_name:
            continue
        if not include_private and attr_name.startswith("_"):
            continue
        if getattr(member, "_evalkit_traced", False):
            continue
        wrapped = inst.trace_function(f"{label}.{attr_name}", capture_io=capture_io)(member)
        try:
            wrapped._evalkit_traced = True
        except Exception:
            pass
        setattr(module, attr_name, wrapped)
        count += 1
    return count


def trace_package(
    package: Any,
    *,
    prefix: Optional[str] = None,
    capture_io: bool = True,
    include_private: bool = False,
) -> int:
    import importlib
    import pkgutil

    total = trace_module(
        package, prefix=prefix, capture_io=capture_io, include_private=include_private
    )
    pkg_path = getattr(package, "__path__", None)
    pkg_name = getattr(package, "__name__", "")
    if not pkg_path:
        return total
    for mod_info in pkgutil.walk_packages(pkg_path, prefix=pkg_name + "."):
        try:
            mod = importlib.import_module(mod_info.name)
        except Exception:
            continue
        try:
            total += trace_module(
                mod, prefix=prefix, capture_io=capture_io, include_private=include_private
            )
        except Exception:
            continue
    return total
