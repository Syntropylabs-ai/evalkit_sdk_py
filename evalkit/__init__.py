from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from ._version import SDK_VERSION
from .core.client import EvalKitClient
from .core.context import SpanContext, _current_span, get_or_create_ambient_trace_id
from .core.types import EvalKitOptions, SpanEvent, TraceEnvelope
from .evaluate import evaluate
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
    "evaluate",
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
) -> EvalKitClient:
    global _instance
    if _instance is not None:
        return _instance  # already initialized — reuse; patch closures reference this client
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
    )
    _instance = EvalKitClient(opts)

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
            pass  # instrumentation must never break init()
    return _instance


def _install_function_tracing(client: EvalKitClient, trace_packages: Optional[List[str]]) -> None:
    """Resolve the caller's source directory and install the import-time tracer."""
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


def flush() -> None:
    _get_instance().flush()


def current_trace_id() -> str:
    ctx = _current_span.get()
    return ctx.trace_id if ctx else get_or_create_ambient_trace_id()


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
    """Register the global LangChain/LangGraph callback handler. Called by
    init() automatically; returns the handler (also usable as callbacks=[h])."""
    return _patch_langchain(_get_instance())


def patch_litellm() -> None:
    """Instrument litellm.completion/acompletion. Called by init() automatically."""
    _patch_litellm(_get_instance())


def patch_ollama() -> None:
    """Instrument ollama.Client/AsyncClient.chat. Called by init() automatically."""
    _patch_ollama(_get_instance())


def patch_pinecone() -> None:
    """Instrument pinecone.Index.query/upsert as db_query spans. Called by init() automatically."""
    _patch_pinecone(_get_instance())


def patch_qdrant() -> None:
    """Instrument qdrant_client.QdrantClient.query_points/upsert as db_query spans. Called by init() automatically."""
    _patch_qdrant(_get_instance())


def patch_chroma() -> None:
    """Instrument chromadb Collection.query/add/upsert as db_query spans. Called by init() automatically."""
    _patch_chroma(_get_instance())


def patch_weaviate() -> None:
    """Instrument weaviate-client near_vector query + insert/insert_many as db_query spans.
    Called by init() automatically."""
    _patch_weaviate(_get_instance())


def patch_llamaindex() -> None:
    """Register evalkit's handlers on LlamaIndex's instrumentation dispatcher, capturing
    llm/embedding/retrieval/query/rerank spans. Called by init() automatically."""
    _patch_llamaindex(_get_instance())


def patch_claude_agent_sdk() -> None:
    """Instrument ``claude_agent_sdk.query`` and ``ClaudeSDKClient.receive_response``
    to capture token/cost/latency data from ``ResultMessage`` as ``llm_call`` spans.

    Called automatically by ``init()`` when ``claude_agent_sdk`` is installed.
    Call explicitly if you install the package after calling ``init()``.
    """
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
    """Explicit drop-in for ``client.aio.live.connect(model=, config=)`` (Gemini Live).

    Returns an async context manager that opens a ``voice.session`` trace and emits
    ``voice.turn`` / ``tool_call`` spans as the WebSocket streams. Use::

        async with evalkit.trace_google_live(genai_client, model=..., config=...) as session:
            ...
    """
    return _trace_google_live(_get_instance(), genai_client, model=model, config=config)


def trace_openai_realtime(
    openai_client: Any,
    *,
    model: Optional[str] = None,
) -> Any:
    """Explicit drop-in for ``client.beta.realtime.connect(model=)`` (OpenAI Realtime).

    Returns an async context manager that opens a ``voice.session`` trace and emits
    ``voice.turn`` / ``tool_call`` spans as the WebSocket streams. Use::

        async with evalkit.trace_openai_realtime(openai_client, model=...) as connection:
            ...
    """
    return _trace_openai_realtime(_get_instance(), openai_client, model=model)


def patch_google_genai_live() -> None:
    """Best-effort: auto-wrap ``Live.connect`` so plain Gemini Live connects are
    traced. Called automatically by ``init()`` when ``google.genai`` is installed."""
    _patch_google_genai_live(_get_instance())


def patch_openai_realtime() -> None:
    """Best-effort: auto-wrap ``AsyncRealtime.connect`` so plain OpenAI Realtime
    connects are traced. Called automatically by ``init()`` when ``openai`` is installed."""
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
) -> Tuple[str, Callable[[str], None], SpanContext]:
    return _get_instance().start_trace(name, attributes, session_id=session_id, device_id=device_id)


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
    """Decorator that traces a function as a first-class ``tool_call`` span
    (renders in the Input/Output panels and counts toward tool metrics). Use it
    on your own client-side tools."""
    return _get_instance().trace_tool(name, attributes, capture_io=capture_io)


def traced(cls: Any = None, *, prefix: Optional[str] = None, capture_io: bool = True) -> Any:
    """Class decorator — trace EVERY user-defined method of the class as a
    function_call span, APM-style, without decorating each method:

        @evalkit.traced
        class OrderService:
            def place(self, ...): ...
            def cancel(self, ...): ...

    Both ``@evalkit.traced`` and ``@evalkit.traced(prefix="orders")`` work.
    Call ``init()`` before the class is defined.
    """
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
    """Trace every function DEFINED IN ``module`` (in place), APM-style. One call
    instruments all of your app's functions in a module:

        import myapp.services as svc
        evalkit.trace_module(svc)

    Skips imported functions (only wraps those defined in the module) and, by
    default, names starting with ``_``. Returns the number of functions wrapped.
    """
    import types as _types
    inst = _get_instance()
    mod_name = getattr(module, "__name__", "")
    label = prefix or mod_name.split(".")[-1]
    count = 0
    for attr_name, member in list(vars(module).items()):
        if not isinstance(member, _types.FunctionType):
            continue
        if getattr(member, "__module__", None) != mod_name:
            continue  # only functions defined here, not imports
        if not include_private and attr_name.startswith("_"):
            continue
        if getattr(member, "_evalkit_traced", False):
            continue
        wrapped = inst.trace_function(f"{label}.{attr_name}", capture_io=capture_io)(member)
        try:
            wrapped._evalkit_traced = True  # type: ignore[attr-defined]
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
    """Recursively trace every function in every module of a package, APM-style.
    Framework-agnostic — works for FastAPI / Flask / Django / any Python app:

        import myapp
        evalkit.trace_package(myapp)   # traces functions across all of myapp.*

    Imports each submodule (so import side effects run) and wraps its functions
    with trace_function. Returns the total number of functions wrapped.
    """
    import importlib
    import pkgutil

    total = trace_module(
        package, prefix=prefix, capture_io=capture_io, include_private=include_private
    )
    pkg_path = getattr(package, "__path__", None)
    pkg_name = getattr(package, "__name__", "")
    if not pkg_path:
        return total  # a plain module, not a package
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
