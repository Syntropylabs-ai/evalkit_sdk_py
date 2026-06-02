from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional, Tuple

from ._version import SDK_VERSION
from .core.client import EvalKitClient
from .core.context import SpanContext, _current_span, get_or_create_ambient_trace_id
from .core.types import EvalKitOptions, SpanEvent, TraceEnvelope
from .evaluate import evaluate
from .simulation import AgentTurnResult, SimContext, generate_scenarios, simulate_user, with_simulation

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

from .instrumentation.http.requests import patch_requests
from .instrumentation.http.httpx import patch_httpx

from .instrumentation.db.sqlalchemy import patch_sqlalchemy

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
    "with_simulation",
    "SimContext",
    "AgentTurnResult",
    "instrument_flask",
    "EvalKitMiddleware",
    "EvalKitStarletteMiddleware",
    "EvalKitDjangoMiddleware",
    "create_litestar_middleware",
    "patch_sqlalchemy_engine",
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
    "patch_mistral_client",
    "patch_async_openai_client",
    "patch_async_anthropic_client",
    "patch_async_mistral_client",
    "start_trace",
    "start_http_trace",
    "start_span",
    "with_trace",
    "trace_function",
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
    environment: Optional[str] = None,
    app_version: Optional[str] = None,
    device_id: Optional[str] = None,
) -> EvalKitClient:
    global _instance
    if _instance is not None:
        _instance.destroy()
    opts = EvalKitOptions(
        subscription_key=subscription_key,
        base_url=base_url,
        api_url=api_url,
        service_name=service_name,
        debug=debug,
        max_export_batch_size=max_export_batch_size,
        scheduled_delay_ms=scheduled_delay_ms,
        max_queue_size=max_queue_size,
        environment=environment,
        app_version=app_version,
        device_id=device_id,
    )
    _instance = EvalKitClient(opts)
    from .instrumentation.auto import setup_auto_instrumentation
    setup_auto_instrumentation(_instance)
    return _instance


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


def patch_mistral_client(
    mistral_client: Any,
    parent_ctx: Optional[SpanContext] = None,
) -> None:
    patch_mistral(_get_instance(), mistral_client, parent_ctx)


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
