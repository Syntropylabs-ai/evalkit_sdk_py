from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from ._version import SDK_VERSION
from .core.client import EvalKitClient
from .core.config import changed_options, config_problems, disabled_by_env, resolve_log_level
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
from .core.ignore import RouteInfo, UrlInfo
from .core.scopes import session
from .core.untraced import UNTRACED_MARKER, untraced
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
    "stats",
    "set_error_fingerprint",
    "clear_error_fingerprint",
    "current_trace_id",
    "set_user",
    "clear_user",
    "current_user",
    "session",
    "set_agent",
    "clear_agent",
    "current_agent",
    "untraced",
    "RouteInfo",
    "UrlInfo",
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
# What the first init() call asked for, to tell a later call with other options that they do not apply.
_init_options: Optional[Dict[str, Any]] = None
_reinit_warned = False

_log = logging.getLogger("evalkit")



def _get_instance() -> EvalKitClient:
    if _instance is None:
        raise RuntimeError("[evalkit] call evalkit.init() before using the SDK")
    return _instance


def _warn_if_reconfigured(options: Dict[str, Any]) -> None:
    global _reinit_warned
    first = _init_options
    if first is None or _reinit_warned:
        return
    changed = changed_options(first, options)
    if not changed:
        return
    _reinit_warned = True
    _log.warning(
        "evalkit.init() was called again with different options (%s); the first call's settings stay in "
        "effect for the life of the process.",
        ", ".join(changed),
    )


def _noop_client(opts: EvalKitOptions) -> EvalKitClient:
    return EvalKitClient(opts, enabled=False)


def _placeholder_options(options: Dict[str, Any]) -> EvalKitOptions:
    service = options.get("service_name")
    return EvalKitOptions(
        subscription_key="",
        service_name=service if isinstance(service, str) else "unknown",
    )


def init(
    subscription_key: str,
    base_url: str = "https://api.syntropylabs.ai",
    api_url: str = "https://api.syntropylabs.ai",
    service_name: str = "unknown",
    debug: bool = False,
    max_export_batch_size: int = 512,
    scheduled_delay_ms: int = 5000,
    max_queue_size: int = 4096,
    max_body_bytes: int = 64 * 1024,
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
    *,
    max_queue_bytes: int = 32 * 1024 * 1024,
    max_value_bytes: Optional[int] = None,
    function_rate_limit: Optional[float] = 100.0,
    log_level: Any = None,
    log_rate_limit: Optional[float] = 50.0,
    sample_rate: Optional[float] = None,
    capture_unhandled: bool = False,
    in_app_include: Optional[List[str]] = None,
    in_app_exclude: Optional[List[str]] = None,
    ignore_routes: Optional[List[Any]] = None,
    ignore_urls: Optional[List[Any]] = None,
    ignore_functions: Optional[List[Any]] = None,
    trace_health_checks: bool = False,
) -> EvalKitClient:
    """Start recording and exporting spans; returns the client.

    Never raises and never stops the application: an invalid option, ``EVALKIT_DISABLED=true`` or
    ``OTEL_SDK_DISABLED=true`` make the SDK a no-op (the first with one warning saying what is wrong),
    and an integration that fails to install is skipped. The first call's options stay in effect; a
    later call returns the same client, with a warning when its options differ.

    ``ignore_routes``, ``ignore_urls`` and ``ignore_functions`` keep requests, outgoing calls and
    functions out of traces (merged with ``EVALKIT_IGNORE_ROUTES``, ``EVALKIT_IGNORE_URLS`` and
    ``EVALKIT_IGNORE_FUNCTIONS``); a pattern that cannot be read is skipped with a warning. Health
    probes are left out unless ``trace_health_checks=True``.
    """
    global _instance, _init_options
    options: Dict[str, Any] = dict(
        subscription_key=subscription_key, base_url=base_url, api_url=api_url, service_name=service_name,
        debug=debug, max_export_batch_size=max_export_batch_size, scheduled_delay_ms=scheduled_delay_ms,
        max_queue_size=max_queue_size, max_body_bytes=max_body_bytes, environment=environment,
        app_version=app_version, device_id=device_id, function_tracing=function_tracing,
        trace_packages=trace_packages, capture_content=capture_content, capture_secrets=capture_secrets,
        capture_sensitive_headers=capture_sensitive_headers, capture_stream_bodies=capture_stream_bodies,
        mask=mask, export_gzip=export_gzip, max_queue_bytes=max_queue_bytes, max_value_bytes=max_value_bytes,
        function_rate_limit=function_rate_limit, log_level=log_level, log_rate_limit=log_rate_limit,
        sample_rate=sample_rate, capture_unhandled=capture_unhandled, in_app_include=in_app_include,
        in_app_exclude=in_app_exclude, ignore_routes=ignore_routes, ignore_urls=ignore_urls,
        ignore_functions=ignore_functions, trace_health_checks=trace_health_checks,
    )
    if _instance is not None:
        try:
            _warn_if_reconfigured(options)
        except Exception:
            pass
        return _instance
    _init_options = options
    try:
        disabled_by = disabled_by_env()
        if disabled_by:
            _log.info("%s is set: EvalKit records and sends nothing in this process.", disabled_by)
            _instance = _noop_client(_placeholder_options(options))
            return _instance
        problems = config_problems(options)
        if problems:
            _log.warning(
                "evalkit.init(): invalid configuration, so EvalKit records and sends nothing in this process "
                "(the application runs as usual): %s.",
                "; ".join(problems),
            )
            _instance = _noop_client(_placeholder_options(options))
            return _instance
        _instance = _start(options)
    except Exception as exc:
        from .core.secrets import scrub_string

        _log.warning("evalkit.init() failed (%s: %s); EvalKit records nothing in this process.",
                     type(exc).__name__, scrub_string(str(exc)))
        if _instance is None:
            try:
                _instance = _noop_client(_placeholder_options(options))
            except Exception:
                pass
    return _instance  # type: ignore[return-value]


def _module_names(value: Any) -> Optional[List[str]]:
    if not value:
        return None
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _start(options: Dict[str, Any]) -> EvalKitClient:
    from .core.errors import configure as configure_errors
    from .core.exporter import resolve_export_gzip
    from .core.limits import resolve_value_limit
    from .core.redact import (
        resolve_capture_content,
        resolve_capture_secrets,
        resolve_capture_stream_bodies,
    )
    from .core.sampling import resolve_sample_rate
    from .utils.headers import set_capture_sensitive_headers

    capture_secrets = options["capture_secrets"]
    secrets_explicit = capture_secrets if capture_secrets is not None else options["capture_sensitive_headers"]
    opts = EvalKitOptions(
        subscription_key=options["subscription_key"].strip(),
        base_url=options["base_url"].strip(),
        api_url=(options["api_url"] or "https://api.syntropylabs.ai").strip(),
        service_name=options["service_name"],
        debug=options["debug"],
        max_export_batch_size=int(options["max_export_batch_size"]),
        scheduled_delay_ms=int(options["scheduled_delay_ms"]),
        max_queue_size=int(options["max_queue_size"]),
        max_queue_bytes=int(options["max_queue_bytes"]),
        max_body_bytes=int(options["max_body_bytes"]),
        max_value_bytes=resolve_value_limit(options["max_value_bytes"]),
        environment=options["environment"],
        app_version=options["app_version"],
        device_id=options["device_id"],
        capture_content=resolve_capture_content(options["capture_content"]),
        capture_secrets=resolve_capture_secrets(secrets_explicit),
        capture_stream_bodies=resolve_capture_stream_bodies(options["capture_stream_bodies"]),
        mask=options["mask"],
        export_gzip=resolve_export_gzip(options["export_gzip"]),
        function_rate_limit=float(options["function_rate_limit"] or 0),
        log_level=resolve_log_level(options["log_level"]),
        log_rate_limit=float(options["log_rate_limit"] or 0),
        sample_rate=resolve_sample_rate(options["sample_rate"]),
        in_app_include=_module_names(options["in_app_include"]),
        in_app_exclude=_module_names(options["in_app_exclude"]),
        ignore_routes=options["ignore_routes"],
        ignore_urls=options["ignore_urls"],
        ignore_functions=options["ignore_functions"],
        trace_health_checks=bool(options["trace_health_checks"]),
    )
    set_capture_sensitive_headers(opts.capture_secrets)
    try:
        # Before anything is instrumented: the rules decide from the first request and the first
        # module function tracing wraps.
        from .core.ignore import configure as configure_ignore

        configure_ignore(opts.ignore_routes, opts.ignore_urls, opts.ignore_functions, opts.trace_health_checks)
    except Exception as exc:
        _log.warning("evalkit.init(): the ignore options could not be read (%s); nothing is ignored by them.",
                     type(exc).__name__)
    try:
        configure_errors(opts.in_app_include, opts.in_app_exclude)
    except Exception:
        pass
    client = EvalKitClient(opts)
    global _instance
    _instance = client
    try:
        from .core.client import flush_on_termination_signals
        flush_on_termination_signals(client)
    except Exception:
        pass

    try:
        from .core.otel_provider import setup_otel
        setup_otel(client)
    except Exception:
        pass

    try:
        from .instrumentation.auto import setup_auto_instrumentation
        setup_auto_instrumentation(client)
    except Exception as exc:
        _log.debug("automatic instrumentation failed: %s", exc)

    if options["capture_unhandled"]:
        try:
            from .core.unhandled import install as install_unhandled
            install_unhandled(client)
        except Exception as exc:
            _log.debug("installing the unhandled-exception hooks failed: %s", exc)

    if options["function_tracing"] and os.environ.get("EVALKIT_FUNCTION_TRACE", "true") != "false":
        try:
            _install_function_tracing(client, options["trace_packages"])
        except Exception:
            pass
    return client


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
    instance = _get_instance()
    if not instance.enabled:
        return
    from .instrumentation.auto import setup_auto_instrumentation
    try:
        setup_auto_instrumentation(instance, logging=logging, requests=requests, httpx=httpx)
    except Exception as exc:
        _log.debug("automatic instrumentation failed: %s", exc)


def flush(timeout: Optional[float] = 5.0) -> bool:
    """Send what is queued now, within ``timeout`` seconds; whether the receiver took all of it.

    Shutdown hooks, test teardown and exit paths call this whether or not `init()` ran, so before it
    there is nothing to send and the answer is False; it never raises.
    """
    try:
        instance = _instance
        return bool(instance.flush(timeout=timeout)) if instance is not None else False
    except Exception:
        return False


_STATS_KEYS = (
    "queued", "dropped_queue_full", "dropped_bytes_budget", "dropped_export_failed", "export_failures",
    "last_error", "circuit_state", "queued_bytes", "dropped_circuit_open", "dropped_rate_limited",
    "dropped_sampled", "exported", "enabled",
)


def stats() -> Dict[str, Any]:
    """How recording and exporting are going: spans queued and dropped (by reason), failed exports,
    the last error and the circuit breaker's state (closed, open, half_open or disabled). Never raises."""
    try:
        instance = _instance
        if instance is not None:
            return instance.stats()
    except Exception:
        pass
    empty: Dict[str, Any] = {key: 0 for key in _STATS_KEYS}
    empty.update(last_error=None, circuit_state="disabled", enabled=False)
    return empty


def set_error_fingerprint(fingerprint: Any) -> Any:
    """Group the errors recorded from here on under ``fingerprint`` (a string, or a list joined by ``|``).

    Scoped like ``set_user``: per thread and per asyncio task; ``None`` clears it. Returns a token for
    ``clear_error_fingerprint`` to restore the previous one. A failed span records it as
    ``evalkit.exception.fingerprint``; without one the receiver groups errors by itself.
    """
    from .core.errors import set_fingerprint

    return set_fingerprint(fingerprint)


def clear_error_fingerprint(token: Optional[Any] = None) -> None:
    from .core.errors import reset_fingerprint

    reset_fingerprint(token)


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


def _guarded_patch(fn: Callable[..., None]) -> Callable[..., None]:
    """An explicit ``patch_*`` call: nothing when the SDK is a no-op, and a warning rather than an
    exception when the object cannot be instrumented, which then works untraced."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        instance = _get_instance()
        if not instance.enabled:
            return None
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            _log.warning("evalkit.%s could not instrument its target (%s: %s); it runs untraced.",
                         fn.__name__, type(exc).__name__, exc)
        return None

    return wrapper


@_guarded_patch
def instrument_flask(app: Any) -> None:
    _instrument_flask(app, _get_instance())


@_guarded_patch
def patch_sqlalchemy_engine(engine: Any) -> None:
    patch_sqlalchemy(_get_instance(), engine)


@_guarded_patch
def patch_async_sqlalchemy_engine(async_engine: Any) -> None:
    patch_sqlalchemy_async(_get_instance(), async_engine)


@_guarded_patch
def patch_openai_client(
    openai_client: Any,
    parent_ctx: Optional[SpanContext] = None,
) -> None:
    patch_openai(_get_instance(), openai_client, parent_ctx)


@_guarded_patch
def patch_anthropic_client(
    anthropic_client: Any,
    parent_ctx: Optional[SpanContext] = None,
) -> None:
    patch_anthropic(_get_instance(), anthropic_client, parent_ctx)


@_guarded_patch
def patch_bedrock_client(bedrock_runtime_client: Any) -> None:
    patch_bedrock(_get_instance(), bedrock_runtime_client)


@_guarded_patch
def patch_anthropic_vertex_client(
    anthropic_vertex_client: Any,
    parent_ctx: Optional[SpanContext] = None,
) -> None:
    patch_anthropic_vertex(_get_instance(), anthropic_vertex_client, parent_ctx)


@_guarded_patch
def patch_vertex_generative_model(model: Any) -> None:
    from .instrumentation.llm.vertex_ai import patch_vertex_generative_model as _patch_vertex
    _patch_vertex(_get_instance(), model)


@_guarded_patch
def patch_cohere_client(cohere_client: Any) -> None:
    patch_cohere(_get_instance(), cohere_client)


@_guarded_patch
def patch_google_ai_model(model: Any) -> None:
    patch_google_ai(_get_instance(), model)


@_guarded_patch
def patch_google_genai() -> None:
    _patch_google_genai(_get_instance())


def patch_langchain() -> Any:
    return _patch_langchain(_get_instance())


@_guarded_patch
def patch_litellm() -> None:
    _patch_litellm(_get_instance())


@_guarded_patch
def patch_ollama() -> None:
    _patch_ollama(_get_instance())


@_guarded_patch
def patch_pinecone() -> None:
    _patch_pinecone(_get_instance())


@_guarded_patch
def patch_qdrant() -> None:
    _patch_qdrant(_get_instance())


@_guarded_patch
def patch_chroma() -> None:
    _patch_chroma(_get_instance())


@_guarded_patch
def patch_weaviate() -> None:
    _patch_weaviate(_get_instance())


@_guarded_patch
def patch_llamaindex() -> None:
    _patch_llamaindex(_get_instance())


@_guarded_patch
def patch_openai_audio() -> None:
    _patch_openai_audio(_get_instance())


@_guarded_patch
def patch_elevenlabs() -> None:
    _patch_elevenlabs(_get_instance())


@_guarded_patch
def patch_assemblyai() -> None:
    _patch_assemblyai(_get_instance())


@_guarded_patch
def patch_claude_agent_sdk() -> None:
    _patch_claude_agent_sdk(_get_instance())


@_guarded_patch
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


@_guarded_patch
def patch_google_genai_live() -> None:
    _patch_google_genai_live(_get_instance())


@_guarded_patch
def patch_openai_realtime() -> None:
    _patch_openai_realtime(_get_instance())


@_guarded_patch
def patch_async_openai_client(
    async_openai_client: Any,
    parent_ctx: Optional[SpanContext] = None,
) -> None:
    patch_async_openai(_get_instance(), async_openai_client, parent_ctx)


@_guarded_patch
def patch_async_anthropic_client(
    async_anthropic_client: Any,
    parent_ctx: Optional[SpanContext] = None,
) -> None:
    patch_async_anthropic(_get_instance(), async_anthropic_client, parent_ctx)


@_guarded_patch
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
    *,
    error_fingerprint: Optional[str] = None,
) -> Callable:
    return _get_instance().trace_function(
        name, attributes, capture_io=capture_io, error_fingerprint=error_fingerprint,
    )


def trace_tool(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    capture_io: bool = True,
    *,
    error_fingerprint: Optional[str] = None,
) -> Callable:
    return _get_instance().trace_tool(
        name, attributes, capture_io=capture_io, error_fingerprint=error_fingerprint,
    )


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


def _ignored_function(fn: Any) -> bool:
    from .core.ignore import ignored_callable

    return ignored_callable(fn)


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
        if getattr(member, "_evalkit_traced", False) or getattr(member, UNTRACED_MARKER, False):
            continue
        if _ignored_function(member):
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
