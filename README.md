# EvalKit Python SDK

LLM observability and tracing for Python apps. One `init()` call auto-instruments
your LLM clients, HTTP calls, database queries, and logging — then streams traces to
[Syntropy Labs](https://syntropylabs.ai).

## Installation

```bash
pip install evalkit
```

Optional provider extras:

```bash
pip install "evalkit[openai]"      # OpenAI
pip install "evalkit[anthropic]"   # Anthropic
pip install "evalkit[all]"         # everything
```

## Quickstart

```python
import evalkit

evalkit.init(
    subscription_key="sk_...",       # your Syntropy Labs key
    service_name="my-service",
)

# That's it — your OpenAI / Anthropic / HTTP / DB calls are now traced automatically.
from openai import OpenAI

client = OpenAI()
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

`init()` sets up auto-instrumentation for you. Context (including trace IDs)
propagates automatically across threads — no manual wiring required.

## Web frameworks

```python
# FastAPI / Starlette
from evalkit import EvalKitMiddleware
app.add_middleware(EvalKitMiddleware)

# Flask
import evalkit
evalkit.instrument_flask(app)

# Django — add to MIDDLEWARE
"evalkit.EvalKitDjangoMiddleware"
```

## Manual spans

```python
import evalkit

end, ctx = evalkit.start_span("my-operation", {"key": "value"})
try:
    ...  # your work
finally:
    end("ok")

# Or as a decorator
@evalkit.trace_function()
def do_work(x):
    return x * 2
```

## SQLAlchemy

```python
import evalkit
evalkit.patch_sqlalchemy_engine(engine)
```

## Flushing

Traces are batched and exported in the background. Flush before exit if needed:

```python
evalkit.flush()
```

## Links

- Website: https://syntropylabs.ai
- Documentation: https://syntropylabs.ai/docs

## License

Proprietary — © 2026 Syntropy Labs. All rights reserved. See [LICENSE](LICENSE).
