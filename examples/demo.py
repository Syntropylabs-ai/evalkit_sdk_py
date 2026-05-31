"""
EvalKit Python SDK — demo script.

Usage:
    pip install openai anthropic
    pip install -e ../  # install evalkit from source
    OPENAI_API_KEY=sk-... python demo.py
"""

import os
import evalkit
from openai import OpenAI

evalkit.init(
    subscription_key=os.environ.get("EVALKIT_KEY", "demo-key-001"),
    base_url=os.environ.get("EVALKIT_URL", "http://localhost:8085"),
    service_name="demo-py",
    debug=True,
    environment="development",
)

openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "demo-key")
evalkit.patch_openai_client(openai)

trace_id, end, ctx = evalkit.start_trace("demo-request", {"user.id": "demo-user"})

with evalkit.with_trace(ctx):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
    )
    print("Response:", response.choices[0].message.content)

end("OK")
print("Trace ID:", trace_id)

evalkit.flush()
print("Done — spans exported.")
