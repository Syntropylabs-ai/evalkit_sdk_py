import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

import evalkit

evalkit.init(
    subscription_key=os.environ.get("EVALKIT_KEY", "demo-key"),
    base_url=os.environ.get("EVALKIT_URL", "http://localhost:8085"),
    service_name="vertex-ai-demo",
    environment="development",
    debug=True,
)

from google import genai

use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "True").lower() == "true"
if use_vertex:
    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )
else:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.0-flash")


def sync_generate(prompt: str) -> str:
    trace_id, end_trace, ctx = evalkit.start_trace("vertex.generate", {"prompt": prompt[:80]})
    with evalkit.with_trace(ctx):
        response = client.models.generate_content(model=MODEL, contents=prompt)
        text = response.text or ""
    end_trace("OK")
    print(f"Trace ID: {trace_id}")
    return text


async def async_generate(prompt: str) -> str:
    trace_id, end_trace, ctx = evalkit.start_trace("vertex.generate_async", {"prompt": prompt[:80]})
    with evalkit.with_trace(ctx):
        response = await client.aio.models.generate_content(model=MODEL, contents=prompt)
        text = response.text or ""
    end_trace("OK")
    print(f"Trace ID: {trace_id}")
    return text


async def main() -> None:
    prompts = [
        "What are the three laws of thermodynamics?",
        "Explain the difference between supervised and unsupervised learning.",
    ]

    print("\n── Sync calls ───────────────────────────────────────────────────")
    for p in prompts:
        print(f"\nPrompt: {p}")
        answer = sync_generate(p)
        print(f"Answer: {answer[:200]}")

    print("\n── Async calls ──────────────────────────────────────────────────")
    for p in prompts:
        print(f"\nPrompt: {p}")
        answer = await async_generate(p)
        print(f"Answer: {answer[:200]}")

    evalkit.flush()
    print("\nDone — spans exported.")


if __name__ == "__main__":
    asyncio.run(main())
