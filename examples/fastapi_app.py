"""
EvalKit + FastAPI demo

Just call evalkit.init() — everything is auto-instrumented:
  • All Anthropic calls traced automatically (sync + async)
  • Every HTTP route traced via EvalKitMiddleware
  • httpx / requests outbound calls traced

Install:
    pip install fastapi uvicorn anthropic python-dotenv
    pip install -e ../../sdk-py

Run:
    EVALKIT_KEY=tk_live_... ANTHROPIC_API_KEY=sk-ant-... uvicorn fastapi_app:app --reload
    curl -X POST http://localhost:8000/chat \\
         -H 'Content-Type: application/json' \\
         -d '{"message": "Explain quantum entanglement in one sentence."}'
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import anthropic
from dotenv import load_dotenv
import evalkit

load_dotenv()

evalkit.init(
    subscription_key=os.environ.get("EVALKIT_KEY", "demo-key"),
    base_url=os.environ.get("EVALKIT_URL", "http://localhost:8085"),
    service_name="fastapi-demo",
    environment="development",
    debug=True,
)

anthropic_client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY") or "demo-key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    evalkit.flush()


app = FastAPI(title="EvalKit FastAPI Demo", lifespan=lifespan)
app.add_middleware(evalkit.EvalKitMiddleware)  # traces every HTTP request


class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 256


class ChatResponse(BaseModel):
    reply: str
    trace_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    trace_id, end_trace, ctx = evalkit.start_trace(
        "chat", {"user.message": body.message[:100]}
    )
    with evalkit.with_trace(ctx):
        response = await anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=body.max_tokens,
            messages=[{"role": "user", "content": body.message}],
        )
        reply = response.content[0].text if response.content else ""

    end_trace("OK")
    return ChatResponse(reply=reply, trace_id=trace_id)


@app.get("/health")
async def health():
    return {"status": "ok"}
