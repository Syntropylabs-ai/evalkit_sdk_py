"""
EvalKit + Google ADK demo

Just call evalkit.init() — auto-instruments google.genai at the class level
so every Gemini call ADK makes internally (text responses AND tool-calling turns)
appears as an llm_call span. No manual patch needed.

ADK runs tools/skills in its own runtime (not the LLM provider), so to see tool
executions — get_weather, load_skill, any FunctionTool — as tool_call spans, spread
evalkit_tool_callbacks() into the agent (see below). Copy that pattern into your agent.

Install:
    pip install google-adk google-genai python-dotenv
    pip install -e ../../sdk-py

Authenticate:
    gcloud auth application-default login

Run:
    EVALKIT_KEY=tk_live_...  \\
    GOOGLE_CLOUD_PROJECT=your-project  \\
    GOOGLE_GENAI_USE_VERTEXAI=True  \\
    python adk_agent.py
"""

import asyncio
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("LOCAL_DEV", "true")
os.environ.setdefault("GCP_STAGING_BUCKET", "")

import evalkit

evalkit.init(
    subscription_key=os.environ.get("EVALKIT_KEY", "demo-key"),
    base_url=os.environ.get("EVALKIT_URL", "http://localhost:8085"),
    service_name="adk-demo",
    environment="development",
    debug=True,
)

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types as genai_types


# ── Demo tool ─────────────────────────────────────────────────────────────────

def get_weather(city: str) -> str:
    """Return a fake weather report (stub for demo purposes)."""
    stubs = {
        "london": "Cloudy, 14°C",
        "tokyo": "Sunny, 22°C",
        "new york": "Partly cloudy, 18°C",
    }
    return stubs.get(city.lower(), f"Weather data unavailable for {city}.")


weather_tool = FunctionTool(func=get_weather)



_tool_starts: dict = {}


def _before_tool(tool, args, tool_context):
    _tool_starts[id(tool_context)] = time.time()
    return None


def _after_tool(tool, args, tool_context, tool_response):
    started = _tool_starts.pop(id(tool_context), None)
    duration_ms = int((time.time() - started) * 1000) if started else 0
    name = getattr(tool, "name", None) or getattr(tool, "__name__", "tool")
    end, _ctx = evalkit.start_span(name, {
        "evalkit.span_type": "tool_call",
        "gen_ai.tool.name": name,
        "gen_ai.tool.call.arguments": json.dumps(args, default=str)[:4000],
        "gen_ai.tool.call.result": json.dumps(tool_response, default=str)[:4000],
        "tool.duration_ms": duration_ms,
    })
    end("OK")
    return None


def evalkit_tool_callbacks() -> dict:
    """Spread into any ADK LlmAgent(...) to trace its tool/skill executions in EvalKit."""
    return {"before_tool_callback": _before_tool, "after_tool_callback": _after_tool}


# ── Agent ─────────────────────────────────────────────────────────────────────

agent = LlmAgent(
    model=os.environ.get("GEMINI_MODEL_NAME", "gemini-2.0-flash"),
    name="weather_agent",
    description="A helpful assistant that can look up weather information.",
    instruction="You are a weather assistant. Use get_weather to answer questions.",
    tools=[weather_tool],
    **evalkit_tool_callbacks(),
)

session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name="adk-demo", session_service=session_service)


# ── Run ───────────────────────────────────────────────────────────────────────

async def run_turn(session_id: str, user_message: str) -> str:
    message = genai_types.Content(
        role="user",
        parts=[genai_types.Part.from_text(text=user_message)],
    )
    # Optional named trace for grouping spans in the dashboard
    trace_id, end_trace, ctx = evalkit.start_trace(
        "adk.turn", {"user.message": user_message[:100]}
    )
    response_text = ""

    with evalkit.with_trace(ctx):
        for event in runner.run(user_id="demo-user", session_id=session_id, new_message=message):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        response_text += part.text
    end_trace("OK")
    return response_text


async def main() -> None:
    session_id = "demo-session-001"
    await session_service.create_session(
        app_name="adk-demo", user_id="demo-user", session_id=session_id
    )

    questions = [
        "What's the weather like in London?",
        "How about Tokyo?",
        "What is the weather in Paris?",
    ]
    for q in questions:
        print(f"\nUser: {q}")
        answer = await run_turn(session_id, q)
        print(f"Agent: {answer}")

    evalkit.flush()
    print("\nDone — spans exported.")


if __name__ == "__main__":
    asyncio.run(main())
