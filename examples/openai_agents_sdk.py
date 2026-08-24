import asyncio
import json
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import evalkit

load_dotenv()

evalkit.init(
    subscription_key=os.environ.get("EVALKIT_KEY", "demo-key"),
    base_url=os.environ.get("EVALKIT_URL", "http://localhost:8085"),
    service_name="openai-agent-demo",
    environment="development",
    debug=True,
)

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "demo-key")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for recent information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
    }
]


def handle_tool(name: str, args: dict) -> str:
    if name == "search_web":
        return f"[stub] Top result for '{args.get('query', '')}': This is a demo answer."
    return '{"error": "unknown tool"}'


async def run_agent(user_message: str) -> str:
    trace_id, end_trace, ctx = evalkit.start_trace(
        "agent.run", {"user.message": user_message[:100]}
    )
    with evalkit.with_trace(ctx):
        messages = [
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": user_message},
        ]
        final_answer = ""
        for _ in range(5):
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            choice = response.choices[0]
            msg = choice.message
            messages.append(msg.model_dump(exclude_none=True))

            if choice.finish_reason == "tool_calls" and msg.tool_calls:
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments or "{}")
                    result = handle_tool(tc.function.name, args)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            else:
                final_answer = msg.content or ""
                break

    end_trace("OK")
    print(f"Trace ID: {trace_id}")
    return final_answer


async def main() -> None:
    questions = [
        "What are the latest developments in quantum computing?",
        "Who won the Nobel Prize in Physics in 2024?",
    ]
    for q in questions:
        print(f"\nUser: {q}")
        answer = await run_agent(q)
        print(f"Agent: {answer}")

    evalkit.flush()
    print("\nDone — spans exported.")


if __name__ == "__main__":
    asyncio.run(main())
