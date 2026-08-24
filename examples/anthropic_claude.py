import asyncio
import json
import os
from dotenv import load_dotenv
import anthropic
import evalkit

load_dotenv()

evalkit.init(
    subscription_key=os.environ.get("EVALKIT_KEY", "demo-key"),
    base_url=os.environ.get("EVALKIT_URL", "http://localhost:8085"),
    service_name="claude-demo",
    environment="development",
    debug=True,
)

sync_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY") or "demo-key")
async_client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY") or "demo-key")


TOOLS = [
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a given ticker symbol.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL, GOOG",
                }
            },
            "required": ["ticker"],
        },
    }
]


def handle_tool_call(tool_name: str, tool_input: dict) -> str:
    if tool_name == "get_stock_price":
        stubs = {"AAPL": "182.50", "GOOG": "175.00", "MSFT": "415.00"}
        ticker = tool_input.get("ticker", "UNKNOWN")
        return json.dumps({"ticker": ticker, "price": stubs.get(ticker, "N/A"), "currency": "USD"})
    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def sync_demo() -> None:
    print("\n── Sync demo (tool use) ─────────────────────────────────────────")
    trace_id, end_trace, ctx = evalkit.start_trace("claude.sync")
    with evalkit.with_trace(ctx):
        messages = [{"role": "user", "content": "What is the current stock price of AAPL?"}]
        while True:
            response = sync_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=512,
                tools=TOOLS,
                messages=messages,
            )
            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = handle_tool_call(block.name, block.input)
                        tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            else:
                text = next((b.text for b in response.content if hasattr(b, "text")), "")
                print(f"Answer: {text}")
                break
    end_trace("OK")
    print(f"Trace ID: {trace_id}")


async def async_demo() -> None:
    print("\n── Async demo ───────────────────────────────────────────────────")
    trace_id, end_trace, ctx = evalkit.start_trace("claude.async")
    with evalkit.with_trace(ctx):
        response = await async_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": "Summarise the benefits of observability in LLM apps in three bullet points.",
                }
            ],
        )
        text = response.content[0].text if response.content else ""
        print(f"Answer:\n{text}")
    end_trace("OK")
    print(f"Trace ID: {trace_id}")


if __name__ == "__main__":
    sync_demo()
    asyncio.run(async_demo())
    evalkit.flush()
    print("\nDone — spans exported.")
