"""
EvalKit + LangGraph demo

Just call evalkit.init() — auto-instruments all OpenAI calls made by LangGraph
(LangChain's ChatOpenAI uses openai under the hood, which is class-level patched).
Every node execution is wrapped in a named span for richer traces.

Install:
    pip install langgraph langchain-openai openai python-dotenv
    pip install -e ../../sdk-py

Run:
    EVALKIT_KEY=tk_live_... OPENAI_API_KEY=sk-... python langgraph_agent.py
"""

import os
from dotenv import load_dotenv
import evalkit

load_dotenv()

evalkit.init(
    subscription_key=os.environ.get("EVALKIT_KEY", "demo-key"),
    base_url=os.environ.get("EVALKIT_URL", "http://localhost:8085"),
    service_name="langgraph-demo",
    environment="development",
    debug=True,
)

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
import operator


# ── Tools (wrapped in trace spans) ───────────────────────────────────────────

@evalkit.trace_function("tool.calculator")
def calculator(expression: str) -> str:
    """Evaluate a simple math expression."""
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


@evalkit.trace_function("tool.lookup")
def lookup(query: str) -> str:
    """Simulate a knowledge lookup (stub for demo)."""
    return f"[stub] Top result for '{query}': This is a demo response."


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    question: str
    thoughts: Annotated[list[str], operator.add]
    answer: str


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── Nodes ─────────────────────────────────────────────────────────────────────

def think(state: AgentState) -> AgentState:
    question = state["question"]
    trace_id, end_trace, ctx = evalkit.start_trace("langgraph.think", {"question": question[:80]})
    with evalkit.with_trace(ctx):
        response = llm.invoke([
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user", "content": question},
        ])
        thought = response.content
    end_trace("OK")
    return {"thoughts": [thought], "answer": thought}


def use_tool(state: AgentState) -> AgentState:
    if "calculate" in state["question"].lower():
        result = calculator("2 + 2 * 10")
        return {"thoughts": [f"Calculator: {result}"], "answer": result}
    result = lookup(state["question"])
    return {"thoughts": [f"Lookup: {result}"], "answer": result}


def router(state: AgentState) -> str:
    if any(w in state["question"].lower() for w in ("calculate", "compute", "math", "lookup", "search")):
        return "use_tool"
    return "answer"


# ── Graph ─────────────────────────────────────────────────────────────────────

builder = StateGraph(AgentState)
builder.add_node("think", think)
builder.add_node("use_tool", use_tool)
builder.set_entry_point("think")
builder.add_conditional_edges("think", router, {"use_tool": "use_tool", "answer": END})
builder.add_edge("use_tool", END)
graph = builder.compile()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    questions = [
        "What is the capital of France?",
        "Calculate 42 * 7 + 3.",
        "Lookup the history of Python.",
    ]
    for q in questions:
        print(f"\nQuestion: {q}")
        result = graph.invoke({"question": q, "thoughts": [], "answer": ""})
        print(f"Answer: {result['answer']}")

    evalkit.flush()
    print("\nDone — spans exported.")
