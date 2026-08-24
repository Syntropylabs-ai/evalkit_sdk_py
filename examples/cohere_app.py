import os
from dotenv import load_dotenv
import cohere
import evalkit

load_dotenv()

evalkit.init(
    subscription_key=os.environ.get("EVALKIT_KEY", "demo-key"),
    base_url=os.environ.get("EVALKIT_URL", "http://localhost:8085"),
    service_name="cohere-demo",
    environment="development",
    debug=True,
)

co = cohere.Client(api_key=os.environ.get("COHERE_API_KEY", ""))


def ask(question: str) -> str:
    trace_id, end_trace, ctx = evalkit.start_trace("cohere.chat", {"q": question[:80]})
    with evalkit.with_trace(ctx):
        response = co.chat(
            model="command-r-plus",
            message=question,
            preamble="You are a helpful assistant.",
        )
        answer = response.text
    end_trace("OK")
    print(f"Trace ID: {trace_id}")
    return answer


if __name__ == "__main__":
    questions = [
        "What is Retrieval-Augmented Generation (RAG)?",
        "Explain embeddings in 2 sentences.",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {ask(q)}")

    evalkit.flush()
    print("\nDone — spans exported.")
