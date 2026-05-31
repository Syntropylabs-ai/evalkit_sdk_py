"""
EvalKit + AWS Bedrock demo

Just call evalkit.init() — auto-instruments botocore at the class level
so every bedrock-runtime Converse call is traced automatically.

Install:
    pip install boto3 python-dotenv
    pip install -e ../../sdk-py

Configure:
    aws configure   # or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY

Run:
    EVALKIT_KEY=tk_live_... AWS_DEFAULT_REGION=us-east-1 python bedrock_app.py
"""

import os
from dotenv import load_dotenv
import boto3
import evalkit

load_dotenv()

evalkit.init(
    subscription_key=os.environ.get("EVALKIT_KEY", "demo-key"),
    base_url=os.environ.get("EVALKIT_URL", "http://localhost:8085"),
    service_name="bedrock-demo",
    environment="development",
    debug=True,
)

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
)


def ask(model_id: str, question: str) -> str:
    trace_id, end_trace, ctx = evalkit.start_trace(
        "bedrock.converse", {"model": model_id, "question": question[:80]}
    )
    with evalkit.with_trace(ctx):
        response = bedrock.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": question}]}],
            inferenceConfig={"maxTokens": 256, "temperature": 0.5},
        )
        output = response.get("output", {})
        content = (output.get("message", {}) or {}).get("content", []) or []
        answer = content[0].get("text", "") if content else ""
    end_trace("OK")
    print(f"Trace ID: {trace_id}")
    return answer


if __name__ == "__main__":
    models = [
        ("anthropic.claude-3-haiku-20240307-v1:0", "What is the capital of Japan?"),
        ("amazon.titan-text-express-v1", "Explain machine learning in one sentence."),
    ]
    for model_id, question in models:
        print(f"\nModel: {model_id}")
        print(f"Q: {question}")
        try:
            answer = ask(model_id, question)
            print(f"A: {answer}")
        except Exception as exc:
            print(f"Error: {exc}")

    evalkit.flush()
    print("\nDone — spans exported.")
