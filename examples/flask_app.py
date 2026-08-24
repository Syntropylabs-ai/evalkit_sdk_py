import os
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
import evalkit

load_dotenv()

evalkit.init(
    subscription_key=os.environ.get("EVALKIT_KEY", "demo-key"),
    base_url=os.environ.get("EVALKIT_URL", "http://localhost:8085"),
    service_name="flask-demo",
    environment="development",
    debug=True,
)

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "demo-key")
app = Flask(__name__)
evalkit.instrument_flask(app)


@app.route("/ask")
def ask():
    question = request.args.get("q", "What is 2+2?")

    trace_id, end_trace, ctx = evalkit.start_trace("ask", {"question": question[:100]})
    with evalkit.with_trace(ctx):
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
        )
        answer = response.choices[0].message.content or ""

    end_trace("OK")
    return jsonify({"answer": answer, "trace_id": trace_id})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
