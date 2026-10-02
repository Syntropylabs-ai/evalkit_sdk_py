# EvalKit Python SDK examples

These are small apps that send traces to SyntropyLabs with the EvalKit Python SDK. Each one uses a different web framework, agent framework or LLM provider.

## Set up

Run these commands from this `examples/` directory.

```bash
# Install evalkit from source
pip install -e ../

# Copy and fill in .env
cp .env.example .env
```

The repository has no `.env.example`, so create `.env` yourself from the template in the next section.

Most examples load `.env` with `python-dotenv`, which the SDK does not install. Install it with `pip install python-dotenv`.

`demo.py` does not load `.env`. It reads the same variables from your shell, so export them before you run it.

## The `.env` template

```env
EVALKIT_KEY=tk_live_...
EVALKIT_URL=http://localhost:8085   # or https://api.syntropylabs.ai

# LLM provider keys (only the ones you use)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
COHERE_API_KEY=...

# Google Cloud (Vertex AI / ADK)
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_GENAI_USE_VERTEXAI=True

# AWS (Bedrock)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
```

`EVALKIT_KEY` is your subscription key. `EVALKIT_URL` is the address the examples send their spans to, and they pass it to `evalkit.init` as `base_url`. When it is unset, they use `http://localhost:8085`, a trace service on your own machine. Set it to `https://api.syntropylabs.ai` to send traces to SyntropyLabs.

Set only the provider keys for the examples you run. `adk_agent.py` and `vertex_ai_app.py` read the model name from `GEMINI_MODEL_NAME`, which defaults to `gemini-2.0-flash`. `vertex_ai_app.py` uses Vertex AI while `GOOGLE_GENAI_USE_VERTEXAI` is `True`. To use the Gemini API, set it to `False` and set `GEMINI_API_KEY`.

## Examples

| File | Framework | LLM provider |
| --- | --- | --- |
| `demo.py` | Standalone | OpenAI, one chat call |
| `flask_app.py` | Flask | OpenAI |
| `fastapi_app.py` | FastAPI | Anthropic, async |
| `django_app.py` | Django | OpenAI |
| `langgraph_agent.py` | LangGraph | OpenAI, with tool use |
| `adk_agent.py` | Google ADK | Gemini on Vertex AI |
| `anthropic_claude.py` | Standalone | Anthropic: sync, async and tool use |
| `openai_agents_sdk.py` | Standalone agent loop | OpenAI: async and tool use |
| `bedrock_app.py` | Standalone | AWS Bedrock |
| `vertex_ai_app.py` | Standalone | Google Vertex AI or the Gemini API, through `google.genai` |
| `cohere_app.py` | Standalone | Cohere |

`openai_agents_sdk.py` runs its own tool-calling loop on the OpenAI client. It does not use the `openai-agents` package.

Two more scripts cover scenario simulation. `generate_scenarios_example.py` generates three test scenarios for a billing-support agent and prints them. `simulate_user_example.py` generates the same scenarios and runs a stub agent through them with `evalkit.simulate_user`. Both read the subscription key from `EVALKIT_SUBSCRIPTION_KEY` in your shell and do not load `.env`. They use the SDK's default address, `https://api.syntropylabs.ai`.

## Run an example

```bash
# Flask
pip install flask openai
python flask_app.py

# FastAPI
pip install fastapi uvicorn anthropic
uvicorn fastapi_app:app --reload

# Django
pip install django openai
python django_app.py

# LangGraph
pip install langgraph langchain-openai openai
python langgraph_agent.py

# Google ADK (needs GCP project)
pip install google-adk google-genai
python adk_agent.py

# Anthropic Claude
pip install anthropic
python anthropic_claude.py

# OpenAI Agents
pip install openai
python openai_agents_sdk.py

# AWS Bedrock
pip install boto3
python bedrock_app.py

# Google Vertex AI / genai
pip install google-genai google-cloud-aiplatform
python vertex_ai_app.py

# Cohere
pip install cohere
python cohere_app.py
```

`django_app.py` starts the Django development server on port 7000.

## See the traces

With `EVALKIT_URL=https://api.syntropylabs.ai`, the traces appear in the SyntropyLabs dashboard at https://syntropylabs.ai. With the default `http://localhost:8085`, they go to the trace service on your machine.
