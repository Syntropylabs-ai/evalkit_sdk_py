# EvalKit Python SDK — Sample Apps

Quick-start examples for every supported framework and LLM provider.

## Prerequisites

```bash
# Install evalkit from source
pip install -e ../

# Copy and fill in .env
cp .env.example .env
```

## .env template

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

## Examples

| File | Framework | LLM Provider |
|---|---|---|
| `demo.py` | Standalone | OpenAI (basic) |
| `flask_app.py` | Flask | OpenAI |
| `fastapi_app.py` | FastAPI | Anthropic (async) |
| `django_app.py` | Django | OpenAI |
| `langgraph_agent.py` | LangGraph | OpenAI + tool use |
| `adk_agent.py` | Google ADK | Gemini via Vertex AI |
| `anthropic_claude.py` | Standalone | Anthropic (sync + async + tool use) |
| `openai_agents_sdk.py` | Standalone agent loop | OpenAI (async + tool use) |
| `bedrock_app.py` | Standalone | AWS Bedrock |
| `vertex_ai_app.py` | Standalone | Google Vertex AI / google.genai |
| `cohere_app.py` | Standalone | Cohere |

## Running each example

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

## View Traces

After running any example, open https://app.evalkit.io to see traces in real time.
