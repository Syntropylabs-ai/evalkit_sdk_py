"""
EvalKit + Django demo (single-file for testing)

Just call evalkit.init() — everything is auto-instrumented:
  • All OpenAI calls traced automatically
  • Every Django request traced via EvalKitDjangoMiddleware
  • DB calls traced via SQLAlchemy patcher (if used)

Install:
    pip install django openai python-dotenv
    pip install -e ../../sdk-py

Run:
    EVALKIT_KEY=tk_live_... OPENAI_API_KEY=sk-... python django_app.py
    curl "http://localhost:7000/ask/?q=What+is+Python"
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

import django
from django.conf import settings
from django.urls import path

settings.configure(
    DEBUG=True,
    SECRET_KEY="evalkit-demo-secret-key",
    ROOT_URLCONF=__name__,
    ALLOWED_HOSTS=["*"],
    MIDDLEWARE=[
        "evalkit.EvalKitDjangoMiddleware",
        "django.middleware.common.CommonMiddleware",
    ],
)

import evalkit

evalkit.init(
    subscription_key=os.environ.get("EVALKIT_KEY", "demo-key"),
    base_url=os.environ.get("EVALKIT_URL", "http://localhost:8085"),
    service_name="django-demo",
    environment="development",
    debug=True,
)

from openai import OpenAI
from django.http import JsonResponse

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "demo-key")


def ask_view(request):
    question = request.GET.get("q", "What is Python?")
    trace_id, end_trace, ctx = evalkit.start_trace("ask", {"q": question[:100]})
    with evalkit.with_trace(ctx):
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": question}],
        )
        answer = resp.choices[0].message.content or ""
    end_trace("OK")
    return JsonResponse({"answer": answer, "trace_id": trace_id})


def health_view(request):
    return JsonResponse({"status": "ok"})


urlpatterns = [
    path("ask/", ask_view),
    path("health/", health_view),
]

django.setup()

if __name__ == "__main__":
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv if len(sys.argv) > 1 else [__file__, "runserver", "7000"])
