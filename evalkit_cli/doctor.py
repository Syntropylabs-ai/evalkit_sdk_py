from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from .config import SharedConfig
from .ids import span_id, trace_id
from .otlp import get_json, log_record, read_base, send_logs, send_traces
from .spans import AGENT, LLM_CALL, OPERATION, STATUS_OK, TOOL_CALL, attributes, make_span, trace_payload

VENDOR = "evalkit-doctor"
PROMPT = "evalkit doctor synthetic turn"
COMPLETION = "evalkit doctor merged this completion into the agent span"
MODEL = "doctor-model"
TOOL = "Bash"
EXPECTED_SPANS = 3
READ_ATTEMPTS = 4
READ_DELAY_S = 0.75
SECOND_NS = 1_000_000_000
DEFAULT_TENANT = "me"


def synthetic_turn(now_ns: int) -> Tuple[str, str, List[Dict[str, Any]]]:
    seed = str(now_ns)
    trace = trace_id(VENDOR, seed)
    root, llm, tool = span_id("doctor-root", seed), span_id("doctor-llm", seed), span_id("doctor-tool", seed)
    start = now_ns - 2 * SECOND_NS
    shared = {
        "agent.vendor": VENDOR,
        "gen_ai.conversation.id": "doctor-session-" + seed,
        "agent.turn.id": "doctor-turn-" + seed,
        "agent.turn.sequence": 1,
        "evalkit.attribution": "hooks",
        "evalkit.doctor": True,
    }
    agent = {
        **shared,
        "evalkit.span_type": AGENT,
        "gen_ai.operation.name": OPERATION[AGENT],
        "agent.role": "main",
        "gen_ai.agent.name": "EvalKit doctor",
        "prompt": PROMPT,
    }
    model_call = {
        **shared,
        "evalkit.span_type": LLM_CALL,
        "gen_ai.operation.name": OPERATION[LLM_CALL],
        "agent.role": "main",
        "gen_ai.request.model": MODEL,
        "gen_ai.usage.input_tokens": 12,
        "gen_ai.usage.output_tokens": 7,
        "gen_ai.usage.cost_usd": 0.000123,
    }
    tool_call = {
        **shared,
        "evalkit.span_type": TOOL_CALL,
        "gen_ai.operation.name": OPERATION[TOOL_CALL],
        "agent.role": "main",
        "gen_ai.tool.name": TOOL,
        "gen_ai.tool.call.id": "doctor-tool-" + seed,
        "success": True,
    }
    spans = [
        make_span(trace, root, None, PROMPT, start, now_ns, agent),
        make_span(trace, llm, root, "chat " + MODEL, start + SECOND_NS // 2, now_ns - SECOND_NS, model_call, STATUS_OK),
        make_span(trace, tool, root, TOOL, now_ns - SECOND_NS, now_ns, tool_call, STATUS_OK),
    ]
    return trace, root, spans


def enrichment(trace: str, root: str, now_ns: int) -> Dict[str, Any]:
    values = {
        "evalkit.enrichment": True,
        "agent.vendor": VENDOR,
        "completion": COMPLETION,
        "evalkit.end_time_unix_nano": now_ns,
    }
    return log_record(VENDOR, "enrichment", attributes(values), prefix="evalkit", trace=trace, span=root)


def _spans_of(body: Any) -> List[Dict[str, Any]]:
    rows = body.get("data") if isinstance(body, dict) else body
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _read(cfg: SharedConfig, tenant: str, trace: str) -> Tuple[bool, str]:
    url = f"{read_base(cfg.endpoint)}/v1/traces/{tenant}/{trace}?fields=full"
    detail = "no response"
    for attempt in range(READ_ATTEMPTS):
        if attempt:
            time.sleep(READ_DELAY_S)
        code, body, error = get_json(url, cfg.key)
        if code == 403:
            return False, f"read back forbidden for tenant {tenant!r}; pass --tenant <your tenant id>"
        if code is None or not 200 <= code < 300:
            detail = f"HTTP {code}" if code is not None else (error or "no response")
            continue
        rows = _spans_of(body)
        prompt = any(str(row.get("prompt") or "") for row in rows)
        if len(rows) >= EXPECTED_SPANS and prompt:
            return True, f"{len(rows)} spans merged, prompt present"
        detail = f"{len(rows)} spans, prompt {'present' if prompt else 'missing'}"
    return False, detail


def run(cfg: SharedConfig, tenant: Optional[str]) -> Tuple[bool, str]:
    now_ns = time.time_ns()
    trace, root, spans = synthetic_turn(now_ns)
    code, error = send_traces(cfg.endpoint, cfg.key, trace_payload(VENDOR, spans))
    if code is None or not 200 <= code < 300:
        return False, f"posting spans failed ({f'HTTP {code}' if code is not None else error or 'no response'})"
    code, error = send_logs(cfg.endpoint, cfg.key, enrichment(trace, root, now_ns))
    if code is None or not 200 <= code < 300:
        return False, f"posting the enrichment failed ({f'HTTP {code}' if code is not None else error or 'no response'})"
    ok, detail = _read(cfg, tenant or DEFAULT_TENANT, trace)
    return ok, f"trace {trace}: {detail}"
