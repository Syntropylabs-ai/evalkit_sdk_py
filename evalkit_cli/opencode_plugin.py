from __future__ import annotations

PLUGIN_FILE_NAME = "evalkit.ts"
PLUGIN_SOURCE_SHA256 = "267a60805676420c36434e88016226a43983e11cca2ee24575f73483070b42c3"
PLUGIN_SOURCE = r"""import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const VENDOR = "opencode";
const AGENT_NAME = "OpenCode";
const SCOPE = "evalkit-cli";
const NEWLINE = String.fromCharCode(10);
const ATTR_MAX = 4096;
const DEFAULT_ENDPOINT = "https://api.syntropylabs.ai/v1/otlp";

function config() {
  const out = { endpoint: DEFAULT_ENDPOINT, key: "", capture: "minimal" };
  let text = "";
  try {
    text = readFileSync(join(homedir(), ".config", "evalkit", "config.env"), "utf8");
  } catch {
    text = "";
  }
  for (const line of text.split(NEWLINE)) {
    const at = line.indexOf("=");
    if (at < 1 || line.trim().startsWith("#")) continue;
    const key = line.slice(0, at).trim();
    const value = line.slice(at + 1).trim();
    if (key === "EVALKIT_OTLP_ENDPOINT" && value) out.endpoint = value;
    if (key === "EVALKIT_API_KEY") out.key = value;
    if (key === "EVALKIT_CONTENT_CAPTURE" && value) out.capture = value;
  }
  if (process.env.EVALKIT_OTLP_ENDPOINT) out.endpoint = process.env.EVALKIT_OTLP_ENDPOINT;
  if (process.env.EVALKIT_API_KEY) out.key = process.env.EVALKIT_API_KEY;
  if (process.env.EVALKIT_CONTENT_CAPTURE) out.capture = process.env.EVALKIT_CONTENT_CAPTURE;
  return out;
}

function digest(parts) {
  return createHash("sha256").update(parts.join("")).digest("hex");
}

function traceId(...parts) {
  return digest(parts).slice(0, 32);
}

function spanId(...parts) {
  return digest(parts).slice(0, 16);
}

function nowNs() {
  return BigInt(Date.now()) * 1000000n;
}

function attr(key, value) {
  if (typeof value === "boolean") return { key, value: { boolValue: value } };
  if (typeof value === "bigint") return { key, value: { intValue: value.toString() } };
  if (typeof value === "number") return { key, value: { intValue: String(Math.round(value)) } };
  if (typeof value === "string") return { key, value: { stringValue: value.slice(0, ATTR_MAX) } };
  return { key, value: { stringValue: JSON.stringify(value).slice(0, ATTR_MAX) } };
}

function attributes(values) {
  const out = [];
  for (const key of Object.keys(values)) {
    const value = values[key];
    if (value === null || value === undefined || value === "") continue;
    out.push(attr(key, value));
  }
  return out;
}

function resource() {
  return { attributes: attributes({ "service.name": VENDOR, "agent.vendor": VENDOR, "telemetry.sdk.name": SCOPE }) };
}

function baseUrl(endpoint) {
  return endpoint.endsWith("/") ? endpoint.slice(0, -1) : endpoint;
}

async function post(signal, body) {
  const cfg = config();
  if (!cfg.key) return;
  try {
    await fetch(baseUrl(cfg.endpoint) + "/v1/" + signal, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: "Bearer " + cfg.key },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(4000),
    });
  } catch {
    return;
  }
}

function span(trace, id, parent, name, startNs, endNs, values, status) {
  const out = {
    traceId: trace,
    spanId: id,
    name: String(name).slice(0, 80),
    kind: 1,
    startTimeUnixNano: String(startNs),
    endTimeUnixNano: String(endNs),
    attributes: attributes(values),
  };
  if (parent) out.parentSpanId = parent;
  if (status) out.status = { code: status };
  return out;
}

async function sendSpans(spans) {
  await post("traces", { resourceSpans: [{ resource: resource(), scopeSpans: [{ scope: { name: SCOPE }, spans }] }] });
}

async function sendEnrichment(trace, id, values) {
  const now = String(nowNs());
  const record = {
    timeUnixNano: now,
    observedTimeUnixNano: now,
    severityText: "INFO",
    body: { stringValue: "evalkit.enrichment" },
    attributes: attributes(Object.assign({ "event.name": "evalkit.enrichment", "evalkit.enrichment": true, "agent.vendor": VENDOR }, values)),
    traceId: trace,
    spanId: id,
  };
  await post("logs", { resourceLogs: [{ resource: resource(), scopeLogs: [{ scope: { name: SCOPE }, logRecords: [record] }] }] });
}

export const EvalkitOpenCodePlugin = async () => {
  if (globalThis.evalkitOpenCodeLoaded) return {};
  globalThis.evalkitOpenCodeLoaded = true;

  const parents = new Map();
  const turns = new Map();
  const toolStarts = new Map();
  const full = () => config().capture === "full";

  function openTurn(sessionId, messageId) {
    const parent = parents.get(sessionId);
    const host = parent ? turns.get(parent) : undefined;
    const turn = {
      turnId: messageId,
      trace: host ? host.trace : traceId(sessionId, messageId),
      root: spanId(sessionId, messageId),
      parent: host ? host.root : undefined,
      subagent: Boolean(host),
      startNs: nowNs(),
      opened: false,
      prompt: "",
      completion: "",
      model: "",
      usage: undefined,
    };
    turns.set(sessionId, turn);
    return turn;
  }

  async function ensureOpen(sessionId) {
    const turn = turns.get(sessionId);
    if (!turn || turn.opened) return turn;
    turn.opened = true;
    const values = {
      "agent.vendor": VENDOR,
      "gen_ai.conversation.id": sessionId,
      "agent.turn.id": turn.turnId,
      "agent.role": turn.subagent ? "subagent" : "main",
      "gen_ai.agent.name": AGENT_NAME,
      "gen_ai.operation.name": "invoke_agent",
      "evalkit.span_type": "agent",
      "evalkit.attribution": "hooks",
      "agent.subagent.id": turn.subagent ? sessionId : undefined,
    };
    let name = "turn";
    if (full() && turn.prompt) {
      values.prompt = turn.prompt;
      name = turn.prompt;
    }
    await sendSpans([span(turn.trace, turn.root, turn.parent, name, turn.startNs, turn.startNs, values, 0)]);
    return turn;
  }

  async function closeTurn(sessionId, error) {
    const turn = turns.get(sessionId);
    if (!turn) return;
    turns.delete(sessionId);
    if (!turn.opened) return;
    const spans = [];
    if (turn.model || turn.usage) {
      const usage = turn.usage || {};
      spans.push(
        span(turn.trace, spanId(turn.turnId, "llm"), turn.root, "chat " + (turn.model || "model"), turn.startNs, nowNs(), {
          "agent.vendor": VENDOR,
          "gen_ai.conversation.id": sessionId,
          "agent.turn.id": turn.turnId,
          "gen_ai.operation.name": "chat",
          "evalkit.span_type": "llm_call",
          "evalkit.attribution": "hooks",
          "gen_ai.request.model": turn.model,
          "gen_ai.usage.input_tokens": usage.input,
          "gen_ai.usage.output_tokens": usage.output,
          "gen_ai.usage.reasoning_tokens": usage.reasoning,
          "gen_ai.usage.cache_read_input_tokens": usage.cacheRead,
          "gen_ai.usage.cache_creation_input_tokens": usage.cacheWrite,
          "gen_ai.usage.cost_usd": usage.cost,
        }, 0),
      );
    }
    if (spans.length) await sendSpans(spans);
    const values = { "gen_ai.conversation.id": sessionId, "agent.turn.id": turn.turnId, "evalkit.end_time_unix_nano": nowNs() };
    if (error) values["error.message"] = String(error).slice(0, ATTR_MAX);
    if (full() && turn.completion) values.completion = turn.completion;
    await sendEnrichment(turn.trace, turn.root, values);
  }

  async function onMessage(info) {
    if (!info || !info.sessionID) return;
    if (info.role === "user") {
      openTurn(info.sessionID, info.id || String(nowNs()));
      return;
    }
    const turn = turns.get(info.sessionID);
    if (!turn) return;
    if (info.modelID) turn.model = info.providerID ? info.providerID + "/" + info.modelID : info.modelID;
    const tokens = info.tokens;
    if (tokens) {
      const cache = tokens.cache || {};
      turn.usage = {
        input: tokens.input,
        output: tokens.output,
        reasoning: tokens.reasoning,
        cacheRead: cache.read,
        cacheWrite: cache.write,
        cost: typeof info.cost === "number" ? info.cost : undefined,
      };
    }
  }

  async function onPart(part) {
    if (!part || part.type !== "text" || !part.sessionID) return;
    const turn = turns.get(part.sessionID);
    if (!turn) return;
    const text = typeof part.text === "string" ? part.text : "";
    if (!text) return;
    if (part.messageID === turn.turnId) {
      turn.prompt = text;
      return;
    }
    turn.completion = text;
    await ensureOpen(part.sessionID);
  }

  return {
    event: async (payload) => {
      const event = payload && payload.event ? payload.event : payload;
      if (!event || typeof event.type !== "string") return;
      const props = event.properties || {};
      if (event.type === "session.created" || event.type === "session.updated") {
        const info = props.info || props.session || {};
        if (info.id && info.parentID) parents.set(info.id, info.parentID);
        return;
      }
      if (event.type === "message.updated") {
        await onMessage(props.info || props.message);
        return;
      }
      if (event.type === "message.part.updated") {
        await onPart(props.part);
        return;
      }
      if (event.type === "session.idle") {
        await closeTurn(props.sessionID || (props.info || {}).id, undefined);
        return;
      }
      if (event.type === "session.error") {
        const info = props.info || {};
        await closeTurn(props.sessionID || info.sessionID, (info.error || {}).message || "session error");
      }
    },
    "tool.execute.before": async (input) => {
      if (!input || !input.sessionID) return;
      await ensureOpen(input.sessionID);
      toolStarts.set(input.sessionID + ":" + input.callID, nowNs());
    },
    "tool.execute.after": async (input, output) => {
      if (!input || !input.sessionID) return;
      const turn = await ensureOpen(input.sessionID);
      if (!turn) return;
      const key = input.sessionID + ":" + input.callID;
      const start = toolStarts.get(key) || nowNs();
      toolStarts.delete(key);
      const result = output || {};
      const failed = result.metadata && result.metadata.error ? true : false;
      const values = {
        "agent.vendor": VENDOR,
        "gen_ai.conversation.id": input.sessionID,
        "agent.turn.id": turn.turnId,
        "agent.role": turn.subagent ? "subagent" : "main",
        "gen_ai.agent.name": AGENT_NAME,
        "gen_ai.operation.name": "execute_tool",
        "evalkit.span_type": "tool_call",
        "evalkit.attribution": "hooks",
        "gen_ai.tool.name": input.tool,
        "gen_ai.tool.call.id": input.callID,
        success: !failed,
      };
      if (full()) {
        if (result.args !== undefined) values["tool.arguments"] = result.args;
        if (result.output !== undefined) values["tool.result"] = result.output;
        if (result.title) values["tool.title"] = result.title;
      }
      await sendSpans([span(turn.trace, spanId(input.callID), turn.root, input.tool || "tool", start, nowNs(), values, failed ? 2 : 1)]);
    },
  };
};

export default EvalkitOpenCodePlugin;
"""
