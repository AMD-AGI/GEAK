#!/usr/bin/env node
// Responses "de-streaming" shim: lets codex (which streams /v1/responses) drive
// claude via the SaFE gateway. The gateway's NON-streaming /v1/responses for
// claude is correct+complete, but its STREAMING variant makes codex reconnect-
// loop. So: take codex's request, call the gateway with stream:false, then
// synthesize the standard OpenAI Responses SSE event sequence from the complete
// result. No protocol translation — everything stays in Responses schema.
//
//   OPENAI_API_KEY=<ak> SSL_CERT_FILE=<ca> SHIM_PORT=8791 GW_BASE=<.../v1> node responses_shim.mjs
// Point codex at base_url = http://127.0.0.1:8791/v1 (wire_api="responses").
//
// Fully env-parameterized (no hardcoded paths) so it is portable across machines:
//   SHIM_PORT       listen port (default 8791)
//   GW_BASE         upstream gateway base_url (default SaFE global)
//   OPENAI_API_KEY  gateway key (falls back to ANTHROPIC_API_KEY)
//   SSL_CERT_FILE   CA bundle for gateway TLS (optional)
//   SHIM_DEBUG      when set, dump last req/upstream JSON for debugging
//   SHIM_DEBUG_DIR  dir for those dumps (default: cwd)

import http from 'node:http';
import https from 'node:https';
import fs from 'node:fs';
import path from 'node:path';

const PORT = parseInt(process.env.SHIM_PORT || '8791', 10);
const GW_BASE = (process.env.GW_BASE || 'https://global.primus-safe.amd.com/api/v1/llm-proxy/v1').replace(/\/$/, '');
const KEY = process.env.OPENAI_API_KEY || process.env.ANTHROPIC_API_KEY;
const ca = process.env.SSL_CERT_FILE ? fs.readFileSync(process.env.SSL_CERT_FILE) : undefined;
const DEBUG_DIR = process.env.SHIM_DEBUG_DIR || process.cwd();
const log = (...a) => process.stderr.write('[shim] ' + a.join(' ') + '\n');

// The gateway proxies claude via LiteLLM's Anthropic transformation, which only
// understands `type:"function"` tools. codex also sends a `type:"namespace"` tool
// (its multi_agent_v1 grouping) -> LiteLLM 500s. We don't want codex spawning its
// own sub-agents anyway (orchestration is external), so drop any non-function tool.
function sanitizeTools(tools) {
  if (!Array.isArray(tools)) return tools;
  return tools.filter((t) => t && t.type === 'function');
}

function callGateway(path, bodyObj) {
  return new Promise((resolve, reject) => {
    const cleaned = bodyObj.tools ? { ...bodyObj, tools: sanitizeTools(bodyObj.tools) } : bodyObj;
    const data = JSON.stringify({ ...cleaned, stream: false });
    const u = new URL(GW_BASE + path);
    const r = https.request(u, {
      method: 'POST', ca,
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${KEY}`, 'Content-Length': Buffer.byteLength(data) },
    }, (res) => {
      let b = ''; res.on('data', (d) => (b += d)); res.on('end', () => {
        try { resolve({ status: res.statusCode, json: JSON.parse(b) }); }
        catch (e) { reject(new Error(`gateway non-JSON (${res.statusCode}): ${b.slice(0, 400)}`)); }
      });
    });
    r.on('error', reject); r.write(data); r.end();
  });
}

const sse = (res, event, obj) => res.write(`event: ${event}\ndata: ${JSON.stringify(obj)}\n\n`);

// The SaFE gateway's /responses returns object:"chat.completion" and other
// chat-ish fields; codex's Responses client expects a canonical Responses object
// (object:"response", proper item ids, text.format, reasoning object). Normalize.
let __idc = 0;
function normalizeResponse(resp) {
  const out = { ...resp };
  out.object = 'response';
  if (!out.id || !String(out.id).startsWith('resp_')) out.id = `resp_${Date.now()}_${__idc++}`;
  out.text = (out.text && out.text.format) ? out.text : { format: { type: 'text' } };
  out.reasoning = out.reasoning || { effort: null, summary: null };
  out.store = out.store === true;
  out.output = (out.output || []).map((item, i) => {
    if (item.type === 'message') {
      return { ...item, id: (item.id && String(item.id).startsWith('msg_')) ? item.id : `msg_${i}_${__idc++}`,
        role: item.role || 'assistant', status: item.status || 'completed',
        content: (item.content || []).map((p) => ({ type: p.type || 'output_text', text: p.text || '', annotations: p.annotations || [] })) };
    }
    if (item.type === 'function_call') {
      return { ...item, id: (item.id && String(item.id).startsWith('fc_')) ? item.id : `fc_${i}_${__idc++}`,
        call_id: item.call_id || item.id || `call_${i}`, name: item.name, arguments: item.arguments || '', status: item.status || 'completed' };
    }
    return item;
  });
  return out;
}

function streamResponsesObject(res, rawResp) {
  const resp = normalizeResponse(rawResp);
  res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' });
  let seq = 0; const S = () => seq++;
  const empty = { ...resp, status: 'in_progress', output: [], usage: null };
  sse(res, 'response.created', { type: 'response.created', sequence_number: S(), response: empty });
  sse(res, 'response.in_progress', { type: 'response.in_progress', sequence_number: S(), response: empty });
  (resp.output || []).forEach((item, oi) => {
    const added = item.type === 'message' ? { ...item, content: [] } : item;
    sse(res, 'response.output_item.added', { type: 'response.output_item.added', sequence_number: S(), output_index: oi, item: added });
    if (item.type === 'message') {
      (item.content || []).forEach((part, pi) => {
        sse(res, 'response.content_part.added', { type: 'response.content_part.added', sequence_number: S(), item_id: item.id, output_index: oi, content_index: pi, part: { ...part, text: '' } });
        if (typeof part.text === 'string') {
          sse(res, 'response.output_text.delta', { type: 'response.output_text.delta', sequence_number: S(), item_id: item.id, output_index: oi, content_index: pi, delta: part.text });
          sse(res, 'response.output_text.done', { type: 'response.output_text.done', sequence_number: S(), item_id: item.id, output_index: oi, content_index: pi, text: part.text });
        }
        sse(res, 'response.content_part.done', { type: 'response.content_part.done', sequence_number: S(), item_id: item.id, output_index: oi, content_index: pi, part });
      });
    } else if (item.type === 'function_call') {
      const args = item.arguments || '';
      sse(res, 'response.function_call_arguments.delta', { type: 'response.function_call_arguments.delta', sequence_number: S(), item_id: item.id, output_index: oi, delta: args });
      sse(res, 'response.function_call_arguments.done', { type: 'response.function_call_arguments.done', sequence_number: S(), item_id: item.id, output_index: oi, arguments: args });
    }
    sse(res, 'response.output_item.done', { type: 'response.output_item.done', sequence_number: S(), output_index: oi, item });
  });
  sse(res, 'response.completed', { type: 'response.completed', sequence_number: S(), response: { ...resp, status: 'completed' } });
  res.end();
}

const server = http.createServer((req, res) => {
  const path = req.url.replace(/^\/v1/, '');
  if (req.method === 'GET') {
    // minimal /models so codex health checks pass
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ object: 'list', data: [{ id: 'claude-opus-4-8', object: 'model' }, { id: 'claude-opus-5', object: 'model' }] }));
    return;
  }
  if (req.method !== 'POST') { res.writeHead(405); res.end(); return; }
  let body = ''; req.on('data', (d) => (body += d)); req.on('end', async () => {
    let reqObj; try { reqObj = JSON.parse(body); } catch { res.writeHead(400); res.end('bad json'); return; }
    const wantStream = reqObj.stream !== false;
    const upstreamPath = path.startsWith('/responses') ? '/responses'
      : path.startsWith('/chat/completions') ? '/chat/completions' : '/responses';
    try {
      log(`${req.method} ${req.url} model=${reqObj.model} stream=${wantStream} -> ${upstreamPath}(stream:false)`);
      if (process.env.SHIM_DEBUG) {
        try { fs.writeFileSync(path_join(DEBUG_DIR, 'shim_req.json'), JSON.stringify(reqObj, null, 2)); } catch {}
      }
      const { status, json } = await callGateway(upstreamPath, reqObj);
      if (process.env.SHIM_DEBUG) {
        try { fs.writeFileSync(path_join(DEBUG_DIR, 'shim_upstream.json'), JSON.stringify({ status, json }, null, 2)); } catch {}
      }
      if (status !== 200) { res.writeHead(status, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(json)); return; }
      if (!wantStream) { res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(json)); return; }
      streamResponsesObject(res, json);
    } catch (e) {
      log('ERROR ' + (e.message || e));
      if (!res.headersSent) res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { message: String(e.message || e) } }));
    }
  });
});

// small helper: join without shadowing the request-scoped `path` string above.
function path_join(dir, file) { return path.join(dir, file); }

server.listen(PORT, '127.0.0.1', () => log(`listening on 127.0.0.1:${PORT} -> ${GW_BASE} (de-stream)`));
