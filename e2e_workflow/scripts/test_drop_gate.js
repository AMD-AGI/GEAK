#!/usr/bin/env node
// Regression guard for normalizeQueues + the `drop_gate` relevance gate (no GPU, no model needed).
//
// Invariants under test:
//   (a) drop_gate=off leaves the queues exactly as the Architect returned them (feature is inert),
//   (b) drop_gate=dryrun records what it WOULD drop and drops nothing,
//   (c) drop_gate=on drops, but NEVER a candidate at or above HEAD_PROTECT_PCT, in any mode,
//   (d) matching is strict: an id we invented ourselves can never satisfy a drop_list entry, and a
//       drop_list entry that matches nothing is recorded rather than silently ignored,
//   (e) the op-identity guard runs inside normalizeQueues, so it survives a re-strategize,
//   (f) inputs are deep-copied — the Architect's own candidate objects are never mutated,
//   (g) every queue-assignment site routes through normalizeQueues,
//   (h) no schema gained a `required` entry (obj() emits additionalProperties:true and nothing validates
//       locally, so a `required` entry would change LLM generation on every run regardless of the flag).
//
// We prove (a)-(f) behaviorally by extracting the ACTUAL normalizeQueues block from the workflow script
// and running it with controlled module-scope deps.
//
// Run:  node e2e_workflow/scripts/test_drop_gate.js
'use strict';
const fs = require('fs');
const path = require('path');
const assert = require('assert');

const ROOT = path.resolve(__dirname, '..', '..');            // .../GEAK
const FILE = path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js');

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };
const eq = (a, b, msg) => { try { assert.deepStrictEqual(a, b); ok(true, msg); } catch (e) { ok(false, `${msg} -> ${e.message.split('\n')[0]}`); } };

console.log(`\n# ${path.relative(ROOT, FILE)}`);
const src = fs.readFileSync(FILE, 'utf8');

// Extract the real block: from `const dropDecisions` through the close of normalizeQueues.
const m = src.match(/const dropDecisions = \(ST\.drop_decisions[\s\S]*?\n\}\n(?=\nasync function ensureFlydslGate)/);
ok(!!m, 'normalizeQueues block found');

if (m) {
  const make = (dropGate, protectPct, st) => {
    const lines = [];
    const fn = new Function('ST', 'DROP_GATE', 'HEAD_PROTECT_PCT', 'log',
      m[0] + '\nreturn { normalizeQueues, dropDecisions };');
    const api = fn(st || {}, dropGate, protectPct, (s) => lines.push(String(s)));
    return { ...api, lines };
  };

  // A queue the Architect might plausibly return.
  const architect = () => ({
    head: [
      { id: 'h0', short_name: 'fp8_gemm', op_kind: 'gemm', pct_gpu_time: 57.0,
        live_call_seam: 'sglang.srt.layers.quantization.fp8:Fp8LinearMethod.apply(x,w,bias)' },
      { id: 'h1', short_name: 'moe_gemm', op_kind: 'gemm', pct_gpu_time: 8.0, is_fused_kernel: true,
        live_call_seam: 'aiter.fused_moe_bf16_asm:asm_moe_tkw1(h,w1,w2,tw,ti)' },
      { id: 'h2', short_name: 'rmsnorm', op_kind: 'norm', pct_gpu_time: 2.0 },
      { short_name: 'nameless_head', op_kind: 'attn', pct_gpu_time: 1.5 },   // Architect omitted the id
    ],
    kernel: [
      { id: 'k0', short_name: 'silu_mul', classification: 'elementwise', pct_gpu_time: 1.2 },
      { id: 'k1', short_name: 'rope', classification: 'elementwise', pct_gpu_time: 0.9 },
    ],
    dropList: [
      { id: 'h2', short_name: 'rmsnorm', pct_gpu_time: 2.0, why: 'below Amdahl threshold' },
      { short_name: 'silu_mul', why: 'too small to move e2e' },
    ],
    origin: 'strategize',
  });

  // ---- (a) OFF is inert -------------------------------------------------------------------------
  {
    const { normalizeQueues, dropDecisions } = make('off', 30);
    const inp = architect();
    const before = JSON.parse(JSON.stringify({ head: inp.head, kernel: inp.kernel }));
    const out = normalizeQueues(inp);
    ok(out.head.length === 4 && out.kernel.length === 2, 'OFF: every candidate survives');
    ok(dropDecisions.length === 0, 'OFF: no drop decisions recorded');
    eq({ head: inp.head, kernel: inp.kernel }, before, 'OFF: the Architect\'s own objects are untouched');
  }

  // ---- (b) DRY RUN records but does not drop ----------------------------------------------------
  {
    const { normalizeQueues, dropDecisions } = make('dryrun', 30);
    const out = normalizeQueues(architect());
    ok(out.head.length === 4 && out.kernel.length === 2, 'DRYRUN: every candidate survives');
    const outcomes = dropDecisions.map((d) => d.outcome).sort();
    eq(outcomes, ['would_drop', 'would_drop'], 'DRYRUN: both matches recorded as would_drop');
  }

  // ---- (c) ON drops, and protection wins in every mode ------------------------------------------
  {
    const { normalizeQueues, dropDecisions } = make('on', 30);
    const out = normalizeQueues(architect());
    eq(out.head.map((c) => c.id), ['h0', 'h1', 'h3'],
      'ON: h2 dropped, the id-less head keeps its invented id');
    ok(out.kernel.map((c) => c.id).join(',') === 'k1', 'ON: silu_mul dropped by short_name');
    ok(dropDecisions.filter((d) => d.outcome === 'dropped').length === 2, 'ON: two drops recorded');
  }
  {
    // The dominant head is on the drop list: refused in EVERY mode.
    for (const mode of ['dryrun', 'on']) {
      const { normalizeQueues, dropDecisions } = make(mode, 30);
      const inp = architect();
      inp.dropList = [{ id: 'h0', short_name: 'fp8_gemm', why: 'architect changed its mind' }];
      const out = normalizeQueues(inp);
      ok(out.head.some((c) => c.id === 'h0'), `${mode}: 57%-GPU head is NOT dropped`);
      ok(dropDecisions.length === 1 && dropDecisions[0].outcome === 'protected',
        `${mode}: the refusal is recorded as 'protected'`);
    }
  }

  // ---- (d) strict matching ----------------------------------------------------------------------
  {
    // 'h3' is the id WE invented for the nameless head. It must not match.
    const { normalizeQueues, dropDecisions, lines } = make('on', 30);
    const inp = architect();
    inp.dropList = [{ id: 'h3', why: 'referring to an id the Architect never issued' }];
    const out = normalizeQueues(inp);
    ok(out.head.length === 4, 'an invented id cannot satisfy a drop_list entry');
    ok(dropDecisions.length === 1 && dropDecisions[0].outcome === 'no_match', 'the miss is recorded');
    ok(lines.some((l) => /NO MATCH/.test(l)), 'the miss is logged loudly, not swallowed');
  }
  {
    // Seam match, ignoring the advisory signature.
    const { normalizeQueues } = make('on', 30);
    const inp = architect();
    inp.dropList = [{ live_call_seam: 'aiter.fused_moe_bf16_asm:asm_moe_tkw1(totally, different, args)',
                      why: 'matched on module:attr, signature ignored' }];
    const out = normalizeQueues(inp);
    ok(!out.head.some((c) => c.id === 'h1'), 'seam match ignores the signature after "("');
  }

  // ---- (e) op-identity guard runs here ----------------------------------------------------------
  {
    const { normalizeQueues } = make('off', 30);
    const out = normalizeQueues(architect());
    const fused = out.head.find((c) => c.id === 'h1');
    ok(fused.op_kind === 'moe', 'fused head forced to op_kind=moe');
    ok(fused.target_callable === 'aiter.fused_moe_bf16_asm:asm_moe_tkw1(h,w1,w2,tw,ti)',
      'fused head bound to its live call seam');
    const standalone = out.head.find((c) => c.id === 'h0');
    ok(standalone.target_callable === 'sglang.srt.layers.quantization.fp8:Fp8LinearMethod.apply(x,w,bias)',
      'a NON-fused head with a live seam is bound too (used to be fused-only)');
    ok(standalone.op_kind === 'gemm', 'a non-fused head keeps its op_kind');
  }
  {
    // Idempotent: a re-strategize re-runs this over an already-normalized queue.
    const { normalizeQueues } = make('off', 30);
    const once = normalizeQueues(architect());
    const twice = normalizeQueues({ head: once.head, kernel: once.kernel, dropList: [], origin: 're-strategize' });
    eq(twice, once, 'normalizeQueues is idempotent (safe at the re-strategize and resume sites)');
  }

  // ---- (f) short_name is never invented ----------------------------------------------------------
  {
    const { normalizeQueues } = make('off', 30);
    const out = normalizeQueues({ head: [{ op_kind: 'gemm', pct_gpu_time: 3 }], kernel: [], dropList: [], origin: 't' });
    ok(out.head[0].short_name === undefined,
      'short_name is left alone — the head/milestone tracks name their own kernels downstream');
    ok(out.head[0].id === 'h0' && out.head[0].id_synthesized === true, 'an invented id is marked as invented');
  }
}

// ---- (g) every queue-assignment site routes through normalizeQueues -------------------------------
// `= normalizeQueues({`, so the function's own declaration is not counted as a call.
const callSites = (src.match(/= normalizeQueues\(\{/g) || []).length;
ok(callSites === 3, `all three queue-assignment sites call normalizeQueues (found ${callSites})`);
for (const origin of ['strategize', 'carried-state', 're-strategize']) {
  ok(src.includes(`origin: '${origin}'`), `call site tagged origin='${origin}'`);
}
ok(!/(kernelQueue|headQueue) = \(?(strategy|restrat|ST)\b/.test(src),
  'no queue-assignment site bypasses normalizeQueues');

// ---- (h) no schema gained a `required` entry ------------------------------------------------------
// obj() emits additionalProperties:true and nothing validates locally, so `required` only shapes LLM
// generation — it would fire on every run regardless of drop_gate, breaking "off behaves like today".
ok(/drop_list: arrObj[\s\S]{0,200}?\}, \['kernel_candidates'\]\);/.test(src),
  "STRATEGY_SCHEMA still requires only ['kernel_candidates']");
ok(!/drop_list: obj\(/.test(src), 'drop_list is still the permissive arrObj (no new required fields)');

// The Architect must be ASKED for the matching keys (properties/prompt only, never `required`).
const arch = fs.readFileSync(path.join(ROOT, 'e2e_workflow', 'roles', 'system_architect.md'), 'utf8');
ok(/"drop_list": \[\{"id":/.test(arch), 'roles/system_architect.md asks for an id on each drop_list entry');
ok(/drop_list[\s\S]{0,300}pct_gpu_time/.test(arch), 'roles/system_architect.md asks for pct_gpu_time on each drop_list entry');

console.log(failures === 0
  ? '\nPASS: drop_gate is inert when off, auditable when on, and never drops a dominant head.'
  : `\nFAILED: ${failures} assertion(s).`);
process.exit(failures === 0 ? 0 : 1);
