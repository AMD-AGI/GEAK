#!/usr/bin/env node
// Regression guard for normalizeQueues + the `drop_gate` relevance gate (no GPU, no model needed).
//
// The gate is ON by default, so these are the checks that make that safe. A drop must clear ALL of:
// exactly one match, a KNOWN size, a size under HEAD_PROTECT_PCT, and a list that is not eating the
// queue. Invariants under test:
//   (a) drop_gate=off leaves the queues exactly as the Architect returned them (kill switch works),
//   (b) drop_gate defaults to 'on' — merging this changes behaviour, which is the point,
//   (c) a candidate at or above HEAD_PROTECT_PCT is never dropped,
//   (d) matching is strict: an id we invented ourselves can never satisfy a drop_list entry; an entry
//       that carries an id matches on that id ALONE (no silent fallback to the name); an entry that
//       matches two candidates is refused rather than guessed at; and a no-match is recorded loudly,
//   (e) a candidate whose pct_gpu_time is missing/blank/NaN is NOT dropped — "too small to matter" is
//       unprovable without a size, and Number(undefined)||0 would have read it as 0%,
//   (f) a drop_list resolving to more than DROP_MAX_FRACTION of the queue is refused WHOLE,
//   (g) the op-identity guard runs inside normalizeQueues, so it survives a re-strategize,
//   (h) inputs are deep-copied — the Architect's own candidate objects are never mutated,
//   (i) every queue-assignment site routes through normalizeQueues,
//   (j) no schema gained a `required` entry (obj() emits additionalProperties:true and nothing validates
//       locally, so a `required` entry would change LLM generation on every run regardless of the flag).
//
// We prove (a)-(h) behaviorally by extracting the ACTUAL normalizeQueues block from the workflow script
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

// ---- (b) the shipped default is `on` --------------------------------------------------------------
// A gate that defaults to inert changes nothing on merge. This pins the default so it cannot drift back.
ok(/A\.drop_gate != null \? A\.drop_gate : 'on'/.test(src), "drop_gate defaults to 'on'");
ok(!/dryrun/i.test(src), "'dryrun' is gone — the only two settings are 'on' and 'off'");
ok(/const DROP_MAX_FRACTION = 0\.5;/.test(src), 'the bulk-drop circuit breaker exists and is 50%');

// normalizeQueues delegates its op-identity step to applyOpIdentityGuard, which lives at module scope
// and is main's, not ours. Extract that block too rather than re-implementing the fused rule here: a
// hand-written double would be free to drift from the rule the orchestrator actually applies, which is
// exactly what invariant (g) is supposed to catch.
const g = src.match(/const _isFusedOp = [\s\S]*?\nfunction applyOpIdentityGuard\(queue, stage\) \{[\s\S]*?\n\}\n/);
ok(!!g, 'applyOpIdentityGuard block found (normalizeQueues delegates the fused rule to it)');

if (m && g) {
  const make = (dropGate, protectPct, st, maxFraction) => {
    const lines = [];
    // admitHeads is injected as identity on purpose. It is main's SEPARATE gate — it requires
    // entity_kind='gpu_kernel' plus a device-kernel identity, neither of which these Architect-shaped
    // fixtures carry, so the real one would reject all four heads and leave nothing for the drop
    // filter to act on. Admission is out of scope here; the drop gate is what is under test.
    const fn = new Function('ST', 'DROP_GATE', 'HEAD_PROTECT_PCT', 'DROP_MAX_FRACTION', 'log', 'admitHeads',
      g[0] + m[0] + '\nreturn { normalizeQueues, dropDecisions };');
    const api = fn(st || {}, dropGate, protectPct,
      maxFraction == null ? 0.5 : maxFraction, (s) => lines.push(String(s)), (q) => q);
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

  // ---- (c) ON drops, and the dominant head is refused -------------------------------------------
  {
    const { normalizeQueues, dropDecisions } = make('on', 30);
    const out = normalizeQueues(architect());
    eq(out.head.map((c) => c.id), ['h0', 'h1', 'h3'],
      'ON: h2 dropped, the id-less head keeps its invented id');
    ok(out.kernel.map((c) => c.id).join(',') === 'k1', 'ON: silu_mul dropped by short_name');
    ok(dropDecisions.filter((d) => d.outcome === 'dropped').length === 2, 'ON: two drops recorded');
  }
  {
    // The dominant head is on the drop list. This is the check that makes ON-by-default defensible.
    const { normalizeQueues, dropDecisions } = make('on', 30);
    const inp = architect();
    inp.dropList = [{ id: 'h0', short_name: 'fp8_gemm', why: 'architect changed its mind' }];
    const out = normalizeQueues(inp);
    ok(out.head.some((c) => c.id === 'h0'), 'ON: the 57%-GPU head is NOT dropped');
    ok(dropDecisions.length === 1 && dropDecisions[0].outcome === 'protected',
      "ON: the refusal is recorded as 'protected'");
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
    // An entry that CARRIES an id is matched on that id alone. A stale id must not quietly fall
    // through to the short_name and take out a candidate the Architect did not name.
    const { normalizeQueues, dropDecisions } = make('on', 30);
    const inp = architect();
    inp.dropList = [{ id: 'h47', short_name: 'rmsnorm', why: 'stale id, real name' }];
    const out = normalizeQueues(inp);
    ok(out.head.some((c) => c.id === 'h2'),
      'an entry with an id does NOT fall back to short_name when the id misses');
    ok(dropDecisions[0].outcome === 'no_match', 'the stale-id entry is recorded as a miss');
  }
  {
    // Two candidates answer to the same name: refuse rather than guess.
    const { normalizeQueues, dropDecisions, lines } = make('on', 30);
    const inp = architect();
    inp.kernel.push({ id: 'k2', short_name: 'rmsnorm', classification: 'norm', pct_gpu_time: 0.4 });
    inp.dropList = [{ short_name: 'rmsnorm', why: 'which one?' }];
    const out = normalizeQueues(inp);
    ok(out.head.some((c) => c.id === 'h2') && out.kernel.some((c) => c.id === 'k2'),
      'an ambiguous entry drops NEITHER candidate');
    ok(dropDecisions[0].outcome === 'ambiguous', "the ambiguity is recorded as 'ambiguous'");
    ok(lines.some((l) => /AMBIGUOUS/.test(l)), 'the ambiguity is logged');
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

  // ---- (e) an unknown size is never dropped -----------------------------------------------------
  // `Number(c.pct_gpu_time) || 0` would read a missing field as 0% — i.e. as the most droppable value
  // possible. A candidate we cannot size is a candidate we cannot justify dropping.
  {
    for (const bad of [undefined, null, '', 'n/a', NaN]) {
      const { normalizeQueues, dropDecisions } = make('on', 30);
      const inp = architect();
      inp.head = [{ id: 'h0', short_name: 'mystery', op_kind: 'gemm', pct_gpu_time: bad }];
      inp.kernel = [{ id: 'k0', short_name: 'filler', pct_gpu_time: 1.0 },
                    { id: 'k1', short_name: 'filler2', pct_gpu_time: 1.0 }];
      inp.dropList = [{ id: 'h0', why: 'looks small to me' }];
      const out = normalizeQueues(inp);
      ok(out.head.length === 1, `pct_gpu_time=${JSON.stringify(bad)}: not dropped (size unknown)`);
      ok(dropDecisions[0].outcome === 'unverified', `pct_gpu_time=${JSON.stringify(bad)}: recorded as unverified`);
    }
  }
  {
    // An explicit 0 is a statement, not a gap: it means "measured, negligible". That one IS droppable.
    const { normalizeQueues, dropDecisions } = make('on', 30);
    const inp = architect();
    inp.head = [{ id: 'h0', short_name: 'genuinely_tiny', pct_gpu_time: 0 }];
    inp.kernel = [{ id: 'k0', short_name: 'filler', pct_gpu_time: 1.0 },
                  { id: 'k1', short_name: 'filler2', pct_gpu_time: 1.0 }];
    inp.dropList = [{ id: 'h0', why: 'measured at 0.0%' }];
    const out = normalizeQueues(inp);
    ok(out.head.length === 0, 'an explicit pct_gpu_time of 0 IS droppable');
    ok(dropDecisions[0].outcome === 'dropped', 'and is recorded as a real drop');
  }
  {
    // The entry may carry the size even when the candidate does not.
    const { normalizeQueues } = make('on', 30);
    const inp = architect();
    inp.head = [{ id: 'h0', short_name: 'sizeless' }];
    inp.kernel = [{ id: 'k0', short_name: 'filler', pct_gpu_time: 1.0 },
                  { id: 'k1', short_name: 'filler2', pct_gpu_time: 1.0 }];
    inp.dropList = [{ id: 'h0', pct_gpu_time: 0.7, why: 'sized by the drop entry' }];
    ok(normalizeQueues(inp).head.length === 0,
      "the drop entry's own pct_gpu_time is accepted when the candidate has none");
  }

  // ---- (f) a drop_list that eats the queue is refused WHOLE --------------------------------------
  {
    const { normalizeQueues, dropDecisions, lines } = make('on', 30);
    const inp = architect();                       // 4 heads + 2 kernels = 6, cap = floor(6*0.5) = 3
    inp.dropList = [
      { id: 'h1', why: 'no' }, { id: 'h2', why: 'no' },
      { short_name: 'silu_mul', why: 'no' }, { short_name: 'rope', why: 'no' },
    ];
    const out = normalizeQueues(inp);
    ok(out.head.length === 4 && out.kernel.length === 2,
      '4 drops out of 6 candidates exceeds the 50% cap -> nothing is dropped');
    ok(dropDecisions.every((d) => d.outcome === 'refused_bulk'),
      "every entry is recorded as 'refused_bulk', not silently half-applied");
    ok(lines.some((l) => /REFUSING THE WHOLE drop_list/.test(l)), 'the bulk refusal is logged');
  }
  {
    // Exactly at the cap is allowed; the breaker is for runaway lists, not ordinary pruning.
    const { normalizeQueues } = make('on', 30);
    const inp = architect();
    inp.dropList = [{ id: 'h1', why: 'ok' }, { id: 'h2', why: 'ok' }, { short_name: 'silu_mul', why: 'ok' }];
    const out = normalizeQueues(inp);
    ok(out.head.length + out.kernel.length === 3, 'a list exactly at the cap (3 of 6) still applies');
  }

  // ---- (g) op-identity guard runs here ----------------------------------------------------------
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

  // ---- (h) short_name is never invented; an invented id is marked ----------------------------------
  {
    const { normalizeQueues } = make('off', 30);
    const out = normalizeQueues({ head: [{ op_kind: 'gemm', pct_gpu_time: 3 }], kernel: [], dropList: [], origin: 't' });
    ok(out.head[0].short_name === undefined,
      'short_name is left alone — the head/milestone tracks name their own kernels downstream');
    ok(out.head[0].id === 'h0' && out.head[0].id_synthesized === true, 'an invented id is marked as invented');
  }
}

// ---- (i) every queue-assignment site routes through normalizeQueues -------------------------------
// `= normalizeQueues({`, so the function's own declaration is not counted as a call.
const callSites = (src.match(/= normalizeQueues\(\{/g) || []).length;
ok(callSites === 4, `all four queue-assignment sites call normalizeQueues (found ${callSites})`);
for (const origin of ['strategize', 'carried-state', 're-strategize', 'post-tuning re-strategize']) {
  ok(src.includes(`origin: '${origin}'`), `call site tagged origin='${origin}'`);
}
ok(!/(kernelQueue|headQueue) = \(?(strategy|restrat|retune|ST)\b/.test(src),
  'no queue-assignment site bypasses normalizeQueues');
// The post-tuning site used to assign the queues separately — the guard on the heads, a bare .slice()
// on the kernels. Pin that it cannot come back: a .slice() there is both a bypassed drop gate and the
// shallow copy of bug 6.
ok(!/(headQueue|kernelQueue) = .*\.slice\(\)/.test(src),
  'no queue is assigned from a shallow .slice() of an Architect response');

// ---- (j) no schema gained a `required` entry ------------------------------------------------------
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
  ? '\nPASS: drop_gate is on by default and refuses any drop it cannot fully justify; off is inert.'
  : `\nFAILED: ${failures} assertion(s).`);
process.exit(failures === 0 ? 0 : 1);
