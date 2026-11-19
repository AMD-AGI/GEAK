#!/usr/bin/env node
// Regression guard for the op-identity guard (no GPU, no model needed).
//
// A fused non-GEMM head (an attention scorer + select chain, is_fused_kernel=true) keeps its
// declared op_kind, so its prepared UT still matches its declaration at dispatch. Fused MoE and
// grouped-GEMM heads are still tagged op_kind=moe.
//
// Run:  node e2e_workflow/scripts/test_op_identity_guard.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..'); // .../GEAK
const src = fs.readFileSync(path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js'), 'utf8');

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

const slice = (startMarker, endMarker) => {
  const start = src.indexOf(startMarker);
  const end = src.indexOf(endMarker, start);
  return start !== -1 && end > start ? src.slice(start, end + endMarker.length) : '';
};
const guardSrc = slice('const _isFusedOp = ', '\n  return admitted;\n}\n');
const defectSrc = slice('function preparedUtDefect(', "\n  return '';\n}\n");
ok(guardSrc !== '' && defectSrc !== '', 'the op-identity guard and preparedUtDefect located in e2e_workflow.js');

const logs = [];
const { applyOpIdentityGuard, preparedUtDefect } = new Function('admitHeads', 'log',
  `${guardSrc}\n${defectSrc}\nreturn { applyOpIdentityGuard, preparedUtDefect };`)((q) => q, (m) => logs.push(m));

const prepared = (kind) => ({ phase: 'extract_op',
  result: { task_dir: '/eval/tasks/t', smoke: 'pass', baseline_frozen: true, op_kind: kind } });
const queue = [
  { short_name: 'fused_attn_score_select', op_kind: 'attn', is_fused_kernel: true, pct_gpu_time: 5, prepared_ut: prepared('attn') },
  { short_name: 'fused_moe_decode', op_kind: 'moe', is_fused_kernel: true, pct_gpu_time: 20, prepared_ut: prepared('moe') },
  { short_name: 'fused_moe_kernel', op_kind: '', pct_gpu_time: 30 },
  { short_name: 'grouped_gemm_head', op_kind: 'gemm', is_fused_kernel: true, pct_gpu_time: 12 },
  { short_name: 'moe_gemm1_leaf', op_kind: 'gemm', pct_gpu_time: 6, prepared_ut: prepared('gemm') },
  { short_name: 'allreduce_rmsnorm', op_kind: 'collective', is_fused_kernel: true, pct_gpu_time: 5 },
  { short_name: 'dense_qkv_gemm', op_kind: 'gemm', pct_gpu_time: 8 },
];
const out = applyOpIdentityGuard(queue, 'strategize');
const head = (name) => out.find((c) => c.short_name === name);
const defect = (name) => preparedUtDefect(head(name), head(name).prepared_ut.result, 'extract_op');

ok(head('fused_attn_score_select').op_kind === 'attn', 'a fused attention chain keeps op_kind=attn');
ok(defect('fused_attn_score_select') === '', "its prepared UT still matches its declaration (no 'moe / attn' defect)");
ok(head('fused_moe_decode').op_kind === 'moe' && defect('fused_moe_decode') === '', 'a fused MoE head stays op_kind=moe');
ok(head('fused_moe_kernel').op_kind === 'moe', 'an untyped MoE-named head is tagged op_kind=moe');
ok(head('grouped_gemm_head').op_kind === 'moe', 'a fused GEMM head is tagged op_kind=moe');
ok(head('moe_gemm1_leaf').op_kind === 'gemm' && defect('moe_gemm1_leaf') === '',
  'a prepared UT validated as gemm is not rewritten to moe');
ok(head('allreduce_rmsnorm').op_kind === 'collective', 'a fused collective keeps its kind');
ok(head('dense_qkv_gemm').op_kind === 'gemm', 'a dense GEMM head is untouched');
ok(logs.some((l) => /keep their declared op_kind: .*fused_attn_score_select=attn/.test(l)),
  'the heads that keep their kind are logged');

console.log(failures === 0 ? '\nPASS: the op-identity guard keeps settled non-GEMM kinds.' : `\nFAILED: ${failures} assertion(s).`);
process.exit(failures === 0 ? 0 : 1);
