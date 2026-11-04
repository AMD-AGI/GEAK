#!/usr/bin/env node
// Regression guard for head admission (no GPU, no model needed).
//
// The Architect restates each head candidate and has emitted the profile row's classification
// (library_gemm, triton_kernel) as entity_kind. admitHeads then refused every head as
// wrong_head_granularity although each profiled row was a gpu_kernel. The profiled row is the
// authority: a head whose device_kernel matches a profiled gpu_kernel row takes that entity_kind,
// and nothing else is promoted.
//
// Run:  node e2e_workflow/scripts/test_head_entity_kind_adoption.js
'use strict';
const fs = require('fs');
const path = require('path');

const WORKFLOW = path.resolve(__dirname, '..', 'e2e_workflow.js');

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

const src = fs.readFileSync(WORKFLOW, 'utf8');
const canonStart = src.indexOf('const SHORT_NAME_LIMIT');
const canonEnd = src.indexOf('function requiredDeviceKernel');
const rdkEnd = src.indexOf('const candidateSpec');
const adoptStart = src.indexOf('function adoptProfiledEntityKind');
const admitEnd = src.indexOf('// A FROZEN baseline is resolvable');
ok(canonStart !== -1 && canonEnd > canonStart, 'canonicalization block located in e2e_workflow.js');
ok(rdkEnd > canonEnd, 'requiredDeviceKernel located in e2e_workflow.js');
ok(adoptStart !== -1 && admitEnd > adoptStart, 'adoptProfiledEntityKind + admitHeads located in e2e_workflow.js');
if (failures) process.exit(1);

const logged = [];
const { adoptProfiledEntityKind, admitHeads, setProfile, PRE_FLAGGED_HEADS } = new Function(
  'log', 'prepareHeadSelection',
  `${src.slice(canonStart, rdkEnd)}\nlet profile = null;\n${src.slice(adoptStart, admitEnd)}\n` +
  'return { adoptProfiledEntityKind, admitHeads, setProfile: (p) => { profile = p; }, PRE_FLAGGED_HEADS };')(
  (msg) => logged.push(msg), (h) => h);

const GEMM = 'Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAB_SCD_SAV_UserArgs_MT128x32x64_MI16x16x1_SN_LDSB1_AFC1';
const rows = [
  { short_name: 'Cijk_MT128x32', classification: 'library_gemm', entity_kind: 'gpu_kernel', device_kernel: GEMM },
  { short_name: 'kernel_paged_attention_2d', classification: 'triton_kernel', entity_kind: 'gpu_kernel',
    device_kernel: 'kernel_paged_attention_2d.kd' },
  { short_name: 'aten::linear', classification: 'host_op', entity_kind: 'host_op', device_kernel: 'aten::linear' },
];

console.log('\n# adopted from the profiled row');
const gemm = { short_name: 'decode_dense_linear_gemm', entity_kind: 'library_gemm', device_kernel: GEMM };
ok(adoptProfiledEntityKind(gemm, rows) && gemm.entity_kind === 'gpu_kernel',
  'a renamed GEMM head whose device_kernel is a profiled gpu_kernel row is admitted');
const attn = { short_name: 'paged_attn', entity_kind: 'triton_kernel', device_kernel: 'kernel_paged_attention_2d' };
ok(adoptProfiledEntityKind(attn, rows) && attn.entity_kind === 'gpu_kernel',
  'the device_kernel matches through canonicalization (.kd suffix)');

console.log('\n# never promoted');
const host = { short_name: 'linear', entity_kind: 'host_op', device_kernel: 'aten::linear' };
ok(!adoptProfiledEntityKind(host, rows) && host.entity_kind === 'host_op',
  'a head matching a non-gpu_kernel row keeps its entity_kind');
const unknown = { short_name: 'x', entity_kind: 'library_gemm', device_kernel: 'not_in_the_profile' };
ok(!adoptProfiledEntityKind(unknown, rows) && unknown.entity_kind === 'library_gemm',
  'a device_kernel absent from the profile is not promoted');
const bare = { short_name: 'y', entity_kind: 'library_gemm' };
ok(!adoptProfiledEntityKind(bare, rows) && bare.entity_kind === 'library_gemm',
  'a head without device_kernel is not promoted');
const resumed = { short_name: 'z', entity_kind: 'library_gemm', device_kernel: GEMM };
ok(!adoptProfiledEntityKind(resumed, []) && resumed.entity_kind === 'library_gemm',
  'no profiled rows (resume) promotes nothing');

console.log('\n# a namespace collision with a non-kernel row is not promoted');
const collision = [
  { name: 'aten::mul', entity_kind: 'dispatcher_op' },
  { name: 'mul', short_name: 'mul', entity_kind: 'gpu_kernel', device_kernel: 'mul' },
];
const hostMul = { short_name: 'mul_head', entity_kind: 'library_gemm', device_kernel: 'aten::mul' };
ok(!adoptProfiledEntityKind(hostMul, collision) && hostMul.entity_kind === 'library_gemm',
  'a head naming aten::mul matches the dispatcher row exactly and is not promoted through mul');
const kernelMul = { short_name: 'mul_kernel', entity_kind: 'triton_kernel', device_kernel: 'mul' };
ok(adoptProfiledEntityKind(kernelMul, collision) && kernelMul.entity_kind === 'gpu_kernel',
  'a head naming mul matches the gpu_kernel row exactly and is promoted');
const foldedMul = { short_name: 'mul_folded', entity_kind: 'library_gemm', device_kernel: 'aten::Mul' };
ok(!adoptProfiledEntityKind(foldedMul, collision) && foldedMul.entity_kind === 'library_gemm',
  'a head that only matches through the shared fold reaches both kinds and is not promoted');
PRE_FLAGGED_HEADS.length = 0;
setProfile({ top_kernels: collision });
const collided = admitHeads([{ short_name: 'mul_head', entity_kind: 'library_gemm', device_kernel: 'aten::mul' }], 'test');
ok(collided.length === 0 && PRE_FLAGGED_HEADS.length === 1 &&
  PRE_FLAGGED_HEADS[0].gate === 'wrong_head_granularity',
  'admitHeads flags the aten::mul head instead of admitting it');
PRE_FLAGGED_HEADS.length = 0;

console.log('\n# a display alias does not override a concrete symbol');
const FILL = 'void at::native::vectorized_elementwise_kernel<16, at::native::FillFunctor<bool>, ' +
  'std::array<char*, 1ul> >(int, at::native::FillFunctor<bool>)';
const POW = 'void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous ' +
  'namespace)::pow_tensor_scalar_kernel_impl<float>, std::array<char*, 2ul> >(int)';
const fillRows = [{ short_name: 'vectorized_elementwise_kernel', entity_kind: 'gpu_kernel', device_kernel: FILL }];
const pow = { short_name: 'pow', entity_kind: 'library_gemm', device_kernel: POW };
ok(!adoptProfiledEntityKind(pow, fillRows) && pow.entity_kind === 'library_gemm',
  'an unprofiled instantiation is not promoted through the shared short_name');
setProfile({ top_kernels: fillRows });
const powAdmitted = admitHeads([{ short_name: 'pow', entity_kind: 'library_gemm', device_kernel: POW }], 'test');
ok(powAdmitted.length === 0 && PRE_FLAGGED_HEADS.length === 1, 'admitHeads flags the unprofiled pow head');
PRE_FLAGGED_HEADS.length = 0;
const aliased = [
  { name: 'aten::mul', short_name: 'mul', entity_kind: 'dispatcher_op' },
  { name: 'mul', short_name: 'mul', entity_kind: 'gpu_kernel', device_kernel: 'mul' },
];
const exactMul = { short_name: 'mul_kernel', entity_kind: 'triton_kernel', device_kernel: 'mul' };
ok(adoptProfiledEntityKind(exactMul, aliased) && exactMul.entity_kind === 'gpu_kernel',
  'a dispatcher row whose short_name is mul does not block an exact mul gpu_kernel head');

console.log('\n# admitHeads uses the profiled rows');
setProfile({ top_kernels: rows });
const admitted = admitHeads([
  { short_name: 'decode_dense_linear_gemm', entity_kind: 'library_gemm', device_kernel: GEMM },
  { short_name: 'linear', entity_kind: 'host_op', device_kernel: 'aten::linear' },
  { short_name: 'ghost', entity_kind: 'library_gemm', device_kernel: 'not_in_the_profile' },
], 'test');
ok(admitted.length === 1 && admitted[0].short_name === 'decode_dense_linear_gemm' &&
  admitted[0].entity_kind === 'gpu_kernel', 'the renamed GEMM head is admitted as gpu_kernel');
const flagged = PRE_FLAGGED_HEADS.map((f) => f.short_name).sort();
ok(JSON.stringify(flagged) === JSON.stringify(['ghost', 'linear']) &&
  PRE_FLAGGED_HEADS.every((f) => f.gate === 'wrong_head_granularity'),
  'the host-op head and the unprofiled head are flagged, not promoted');

console.log('\n# a gpu_kernel row without device_kernel is reported');
logged.length = 0;
PRE_FLAGGED_HEADS.length = 0;
setProfile({ top_kernels: [{ short_name: 'Cijk_MT128x32', entity_kind: 'gpu_kernel' }] });
const none = admitHeads([{ short_name: 'g', entity_kind: 'library_gemm', device_kernel: GEMM }], 'test');
ok(none.length === 0 && PRE_FLAGGED_HEADS.length === 1, 'the head is not admitted from a row without device_kernel');
ok(logged.some((m) => m.includes('profile contract') && m.includes('Cijk_MT128x32')),
  'admitHeads names the gpu_kernel row that carries no device_kernel');

console.log(failures ? `\n${failures} failure(s)` : '\nall passed');
process.exit(failures ? 1 : 0);
