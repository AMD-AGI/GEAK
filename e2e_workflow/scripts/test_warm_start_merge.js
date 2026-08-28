#!/usr/bin/env node
// Regression guard for how a RECOVERED configuration is combined with the one this run was handed
// (no GPU, no model, no KB needed).
//
// A stored e2e record is a whole launch configuration, and warm start replays it on top of whatever
// baseline the run was given — under Hyperloom, a full flag string the run does not own. That
// combination used to be a string concatenation written by an agent, and a duplicated key is
// resolved by argparse (and by `env`) as last-wins. So which value applied depended on the order the
// agent happened to write, and when the baseline won, the server ran the baseline configuration
// twice: ~0% delta, nothing wrong in any log, and the record filed as `rejected`. A record that wins
// recorded as a loss is the one outcome the warm start exists to prevent.
//
// The merge functions are EXTRACTED from the shipped source rather than reimplemented here, so this
// cannot pass while e2e_workflow.js drifts.
//
// Run:  node e2e_workflow/scripts/test_warm_start_merge.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..'); // .../GEAK
const src = fs.readFileSync(path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js'), 'utf8');

let failures = 0;
const eq = (got, want, msg) => {
  const a = JSON.stringify(got), b = JSON.stringify(want);
  if (a !== b) { console.error('  FAIL:', msg, '\n    got: ', a, '\n    want:', b); failures++; }
  else console.log('  ok:', msg);
};

const mStart = src.indexOf('const REPEATABLE_FLAGS');
const mEnd = src.indexOf('// PHASE: Setup + Baseline profile', mStart);
if (mStart < 0 || mEnd < 0) { console.error('FAIL: merge block not found in e2e_workflow.js'); process.exit(1); }
const { mergeFlags, mergeEnv, describeOverrides } = new Function(
  `${src.slice(mStart, mEnd)}\nreturn { mergeFlags, mergeEnv, describeOverrides };`)();

// ------------------------------------------------------------------ the collision this exists for
console.log('collisions');
{
  // The real pair: the Kimi-K3 record pins --context-length 13312, the run's baseline pins 11264.
  const r = mergeFlags('--context-length 11264 --mem-fraction-static 0.96',
    '--context-length 13312 --cuda-graph-max-bs 256');
  eq(r.merged, '--context-length 13312 --mem-fraction-static 0.96 --cuda-graph-max-bs 256',
    'the recorded value replaces the baseline IN PLACE, and what is new is appended');
  eq((r.merged.match(/--context-length/g) || []).length, 1,
    'the key appears exactly once, so last-wins never gets a vote');
  eq(r.overrides, [{ flag: '--context-length', baseline_value: '11264', recalled_value: '13312' }],
    'the disagreement is reported, not just resolved');
  eq(r.added, ['--cuda-graph-max-bs'], 'what the record contributed outright is reported too');
}
{
  const r = mergeEnv('TP=8 MAX_MODEL_LEN=13312', 'MAX_MODEL_LEN=16384 SGLANG_USE_AITER=1');
  eq(r.merged, 'TP=8 MAX_MODEL_LEN=16384 SGLANG_USE_AITER=1', 'env merges by key, same rule');
  eq(r.overrides, [{ flag: 'MAX_MODEL_LEN', baseline_value: '13312', recalled_value: '16384' }],
    'env override reported');
}

// ------------------------------------------------------------------ the spellings that show up
console.log('spellings');
eq(mergeFlags('--foo=1 --bar 2', '--foo=3').merged, '--foo=3 --bar 2',
  '`--flag=value` is the same key as `--flag value`');
eq(mergeFlags('--tp 8', '--enable-torch-compile').merged, '--tp 8 --enable-torch-compile',
  'a bare boolean flag is appended');
eq(mergeFlags('--attention-backend triton', '--extra a b --tp 8').merged,
  '--attention-backend triton --extra a b --tp 8',
  'a multi-valued flag stays one item, so it is overridden whole or not at all');
eq(mergeFlags('', '--tp 8').merged, '--tp 8', 'empty baseline');
eq(mergeFlags('--tp 8', '').merged, '--tp 8', 'empty recovered config changes nothing');
eq(mergeFlags('  --tp   8  ', '').merged, '--tp 8', 'whitespace is normalized');

// ------------------------------------------------------------------ what must NOT be an override
console.log('agreement is not an override');
{
  const r = mergeFlags('--disable-radix-cache --tp 8', '--disable-radix-cache');
  eq(r.overrides, [], 'the two agreeing on a bare flag is not a disagreement');
  eq(r.unchanged, ['--disable-radix-cache'], 'and it is reported as agreement');
  eq(r.merged, '--disable-radix-cache --tp 8', 'nor is it duplicated');
}
eq(mergeEnv('A=1', 'A=1').overrides, [], 'the two agreeing on an env value is not a disagreement');
eq(mergeFlags('--tp 8', '--tp 8').overrides, [], 'identical valued flag is not a disagreement');

console.log('repeatable flags');
{
  const r = mergeFlags('--lora-path a=/a', '--lora-path b=/b');
  eq(r.merged, '--lora-path a=/a --lora-path b=/b',
    '--lora-path is a list; deduping it would silently drop an adapter');
  eq(r.overrides, [], 'a repeatable flag is never an override');
}

console.log('prose');
eq(describeOverrides([{ flag: '--x', baseline_value: '1', recalled_value: '2' }]), '--x 1 -> 2',
  'overrides render for a human');
eq(describeOverrides([{ flag: '--x', baseline_value: null, recalled_value: '2' }]), '--x (unset) -> 2',
  'an absent baseline value says so rather than rendering empty');
eq(describeOverrides([]), '', 'no overrides renders as nothing at all');

// ------------------------------------------------- the orchestrator must USE what it merged
console.log('wiring');
{
  const wStart = src.indexOf('const mf = mergeFlags(curFlags, storedFlags);');
  const wEnd = src.indexOf('if (accept) break;', wStart);
  const block = wStart < 0 ? '' : src.slice(wStart, wEnd);
  eq(block.includes('flags: mf.merged, env: me.merged'), true,
    'the direction handed to config_tuner carries the MERGED strings, not the raw stored ones');
  eq(/storedFlags\s*\|\|\s*curFlags/.test(block), false,
    'adoption falls back to the merged config, never to the unmerged stored one');
  eq(block.includes("'inapplicable'"), true,
    'a launch failure the repair blamed on the baseline is not counted against the record');
}

console.log(failures ? `\n${failures} FAILED` : '\nall passed');
process.exit(failures ? 1 : 0);
