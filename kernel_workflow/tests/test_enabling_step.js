#!/usr/bin/env node
// A step that ENABLES a fusion is not a step that speeds one up.
//
// The round filter is `primSpeedup(ver) > 1.0`. For a fusion built in stages that is not a quality
// bar, it is a structural block: the producer half adds completion signalling and a second buffer,
// has no consumer yet, and can only measure its own overhead. It is supposed to be slower. Under a
// speed-only filter it is discarded, never enters the canonical tree, and the consumer half is then
// written against a tree where the producer half does not exist -- so it is never written at all.
// That is how this project reached a half-fused kernel and stopped there.
//
// Each assertion below is a way the fix could be wrong in a way nobody would notice:
//   - `enabling` becomes a way to commit anything, including a step that does not work
'use strict';

const fs = require('fs');
const path = require('path');

const WF = path.resolve(__dirname, '..');
const src = require('./mega_source')();

let failures = 0;
const ok = (cond, msg) => {
  if (cond) console.log('  ok:', msg);
  else { console.error('  FAIL:', msg); failures++; }
};

const m = src.match(/\/\/ <<REPLAY:enabling_step>>([\s\S]*?)\/\/ <<\/REPLAY:enabling_step>>/);
if (!m) { console.error('  FAIL: no <<REPLAY:enabling_step>> region — nothing to test'); process.exit(1); }
const { stepRoleOf } =
  // eslint-disable-next-line no-new-func
  new Function(`${m[1]}\nreturn { stepRoleOf };`)();

console.log('\n# the default does not change');
{
  ok(stepRoleOf({}) === 'terminal' && stepRoleOf(null) === 'terminal',
     'a direction that says nothing is terminal — the speed bar stays the default for everything else');
  ok(stepRoleOf({ step_role: 'ENABLING' }) === 'enabling',
     'the role is read case-insensitively; a direction is not silently demoted by capitalisation');
  ok(stepRoleOf({ step_role: 'prerequisite' }) === 'terminal',
     'an unrecognised role falls back to the STRICTER bar, not the looser one');
}

console.log('\n# wired into the round');
{
  ok(/CHAIN_CAVEATS\.length \? \{ CHAIN_CAVEATS \}/.test(src),
     'and the report, so a wave that stopped half-fused does not read as a failed search');
}

console.log(failures ? `\n${failures} FAILED` : '\nall passed');
process.exit(failures ? 1 : 0);
