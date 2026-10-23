#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.resolve(__dirname, '..', 'kernel_mega.js'), 'utf8');
const engineer = fs.readFileSync(
  path.resolve(__dirname, '..', 'roles', 'mega', 'engineer.md'), 'utf8',
);

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

const schema = src.match(
  /const MEGA_CANDIDATE_SCHEMA = obj\(\{([\s\S]*?)\n\}, \['candidate_id'/,
);
if (!schema) throw new Error('missing Mega candidate schema');
const authorPrompt = src.slice(
  src.indexOf('AUTHORING CONTRACT PREFLIGHT IS REQUIRED'),
  src.indexOf('Before any source edit', src.indexOf('AUTHORING CONTRACT PREFLIGHT IS REQUIRED')),
);

console.log('\n# static checkpoint completion is separate from hardware proof');
ok(/claim_complete: \{[\s\S]*GPU-free static checkpoint/.test(schema[0]),
  'schema defines claim_complete as turn-artifact completion');
ok(/candidate_status: \{[\s\S]*GPU-free source checkpoint remains authoring/.test(schema[0]),
  'schema keeps a static checkpoint in authoring state');
ok(/runtime_verified: \{ type: 'boolean' \}/.test(schema[0]) &&
   /gpu_executed: \{ type: 'boolean' \}/.test(schema[0]) &&
   /score_complete: \{ type: 'boolean' \}/.test(schema[0]),
  'runtime, GPU, and scoring gates remain independent fields');
ok(/claim_complete=true[\s\S]*candidate_status=authoring/.test(authorPrompt) &&
   /runtime_verified, gpu_executed, and score_complete remain[\s\S]*false separately/.test(authorPrompt),
  'the dispatched Author prompt states the exact static checkpoint result');
ok(/claim_complete` describes this turn's[\s\S]*set it true with[\s\S]*candidate_status:"authoring"/.test(engineer),
  'the shared Engineer role retains the same checkpoint semantics');

console.log(failures
  ? `\nFAIL: ${failures} checkpoint assertion(s)`
  : '\nPASS: static checkpoint completion is explicit.');
process.exit(failures ? 1 : 0);
