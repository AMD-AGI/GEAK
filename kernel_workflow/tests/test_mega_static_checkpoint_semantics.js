#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const megaRole = require('./mega_roles');

const src = require('./mega_source')();
const engineer = megaRole('engineer');

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

const schema = src.match(
  /const MEGA_CANDIDATE_SCHEMA = obj\(\{([\s\S]*?)\n\}, \['candidate_id'/,
);
if (!schema) throw new Error('missing Mega candidate schema');
// The Author's GPU-mode and checkpoint rules live in the engineer's mode=mega fragment.
const authorPrompt = engineer;

console.log('\n# static checkpoint completion is separate from hardware proof');
ok(/claim_complete: \{[\s\S]*GPU-free static checkpoint/.test(schema[0]),
  'schema defines claim_complete as turn-artifact completion');
ok(/candidate_status: \{[\s\S]*GPU-free source checkpoint remains authoring/.test(schema[0]),
  'schema keeps a static checkpoint in authoring state');
ok(/runtime_verified: \{ type: 'boolean' \}/.test(schema[0]) &&
   /gpu_executed: \{ type: 'boolean' \}/.test(schema[0]) &&
   /score_complete: \{ type: 'boolean' \}/.test(schema[0]),
  'runtime, GPU, and scoring gates remain independent fields');
ok(/candidate_status:"authoring"[\s\S]*claim_complete:true/.test(authorPrompt) &&
   /runtime_verified, gpu_executed, and score_complete remain[\s\S]*false separately/.test(authorPrompt),
  'the dispatched Author prompt states the exact static checkpoint result');
ok(/claim_complete` describes this turn's[\s\S]*set it true with[\s\S]*candidate_status:"authoring"/.test(engineer),
  'the shared Engineer role retains the same checkpoint semantics');

console.log(failures
  ? `\nFAIL: ${failures} checkpoint assertion(s)`
  : '\nPASS: static checkpoint completion is explicit.');
process.exit(failures ? 1 : 0);
