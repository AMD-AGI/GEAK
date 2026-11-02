// Tests that every workflow script fits Claude Code's workflow tool.
//
// Run: node e2e_workflow/scripts/test_workflow_script_size.js
//
// The tool refuses a script longer than 524288 characters before anything runs ("workflow script
// exceeds 524288 bytes": it counts the JS string's length, so UTF-16 code units, not bytes), so a
// run whose script is over it ends with no phase started.

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const WORKFLOW_SCRIPT_MAX_CHARS = 524288;

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

const scripts = ['e2e_workflow', 'kernel_workflow'].flatMap((dir) => fs.readdirSync(path.join(ROOT, dir))
  .filter((f) => f.endsWith('.js')).map((f) => `${dir}/${f}`));
ok(scripts.includes('e2e_workflow/e2e_workflow.js') && scripts.includes('kernel_workflow/kernel_lane.js'),
  'the workflow scripts are located');
for (const rel of scripts) {
  const chars = fs.readFileSync(path.join(ROOT, rel), 'utf8').length;
  ok(chars <= WORKFLOW_SCRIPT_MAX_CHARS, `${rel} is ${chars} characters (limit ${WORKFLOW_SCRIPT_MAX_CHARS})`);
}

console.log(failures === 0
  ? '\nPASS: every workflow script fits the workflow tool.'
  : `\nFAILED: ${failures} assertion(s).`);
process.exit(failures === 0 ? 0 : 1);
