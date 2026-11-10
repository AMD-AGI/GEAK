// Tests that every workflow script fits Claude Code's workflow tool.
//
// Run: node e2e_workflow/scripts/test_workflow_script_size.js
//
// The tool refuses a scriptPath file larger than 524288 bytes before anything runs ("Workflow script
// file ... exceeds 524288 bytes": it checks the file's size, so bytes, not characters), so a run whose
// script is over it ends with no phase started. Each script must stay MARGIN_BYTES under the limit, so
// the change that would cross it fails here rather than at launch.

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const WORKFLOW_SCRIPT_MAX_BYTES = 524288;
const MARGIN_BYTES = 4096;

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

const scripts = ['e2e_workflow', 'kernel_workflow'].flatMap((dir) => fs.readdirSync(path.join(ROOT, dir))
  .filter((f) => f.endsWith('.js')).map((f) => `${dir}/${f}`));
ok(scripts.includes('e2e_workflow/e2e_workflow.js') && scripts.includes('kernel_workflow/kernel_lane.js'),
  'the workflow scripts are located');
for (const rel of scripts) {
  const bytes = fs.statSync(path.join(ROOT, rel)).size;
  ok(bytes <= WORKFLOW_SCRIPT_MAX_BYTES - MARGIN_BYTES,
    `${rel} is ${bytes} bytes (the tool refuses over ${WORKFLOW_SCRIPT_MAX_BYTES}; keep ${MARGIN_BYTES} under it)`);
}

console.log(failures === 0
  ? '\nPASS: every workflow script fits the workflow tool.'
  : `\nFAILED: ${failures} assertion(s).`);
process.exit(failures === 0 ? 0 : 1);
