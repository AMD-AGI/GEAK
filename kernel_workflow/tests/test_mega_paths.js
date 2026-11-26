// Every workflow-relative file path mode=mega hands an agent exists: the ${WORKFLOW_DIR}/... paths in
// kernel_lane.js and the knowledge/... and scripts/mega/... paths in the mega role fragments and knowledge
// (a bare scripts/... path there is the task's own, not the workflow's).
// An agent told to read or run a missing file does not fail; it silently works without it.
//
// Run: node kernel_workflow/tests/test_mega_paths.js

const fs = require('fs');
const path = require('path');

const WF = path.resolve(__dirname, '..');
let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

const exists = (rel) => fs.existsSync(path.join(WF, rel.replace(/[.,;:)]+$/, '')));
const lane = fs.readFileSync(path.join(WF, 'kernel_lane.js'), 'utf8');
const lanePaths = [...lane.matchAll(/\$\{WORKFLOW_DIR\}\/([A-Za-z0-9_./-]+\.(?:md|py|json|js))/g)].map((m) => m[1]);
ok(lanePaths.length > 10, `kernel_lane.js names ${lanePaths.length} workflow files`);
for (const p of new Set(lanePaths)) ok(exists(p), `kernel_lane.js: ${p}`);

const docs = [
  ...fs.readdirSync(path.join(WF, 'roles', '_fragments')).map((f) => path.join('roles', '_fragments', f)),
  ...fs.readdirSync(path.join(WF, 'knowledge', 'mega')).map((f) => path.join('knowledge', 'mega', f)),
];
for (const doc of docs) {
  const text = fs.readFileSync(path.join(WF, doc), 'utf8');
  const named = [...text.matchAll(/(?:^|[\s`(/$])((?:knowledge|scripts\/mega)\/[A-Za-z0-9_./-]+\.(?:md|py|json|js))/g)]
    .map((m) => m[1]).filter((p) => !p.includes('learned/'));
  for (const p of new Set(named)) ok(exists(p), `${doc}: ${p}`);
}

console.log(failures === 0 ? '\nPASS: every mode=mega workflow path exists.' : `\nFAIL: ${failures} missing path(s).`);
process.exit(failures === 0 ? 0 : 1);
