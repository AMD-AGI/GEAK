// Offline unit tests for the pure routing tier map + verbatim-write validator.
// No API call, no GPU, no fs writes. Run with any node:
//   NODE=$(ls -t /home/aditysin/.cursor-server/bin/linux-x64/*/node | head -1)
//   "$NODE" e2e_workflow/routing/tier_map.test.js
'use strict';
const R = require('./tier_map.js');

let fails = 0, n = 0;
function eq(name, got, want) {
  n++;
  const ok = JSON.stringify(got) === JSON.stringify(want);
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}: got ${JSON.stringify(got)} want ${JSON.stringify(want)}`);
  if (!ok) fails++;
}
function ok_(name, cond) { n++; const ok = !!cond; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}`); if (!ok) fails++; }

// --- labelPrefix: static prefix, colon kept, first space splits -------------------
eq('labelPrefix keeps whole static colon label',
   R.labelPrefix('a:b:c'), 'a:b:c');
eq('labelPrefix strips dynamic tag after space', R.labelPrefix('clock r1'), 'clock');
eq('labelPrefix strips reclaim round', R.labelPrefix('storage:reclaim r3'), 'storage:reclaim');
eq('labelPrefix empty on null', R.labelPrefix(null), '');

// --- routeFor: OFF by default (byte-identical) ------------------------------------
// NOTE: the scope label is the ACTUAL call-site literal 'persist-workflow-return'
// (e2e_workflow.js), not a role:sub_phase string — the map keys on what the call passes.
eq('routing disabled -> undefined for a mapped scope',
   R.routeFor({ phase: 'Validate', label: 'persist-workflow-return' },
              { enabled: false }),
   undefined);
eq('env without GEAK_ROUTING -> disabled -> undefined',
   R.routeFor({ phase: 'Validate', label: 'persist-workflow-return' },
              { env: {} }),
   undefined);

// --- routeFor: ON, only allowlisted scopes get the cheap model --------------------
eq('enabled + mapped workflow-return -> Sonnet',
   R.routeFor({ phase: 'Validate', label: 'persist-workflow-return' },
              { enabled: true }),
   'claude-sonnet-5');
eq('enabled + mapped record-measurements -> Sonnet',
   R.routeFor({ phase: 'WarmStart', label: 'warm_start:record-measurements' },
              { enabled: true }),
   'claude-sonnet-5');
eq('enabled + UN-mapped reasoning role -> undefined (pinned)',
   R.routeFor({ phase: 'Optimize', label: 'system_architect:strategize' },
              { enabled: true }),
   undefined);
eq('enabled + right label but WRONG phase -> undefined (scope key is phase+label)',
   R.routeFor({ phase: 'Optimize', label: 'persist-workflow-return' },
              { enabled: true }),
   undefined);
eq('enabled + mapped label carrying dynamic suffix still matches on static prefix',
   R.routeFor({ phase: 'WarmStart', label: 'warm_start:record-measurements extra' },
              { enabled: true }),
   'claude-sonnet-5');

// --- routeFor: a STRONG-tier map entry produces NO override -----------------------
eq('strong tier -> no override (fall through to pinned)',
   R.routeFor({ phase: 'X', label: 'y' },
              { enabled: true, map: (() => { const m = {}; m[R.scopeKey('X', 'y')] = { tier: 'strong', kind: 'verbatim_write' }; return m; })() }),
   undefined);

// --- decideFor: returns model + validator for a mapped scope ----------------------
const dec = R.decideFor({ phase: 'Validate', label: 'persist-workflow-return' }, { enabled: true });
ok_('decideFor mapped -> model+validator', dec && dec.model === 'claude-sonnet-5' && typeof dec.validate === 'function' && dec.kind === 'verbatim_write');
eq('decideFor un-mapped -> null', R.decideFor({ phase: 'Optimize', label: 'director:plan' }, { enabled: true }), null);
eq('decideFor disabled -> null', R.decideFor({ phase: 'Validate', label: 'persist-workflow-return' }, { enabled: false }), null);

// --- checkVerbatimWrite: artifact oracle ------------------------------------------
const PROMPT_JSON =
  'You are a file writer. Use the Write tool to create the file ' +
  '"/tmp/eval/workflow_return.json" with EXACTLY the content below, verbatim. ' +
  'Do NOT reformat, truncate, summarize, or add/remove any keys or values:\n\n' +
  '```json\n{\n  "a": 1\n}\n```\n\n' +
  'Then return {"written": true, "path": "/tmp/eval/workflow_return.json"}.';

const intent = R.extractVerbatimIntent(PROMPT_JSON);
eq('extract path', intent.path, '/tmp/eval/workflow_return.json');
eq('extract block', intent.block, '{\n  "a": 1\n}');

// exact match -> ok
eq('validator: exact bytes -> ok',
   R.checkVerbatimWrite(PROMPT_JSON, { written: true, path: '/tmp/eval/workflow_return.json' },
                        () => '{\n  "a": 1\n}').ok,
   true);
// trailing newline tolerated
eq('validator: one trailing newline tolerated -> ok',
   R.checkVerbatimWrite(PROMPT_JSON, { written: true, path: '/tmp/eval/workflow_return.json' },
                        () => '{\n  "a": 1\n}\n').ok,
   true);
// bytes differ -> fail
eq('validator: differing bytes -> fail',
   R.checkVerbatimWrite(PROMPT_JSON, { written: true, path: '/tmp/eval/workflow_return.json' },
                        () => '{\n  "a": 2\n}').ok,
   false);
// artifact absent -> fail
eq('validator: absent artifact -> fail',
   R.checkVerbatimWrite(PROMPT_JSON, { written: true, path: '/tmp/eval/workflow_return.json' },
                        () => null).ok,
   false);
// receipt path mismatch -> fail before reading
eq('validator: receipt path mismatch -> fail',
   R.checkVerbatimWrite(PROMPT_JSON, { written: true, path: '/tmp/OTHER.json' },
                        () => '{\n  "a": 1\n}').ok,
   false);
// no readFile (Path A / no fs) -> UNVERIFIED == FAILURE (Astra: weak must NEVER pass the gate)
const weak = R.checkVerbatimWrite(PROMPT_JSON, { written: true, path: '/tmp/eval/workflow_return.json' }, null);
ok_('validator: no-fs receipt-only ok=FALSE (unverified != pass), flagged weak',
    weak.ok === false && weak.verified === false && weak.weak === true);
const weakNo = R.checkVerbatimWrite(PROMPT_JSON, { written: false }, null);
ok_('validator: no-fs no-receipt ok=false', weakNo.ok === false);
// non-contract prompt -> fail cleanly (no expected, prompt has no verbatim contract)
eq('validator: non-verbatim prompt -> not a contract',
   R.checkVerbatimWrite('do some reasoning', { written: true }, () => 'x').ok,
   false);

// --- checkVerbatimWrite: host-supplied `expected` is authoritative -----------------
// A prompt whose fence would MISPARSE (nested markdown fence) is bypassed by host-held content.
const NESTED_PROMPT =
  'Use the Write tool to create the file "/tmp/eval/measured_on_this_box.md" with EXACTLY ' +
  'the content below, verbatim:\n\n' +
  '````markdown\n# Measured\n\n```\ninner fence\n```\n\n| a | b |\n````\n\n' +
  'Then return {"written": true}.';
const EXP = { path: '/tmp/eval/measured_on_this_box.md', content: '# Measured\n\n```\ninner fence\n```\n\n| a | b |' };
// fence-length aware extraction recovers the FULL nested block from the prompt
const nestedIntent = R.extractVerbatimIntent(NESTED_PROMPT);
eq('fence-length aware: nested ``` inside ```` not truncated',
   nestedIntent.block, '# Measured\n\n```\ninner fence\n```\n\n| a | b |');
// host-supplied expected: exact on-disk bytes -> ok
eq('validator: host expected + exact bytes -> ok',
   R.checkVerbatimWrite(NESTED_PROMPT, { written: true }, () => EXP.content, EXP).ok, true);
// host-supplied expected: differing bytes -> fail (authoritative content wins over any prompt parse)
eq('validator: host expected + differing bytes -> fail',
   R.checkVerbatimWrite(NESTED_PROMPT, { written: true }, () => EXP.content + 'X', EXP).ok, false);

// --- escalate(): cheap-first, strong-fallback, all attempts recorded ---------------
async function runEscalationTests() {
  const decision = R.decideFor({ phase: 'Validate', label: 'persist-workflow-return' }, { enabled: true });
  const EXPECTED = { path: '/tmp/eval/workflow_return.json', content: '{\n  "a": 1\n}' };

  // (A) cheap writes correct bytes -> accepted at cheap, ONE attempt, no escalation.
  {
    const calls = [];
    const attempts = [];
    let disk = null;
    const run = async (p, o) => { calls.push(o.model); disk = EXPECTED.content; return { written: true, path: EXPECTED.path }; };
    const out = await R.escalate('p', { phase: 'Validate', label: 'persist-workflow-return' }, decision, run,
      { readFile: () => disk, expected: EXPECTED, record: a => attempts.push(a) });
    ok_('escalate: cheap-correct accepted at cheap', out.accepted === 'cheap');
    ok_('escalate: cheap-correct made ONE call (no fallback)', calls.length === 1 && calls[0] === 'claude-sonnet-5');
    ok_('escalate: cheap-correct recorded exactly 1 attempt', attempts.length === 1 && attempts[0].ok === true);
  }

  // (B) cheap writes WRONG bytes -> strong fallback (bypasses cheap map), strong fixes it.
  //     Both attempts recorded; cap of 2; second call is the strong model.
  {
    const calls = [];
    const attempts = [];
    let disk = null;
    const run = async (p, o) => {
      calls.push(o.model);
      disk = (o.model === R.MODEL_STRONG) ? EXPECTED.content : '{\n  "a": 999\n}'; // cheap writes wrong, strong writes right
      return { written: true, path: EXPECTED.path };
    };
    const out = await R.escalate('p', { phase: 'Validate', label: 'persist-workflow-return' }, decision, run,
      { readFile: () => disk, expected: EXPECTED, record: a => attempts.push(a) });
    ok_('escalate: cheap-wrong -> accepted at strong', out.accepted === 'strong');
    ok_('escalate: exactly 2 calls, cheap then strong', calls.length === 2 && calls[0] === 'claude-sonnet-5' && calls[1] === R.MODEL_STRONG);
    ok_('escalate: BOTH attempts recorded (fail then pass)', attempts.length === 2 && attempts[0].ok === false && attempts[1].ok === true && attempts[1].escalated === true);
  }

  // (C) both fail -> strong-unverified, still capped at 2, all recorded.
  {
    const calls = [];
    const attempts = [];
    const run = async (p, o) => { calls.push(o.model); return { written: true, path: EXPECTED.path }; }; // disk never matches (null read)
    const out = await R.escalate('p', { phase: 'Validate', label: 'persist-workflow-return' }, decision, run,
      { readFile: () => null, expected: EXPECTED, record: a => attempts.push(a) });
    ok_('escalate: both fail -> strong-unverified', out.accepted === 'strong-unverified');
    ok_('escalate: capped at 2 attempts even on double-fail', calls.length === 2 && attempts.length === 2);
  }

  // (D) a THROWN dispatch is still recorded, then re-thrown (audit trail survives a transport error),
  //     and the exception MESSAGE (which can carry prompt text / credentials) never leaks into the row.
  {
    const attempts = [];
    const secret = 'BEARER sk-leak-me-please';
    const run = async () => { const e = new TypeError(secret); throw e; };
    let threw = false, caught = null;
    try {
      await R.escalate('p', { phase: 'Validate', label: 'persist-workflow-return' }, decision, run,
        { readFile: () => null, expected: EXPECTED, record: a => attempts.push(a) });
    } catch (e) { threw = true; caught = e; }
    ok_('escalate: a thrown cheap dispatch propagates (transport-retry policy preserved)',
        threw && caught && caught.message === secret);
    ok_('escalate: the thrown attempt is recorded (not silent)',
        attempts.length === 1 && attempts[0].attempt === 1 && attempts[0].threw === true && attempts[0].ok === false);
    ok_('escalate: throw record carries exception CLASS + correlation id, NOT the message',
        attempts[0].reason === 'dispatch threw (TypeError)' &&
        typeof attempts[0].cid === 'string' && attempts[0].cid.length > 0 &&
        !JSON.stringify(attempts[0]).includes('sk-leak-me-please'));
    ok_('escalate: a throw on attempt 1 makes NO second dispatch (cap holds as a ceiling)',
        attempts.length === 1);
  }

  console.log(`\n${fails === 0 ? 'ALL PASS' : 'FAIL'} — ${n - fails}/${n} checks`);
  process.exit(fails === 0 ? 0 : 1);
}
runEscalationTests();
