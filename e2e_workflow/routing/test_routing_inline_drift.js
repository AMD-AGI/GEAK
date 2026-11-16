'use strict';
// ROUTING-INLINE drift guard (PR #574 review: amd-ethany). Offline, pure node (fs/path + the
// canonical module) -- no GPU, model or network -- an L0 check in the same spirit as the Eikos
// registry lint.
//
// WHAT THE REVIEWER ASKED. "The ROUTING-INLINE region is copied byte for byte into e2e_workflow.js,
// kernel_lane.js and routing/tier_map.js. Add a check that fails when the three copies differ."
//
// WHAT IS ACTUALLY ON DISK. The three files are NOT a single byte-identical block:
//   * routing/tier_map.js is the CANONICAL pure module (the single source of truth). It has NO
//     ROUTING-INLINE sentinels -- it is the thing the other two inline FROM.
//   * e2e_workflow.js inlines the verbatim-write CASCADE seam between the sentinels.
//   * kernel_lane.js inlines the cost-LADDER seam between the sentinels (a different, larger body).
// The two inline seams are deliberately different code, so literal whole-region byte-equality across
// all three is impossible. What the three copies MUST keep in lockstep is the routing CONTRACT they
// share: the model/lane identities and the U+0000 scope-key separator. A silent change to a model
// name or the lane ladder in ONE copy but not the others is exactly the drift the reviewer flagged
// (cf. the GEAK_CLAUDE_MODEL bump). THIS guard fails when that shared contract diverges from the
// canonical module. Deeper per-seam drift (inline ROUTE_TIER_MAP vs canonical TIER_MAP, ROUTE_LANES
// vs canonical LANES, the full reducer) is asserted by routing/routing_dryrun.js.
//
//   node e2e_workflow/routing/test_routing_inline_drift.js
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const CANON = require('./tier_map.js');          // the single source of truth
const START = '// <<ROUTING-INLINE-START>>';
const END = '// <<ROUTING-INLINE-END>>';

function regionOf(rel) {
  const src = fs.readFileSync(path.join(ROOT, rel), 'utf8');
  const i = src.indexOf(START);
  const j = src.indexOf(END);
  if (i < 0 || j < 0) throw new Error(`ROUTING-INLINE sentinels not found in ${rel}`);
  return src.slice(i, j + END.length);
}

const E2E = regionOf(path.join('e2e_workflow', 'e2e_workflow.js'));
const KERNEL = regionOf(path.join('kernel_workflow', 'kernel_lane.js'));
const TIER = fs.readFileSync(path.join(ROOT, 'e2e_workflow', 'routing', 'tier_map.js'), 'utf8');

let fails = 0, n = 0;
function ok(name, cond) { n++; if (cond) { console.log(`PASS  ${name}`); } else { console.log(`FAIL  ${name}`); fails++; } }

// The canonical routing contract, taken from the module every copy inlines from.
const LANES = CANON.LANES;
const STRONG = CANON.MODEL_STRONG;
const CHEAP = CANON.MODEL_CHEAP;
const q = (s) => `'${s}'`;
const ladderLiteral = '[' + LANES.map(q).join(', ') + ']';   // e.g. ['claude-haiku-5-5', 'claude-sonnet-5-5', 'claude-opus-5-5']

// 1) The canonical ladder is a cheapest-first 3-lane list whose top is MODEL_STRONG and whose
//    middle is MODEL_CHEAP. If this ever changes, every copy below must change with it.
ok('canonical LANES is a 3-lane cheapest-first ladder (top=MODEL_STRONG, mid=MODEL_CHEAP)',
  Array.isArray(LANES) && LANES.length === 3 && LANES[2] === STRONG && LANES[1] === CHEAP);

// 2) The kernel ladder inline copy carries the canonical lane list BYTE-FOR-BYTE.
ok('kernel_lane.js inline ladder == canonical LANES (byte-for-byte)', KERNEL.indexOf(ladderLiteral) >= 0);

// 3) Each inline copy pins the canonical model identities it uses (no silent per-copy bump).
ok('e2e_workflow.js inline pins the canonical strong model', E2E.indexOf(q(STRONG)) >= 0);
ok('e2e_workflow.js inline pins the canonical cheap model', E2E.indexOf(q(CHEAP)) >= 0);
ok('kernel_lane.js inline pins all canonical lane identities', LANES.every((m) => KERNEL.indexOf(q(m)) >= 0));
ok('tier_map.js canonical pins the strong and cheap model identities',
  TIER.indexOf(q(STRONG)) >= 0 && TIER.indexOf(q(CHEAP)) >= 0);

// 4) All three build the scope-key separator the SAME way: textual in source, U+0000 at runtime.
const SEP = 'String.fromCharCode(0)';
ok('canonical, e2e and kernel all build the U+0000 scope separator identically',
  TIER.indexOf(SEP) >= 0 && E2E.indexOf(SEP) >= 0 && KERNEL.indexOf(SEP) >= 0);

console.log(`\n${fails === 0 ? 'ROUTING-INLINE DRIFT OK' : 'ROUTING-INLINE DRIFT FAIL'} - ${n - fails}/${n} checks`);
process.exit(fails === 0 ? 0 : 1);
