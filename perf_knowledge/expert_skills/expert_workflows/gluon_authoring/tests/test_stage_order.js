#!/usr/bin/env node
// The stage machine is code: each entry mode runs its stages in order, and the reused GEAK roles
// (director, benchmark_engineer, verify_engineer) stay where acceptance needs them; the baseline
// profile is the pack's gluon_profiler.
'use strict';

const { launch, ok, done, inOrder, engineerReturn, refsIn } = require('./harness');

(async () => {
  console.log('\n# 1. port (plain Triton source): setup -> entry -> budget -> transcribe -> recover -> climb -> close -> validate');
  {
    const x = await launch({ budget: 4, entry_mode: 'auto' }, (label, prompt) => {
      // r1 anchor (0.8x), r2 recovery to 0.97x of the champion clears parity, r3/r4 climb above it.
      if (label === 'verify r1') return { status: 'regression', correctness: 'pass', verified_geomean: 0.8 };
      if (label === 'verify r2_d0') return { status: 'regression', correctness: 'pass', verified_geomean: 0.97 };
      if (label === 'verify r3_d0') return { status: 'verified', correctness: 'pass', verified_geomean: 1.12 };
      if (label === 'verify r4_d0') return { status: 'verified', correctness: 'pass', verified_geomean: 1.2 };
      return undefined;
    }, { entry: 'port' });
    ok(!x.error, 'runs to completion', x.error);
    ok(inOrder(x.labels, ['director:setup', 'benchmark_engineer', 'gluon_profiler:baseline', 'gluon_lead:entry',
      'gluon_lead:budget', 'gluon_engineer:transcribe r1', 'evidence_audit r1', 'verify r1',
      'tech_lead:plan r2', 'gluon_engineer:recover r2_d0', 'verify r2_d0', 'gluon_engineer:climb r3_d0', 'verify r3_d0',
      'gluon_engineer:climb r4_d0', 'verify r4_d0', 'gluon_lead:close', 'director:validate']), 'port stage order', x.labels);
    ok(!x.labels.includes('verify champion'), 'champion == frozen baseline: no separate champion measurement');
    ok(x.result.entry_mode === 'port' && x.result.parity === 'cleared', 'parity cleared before the climb', x.result.parity);
    ok(x.result.outcome === 'win' && x.result.accepted.round === 4, 'the last verified best is accepted', x.result.accepted);
    ok(x.result.rounds.map(r => r.stage).join() === 'transcribe,recover,climb,climb', 'rounds recorded per stage');
    const r1 = x.result.rounds[0];
    ok(r1.kept && !r1.accepted, 'the anchor below the champion is kept on the track but never accepted');
  }

  console.log('\n# 2. port where the task ships the Gluon anchor and the plain champion as golden copy');
  {
    const x = await launch({ budget: 3 }, (label, prompt) => {
      if (label === 'gluon_lead:entry') {
        return { entry_mode: 'port', references_read: refsIn(prompt), source_is_anchor: true,
          champion_patch: '/x/champion.diff', champion_bundle: '/x/b.json', track_ready: true,
          champion_gate: { command: 'python3 champion_gate.py --champion /x/b.json --allow-shallow-climb', exit_code: 0, flags: ['--allow-shallow-climb'] } };
      }
      if (label === 'verify champion') return { status: 'verified', correctness: 'pass', verified_geomean: 1.25 };
      if (label === 'gluon_engineer:transcribe r1') return engineerReturn(prompt, { anchor_provided: true, patch_file: '', track_tag: 'geak_base' });
      if (label === 'verify r2_d0') return { status: 'verified', correctness: 'pass', verified_geomean: 1.22 };
      if (label === 'verify r3_d0') return { status: 'verified', correctness: 'pass', verified_geomean: 1.4 };
      return undefined;
    });
    ok(!x.error, 'runs', x.error);
    ok(inOrder(x.labels, ['gluon_lead:entry', 'gluon_lead:budget', 'verify champion', 'gluon_engineer:transcribe r1',
      'gluon_engineer:recover r2_d0', 'verify r2_d0', 'gluon_engineer:climb r3_d0', 'verify r3_d0', 'gluon_lead:close']), 'champion measured by verify before the track', x.labels);
    ok(!x.labels.includes('verify r1'), 'an anchor that is already the frozen baseline is not re-measured');
    ok(x.result.champion_vs_baseline === 1.25, 'champion speedup vs the frozen anchor is carried', x.result.champion_vs_baseline);
    const r2 = x.result.rounds[1];
    ok(Math.abs(r2.vs_champion - 1.22 / 1.25) < 1e-9, 'every round carries vs_champion', r2.vs_champion);
    ok(x.result.parity === 'cleared', 'parity: 1.22/1.25 = 0.976 >= 0.95 with the gate at exit 0');
    ok(x.result.outcome === 'win' && Math.abs(x.result.accepted.vs_champion - 1.12) < 1e-9, 'win quoted vs the champion', x.result.accepted);
    ok(x.result.caveats.includes('champion_gate --allow-shallow-climb'), 'a G1 escape flag becomes a caveat', x.result.caveats);
  }

  console.log('\n# 3. incumbent: no transcribe, no recover, no parity gate; the climb starts at round 1');
  {
    const x = await launch({ budget: 2, entry_mode: 'incumbent' }, (label) => {
      if (label === 'verify r1_d0') return { status: 'verified', correctness: 'pass', verified_geomean: 1.08 };
      if (label === 'verify r2_d0') return { status: 'verified', correctness: 'pass', verified_geomean: 1.05 };
      return undefined;
    }, { entry: 'incumbent' });
    ok(!x.error, 'runs', x.error);
    ok(inOrder(x.labels, ['gluon_lead:entry', 'gluon_lead:budget', 'tech_lead:plan r1', 'gluon_engineer:climb r1_d0', 'verify r1_d0', 'tech_lead:memory r1',
      'gluon_engineer:climb r2_d0', 'verify r2_d0', 'gluon_lead:close', 'director:validate']), 'incumbent stage order', x.labels);
    ok(!x.labels.some(l => /transcribe|recover|verify champion/.test(l)), 'no transcribe/recover/champion steps');
    ok(x.result.parity === 'not_applicable' && x.result.outcome === 'win', 'incumbent win, parity not applicable', x.result);
    ok(x.result.accepted.round === 1, 'r2 (1.05 < 1.08 * 1.02) does not displace r1', x.result.accepted);
  }

  console.log('\n# 4. fresh: author the first Gluon kernel, then climb');
  {
    const x = await launch({ budget: 2, entry_mode: 'fresh' }, (label, prompt) => {
      if (label === 'gluon_lead:entry') return { entry_mode: 'fresh', references_read: refsIn(prompt),
        champion_gate: { exit_code: null, not_applicable_reason: 'baseline is a torch reference' }, track_ready: true };
      if (label === 'verify r1') return { status: 'regression', correctness: 'pass', verified_geomean: 0.6 };
      if (label === 'verify r2_d0') return { status: 'verified', correctness: 'pass', verified_geomean: 1.3 };
      return undefined;
    });
    ok(!x.error, 'runs', x.error);
    ok(inOrder(x.labels, ['gluon_lead:entry', 'gluon_engineer:author r1', 'verify r1', 'gluon_engineer:climb r2_d0', 'verify r2_d0']), 'fresh stage order', x.labels);
    ok(x.result.outcome === 'win', 'fresh win', x.result.outcome);
  }

  console.log('\n# 5. GEAK roles keep their own role files; method roles come from this package');
  {
    const x = await launch({ budget: 1 }, undefined, { entry: 'incumbent' });
    const roleOf = (label) => ((x.calls.find(c => c.label === label) || {}).prompt || '').match(/Then Read (\S+\.md)/);
    ok(/kernel_workflow\/roles\/director\.md$/.test(roleOf('director:setup')[1]), 'director from kernel_workflow');
    ok(/kernel_workflow\/roles\/verify_engineer\.md$/.test(roleOf('verify r1_d0')[1]), 'verify_engineer from kernel_workflow');
    ok(/expert_workflows\/gluon_authoring\/roles\/gluon_engineer\.md$/.test(roleOf('gluon_engineer:climb r1_d0')[1]), 'gluon_engineer from this package');
  }
  done('stage-order');
})();
