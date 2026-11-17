#!/usr/bin/env node
// Acceptance is GEAK's (verify_engineer's number + the MIN_IMPROVE commit gate + director validate),
// never the engineer's claim; and the loop stops in code on the parity gate, G1, the deadline and the
// token budget.
'use strict';

const { launch, ok, done, inOrder, engineerReturn, gate, refsIn, goodAtt } = require('./harness');

(async () => {
  console.log('\n# 1. the engineer\'s claim never decides: verify does');
  {
    const x = await launch({ budget: 1, entry_mode: 'incumbent' }, (label, prompt) => {
      if (/^gluon_engineer:/.test(label)) return engineerReturn(prompt, { g4_screen: { command: 'ab_bench.py --permute', exit_code: 0,
        json_path: '/x', delta_pct: 60, resolved: true, valid: true }, claim: 'win' });
      if (label === 'verify r1') return { status: 'correctness_failed', correctness: 'fail', verified_geomean: 0 };
      return undefined;
    }, { entry: 'incumbent' });
    ok(!x.result.accepted && x.result.outcome === 'negative_keep_baseline', 'a 60% screen with a failed verify is not accepted', x.result);
    ok(!x.labels.includes('director:validate') && x.result.validation_status === 'no_candidate', 'nothing to validate');
  }
  {
    const x = await launch({ budget: 1, entry_mode: 'incumbent' }, (label) => {
      if (label === 'verify r1') return { status: 'verified', correctness: 'pass', verified_geomean: 1.015 };
      return undefined;
    }, { entry: 'incumbent' });
    ok(!x.result.accepted, 'a verified 1.5% is under the 2% commit gate', x.result.accepted);
  }
  {
    const x = await launch({ budget: 1, entry_mode: 'incumbent' }, (label, prompt) => {
      if (/^gluon_engineer:/.test(label)) return engineerReturn(prompt, { correctness: 'fail' });
      return undefined;
    }, { entry: 'incumbent' });
    ok(!x.labels.includes('verify r1') && !x.result.accepted, 'a self-reported correctness failure is not a candidate', x.labels);
  }
  {
    const x = await launch({ budget: 1, entry_mode: 'incumbent', min_improve: 0.1 }, (label) => {
      if (label === 'verify r1') return { status: 'verified', correctness: 'pass', verified_geomean: 1.12, verified_weighted: 1.09 };
      return undefined;
    }, { entry: 'incumbent' });
    ok(!x.result.accepted, 'the weighted metric is primary when the oracle provides it (1.09 < 1.10)', x.result.accepted);
  }
  {
    const x = await launch({ budget: 1, entry_mode: 'incumbent' }, (label) => {
      if (label === 'verify r1') return { status: 'verified', correctness: 'pass', verified_geomean: 1.3 };
      if (label === 'director:validate') return { validation_status: 'flagged', director_verified_speedup_geomean: 1.0 };
      return undefined;
    }, { entry: 'incumbent' });
    const v = x.calls.find(c => c.label === 'director:validate');
    ok(v && /FINAL_PATCH: \/tmp\/eval\/final_patch\.diff/.test(v.prompt), 'director validates the delivered patch');
    ok(x.result.validation_status === 'flagged' && x.result.director_verified_speedup_geomean === 1.0,
      'the director\'s verdict is the headline, not the round\'s', x.result);
  }

  console.log('\n# 2. port: accepted vs the frozen anchor but slower than the champion -> revert to plain');
  {
    const x = await launch({ budget: 3 }, (label, prompt) => {
      if (label === 'gluon_lead:entry') return { entry_mode: 'port', references_read: refsIn(prompt),
        source_is_anchor: true, champion_patch: '/x/champion.diff', track_ready: true,
        champion_gate: { command: 'python3 champion_gate.py --champion b.json', exit_code: 0 } };
      if (label === 'verify champion') return { status: 'verified', correctness: 'pass', verified_geomean: 1.3 };
      if (label === 'gluon_engineer:transcribe r1') return engineerReturn(prompt, { anchor_provided: true, patch_file: '', track_tag: 'geak_base' });
      if (label === 'verify r2') return { status: 'verified', correctness: 'pass', verified_geomean: 1.25 };
      if (label === 'verify r3') return { status: 'verified', correctness: 'pass', verified_geomean: 1.26 };
      return undefined;
    });
    ok(x.result.accepted && x.result.accepted.vs_baseline === 1.25, 'GEAK gate accepts 1.25x vs the anchor', x.result.accepted);
    ok(x.result.outcome === 'negative_revert_plain' && x.result.reverted_to_champion, 'slower than the champion: revert', x.result.outcome);
    const close = x.calls.find(c => c.label === 'gluon_lead:close').prompt;
    ok(/FINAL_PATCH_SOURCE: \/x\/champion\.diff/.test(close), 'the champion patch is what gets delivered');
    ok(x.labels.includes('director:validate'), 'the revert is validated by the director too');
  }

  console.log('\n# 2b. a provided anchor is measured: the frozen baseline may be the plain champion');
  {
    // The benchmark froze source_golden/ (the champion) as the baseline; the anchor runs at 0.786 of it.
    const x = await launch({ budget: 3, max_recover_rounds: 2 }, (label, prompt) => {
      if (label === 'gluon_lead:entry') return { entry_mode: 'port', references_read: refsIn(prompt),
        source_is_anchor: true, champion_patch: '/x/champion.diff', track_ready: true,
        champion_gate: { command: 'python3 champion_gate.py --champion b.json', exit_code: 0 } };
      if (label === 'verify champion') return { status: 'verified', correctness: 'pass', verified_geomean: 0.999 };
      if (label === 'gluon_engineer:transcribe r1') return engineerReturn(prompt, { anchor_provided: true, patch_file: '', track_tag: 'geak_base' });
      if (label === 'verify r1 (anchor as is)') return { status: 'regression', correctness: 'pass', verified_geomean: 0.786 };
      if (/^gluon_engineer:recover/.test(label)) return engineerReturn(prompt, { suspect: 'lost_pipeline', layer: '4', att: goodAtt, parity_gate: gate(2) });
      if (label === 'verify r2') return { status: 'regression', correctness: 'pass', verified_geomean: 0.940 };
      if (label === 'verify r3') return { status: 'regression', correctness: 'pass', verified_geomean: 0.936 };
      return undefined;
    });
    const anchorVerify = x.calls.find(c => c.label === 'verify r1 (anchor as is)');
    ok(anchorVerify && /PATCH is intentionally empty/.test(anchorVerify.prompt) && /MEASURE_AS_IS/.test(anchorVerify.prompt),
      'the anchor is verified as it is, not assumed', x.labels);
    const [r1, r2, r3] = x.result.rounds;
    ok(r1.vs_baseline === 0.786 && r1.kept && !r1.accepted, 'the anchor sets the track line at its measured 0.786 and is not a deliverable', r1);
    ok(r2.kept && r2.vs_baseline === 0.94, 'a recovery from 0.786 to 0.940 is kept, not read as a regression', r2);
    const r3prompt = x.calls.find(c => c.label === 'gluon_engineer:recover r3').prompt;
    ok(/TRACK_START: r2/.test(r3prompt) && /TRACK_BEST_VS_BASELINE: 0\.94/.test(r3prompt), 'the next round starts from r2', r3prompt.slice(0, 0));
    ok(!r3.kept, 'a round below the kept line is not kept', r3);
    ok(!x.result.accepted && x.result.outcome === 'negative_revert_plain', 'nothing beat the frozen champion: revert', x.result.outcome);
  }
  {
    // What the engineer actually returns for a provided anchor (run5): no_change, no correctness run.
    const x = await launch({ budget: 2, max_recover_rounds: 1 }, (label, prompt) => {
      if (label === 'gluon_engineer:transcribe r1') return engineerReturn(prompt, { anchor_provided: true, no_change: true,
        correctness: 'not_run', patch_file: '', track_tag: 'geak_base' });
      if (label === 'verify r1 (anchor as is)') return { status: 'regression', correctness: 'pass', verified_geomean: 0.786 };
      return undefined;
    }, { entry: 'port' });
    ok(x.labels.includes('verify r1 (anchor as is)'), 'a no_change anchor is still verified', x.labels);
    ok(x.result.rounds[0].status !== 'anchor_incorrect' && x.result.rounds[0].vs_baseline === 0.786 && x.labels.includes('gluon_engineer:recover r2'),
      'and the run goes on to recover from its measured line', x.result.rounds[0]);
  }

  console.log('\n# 3. parity_gate exit 2 blocks climbing');
  {
    const x = await launch({ budget: 6, max_recover_rounds: 2 }, (label, prompt) => {
      if (label === 'verify r1') return { status: 'regression', correctness: 'pass', verified_geomean: 0.7 };
      if (/^gluon_engineer:recover/.test(label)) return engineerReturn(prompt, { parity_gate: gate(2) });
      if (/^verify r[23]$/.test(label)) return { status: 'regression', correctness: 'pass', verified_geomean: 0.75 };
      return undefined;
    }, { entry: 'port' });
    ok(!x.labels.some(l => /climb/.test(l)), 'no climb round runs', x.labels);
    ok(x.result.stopped_by === 'parity_unreached' && x.result.parity === 'parity_unreached', 'stops at parity_unreached', x.result.stopped_by);
    ok(x.result.caveats.includes('parity_unreached'), 'carried as a caveat');
    ok(x.result.outcome === 'negative_revert_plain' && x.result.rounds_used === 3, 'transcribe + 2 recover rounds, then close', x.result.rounds_used);
    ok(x.calls.some(c => c.label === 'gluon_lead:close' && /OUTCOME: negative_revert_plain/.test(c.prompt)), 'close gets the outcome the code decided');
  }
  {
    const x = await launch({ budget: 4, max_recover_rounds: 1 }, (label, prompt) => {
      if (label === 'verify r1') return { status: 'regression', correctness: 'pass', verified_geomean: 0.7 };
      if (label === 'gluon_engineer:recover r2') return engineerReturn(prompt, { parity_gate: gate(0) });
      if (label === 'verify r2') return { status: 'regression', correctness: 'pass', verified_geomean: 0.8 };
      return undefined;
    }, { entry: 'port' });
    ok(!x.labels.some(l => /climb/.test(l)) && x.result.parity === 'parity_unreached',
      'parity_gate exit 0 contradicted by a verified 0.80x does not clear parity', x.labels);
    ok(x.logs.some(l => /contradicted by verify/.test(l)), 'the contradiction is logged');
  }

  console.log('\n# 4. G1: a failing champion gate stops before any edit');
  {
    const x = await launch({ budget: 4 }, (label, prompt) => {
      if (label === 'gluon_lead:entry') return { entry_mode: 'port', references_read: refsIn(prompt), track_ready: true,
        champion_gate: { command: 'python3 champion_gate.py --champion b.json', exit_code: 2, output_path: '/x/g1.txt' } };
      return undefined;
    });
    ok(!x.labels.some(l => /gluon_engineer|verify|budget/.test(l)), 'no budget, engineer or verify', x.labels);
    ok(/champion_gate exit 2/.test(x.result.blocked) && x.result.stopped_by === 'blocked', 'blocked with the reason', x.result.blocked);
    ok(x.labels.includes('gluon_lead:close') && !x.labels.includes('director:validate'), 'still closes, nothing to validate');
  }
  {
    const x = await launch({ budget: 4 }, (label, prompt) => {
      if (/^gluon_lead:entry/.test(label)) return { entry_mode: 'port', references_read: refsIn(prompt), track_ready: true,
        champion_gate: { exit_code: 0 } };
      return undefined;
    });
    ok(x.labels.includes('gluon_lead:entry (reask 1)') && /entry_incomplete/.test(x.result.blocked),
      'a G1 result without the champion_gate.py command is re-asked, then blocks', x.result.blocked);
  }

  console.log('\n# 5. deadline: no round starts inside the close reserve');
  {
    const x = await launch({ budget: 5, entry_mode: 'incumbent', deadline_epoch: 1000 + 7200, deadline_reserve_s: 1800 }, (label) => {
      if (label === 'clock r1') return { epoch: 1000 };
      if (label === 'clock r2') return { epoch: 1000 + 5500 };
      return undefined;
    }, { entry: 'incumbent' });
    ok(inOrder(x.labels, ['clock r1', 'gluon_engineer:climb r1', 'verify r1', 'clock r2', 'gluon_lead:close']), 'round 1 runs, round 2 does not', x.labels);
    ok(!x.labels.includes('gluon_engineer:climb r2'), 'no engineer after the reserve is reached');
    ok(x.result.stopped_by === 'deadline' && x.result.deadline_hit, 'stopped_by deadline', x.result.stopped_by);
    const eng = x.calls.find(c => c.label === 'gluon_engineer:climb r1').prompt;
    ok(/ROUND_DEADLINE_EPOCH: 4600\b/.test(eng), 'the engineer gets a round deadline (now + min(cap, left - reserve))');
    const y = await launch({ budget: 5, entry_mode: 'incumbent', deadline_epoch: 2000 }, () => undefined, { entry: 'incumbent' });
    ok(y.result.outcome === 'timeout' && y.result.rounds_used === 0, 'no time at all: timeout without a round', y.result);
    const z = await launch({ budget: 2, entry_mode: 'incumbent', deadline_epoch: 99999 }, (label) =>
      label.startsWith('clock') ? null : undefined, { entry: 'incumbent' });
    ok(z.result.rounds_used === 2 && z.result.stopped_by !== 'deadline', 'an unreadable clock never truncates the run', z.result.stopped_by);
  }

  console.log('\n# 6. token budget and round budget');
  {
    let spent = 0;
    const tokens = { total: 1000000, spent: () => spent, remaining: () => 1000000 - spent };
    const x = await launch({ budget: 5, entry_mode: 'incumbent' }, (label) => {
      if (label === 'verify r1') { spent = 900000; }
      return undefined;
    }, { entry: 'incumbent', tokens });
    ok(x.result.stopped_by === 'token_budget' && x.result.rounds_used === 1, 'stops when the token reserve is reached', x.result.stopped_by);
    const y = await launch({ budget: 2, entry_mode: 'incumbent', max_no_improve: 9 }, () => undefined, { entry: 'incumbent' });
    ok(y.result.stopped_by === 'budget' && y.result.rounds_used === 2, 'budget counts engineer rounds', y.result);
    const z = await launch({ budget: 6, entry_mode: 'incumbent' }, (label) =>
      /^verify/.test(label) ? { status: 'regression', correctness: 'pass', verified_geomean: 0.95 } : undefined, { entry: 'incumbent' });
    ok(z.result.stopped_by === 'max_no_improve' && z.result.rounds_used === 2, 'incumbent stalls after 2 non-improving rounds', z.result);
  }
  done('acceptance/stop');
})();
