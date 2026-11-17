#!/usr/bin/env node
// Evidence is a contract checked in code: each stage's required references open its prompt, and a
// round that is missing G3 / G4 / ATT (pipeline layers) / its gluon_profile.py re-profile or fails the
// artifact audit is sent back by name once, then rejected without a verify. The baseline profile is
// the pack's gluon_profile.py with every layer's state stated.
'use strict';

const { launch, ok, done, engineerReturn, refsIn, goodAtt, goodProfileCmd, SKILL } = require('./harness');

const climbOnly = { budget: 1, entry_mode: 'incumbent' };

(async () => {
  console.log('\n# 1. each stage prompt opens with its required references, before the role file');
  {
    const port = await launch({ budget: 3 }, (label) => {
      if (label === 'verify r1') return { status: 'regression', correctness: 'pass', verified_geomean: 0.9 };
      if (label === 'verify r2') return { status: 'regression', correctness: 'pass', verified_geomean: 0.97 };
      return undefined;
    }, { entry: 'port' });
    const prompt = (re) => (port.calls.find(c => re.test(c.label)) || {}).prompt || '';
    const expect = {
      'gluon_lead:entry': ['skill.md', 'references/method/entry.md'],
      'gluon_lead:budget': ['references/method/budget.md'],
      'gluon_profiler:baseline': ['references/method/profile.md'],
      'gluon_engineer:transcribe r1': ['references/method/transcribe.md'],
      'gluon_engineer:recover r2': ['references/method/recover.md', 'references/tile-programming/pipeline.md'],
      'gluon_engineer:climb r3': ['references/method/climb.md', 'references/method/profile.md'],
      'gluon_lead:close': ['references/method/close.md'],
      'evidence_audit r1': ['references/method/index.md'],
    };
    for (const [label, rels] of Object.entries(expect)) {
      const p = prompt(new RegExp(`^${label.replace(/[.+]/g, '\\$&')}$`));
      const roleAt = p.indexOf('Then Read ');
      for (const rel of rels) {
        const at = p.indexOf(`${SKILL}/${rel}`);
        ok(at > 0 && at < roleAt && at < 1500, `${label}: ${rel} listed before the role file, near the top`, at);
      }
    }
    const climb = prompt(/^gluon_engineer:climb r3$/);
    ok(/- layer 4: \S+pipeline\.md/.test(climb), 'climb lists each layer\'s references (layer 4 -> pipeline.md)');
    for (const label of ['director:setup', 'benchmark_engineer', 'verify r1']) {
      ok(!prompt(new RegExp(`^${label}$`)).includes('Required reading'), `${label} (GEAK role) gets no method reading list`);
    }
  }

  console.log('\n# 2. a climb round missing G3 is re-asked; the re-ask that supplies it is verified');
  {
    const x = await launch(climbOnly, (label, prompt) => {
      if (label === 'gluon_engineer:climb r1') { const r = engineerReturn(prompt); delete r.g3_occupancy; return r; }
      if (/^gluon_engineer:evidence r1/.test(label)) {
        ok(/g3_occupancy/.test(prompt) && /MISSING/.test(prompt), 're-ask names the missing gate');
        return engineerReturn(prompt.replace('PHASE=climb', 'PHASE=climb'), { stage: 'climb' });
      }
      return undefined;
    }, { entry: 'incumbent' });
    ok(x.labels.includes('gluon_engineer:evidence r1 (reask 1)') && x.labels.includes('verify r1'), 'reask then verify', x.labels);
    ok(x.result.rounds[0].status === 'verified', 'round counted normally', x.result.rounds[0]);
  }

  console.log('\n# 3. still missing after the re-ask -> rejected: no verify, track not moved');
  {
    const x = await launch(climbOnly, (label, prompt) => {
      if (/^gluon_engineer:/.test(label)) { const r = engineerReturn(prompt, { stage: 'climb' }); delete r.g4_screen; return r; }
      return undefined;
    }, { entry: 'incumbent' });
    ok(!x.labels.includes('verify r1'), 'no verify for a rejected round', x.labels);
    const r1 = x.result.rounds[0];
    ok(r1.status === 'rejected_missing_evidence' && r1.missing.some(m => /g4_screen/.test(m)), 'rejection names G4', r1);
    ok(x.result.outcome === 'negative_keep_baseline' && !x.result.final_patch, 'nothing delivered', x.result.outcome);
    ok(!x.labels.includes('director:validate'), 'no validation without a candidate');
  }

  console.log('\n# 4. G4 must be the permuted screen');
  {
    const x = await launch(climbOnly, (label, prompt) => {
      if (/^gluon_engineer:/.test(label)) {
        const r = engineerReturn(prompt, { stage: 'climb' });
        r.g4_screen = { ...r.g4_screen, command: 'python3 ab_bench.py --module a.py --json x.json' };
        return r;
      }
      return undefined;
    }, { entry: 'incumbent' });
    ok(x.result.rounds[0].status === 'rejected_missing_evidence', 'ab_bench without --permute is rejected', x.result.rounds[0]);
  }

  console.log('\n# 5. G3 must carry both limiters (or say why one is missing)');
  {
    const x = await launch(climbOnly, (label, prompt) => {
      if (/^gluon_engineer:/.test(label)) {
        const r = engineerReturn(prompt, { stage: 'climb' });
        r.g3_occupancy = { ...r.g3_occupancy }; delete r.g3_occupancy.lds_wg_per_cu;
        return r;
      }
      return undefined;
    }, { entry: 'incumbent' });
    ok(x.result.rounds[0].missing.some(m => /lds_wg_per_cu/.test(m)), 'LDS limiter missing is named', x.result.rounds[0]);
    const y = await launch(climbOnly, (label, prompt) => {
      if (/^gluon_engineer:/.test(label)) {
        const r = engineerReturn(prompt, { stage: 'climb' });
        r.g3_occupancy = { ...r.g3_occupancy, lds_wg_per_cu: null, missing_reason: 'no shared bytes in the metadata' };
        return r;
      }
      return undefined;
    }, { entry: 'incumbent' });
    ok(y.result.rounds[0].status === 'verified', 'null with a missing_reason is accepted', y.result.rounds[0]);
  }

  console.log('\n# 6. a pipeline/scheduling layer owes ATT evidence (captured, or unavailable with a reason)');
  {
    const x = await launch(climbOnly, (label, prompt) => {
      if (label === 'gluon_engineer:climb r1') return engineerReturn(prompt, { layer: '4' });
      if (/^gluon_engineer:evidence r1/.test(label)) {
        return engineerReturn(prompt, { stage: 'climb', layer: '4',
          att: { status: 'unavailable', tool: 'capture.sh', command: 'bash capture.sh ...', exit_code: 0,
            reason: 'capture.json att_state=decoder_absent: ROCPROF_ATT_LIBRARY_PATH unset in this container' } });
      }
      return undefined;
    }, { entry: 'incumbent' });
    ok(x.labels.includes('gluon_engineer:evidence r1 (reask 1)'), 'layer 4 without ATT is re-asked', x.labels);
    ok(x.result.rounds[0].status === 'verified', 'a reasoned "unavailable" is accepted', x.result.rounds[0]);
    const y = await launch(climbOnly, (label, prompt) => {
      if (/^gluon_engineer:/.test(label)) return engineerReturn(prompt, { stage: 'climb', layer: '1.5',
        att: { status: 'unavailable', reason: 'n/a' } });
      return undefined;
    }, { entry: 'incumbent' });
    ok(y.result.rounds[0].status === 'rejected_missing_evidence', 'an unreasoned "unavailable" is rejected', y.result.rounds[0]);
    const z = await launch(climbOnly, (label, prompt) => {
      if (/^gluon_engineer:/.test(label)) return engineerReturn(prompt, { stage: 'climb', layer: '4',
        att: { ...goodAtt, command: 'bash capture.sh r /x --kernel k --source k.py --no-att -- python run.py' } });
      return undefined;
    }, { entry: 'incumbent' });
    ok(z.result.rounds[0].status === 'rejected_missing_evidence', 'capture.sh --no-att is not ATT evidence', z.result.rounds[0]);
    const w = await launch(climbOnly, (label, prompt) => {
      if (/^gluon_engineer:/.test(label)) return engineerReturn(prompt, { stage: 'climb', layer: '5' });
      return undefined;
    }, { entry: 'incumbent' });
    ok(w.result.rounds[0].status === 'verified', 'layer 5 does not owe ATT', w.result.rounds[0]);
  }

  console.log('\n# 7. a layer\'s own references are required for that layer');
  {
    const x = await launch(climbOnly, (label, prompt) => {
      if (/^gluon_engineer:/.test(label)) return engineerReturn(prompt, { stage: 'climb', layer: '5',
        references_read: refsIn(prompt) });
      return undefined;
    }, { entry: 'incumbent' });
    ok(x.result.rounds[0].missing.some(m => /slicing\.md/.test(m)), 'layer 5 without slicing.md is rejected', x.result.rounds[0]);
  }

  console.log('\n# 8. an audit finding is sent back like a missing field');
  {
    let audits = 0;
    const x = await launch(climbOnly, (label) => {
      if (/^evidence_audit/.test(label)) return ++audits === 1
        ? { ok: false, findings: ['g3 re-run: binding lds, evidence says registers'] } : { ok: true, findings: [] };
      return undefined;
    }, { entry: 'incumbent' });
    ok(audits === 2 && x.labels.includes('gluon_engineer:evidence r1 (reask 1)'), 'audit -> reask -> re-audit', x.labels);
    const reask = x.calls.find(c => c.label === 'gluon_engineer:evidence r1 (reask 1)').prompt;
    ok(/audit: g3 re-run/.test(reask), 'the finding is quoted in the re-ask');
    ok(x.result.rounds[0].status === 'verified', 'verified after the clean audit');
    const y = await launch(climbOnly, (label) =>
      /^evidence_audit/.test(label) ? { ok: false, findings: [] } : undefined, { entry: 'incumbent' });
    ok(y.result.rounds[0].status === 'rejected_missing_evidence' && !y.labels.includes('verify r1'),
      'a failed audit that names nothing still rejects the round', y.result.rounds[0]);
  }

  console.log('\n# 9. incumbent: parity_gate may only be a diagnostic');
  {
    const x = await launch(climbOnly, (label, prompt) => {
      if (/^gluon_engineer:/.test(label)) return engineerReturn(prompt, { stage: 'climb',
        parity_gate: { command: 'python3 parity_gate.py --champion-ms 1 --anchor-ms 1', exit_code: 0, mode: 'gate' } });
      return undefined;
    }, { entry: 'incumbent' });
    ok(x.result.rounds[0].status === 'rejected_missing_evidence', 'a vacuous parity gate on an incumbent is rejected', x.result.rounds[0]);
  }
  console.log('\n# 10. a pipeline layer\'s ATT must be a usable collection');
  {
    const partial = await launch(climbOnly, (label, prompt) => {
      if (/^gluon_engineer:/.test(label)) return engineerReturn(prompt, { stage: 'climb', layer: '4',
        att: { ...goodAtt, state: 'partial' } });
      return undefined;
    }, { entry: 'incumbent' });
    ok(partial.result.rounds[0].status === 'rejected_missing_evidence' &&
      partial.result.rounds[0].missing.some(m => /att\.state/.test(m)), 'a partial ATT without a reason is rejected', partial.result.rounds[0]);
    const reasoned = await launch(climbOnly, (label, prompt) => {
      if (/^gluon_engineer:/.test(label)) return engineerReturn(prompt, { stage: 'climb', layer: '4',
        att: { ...goodAtt, state: 'partial', reason: 'kernel_identity_unverified: two dispatches share the name' } });
      return undefined;
    }, { entry: 'incumbent' });
    ok(reasoned.result.rounds[0].status === 'verified', 'a partial ATT with its reason is accepted', reasoned.result.rounds[0]);
    const viaProfile = await launch(climbOnly, (label, prompt) => {
      if (/^gluon_engineer:/.test(label)) return engineerReturn(prompt, { stage: 'climb', layer: '4',
        att: { status: 'captured', tool: 'gluon_profile.py', command: goodProfileCmd, exit_code: 0,
          output_dir: '/x/prof/capture/att', state: 'ok' } });
      return undefined;
    }, { entry: 'incumbent' });
    ok(viaProfile.result.rounds[0].status === 'verified', 'the round\'s gluon_profile.py ATT layer is ATT evidence', viaProfile.result.rounds[0]);
    const noAttLayer = await launch(climbOnly, (label, prompt) => {
      if (/^gluon_engineer:/.test(label)) return engineerReturn(prompt, { stage: 'climb', layer: '4',
        att: { status: 'captured', tool: 'gluon_profile.py', command: goodProfileCmd.replace('--gpu 6', '--gpu 6 --layers trace,pmc'),
          exit_code: 0, output_dir: '/x/prof', state: 'ok' } });
      return undefined;
    }, { entry: 'incumbent' });
    ok(noAttLayer.result.rounds[0].status === 'rejected_missing_evidence', 'gluon_profile.py without its att layer is not ATT evidence', noAttLayer.result.rounds[0]);
  }

  console.log('\n# 11. a climb round re-profiles with gluon_profile.py');
  {
    const x = await launch(climbOnly, (label, prompt) => {
      if (/^gluon_engineer:/.test(label)) return engineerReturn(prompt, { stage: 'climb',
        profile_command: 'bash kernel_workflow/scripts/profile_kernel.sh 6 "python run.py" /x/prof' });
      return undefined;
    }, { entry: 'incumbent' });
    ok(x.result.rounds[0].missing.some(m => /profile_command/.test(m)), 'a profile_kernel.sh re-profile is sent back', x.result.rounds[0]);
    const prompt = x.calls.find(c => c.label === 'gluon_engineer:climb r1').prompt;
    ok(/LAST_PROFILE: \/tmp\/eval\/gluon\/profile_baseline/.test(prompt), 'the first climb compares against the baseline profile dir');
  }

  console.log('\n# 12. the baseline profile states every layer; a degraded one needs its reason');
  {
    const plain = await launch(climbOnly, (label) => label.startsWith('gluon_profiler:')
      ? { bottleneck: 'latency', command: 'bash profile_kernel.sh 6 "python run.py" /x', exit_code: 0, profile_dir: '/x', layers: {}, references_read: [] }
      : undefined, { entry: 'incumbent' });
    ok(plain.labels.includes('gluon_profiler:baseline (reask 1)'), 'an incomplete baseline profile is re-asked', plain.labels);
    ok(plain.result.caveats.some(c => /baseline_profile_incomplete: .*gluon_profile\.py.*layers\./.test(c)),
      'what stayed missing is a caveat', plain.result.caveats);
    const degraded = await launch(climbOnly, (label, prompt) => label.startsWith('gluon_profiler:')
      ? { bottleneck: 'latency', command: 'python3 s/gluon_profile.py /x --kernel k --source k.py -- python run.py', exit_code: 5,
          profile_dir: '/x', references_read: refsIn(prompt), layers: { trace: 'ok', pmc: 'partial', att: 'ok', ir: 'ok' },
          degraded_reason: 'pmc: SQ_WAIT_ANY aborts alone on this box (dropped by bisection)' }
      : undefined, { entry: 'incumbent' });
    ok(!degraded.labels.includes('gluon_profiler:baseline (reask 1)'), 'a reasoned degraded layer is accepted', degraded.labels);
    ok(degraded.result.caveats.some(c => /baseline_profile_degraded: pmc/.test(c)), 'and recorded as a caveat', degraded.result.caveats);
  }
  done('evidence-gate');
})();
