#!/usr/bin/env node
// Execute the actual workflow body with explicitly synthetic role responses.
// This checks control flow and data handoff; no agents, GPU work or git commits run.
'use strict';
const assert = require('assert');
const {run} = require('./test_gluon_skill_routing');
const marker = '## Triton/Gluon method (automatic language reference)';
const direction = {id:'smoke', title:'Synthetic candidate', specialty:'compute', prompt:'offline smoke'};

function response(accept = true) {
  return (label) => {
    if (label === 'author:gluon') return {authored:true, correctness:'pass'};
    if (/^eng /.test(label)) return {status:'success', measurement_valid:true, speedup_geomean:1.2};
    if (/^verify /.test(label)) return {status:'verified', correctness:accept ? 'pass' : 'fail', verified_geomean:1.2};
    if (/^commit /.test(label)) return {committed:true};
    if (/^reprofile /.test(label)) return {summary_path:'/tmp/reprofile.json'};
    if (label === 'tech_lead:report') return {final_speedup_geomean:1.2, report_path:'/tmp/report.md', final_patch:'/tmp/final.diff'};
    if (label === 'director:validate') return {validation_status:accept ? 'accepted' : 'rejected',
      director_verified_speedup_geomean:accept ? 1.2 : 1, timing_basis:'synthetic-fixture'};
  };
}

(async () => {
  const rows = [];
  for (const mode of ['optimize', 'author']) {
    for (const useGluon of ['auto', 'false']) {
      const x = await run('gfx950', '', {mode, language:'gluon', useGluon,
        budget:1, planDirections:[direction], agentResponse:response()});
      const labels = x.prompts.map(p => p.label);
      const sequence = ['director:setup', ...(mode === 'author' ? ['author:gluon'] : []),
        'tech_lead:analyze', 'benchmark_engineer', 'profile_engineer:baseline',
        'eng smoke:compute', 'verify smoke', 'commit r1', 'reprofile r1', 'tech_lead:report', 'director:validate'];
      let last = -1;
      for (const label of sequence) {
        const index = labels.indexOf(label);
        assert(index > last, `${mode}/${useGluon}: missing or out-of-order ${label}`);
        last = index;
      }
      for (const label of ['tech_lead:analyze', 'profile_engineer:baseline', 'eng smoke:compute', 'reprofile r1']) {
        assert.strictEqual(x.prompts.find(p => p.label === label).prompt.includes(marker), useGluon !== 'false');
      }
      for (const label of ['benchmark_engineer', 'verify smoke', 'director:validate']) {
        assert(!x.prompts.find(p => p.label === label).prompt.includes(marker), `method unexpectedly controls ${label}`);
      }
      assert.strictEqual(x.result.validation_status, 'accepted');
      assert.strictEqual(x.result.timing_basis, 'synthetic-fixture');
      rows.push({mode, useGluon, sequence, result:x.result.validation_status});
    }
  }
  const badAuthor = await run('gfx950', '', {mode:'author', language:'gluon',
    agentResponses:{'author:gluon':{authored:true, correctness:'fail'}}});
  assert.strictEqual(badAuthor.result.validation_status, 'author_failed');
  assert(!badAuthor.prompts.some(p => p.label === 'benchmark_engineer'));
  const rejected = await run('gfx950', '', {language:'gluon', budget:1, planDirections:[direction], agentResponse:response(false)});
  assert(!rejected.prompts.some(p => /^commit |^reprofile /.test(p.label)));
  const incumbent = await run('gfx950', '', {sourceLanguage:'gluon', useGluon:'auto'});
  assert(incumbent.prompts.find(p => p.label === 'profile_engineer:baseline').prompt.includes('skips transcription/parity'));
  await assert.rejects(run('gfx950', '', {useGluon:'typo'}), /use_gluon_skill/);
  console.log(JSON.stringify({status:'PASS', synthetic_role_responses:true, flows:rows,
    negative_cases:['author_failed','verification_failed','invalid_switch'],
    incumbent:'skip transcription/parity'}, null, 2));
})().catch(error => {console.error(error); process.exit(1);});
