'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const src = fs.readFileSync(path.join(__dirname, '../e2e_workflow.js'), 'utf8');
const helper = src.slice(src.indexOf('function tuningAccepted('), src.indexOf('let tuning = ST.tuning'));
const accepted = new Function('ACCURACY_GATE', helper + '\nreturn tuningAccepted;')('none');
const valid = {gate:'accepted', engagement_verified:true, ab_complete:true, correctness_gate:'none',
  pre_tune_throughput_tok_s:1000, post_tune_throughput_tok_s:1033.48, tuning_delta_pct:3.348};
assert(accepted(valid));
assert(!new Function('ACCURACY_GATE', helper + '\nreturn tuningAccepted;')('gsm8k')(valid));
const bad=[];
for (const field of ['ab_complete','engagement_verified','correctness_gate','pre_tune_throughput_tok_s','post_tune_throughput_tok_s']) {
  const value={...valid}; delete value[field]; bad.push(value);
}
for (const value of [false,1,'true',null]) bad.push({...valid,ab_complete:value});
for (const field of ['pre_tune_throughput_tok_s','post_tune_throughput_tok_s'])
  for (const value of [0,-1,NaN,Infinity,-Infinity,'1000',true]) bad.push({...valid,[field]:value});
for (const value of ['fail','unknown','',null]) bad.push({...valid,correctness_gate:value});
const reportCode=src.slice(src.indexOf('function tuningReturn()'),src.indexOf('\nconst wfReturn ='));
const report = new Function('tuning','tuningAccepted', 'ACCURACY_GATE', `
 const TUNING_SKILLSET_ENABLED=true,TUNING_SKILLSET_DIR='/skills',TUNING_KB_ENABLED=false;
 const validatedOk=false,validation=null,finalTput=1033.48,BASELINE_TPUT=1000,finalize=null,EVAL_DIR='/eval',want=()=>true;
 ${reportCode};return tuningReturn();`);
for (const value of bad) {
  assert.strictEqual(accepted(value),false);
  const result=report(value,accepted,'none');assert.notStrictEqual(result.gate,'accepted');assert.strictEqual(result.share_of_total_gain_pct,null);
}
const missing={...valid};delete missing.ab_complete;assert.strictEqual(report(missing,accepted,'none').ab_complete,false);
const good=report(valid,accepted,'none');assert.strictEqual(good.gate,'accepted');assert.strictEqual(good.tuning_delta_pct,3.348);
const resume=src.slice(src.indexOf('let tuning = ST.tuning'),src.indexOf('if (FAST_MODE) log',src.indexOf('let tuning = ST.tuning')));
const resumeFn=new Function('ST','tuningAccepted',resume+'\nreturn tuning;');
assert.strictEqual(resumeFn({tuning:valid},accepted),valid);
for (const value of bad) assert.throws(()=>resumeFn({tuning:value},accepted),/Carried accepted tuning/);
assert(src.includes('const TUNING_FINALIZE_INPUTS = (TUNING_SKILLSET_ENABLED && tuningAccepted(tuning))'));
assert(src.includes("if (!(TUNING_SKILLSET_ENABLED && tuningAccepted(tuning))) return {};"));
console.log(`PASS ${bad.length} malformed pairs rejected by banking/report/resume; explicit none +3.348% retained`);
