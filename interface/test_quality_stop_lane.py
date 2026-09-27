# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual lane in a CPU VM with synthetic role responses.

These are control-flow checks, not native-runtime or scientific qualification.
The host signer stays outside the VM. No model or GPU command executes.
"""

import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = r"""
const fs=require('fs'),vm=require('vm'),crypto=require('crypto');
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const {publicKey,privateKey}=crypto.generateKeyPairSync('rsa',{modulusLength:2048});
const key={...publicKey.export({format:'jwk'}),alg:'RS256',ext:true,key_ops:['verify']};
const config={protocol:'geak-fixed-floor-stop-v2',trial_id:'cpu-lane-fixture-20260927',public_key:key,candidate_root:'/fixture/workspace'};
const calls=[],logs=[],prompts=[];
if(input.control) config.stopping_enabled=false;
const deadline=Math.floor(Date.now()/1000)+3600;
const args={kernel_path:'/fixture/task',workflow_dir:input.root+'/kernel_workflow',warm_start:'off',
  budget:6,max_no_improve:input.no_improve?2:100,update_experience:'off',use_learned_kb:false,
  agent_retries:1,agent_timeout_ms:0,deadline_epoch:input.deadline?deadline:0,
  ...(input.off?{}:{quality_stop:config})};
if(input.wrong_source) config.candidate_root='/fixture/other';
function signed(value){const payload=JSON.stringify(value);return {payload,signature:crypto.sign('sha256',Buffer.from(payload,'ascii'),privateKey).toString('base64')};}
async function agent(task,options){
  const label=options.label; calls.push(label); prompts.push({label,task});
  const match=label.match(/(?:r|r=)([0-9]+)/),round=match?Number(match[1]):1;
  const speed=input.no_improve?0.9:1+round*0.1;
  const row={name:'synthetic_case',baseline_ms:1,optimized_ms:1/speed,speedup:speed};
  const profile={bottleneck:'synthetic',device:'synthetic gfx950 fixture',top_opportunities:[]};
  if(label.startsWith('clock ')) return {epoch:input.deadline===label?deadline+1:deadline-3600};
  if(label==='director:setup') return {eval_dir:'/fixture/eval',workspace:'/fixture/workspace',baseline_dir:'/fixture/baseline',kernel_name:'fixture',baseline_frozen:true};
  if(label==='tech_lead:analyze') return {kernel_type:'synthetic',kernel_file:'fixture.py',modifiable_files:['fixture.py']};
  if(label==='benchmark_engineer') return {commandment_path:'/fixture/eval/COMMANDMENT.md',baseline_per_case:[row],baseline_geomean_ms:1};
  if(label==='profile_engineer:baseline'||label.startsWith('reprofile ')) return profile;
  if(label.startsWith('tech_lead:plan ')||label.startsWith('tech_lead:replan ')) return {stop:!!input.planner_stop,
    directions:input.planner_stop?[]:[{id:'r'+round+'_d0',specialty:'algorithm',title:'synthetic direction',focus_files:['fixture.py']}]};
  if(label.startsWith('eng ')) return {status:'success',speedup_geomean:speed,measurement_valid:true,per_case:[row],patch_file:'/fixture/patch.diff'};
  if(label.startsWith('verify ')) return {status:'verified',correctness:'pass',verified_geomean:speed,per_case:[row]};
  if(label.startsWith('commit ')) return {committed:true,current_best_diff:'/fixture/current.diff'};
  if(label.startsWith('tech_lead:memory ')) return {insights:[],ledger:[]};
  if(label.startsWith('storage:reclaim ')) return {ok:true,note:'reclaimed'};
  if(label.startsWith('quality_stop:')){
    if(input.throw_route) throw new Error('synthetic mandatory route refusal');
    if(input.forged) return {certified:true,qualifying:true};
    const request=JSON.parse(task.split('\n').slice(1).join('\n'));
    const final=request.stage==='finalize',consume=request.stage==='consume';
    if(consume&&input.consume_throw) throw new Error('synthetic consume refusal');
    const value={protocol:config.protocol,task,certified:!final&&request.look_index===input.certify_at&&!(consume&&input.consume_reject),
      ...(consume?{consumed:!input.consume_reject}:{}),
      qualifying:final&&!input.final_changed,reason:'synthetic_control_flow_only',snapshot_sha256:'a'.repeat(64),
      look_index:request.look_index,round:request.round,issued_at:Date.now()/1000,deadline_epoch:args.deadline_epoch,
      native_binding:{agent_id:label,root_tool:'tool',root_task:'task',run_id:'run',session_id:'session'}};
    const result=signed(value);
    if(input.tampered) result.payload=result.payload.replace('"certified":false','"certified":true');
    return result;
  }
  if(label==='tech_lead:report') return {final_speedup_geomean:1.6,rounds:6,report_path:'/fixture/report.md',final_patch:'/fixture/final.diff'};
  if(label==='director:validate') return {director_verified_speedup_geomean:1.6,validation_status:'accepted',timing_basis:'synthetic_fixture',final_patch:'/fixture/final.diff'};
  if(label==='kb:write') return {written:true};
  if(label==='kb:cite') return {filed:0};
  throw new Error('unregistered synthetic role '+label);
}
async function evaluate(scriptPath,childArgs){
 const source=fs.readFileSync(scriptPath,'utf8').replace('export const meta =','const meta =');
 class ForbiddenDate {constructor(){throw new Error('Native Workflow forbids Date');} static now(){throw new Error('Native Workflow forbids Date.now');}}
 const context=vm.createContext({args:childArgs,agent,Date:ForbiddenDate,phase:()=>{},log:value=>logs.push(value),
   workflow:async(request,next)=>evaluate(request.scriptPath,next),
   parallel:async(values,fn)=>Promise.all(values.map(fn)),
   pipeline:async(values,...stages)=>Promise.all(values.map(async value=>{for(const stage of stages)value=await stage(value);return value;}))});
 return await new vm.Script('(async()=>{'+source+'\n})()').runInContext(context,{timeout:3000});
}
evaluate(input.root+'/kernel_workflow/kernel_workflow.js',args).then(result=>console.log(JSON.stringify({result,calls,logs,prompts})))
 .catch(error=>{console.error(error.stack);process.exitCode=1;});
"""


def lane(**case):
    result = subprocess.run(["node", "-e", SCRIPT], input=json.dumps({"root": str(ROOT), **case}),
                            text=True, capture_output=True, timeout=15, check=True)
    return json.loads(result.stdout)


class LaneTests(unittest.TestCase):
    def test_runtime_contract_covers_both_arms_and_preserves_local_requests(self):
        contract = '## Quality-stop runtime contract'
        runs = [lane(certify_at=1), lane(certify_at=1, control=True)]
        for value in runs:
            for prompt in value['prompts']:
                if prompt['label'].startswith('quality_stop:'):
                    self.assertNotIn(contract, prompt['task'])
                    prefix, request = prompt['task'].split('\n', 1)
                    self.assertEqual(prefix, 'GEAK_QUALITY_STOP_V2')
                    self.assertIsInstance(json.loads(request), dict)
                else:
                    self.assertEqual(prompt['task'].count(contract), 1)
                    self.assertIn('isolated Bash /tmp', prompt['task'])
                    self.assertIn('Do not create a replacement rocminfo', prompt['task'])
        for label in ('benchmark_engineer', 'profile_engineer:baseline'):
            tasks = [next(row['task'] for row in value['prompts'] if row['label'] == label) for value in runs]
            self.assertEqual(tasks[0], tasks[1])

    def test_runtime_contract_is_absent_outside_quality_stop(self):
        value = lane(off=True)
        self.assertTrue(value['prompts'])
        self.assertFalse(any('## Quality-stop runtime contract' in row['task'] for row in value['prompts']))

    def test_disabled_path_retains_budget_stop_and_no_new_calls(self):
        value = lane(off=True)
        self.assertEqual(value["result"]["stopped_by"], "budget")
        self.assertEqual(value["result"]["budget_used"], 6)
        self.assertNotIn("quality_stop", value["result"])
        self.assertFalse(any(label.startswith("quality_stop:") for label in value["calls"]))

    def test_certificate_exits_with_remaining_search_and_preserves_final_duties(self):
        value = lane(certify_at=1)
        result, calls = value["result"], value["calls"]
        self.assertEqual(result["stopped_by"], "quality_certificate")
        self.assertEqual(result["budget_used"], 1)
        self.assertEqual(result["budget_total"], 6)
        self.assertTrue(result["quality_stop"]["qualifying"])
        boundary = calls.index("quality_stop:boundary r1 l1")
        self.assertLess(calls.index("tech_lead:memory r1"), boundary)
        self.assertLess(calls.index("storage:reclaim r1"), boundary)
        self.assertLess(boundary, calls.index("quality_stop:consume r1 l1"))
        self.assertLess(boundary, calls.index("tech_lead:report"))
        self.assertLess(calls.index("director:validate"), calls.index("quality_stop:finalize r1 l1"))
        if "kb:write" in calls:
            self.assertLess(calls.index("kb:write"), calls.index("quality_stop:finalize r1 l1"))
        self.assertFalse(any(label.startswith("tech_lead:plan r2") for label in calls))

    def test_third_eligible_look_can_certify(self):
        value = lane(certify_at=3)
        self.assertEqual(value["result"]["stopped_by"], "quality_certificate")
        self.assertEqual(value["result"]["budget_used"], 3)
        self.assertEqual(value["result"]["quality_stop"]["eligible_looks"], 3)

    def test_failed_routes_or_forged_results_never_authorize_stopping(self):
        for options in ({}, {"forged": True}, {"tampered": True}, {"throw_route": True}, {"wrong_source": True}):
            with self.subTest(options=options):
                value = lane(**options)
                self.assertEqual(value["result"]["stopped_by"], "budget")
                self.assertEqual(value["result"]["quality_stop"]["eligible_looks"], 3)
                self.assertFalse(value["result"]["quality_stop"]["qualifying"])
                self.assertLessEqual(sum(label.startswith("quality_stop:boundary") for label in value["calls"]), 3)

    def test_final_source_failure_invalidates_exit(self):
        value = lane(certify_at=1, final_changed=True)
        self.assertEqual(value["result"]["stopped_by"], "quality_certificate_invalidated")
        self.assertEqual(value["result"]["budget_used"], 1)
        self.assertFalse(value["result"]["quality_stop"]["qualifying"])

    def test_positive_certificate_requires_a_fresh_signed_consumption(self):
        for change in ({"consume_reject": True}, {"consume_throw": True}):
            with self.subTest(change=change):
                value = lane(certify_at=1, **change)
                self.assertEqual(value["result"]["stopped_by"], "budget")
                self.assertFalse(value["result"]["quality_stop"]["qualifying"])
                self.assertEqual(value["calls"].count("quality_stop:consume r1 l1"), 1)
                self.assertFalse(any(label.startswith("quality_stop:finalize") for label in value["calls"]))

    def test_native_no_improvement_rule_retains_priority(self):
        value = lane(no_improve=True)
        self.assertEqual(value["result"]["stopped_by"], "no_improve")
        self.assertEqual(value["result"]["budget_used"], 2)
        self.assertEqual(value["result"]["quality_stop"]["eligible_looks"], 1)

    def test_native_deadline_retains_priority_before_and_after_measurement(self):
        for label, looks in (("clock pre-r1", 0), ("clock quality-r1", 0), ("clock post-quality-r1", 1)):
            with self.subTest(label=label):
                value = lane(certify_at=1, deadline=label)
                self.assertEqual(value["result"]["stopped_by"], "deadline")
                self.assertEqual(value["result"]["quality_stop"]["eligible_looks"], looks)
                self.assertFalse(value["result"]["quality_stop"]["qualifying"])

    def test_native_forced_replans_remain_six(self):
        value = lane(planner_stop=True, deadline=True)
        self.assertEqual(value["result"]["forced_replans"], 6)
        self.assertEqual(value["result"]["budget_used"], 0)
        self.assertEqual(value["result"]["stopped_by"], "tech_lead_stop")
        self.assertEqual(value["result"]["quality_stop"]["eligible_looks"], 0)


if __name__ == "__main__":
    unittest.main()
