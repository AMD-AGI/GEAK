"""CPU-only split-stage integration. Synthetic receipts here are not GPU evidence."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

WORKLOAD = Path(__file__).resolve().parents[3]
PROFILE = WORKLOAD / "stages/profile"
UT = WORKLOAD / "stages/ut"
sys.path.insert(0, str(PROFILE))
import handoff as H
sys.path.insert(0, str(UT))
from prepare import prepare, verify_input

KERNEL = "_rms_norm_kernel"
PARAMS = {"output_kind": "dispatch_evidence", "selection_top_k": 10,
          "selection_overall_pct_gt": 3, "selection_phases": ["PREFILL", "DECODE"]}


def profile_fixture(root):
    H.write(root / "serving.json", {"model": "unit-test", "tp": 1, "gpus": "0"})
    ranking = {"denominators": {"total_device_us": 1000},
               "ranking": [{"id": KERNEL, "name": KERNEL, "selection_pct_gpu_time": 20,
                            "share_of_phase_pct": {"PREFILL": 20}, "kind": "compute"}],
               "selection_scope": {"basis": "test fixture"},
               "candidate_inventory": {"all": [KERNEL]}, "coverage_summary": {"basis": "measured points"}}
    H.write(root / "kernel_ranking.json", ranking)
    H.write(root / "selection_targets.json", {"policy": {"top_k_per_phase": 10, "overall_pct_gt": 3,
            "phases": ["PREFILL", "DECODE"]}, "targets": [{"id": KERNEL, "name": KERNEL, "overall_pct": 20}]})
    H.write(root / "evidence/source_map.json", {"kernels": {KERNEL: {"op": {"key": "fixture:norm@Layer",
            "entry": {"target_callable": "fixture:norm", "resolved": True}}}}})
    H.write(root / "evidence/adjacency.json", {"edges": []})
    H.write(root / "evidence/observed_workload.json", {"raw_prefill_rows": [
        {"new_token": 32, "new_seq": 1, "cached_token": 0}], "raw_decode_rows": []})
    points = []
    for n in (32, 64):
        pid = "p" + str(n)
        census = root / "evidence/census" / (pid + ".json")
        H.write(census, {"phase": "PREFILL", "achieved_shape": n, "instances": 1,
                         "per_kernel": {KERNEL: 20}})
        H.write(root / "evidence/shapes" / (pid + ".json"), {"phase": "PREFILL", "achieved_shape": n,
            "kernels": {KERNEL: {"launches_per_instance": 1, "variants": []}}})
        points.append({"point_id": pid, "phase": "PREFILL", "achieved_shape": n, "new_tokens": n,
                       "achieved_context": n, "census": str(census), "counted": True,
                       "represented": {"instances": 5}})
    H.write(root / "sampling_points_final.json", {"points": points, "shape_tolerance": 1.25,
            "phase_summary": {"PREFILL": {"total_instances": 10, "unrepresented_bands": []}}})


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="profile_ut_split_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.profile, self.ut = self.root / "profile", self.root / "ut"
        profile_fixture(self.profile)
        self.params = self.root / "params.json"
        H.write(self.params, PARAMS)
        self.handoff = self.profile / "profile_handoff.json"

    def export(self):
        doc = H.build(self.profile, PARAMS)
        H.write(self.handoff, doc)
        return doc

    def finalize(self, stage, out, params=None):
        env = dict(os.environ, OUT=str(out), STAGE=stage, STAGE_SKILL="kernel-" + stage,
                   PARAMS=str(params or self.params), WORKLOAD_ROOT=str(WORKLOAD))
        p = subprocess.run([sys.executable, "-B", str(WORKLOAD / "stages" / stage / "finalize.py")],
                           env=env, capture_output=True, text=True)
        self.assertIn(p.returncode, (0, 1), p.stdout + p.stderr)
        return H.read(out / "result.json")

    def test_profile_can_finish_without_any_ut(self):
        result = self.finalize("profile", self.profile)
        self.assertEqual(result["status"], "ok", result)
        self.assertFalse((self.profile / "tasks").exists())
        self.assertIs(result["metrics"]["ut_generation_started"], False)
        self.assertEqual(len(H.verify(self.handoff)["points"]), 2)

    def test_old_combined_contract_still_requires_uts(self):
        H.write(self.params, {k: v for k, v in PARAMS.items() if k != "output_kind"})
        result = self.finalize("profile", self.profile)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("no achieved coverage", result["notes"])

    def test_changed_dispatch_changes_upstream_receipt_for_resume(self):
        first=self.finalize('profile',self.profile)
        source=H.read(self.profile/'evidence/source_map.json')
        source['kernels'][KERNEL]['op']['entry']['target_callable']='fixture:other_norm'
        H.write(self.profile/'evidence/source_map.json',source)
        second=self.finalize('profile',self.profile)
        self.assertEqual(first['metrics']['points_measured'],second['metrics']['points_measured'])
        self.assertNotEqual(first['metrics']['handoff_sha256'],second['metrics']['handoff_sha256'])

    def test_preparation_reuses_evidence_without_old_tasks_or_source_mutation(self):
        self.export()
        old_task = self.profile / "tasks/old/unittest.py"
        old_task.parent.mkdir(parents=True);old_task.write_text("raise RuntimeError('do not copy')\n")
        digest = H.sha(self.profile / "kernel_ranking.json")
        prepare(self.handoff, self.ut)
        prepare(self.handoff, self.ut)
        self.assertTrue((self.ut / "evidence").is_symlink())
        self.assertFalse((self.ut / "tasks").exists())
        self.assertEqual(H.sha(self.profile / "kernel_ranking.json"), digest)
        self.assertEqual(verify_input(self.ut)["selection_params"]["selection_overall_pct_gt"], 3)

    def test_changed_profile_or_local_descriptor_invalidates_ut_input(self):
        self.export();prepare(self.handoff, self.ut)
        (self.ut / "selection_targets.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "local profile evidence"):
            verify_input(self.ut)
        (self.ut / "selection_targets.json").write_bytes((self.profile / "selection_targets.json").read_bytes())
        (self.profile / "evidence/shapes/p32.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            verify_input(self.ut)

    def test_changed_handoff_cannot_replace_an_existing_ut_run(self):
        self.export();prepare(self.handoff, self.ut)
        doc=H.read(self.handoff);doc["note"]="new invocation";H.write(self.handoff,doc)
        with self.assertRaisesRegex(ValueError, "different profile evidence"):
            prepare(self.handoff,self.ut)

    def test_counted_points_require_matching_shape_and_census(self):
        (self.profile / "evidence/shapes/p32.json").unlink()
        with self.assertRaisesRegex(ValueError, "no shape/dispatch"):
            self.export()
        profile_fixture(self.profile)
        H.write(self.profile / "evidence/census/p32.json", {"phase":"PREFILL","achieved_shape":99})
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.export()

    def test_scope_mismatch_is_not_fixed_by_splitting(self):
        bad = dict(PARAMS, selection_overall_pct_gt=1)
        with self.assertRaisesRegex(ValueError, "configured"):
            H.build(self.profile, bad)

    def test_edited_summary_cannot_change_verified_shapes(self):
        doc=self.export();doc['points'][0]['shape']=999;H.write(self.handoff,doc)
        with self.assertRaisesRegex(ValueError,'handoff points differs'):
            H.verify(self.handoff)

    def test_omitting_a_shape_hash_cannot_narrow_the_manifest(self):
        doc=self.export();doc['files'].pop('evidence/shapes/p32.json');H.write(self.handoff,doc)
        with self.assertRaisesRegex(ValueError,'handoff files differs'):
            H.verify(self.handoff)

    def test_unresolved_callable_is_an_explicit_gap(self):
        H.write(self.profile / "evidence/source_map.json", {"kernels": {}})
        doc=self.export()
        self.assertEqual(doc["kernels"][0]["status"], "needs_review")
        self.assertTrue(doc["kernels"][0]["gaps"])
        self.assertIsNone(doc["coverage"]["ut_coverage"])

    def test_ut_still_requires_full_task_and_delivery_receipts(self):
        self.export();prepare(self.handoff,self.ut)
        task=self.ut/'tasks/norm';(task/'kernel_src').mkdir(parents=True)
        for name in ('unittest.py','cases.py','meta.json'):(task/name).write_text('{}')
        H.write(self.ut/'task_plan.json',{'tasks':[{'label':'norm','selected_identities':[KERNEL]}]})
        self.assertEqual(self.finalize('ut',self.ut)['status'],'blocked')
        H.write(task/'run/ut_result.json',{'ok':True,'failed_gates':[]})
        self.assertEqual(self.finalize('ut',self.ut)['status'],'blocked')
        # Synthetic receipt fixture for the slot adapter, not a real GPU pass.
        H.write(task/'run/RETURN.json',{'short_name':'norm'})
        result=self.finalize('ut',self.ut)
        self.assertEqual(result['status'],'ok',result)
        self.assertFalse(result['metrics']['serving_coverage_known'])

    def test_ut_workspace_must_not_be_inside_profile(self):
        self.export()
        with self.assertRaisesRegex(ValueError,'separate'):
            prepare(self.handoff,self.profile/'ut')

    def test_all_blocked_identities_are_not_a_ut_delivery(self):
        self.export();prepare(self.handoff,self.ut)
        H.write(self.ut/'task_plan.json',{'tasks':[],'blocked_identities':[
            {'kernel':KERNEL,'blocker':'native source unavailable'}]})
        result=self.finalize('ut',self.ut)
        self.assertEqual(result['status'],'blocked')
        self.assertIn('No standalone UT tasks',result['notes'])

    def test_profile_only_collects_evidence_without_drafting_uts(self):
        scripts=PROFILE/'kernel-extract/scripts';sys.path.insert(0,str(scripts))
        import run_points
        points=H.read(self.profile/'sampling_points_final.json')['points']
        censuses={p['point_id']:p['census'] for p in points}
        for path in censuses.values():
            doc=H.read(path);doc['trace_dir']=str(self.profile/'traces');H.write(path,doc)
        args=SimpleNamespace(select_by_kernel=True,select_arg=[],profile_only=True,
                             all_evidence=False,draft_shares_name=False)
        seen=[]
        def steps(rows):
            seen.extend(name for name,_ in rows)
            return True
        with patch.object(run_points,'run',return_value=(0,0,'fixture')):
            ok=run_points.evidence(args,str(self.profile),censuses,points,
                str(self.profile/'selection_targets.json'),sys.executable,str(self.root/'log'),
                {'post':{}},steps)
        self.assertTrue(ok)
        self.assertIn('source_map',seen)
        self.assertNotIn('draft_task_decls',seen)

    def test_shared_ut_entrypoint_and_templates_resolve(self):
        p=subprocess.run([sys.executable,str(UT/'kernel-ut/scripts/ut_wave.py'),'--help'],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertTrue((UT/'kernel-ut/ut_contracts/templates/unittest.py.tmpl').is_file())

    def test_new_skill_scaffolds_the_existing_fixed_contract_and_driver(self):
        manifest=self.root/'manifest.json'
        H.write(manifest,{'cases':[{'sig':'PREFILL_m32','regime':'PREFILL','axes':{'m':32},
                                   'effective_calls':1,'timed':True,'random':True}]})
        task=self.root/'scaffold'
        p=subprocess.run([sys.executable,str(UT/'kernel-ut/ut_contracts/scaffold_ut.py'),'new',
            '--family','gemm','--task',str(task),'--target','fixture:gemm',
            '--bind-module','fixture','--bind-file','kernel_src/kernel.py',
            '--axis','n=32','--axis','k=64','--manifest',str(manifest)],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        for name in ['unittest.py','cases.py','meta.json','run/ut_cases.json','run/ut_contract.json']:
            self.assertTrue((task/name).is_file(),name)
        self.assertEqual(H.sha(task/'ut_driver.py'),H.sha(UT/'kernel-ut/ut_contracts/ut_driver.py'))
        self.assertEqual(H.read(task/'run/ut_contract.json')['family'],'gemm')

    def test_stage_order_and_task_source_are_configurable(self):
        scripts=WORKLOAD/'scripts';sys.path.insert(0,str(scripts))
        import stage
        cfg=H.read(WORKLOAD/'config/default.json')
        cfg['order']=['baseline','profile','ut','roofline','optimize','applyback']
        cfg['task_source_stage']='ut';cfg['run_root']=str(self.root/'run')
        cfg['stages']['ut']={'skill':'kernel-ut','mode':'agent','params':{}}
        layout=stage.stage_layout(cfg,'roofline')
        self.assertEqual(Path(layout['in']),self.root/'run/ut/out')
        self.assertEqual(Path(stage.stage_env(cfg,layout)['TASKS_DIR']),self.root/'run/ut/out/tasks')

    def test_ut_skill_runs_the_shared_wave_with_cpu_stub_tasks(self):
        spec=importlib.util.spec_from_file_location('wave_fixtures',PROFILE/'kernel-extract/tests/test_ut_wave.py')
        fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
        self.export();prepare(self.handoff,self.ut)
        runner=self.root/'stub.sh';runner.write_text(fixtures.STUB_RUNNER)
        rows=[]
        for name in ['a','b']:
            task=self.ut/'tasks'/name;(task/'kernel_src').mkdir(parents=True)
            (task/'run').mkdir()
            (task/'unittest.py').write_text(fixtures.FAKE_UT)
            (task/'mode').write_text('ok')
            H.write(task/'run/requested_kernel.json',{'required_gpu_count':1})
            rows.append({'label':name,'task_dir':str(task),'ut_family':'gemm','planned_cases':{'total':2}})
        H.write(self.ut/'task_plan.json',{'tasks':rows})
        p=subprocess.run([sys.executable,str(UT/'kernel-ut/scripts/ut_wave.py'),'start',
            '--eval-dir',str(self.ut),'--runner',str(runner),'--backend','local','--gpus','0,1',
            '--foreground','--tick-sec','1','--lane-sec','120','--timing-workers','2'],
            capture_output=True,text=True,timeout=30)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        summary=H.read(self.ut/'ut_wave/summary.json')['tasks']
        self.assertEqual({r['status'] for r in summary.values()},{'passed'},p.stdout)
        prompt=(self.ut/'ut_wave/prompts/a.1.txt').read_text()
        self.assertIn(str(UT/'kernel-ut'),prompt)


if __name__ == '__main__':
    unittest.main()
