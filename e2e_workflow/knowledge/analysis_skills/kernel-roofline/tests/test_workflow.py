"""CPU regressions for the campaign's measurement, scoring, and coverage bugs."""
import importlib.util
import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import counter_metrics
import fold
import measurement_runtime as runtime
import peak_audit
import rocprof_window
import run_roofline as rr
import trace_scope
import tl_only
import workload_adapter


def row(case,roof=1.,time=2.,weight=1.,**extra):
    return dict(kernel='op',regime='decode',case=case,roofline_us=roof,measured_us=time,w=weight,**extra)


class FoldTests(unittest.TestCase):
    def test_different_cache_conditions_rejected(self):
        with self.assertRaisesRegex(ValueError,'conditions differ'):
            fold.compare_cases([row('a',measurement_contract={'cache':'warm'})],
                               [row('a',measurement_contract={'cache':'cold-read'})])
    def test_unsupported_model_still_has_latency_comparison(self):
        result=fold.compare_cases([row('a',None,8)],[row('a',None,2)])['op/decode']
        self.assertEqual(result['ab_speedup'],4)
        self.assertEqual(result['candidate']['case_coverage'],0)
        self.assertFalse(result['tl_ratio_is_speedup'])
    def test_missing_cases_keep_their_weight(self):
        result=fold.fold_cases([row(str(i),roof=1 if i<2 else None) for i in range(6)],roof_key='roofline_us')[('op','decode')]
        self.assertAlmostEqual(result['case_coverage'],1/3)
        self.assertEqual(result['n_unscored'],4)
        self.assertEqual(result['status'],'PARTIAL')
    def test_manifest_restores_dropped_cases(self):
        cases=[{'kernel':'op','regime':'decode','case':str(i),'w':1} for i in range(6)]
        result=fold.with_manifest([row('0'),row('1')],cases)
        self.assertEqual(len(result),6)
        self.assertAlmostEqual(fold.fold_cases(result,roof_key='roofline_us')[('op','decode')]['case_coverage'],1/3)
    def test_bad_case_cannot_hide_in_average(self):
        result=fold.fold_cases([row('bad',2,1),row('good',1,100)],roof_key='roofline_us')[('op','decode')]
        self.assertLess(result['pct'],1)
        self.assertTrue(result['suspect'])
        self.assertIsNone(result['distance_to_reference'])
    def test_unequal_case_durations_fold_in_time(self):
        result=fold.fold_cases([row('a',1,2),row('b',10,100)],roof_key='roofline_us')[('op','decode')]
        self.assertAlmostEqual(result['pct'],11/102)
    def test_zero_weight_stays_zero(self):
        result=fold.fold_cases([row('a'),row('b',100,1,weight=0)],roof_key='roofline_us')[('op','decode')]
        self.assertEqual(result['pct'],.5)
        self.assertFalse(result['suspect'])
    def test_partial_kernel_reference_is_not_extrapolated(self):
        reference,cov,missing=fold.fold_kernels([{'kernel':'a','ms':1},{'kernel':'b','ms':3}],lambda x:.5 if x['kernel']=='a' else None,40)
        self.assertEqual((reference,cov),(5,.25))
        r=row('a',reference,40,kernel_coverage=cov,covered_measured_us=10)
        result=fold.fold_cases([r],roof_key='roofline_us')[('op','decode')]
        self.assertEqual(result['pct'],.5)
        self.assertEqual(result['status'],'PARTIAL')
    def test_compare_identity_and_missing_arm(self):
        result=fold.compare_cases([row('a',1,8),row('b',1,4)],[row('a',1,2)])['op/decode']
        self.assertEqual(result['ab_speedup'],4)
        self.assertTrue(result['tl_ratio_is_speedup'])
        self.assertEqual(result['candidate']['case_coverage'],.5)
    def test_changed_reference_breaks_identity(self):
        self.assertFalse(fold.compare_cases([row('a')],[row('a',2,4)])['op/decode']['tl_ratio_is_speedup'])
    def test_mismatched_weights_rejected(self):
        with self.assertRaises(ValueError):
            fold.compare_cases([row('a')],[row('a',weight=2)])
    def test_sample_envelope_crosses_target(self):
        rows=fold.with_manifest([row('a',.85,1,measured_us_reps=[.99,1.01])],[row('a')])
        result=fold.fold_cases(rows,roof_key='roofline_us',target_efficiency=.85)[('op','decode')]
        self.assertEqual(result['status'],'BORDERLINE')


class CounterTests(unittest.TestCase):
    def test_all_six_original_topk_reports_score_without_rounded_ai(self):
        paths=sorted((Path(__file__).parent/'fixtures').glob('*baseline_roofline_raw.txt'))
        self.assertEqual(len(paths),6)
        for path in paths:
            blocks=rr.parse_kernel_blocks(path.read_text())
            values=rr.efficiency_records([b[0] for b in blocks],[b[1] for b in blocks],[b[2] for b in blocks],arch='gfx950')
            target=next(v for v in values if 'topk_main_kernel' in v['kernel'])['references']['empirical']
            self.assertAlmostEqual(target['efficiency'],target['achieved_bw_gbps']/6857)
            if 'ctx418k' in path.name:
                self.assertAlmostEqual(target['efficiency']*100,3.4688639347)
    def test_display_ai_zero_still_scores_topk(self):
        r=counter_metrics.score(0,.57,237.86,6857,153392)
        self.assertEqual(r['bound'],'memory-bound')
        self.assertAlmostEqual(r['efficiency'],237.86/6857)
        self.assertAlmostEqual(r['ai_derived'],.57/237.86)
    def test_rounded_nonzero_ai_does_not_bias_score(self):
        r=counter_metrics.score(.01,1.,200.,6857,153392)
        self.assertAlmostEqual(r['efficiency'],200/6857)
    def test_missing_or_integer_work_not_fabricated(self):
        for perf,integer in [(0,False),(1,True)]:
            r=counter_metrics.score(0,perf,200,6857,153392,integer)
            self.assertIsNone(r['efficiency'])
            self.assertIsNotNone(r['memory_efficiency'])
            self.assertIsNotNone(r['reason'])
    def test_redundancy_removal_is_not_regression(self):
        r=counter_metrics.attribute_change({'bytes':1000,'flops':100,'measured_us':100},{'bytes':200,'flops':100,'measured_us':30})
        self.assertEqual(r['status'],'AI_SHIFT_NOT_REGRESSION')
        self.assertAlmostEqual(r['ai_ratio'],5)
    def test_fp8_scaled_sibling_is_flagged_not_substituted(self):
        values={'mfma_f8':2455000,'mfma_f6f4':9748296}
        peak,label=rr.sanity_check_peak({},'mfma_f8',2455000,'gfx950',values.get)
        self.assertEqual(peak,2455000)
        self.assertIn('suspect',label)

    def test_profile_driver_applies_window_and_keeps_provenance(self):
        fixture=next((Path(__file__).parent/'fixtures').glob('*ctx418k*raw.txt')).read_text()
        lines=fixture.splitlines()
        start=next(i for i,line in enumerate(lines) if rr._KERNEL_HEADER_RE.match(line.strip()) and 'topk_main' in line)
        finish=next((i for i in range(start+1,len(lines)) if rr._KERNEL_HEADER_RE.match(lines[i].strip())),len(lines))
        lines=lines[start:finish]
        lines[0]='Kernel 0: target (100.0%)'
        text='\n'.join(lines)
        analyze=[]
        real_run = subprocess.run
        with tempfile.TemporaryDirectory(prefix='krf quoting ') as folder:
            root=Path(folder)
            cache=root/'frozen.csv'
            cache.write_text('device,HBMBw,MFMABF16Flops,MFMAF16Flops,MFMAF8Flops,MFMAF6F4Flops\n0,6857,2400000,2400000,4800000,9600000\n')
            def run(command,**kwargs):
                if command[1]=='profile':
                    profile=Path(command[command.index('--path')+1])
                    # Reproduce the real profiler's argv flattening. The child
                    # must still run with quoted arguments and the requested cwd.
                    real_run(' '.join(command[command.index('--')+1:]), shell=True,
                             cwd=kwargs['cwd'], env=kwargs['env'], check=True)
                    names=['target']*10+['_krf_read_evict','_krf_begin_marker','target','_krf_end_marker']*2
                    (profile/'pmc_perf.csv').write_text('Dispatch_ID,Kernel_Name,GPU_ID,Start_Timestamp,End_Timestamp\n'+
                         ''.join(f'{i},{n},0,{i*2000},{i*2000+1000}\n' for i,n in enumerate(names)))
                    Path(kwargs['env']['KRF_MEASUREMENT_MANIFEST']).write_text(json.dumps({'status':'OK','measured_calls':2,
                        'cache':{'conditioning_kernel':'_krf_read_evict'},'task':'op','case':'case','arm':'baseline','case_metadata':{'regime':'decode'}}))
                    return SimpleNamespace(returncode=0,stdout='profile complete',stderr='')
                analyze.append(command)
                return SimpleNamespace(returncode=0,stdout=text,stderr='')
            with patch.object(rr,'detect_gpu_arch',return_value='gfx950'),patch.object(rr,'rocprof_compute_version',return_value='3.4.0'),\
                 patch.object(rr.subprocess,'run',side_effect=run),patch.dict(rr.os.environ,{'ROOFLINE_MIBENCH_CACHE':str(cache)}):
                cmd=shlex.join([sys.executable,'-c','from pathlib import Path; Path("ran with spaces.txt").write_text("quoted value")'])
                rc=rr.run_roofline_mode(root,cmd,root,'case','auto',None,expected_calls=2)
            self.assertEqual(rc,0)
            self.assertEqual((root/'ran with spaces.txt').read_text(),'quoted value')
            self.assertEqual(analyze[0][analyze[0].index('-d')+1:],['12','16'])
            result=json.loads((root/'case_roofline.json').read_text())
            self.assertAlmostEqual(result['rows'][0]['measured_us'],1)
            self.assertEqual(result['rows'][0]['kernel_coverage'],1)
            self.assertTrue(json.loads((root/'case_provenance.json').read_text())['peak_reference']['sha256'])

    def test_profiler_cannot_silently_replace_frozen_peaks(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            cache=root/'frozen.csv'
            cache.write_text('device,HBMBw\n0,6857\n')
            def run(command, **kwargs):
                profile=Path(command[command.index('--path')+1])
                (profile/'roofline.csv').write_text('device,HBMBw\n0,1234\n')
                return SimpleNamespace(returncode=0,stdout='',stderr='')
            with patch.object(rr,'detect_gpu_arch',return_value='gfx950'),patch.object(rr,'rocprof_compute_version',return_value='3.4.0'),\
                 patch.object(rr.subprocess,'run',side_effect=run),patch.dict(rr.os.environ,{'ROOFLINE_MIBENCH_CACHE':str(cache)}):
                rc=rr.run_roofline_mode(root,'true',root,'changed','auto',None,expected_calls=1)
            self.assertEqual(rc,1)
            self.assertIn('FROZEN_PEAK_CHANGED',(root/'changed_roofline_error.txt').read_text())


class WindowTests(unittest.TestCase):
    def write(self,path,names):
        path.write_text('Dispatch_ID,Kernel_Name,GPU_ID\n'+''.join(f'{i},{name},0\n' for i,name in enumerate(names)))
    def test_same_named_autotune_and_warmups_are_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            names=['target']*20+['_krf_read_evict','_krf_begin_marker','target','reduce','_krf_end_marker']*2
            self.write(Path(d)/'pmc_perf.csv',names)
            r=rocprof_window.select_dispatch_windows(d,2,{'target':1,'reduce':1})
            self.assertEqual(len(r['selected_ids']),4)
            self.assertGreater(min(r['selected_ids']),20)
    def test_missing_marker_fails(self):
        with tempfile.TemporaryDirectory() as d:
            self.write(Path(d)/'pmc_perf.csv',['target'])
            with self.assertRaises(ValueError): rocprof_window.select_dispatch_windows(d,1)
    def test_missing_cache_dispatch_fails(self):
        with tempfile.TemporaryDirectory() as d:
            self.write(Path(d)/'pmc_perf.csv',['_krf_begin_marker','target','_krf_end_marker'])
            with self.assertRaisesRegex(ValueError,'CACHE_DISPATCH'):
                rocprof_window.select_dispatch_windows(d,1,conditioning_kernel='_krf_read_evict')
    def test_repeated_dispatch_id_fails(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'pmc_perf.csv').write_text('Dispatch_ID,Kernel_Name,GPU_ID\n0,target,0\n0,target,0\n')
            with self.assertRaisesRegex(ValueError,'repeated dispatch ID'):
                rocprof_window.select_dispatch_windows(d,1)
    def test_changed_multiplicity_fails(self):
        with tempfile.TemporaryDirectory() as d:
            self.write(Path(d)/'pmc_perf.csv',['_krf_begin_marker','target','_krf_end_marker','_krf_begin_marker','target','target','_krf_end_marker'])
            with self.assertRaises(ValueError): rocprof_window.select_dispatch_windows(d,2)
    def test_counter_pass_id_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as d:
            names=['_krf_begin_marker','target','_krf_end_marker']
            self.write(Path(d)/'pmc_perf_0.csv',names)
            self.write(Path(d)/'pmc_perf_1.csv',['setup']+names)
            with self.assertRaisesRegex(ValueError,'COUNTER_PASS'): rocprof_window.select_dispatch_windows(d,1)
    def test_merged_csv_cannot_replace_deleted_pass_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            self.write(p/'pmc_perf.csv',['_krf_begin_marker','target','_krf_end_marker'])
            (p/'perfmon').mkdir()
            (p/'perfmon/pmc_perf_0.txt').write_text('pmc: COUNTER')
            with self.assertRaisesRegex(ValueError,'COUNTER_PASSES_MISSING'):
                rocprof_window.select_dispatch_windows(d,1)


class RunnerTests(unittest.TestCase):
    def test_adapter_validation_metadata_reaches_measurement(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'adapter.py'
            path.write_text('def build_case(task, case, arm):\n'
                '    metadata={"operator_boundary":"test operator"}\n'
                '    def validate():\n'
                '        metadata["correctness_detail"]={"passed":True}\n'
                '        return True\n'
                '    return {"run":lambda:None,"validate":validate,"metadata":metadata}\n')
            args=SimpleNamespace(adapter=path,task='test',case='one',arm='candidate')
            built=workload_adapter.build_case(args,None)
            self.assertTrue(built['validate']())
            self.assertTrue(built['metadata']['correctness_detail']['passed'])
    def test_empty_native_graph_is_unavailable_with_actionable_reason(self):
        self.assertIn('native_graph_execution_roots_unrecognized', tl_only.missing_rows_reason(
            {'execution':'graph'},'Found 0 execution roots in graph tree\nDetected GPU-only trace.'))
        self.assertIn('native_cpu_operation_metadata_unavailable', tl_only.missing_rows_reason(
            {'execution':'eager'},'Detected GPU-only trace.'))
    def test_arm_guard_discriminates_baseline_and_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            modules={}
            for arm,folder in [('baseline','baseline_src'),('candidate','kernel_src')]:
                p=Path(d)/folder/'body.py';p.parent.mkdir();p.write_text('def run(a): return a\n')
                modules[arm]=workload_adapter.load_module(p,'fixture_'+arm)
            ut=SimpleNamespace(_BASE=modules['baseline'].run,_CAND=modules['candidate'].run,_invoke=lambda fn,a:fn(a))
            workload_adapter.verify_arm(ut,'baseline')
            workload_adapter.verify_arm(ut,'candidate')
            ut._CAND=ut._BASE
            with self.assertRaisesRegex(ValueError,'cannot verify'):
                workload_adapter.verify_arm(ut,'candidate')
    def test_module_based_arm_uses_module_source(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'kernel_src'/'body.py';p.parent.mkdir();p.write_text('def run(a): return a\n')
            module=workload_adapter.load_module(p,'fixture_module_arm')
            ut=SimpleNamespace(_CANDM=module,_launch=lambda mod,a:mod.run(a))
            target,source=workload_adapter.verify_arm(ut,'candidate')
            self.assertEqual(target,module)
            self.assertEqual(source,str(p))
    def test_launch_not_gather_selected(self):
        calls=[]
        ut=SimpleNamespace(_launch=lambda mod,a:calls.append('launch'),_gather=lambda a:calls.append('gather'),
                           _invoke=lambda mod,a:calls.extend(['launch','gather']))
        run,name=workload_adapter.timed_closure(ut,object(),{})
        run(); self.assertEqual(calls,['launch']); self.assertEqual(name,'_launch')
    def test_ambiguous_correctness_wrapper_rejected(self):
        ut=SimpleNamespace(_gather=lambda a:None,_invoke=lambda m,a:None)
        with self.assertRaises(ValueError): workload_adapter.timed_closure(ut,object(),{})
    def test_cache_before_every_call_and_warmup_outside(self):
        events=[]
        cache=lambda:events.append('cache')
        run=lambda:events.append('run')
        sync=lambda:None
        runtime.prime(run,cache,2,sync)
        runtime.execute_calls(run,cache,2,lambda:events.append('begin'),lambda:events.append('end'),sync)
        self.assertEqual(events,['cache','run']*2+['cache','begin','run','end']*2)
    def test_cpu_list(self):
        self.assertEqual(runtime.parse_cpu_list('0-2,5,8-9'),{0,1,2,5,8,9})
    def test_legal_elementwise_and_ac2g_endpoints_survive(self):
        e=[{'cat':'user_annotation','name':'krf.seam:0','ph':'X','pid':1,'tid':1,'ts':10,'dur':10},
           {'cat':'cpu_op','name':'aten::add','ph':'X','pid':1,'tid':1,'ts':11,'dur':4},
           {'cat':'cuda_runtime','name':'launch','ph':'X','pid':1,'tid':1,'ts':12,'dur':1,'args':{'correlation':42}},
           {'cat':'kernel','name':'_krf_begin_marker','ph':'X','pid':0,'tid':7,'ts':8,'dur':1},
           {'cat':'kernel','name':'vectorized_elementwise_kernel','ph':'X','pid':0,'tid':7,'ts':15,'dur':3,'args':{'correlation':42}},
           {'cat':'kernel','name':'_krf_end_marker','ph':'X','pid':0,'tid':7,'ts':22,'dur':1},
           {'cat':'kernel','name':'CatArrayBatchedCopy','ph':'X','pid':0,'tid':7,'ts':25,'dur':3},
           {'cat':'ac2g','name':'ac2g','ph':'s','pid':1,'tid':1,'ts':12,'id':42},
           {'cat':'ac2g','name':'ac2g','ph':'f','pid':0,'tid':7,'ts':15,'id':42}]
        trace,audit=trace_scope.slice_trace({'traceEvents':e},1)
        names=[x['name'] for x in trace['traceEvents']]
        self.assertIn('vectorized_elementwise_kernel',names)
        self.assertNotIn('CatArrayBatchedCopy',names)
        self.assertEqual(sum(x.get('cat')=='ac2g' for x in trace['traceEvents']),2)
        self.assertEqual(audit['calls'][0]['device_busy_us'],3)


if __name__=='__main__': unittest.main()
