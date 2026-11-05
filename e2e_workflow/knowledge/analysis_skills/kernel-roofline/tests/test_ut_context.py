"""Source binding and complete-scope regressions for UT metadata enrichment."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import ut_context as uc
from tl_only import verify_enriched_rows


class UTContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ut = self.root/'unittest.py'
        self.ut.write_text('raise RuntimeError("must not import UT during source review")\n'
            'def _make_args():\n    return {"input": torch.empty_strided((4,8),(8,1),dtype=torch.bfloat16), "out": torch.empty((4,8))}\n'
            'def _launch(mod, a): mod.run(a["input"], a["out"], False)\n')
        self.impl = self.root/'kernel.py'
        self.impl.write_text('def run(a): pass\n')
        self.measurement = {'status':'OK','task':'op','case':'shape','arm':'candidate',
            'execution':'eager','implementation_source':str(self.impl),
            'implementation_sha256':uc.sha256(self.impl),'native_trace_sha256':'frozen-trace-hash',
            'measured_calls':1,'timed_closure':'_launch','counted_dispatches_per_call':{'main':1,'reduce':1}}
        self.manifest = self.root/'measurement.json'
        self.manifest.write_text(json.dumps(self.measurement))
        self.context = uc.draft_context(self.manifest, self.ut)
        self.context.update(review_status='source_reviewed',semantic_name='complete two-stage operation',
            semantics={'outputs':['out']},native_operation='aiter::test',arguments=[
                {'name':'input','source':'input','kind':'tensor','shape':[4,8],'stride':[8,1],
                 'dtype':'c10::BFloat16','access':'read'}])
        self.context['citations'] = {'inputs':uc.source_citation(self.ut,2,3),
                                     'call':uc.source_citation(self.ut,4,4)}
        self.context['field_evidence'] = {
            field:{'citations':['inputs','call'],'derivation':description} for field,description in {
                'semantic_name':'Synthetic test fixture covering the complete call.',
                'native_operation':'Synthetic test-only operation identity; no native model is asserted here.',
                'semantics.outputs':'The UT passes a separately allocated out to the implementation.',
                'arguments.0.binding':'First tensor passed by _launch is payload input.',
                'arguments.0.access':'Synthetic fixture run reads input and writes its separate out.',
                'arguments.0.shape':'empty_strided allocates the explicit shape (4,8).',
                'arguments.0.stride':'empty_strided specifies stride (8,1).',
                'arguments.0.dtype':'The construction explicitly uses torch.bfloat16.',
            }.items()}

    def trace(self):
        events=[{'ph':'X','cat':'user_annotation','name':'krf.seam:0','pid':1,'tid':1,'ts':10,'dur':50},
                {'ph':'X','cat':'python_function','name':f'{self.ut}(2): _launch','pid':1,'tid':1,'ts':20,'dur':20}]
        for corr, name, host, gpu in [(11,'main',22,30),(12,'reduce',25,40)]:
            events += [{'ph':'X','cat':'cuda_runtime','name':'hipLaunchKernel','pid':1,'tid':1,
                        'ts':host,'dur':1,'args':{'correlation':corr}},
                       {'ph':'X','cat':'kernel','name':name,'pid':0,'tid':7,
                        'ts':gpu,'dur':3,'args':{'correlation':corr}},
                       {'ph':'s','cat':'ac2g','name':'ac2g','pid':1,'tid':1,'ts':host,'id':corr},
                       {'ph':'f','cat':'ac2g','name':'ac2g','pid':0,'tid':7,'ts':gpu,'id':corr}]
        return {'traceEvents':events}

    def validate(self):
        return uc.validate_context(self.context, self.measurement, self.manifest)

    def test_draft_reads_evidence_without_executing_ut(self):
        draft = uc.draft_context(self.manifest, self.ut)
        self.assertEqual(draft['review_status'], 'needs_source_review')
        self.assertIsNone(draft['native_operation'])
        self.assertIn('_launch',[f['name'] for f in draft['sources'][0]['functions']])
        with self.assertRaisesRegex(ValueError,'source_reviewed'):
            uc.validate_context(draft,self.measurement,self.manifest)

    def test_reject_changed_case_manifest_or_source(self):
        self.validate()
        self.context['binding']['case'] = 'another-shape'
        with self.assertRaisesRegex(ValueError,'case'): self.validate()
        self.context['binding']['case'] = 'shape'
        self.manifest.write_text(self.manifest.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'measurement hash'): self.validate()
        self.context['binding']['measurement_sha256'] = uc.sha256(self.manifest)
        self.impl.write_text('def run(a): different(a)\n')
        with self.assertRaisesRegex(ValueError,'source changed'): self.validate()

    def test_runtime_shapes_dtypes_scalars_checked(self):
        self.measurement['input_metadata']={'input':{'kind':'tensor','shape':[4,8],'stride':[8,1],'dtype':'torch.bfloat16'}}
        self.validate()
        self.context['arguments'][0]['shape'] = [8,4]
        with self.assertRaisesRegex(ValueError,'runtime argument mismatch'): self.validate()
        self.context['arguments'][0]['shape'] = [4,8]
        self.context['arguments'].append({'name':'flag','source':'literal:false','kind':'scalar','value':False})
        for field in ['arguments.1.binding','arguments.1.value']:
            self.context['field_evidence'][field]={'citations':['call'],'derivation':'_launch passes literal False as the third argument.'}
        self.validate()
        self.context['arguments'][-1]['value'] = True
        with self.assertRaisesRegex(ValueError,'literal source'): self.validate()

    def test_review_flag_and_file_hashes_do_not_authorize_unexplained_values(self):
        self.context['field_evidence'].pop('arguments.0.stride')
        with self.assertRaisesRegex(ValueError,'UT_FIELD_EVIDENCE_MISSING: arguments.0.stride'):
            self.validate()

    def test_literal_and_quantization_need_actual_ut_citations(self):
        self.context['arguments'].append({'name':'flag','source':'literal:false','kind':'scalar','value':False})
        with self.assertRaisesRegex(ValueError,'UT_FIELD_EVIDENCE_MISSING'):
            self.validate()
        self.context['arguments'].pop()
        self.context['semantics']['quantization']=False
        with self.assertRaisesRegex(ValueError,'semantics.quantization'):
            self.validate()

    def test_changed_quote_unknown_fields_and_model_as_evidence_rejected(self):
        original=copy.deepcopy(self.context)
        self.context['citations']['inputs']['quote']='made-up shape and dtype'
        with self.assertRaisesRegex(ValueError,'citation text mismatch'): self.validate()
        self.context=copy.deepcopy(original)
        self.context['unknown_fields']=['semantics.quantization']
        with self.assertRaisesRegex(ValueError,'UT_FIELDS_UNKNOWN'): self.validate()
        self.context=copy.deepcopy(original)
        model=self.root/'model.py';model.write_text('DEFAULT_SHAPE=(4,8)\n')
        self.context['sources'].append(uc.source_evidence(model,'native_model'))
        self.context['citations']['model']=uc.source_citation(model,1,1)
        with self.assertRaisesRegex(ValueError,'outside the reviewed source chain'): self.validate()

    def test_snapshot_preserves_aliases_and_does_not_read_tensor_contents(self):
        class Tensor:
            shape = (4,8)
            dtype = 'torch.bfloat16'
            def stride(self): return (8,1)
            def storage_offset(self): return 0
            def untyped_storage(self): return SimpleNamespace(_cdata=42)
        x = Tensor()
        result = uc.snapshot_inputs({'input':x,'view':Tensor(),'eps':1e-6},SimpleNamespace(is_tensor=lambda x:isinstance(x,Tensor)))
        self.assertEqual(result['input']['alias_group'],result['view']['alias_group'])
        self.assertEqual(result['input']['shape'],[4,8])
        self.assertEqual(result['eps']['value'],1e-6)

    def test_enrichment_retains_every_original_event_and_two_subkernels(self):
        trace = self.trace()
        original = copy.deepcopy(trace)
        enriched, count = uc.enrich_eager_trace(trace,self.context,self.measurement)
        self.assertEqual(trace,original)
        self.assertEqual(count,1)
        self.assertEqual(enriched['traceEvents'][1:],original['traceEvents'])
        new = enriched['traceEvents'][0]
        self.assertEqual((new['ts'],new['dur']),(20,20))
        self.assertEqual(new['args']['Input Dims'],[[4,8]])
        self.assertIn('added for analysis',new['args']['KRF metadata origin'])

    def test_reject_graph_existing_cpu_ops_and_missing_launch_frame(self):
        with self.assertRaisesRegex(ValueError,'eager only'):
            uc.enrich_eager_trace(self.trace(),self.context,{**self.measurement,'execution':'graph'})
        trace=self.trace()
        trace['traceEvents'].append({'cat':'cpu_op','name':'aten::add','pid':1,'tid':1,'ts':21,'dur':20})
        with self.assertRaisesRegex(ValueError,'existing CPU operators'):
            uc.enrich_eager_trace(trace,self.context,self.measurement)
        self.context['launch_frame']['function']='unrelated'
        with self.assertRaisesRegex(ValueError,'launch_frame'):
            uc.enrich_eager_trace(self.trace(),self.context,self.measurement)

    def test_host_only_view_ops_are_preserved_without_blocking_metadata(self):
        trace=self.trace()
        view={'cat':'cpu_op','name':'aten::transpose','pid':1,'tid':1,'ts':20.5,'dur':.5}
        trace['traceEvents'].append(view)
        result,count=uc.enrich_eager_trace(trace,self.context,self.measurement)
        self.assertEqual(count,1)
        self.assertIn(view,result['traceEvents'])
        self.assertEqual(result['traceEvents'][1:],trace['traceEvents'])

    def test_reject_missing_correlation_and_unmodeled_device_work(self):
        trace=self.trace()
        next(e for e in trace['traceEvents'] if e['cat']=='kernel')['args']['correlation']=99
        with self.assertRaisesRegex(ValueError,'correlation'):
            uc.enrich_eager_trace(trace,self.context,self.measurement)
        trace=self.trace()
        trace['traceEvents'].append({'cat':'kernel','name':'extra','ts':100,'dur':2})
        with self.assertRaisesRegex(ValueError,'outside UT call ranges'):
            uc.enrich_eager_trace(trace,self.context,self.measurement)

    def test_report_cannot_drop_a_required_subkernel(self):
        row={'operation':'aiter::test','context':{'native_row':{'operation_count':'1'},
            'kernels':[{'name':'main','detail':{'count':1},'call_stack_status':'matched'}]}}
        with self.assertRaisesRegex(ValueError,'kernel coverage'):
            verify_enriched_rows([row],self.context,self.measurement)
        row['context']['kernels'].append({'name':'reduce','detail':{'count':1},'call_stack_status':'matched'})
        verify_enriched_rows([row],self.context,self.measurement)


if __name__ == '__main__':
    unittest.main()
