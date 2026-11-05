#!/usr/bin/env python3
"""Native TL integration for a synthetic two-kernel UT; not GPU performance data."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
from ut_context import draft_context, sha256, source_citation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tl-source',required=True,type=Path)
    parser.add_argument('--tl-commit',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    ut=root/'ut.py';ut.write_text('''# Synthetic metadata fixture; no GPU performance claim.
def _make_args():
    return dict(out=torch.empty_strided((4,8),(8,1),dtype=torch.bfloat16),
                input=torch.empty_strided((4,8),(8,1),dtype=torch.bfloat16),
                weight=torch.ones(8,dtype=torch.bfloat16), epsilon=1e-6)
def _launch(mod, args):
    mod.reduce(args)
    mod.apply(args)
def _ref(a):
    x = a['input'].float()
    return (x * torch.rsqrt(x.square().mean(-1,keepdim=True)+a['epsilon']) * a['weight'].float()).to(torch.bfloat16)
''')
    impl=root/'impl.py';impl.write_text('# Synthetic two-stage RMSNorm metadata fixture; no executable GPU implementation.\n')
    events=[{'ph':'X','cat':'user_annotation','name':'krf.seam:0','pid':1,'tid':1,'ts':10,'dur':70},
            {'ph':'X','cat':'python_function','name':f'{ut}(6): _launch','pid':1,'tid':1,'ts':20,'dur':25}]
    for corr,name,host,gpu in [(11,'synthetic_reduce',22,50),(12,'synthetic_apply',25,55)]:
        events.extend([
            {'ph':'X','cat':'cuda_runtime','name':'hipLaunchKernel','pid':1,'tid':1,'ts':host,'dur':1,'args':{'correlation':corr}},
            {'ph':'X','cat':'kernel','name':name,'pid':0,'tid':7,'ts':gpu,'dur':3,'args':{'correlation':corr,'stream':7}},
            {'ph':'s','cat':'ac2g','name':'ac2g','pid':1,'tid':1,'ts':host,'id':corr},
            {'ph':'f','cat':'ac2g','name':'ac2g','pid':0,'tid':7,'ts':gpu,'id':corr}])
    trace=root/'trace.json';trace.write_text(json.dumps({'traceEvents':events}))
    m={'status':'OK','task':'fixture','case':'two_stages','arm':'candidate','execution':'eager',
       'implementation_source':str(impl),'implementation_sha256':sha256(impl),'timed_closure':'_launch',
       'native_trace':str(trace),'native_trace_sha256':sha256(trace),'measured_calls':1,
       'device_busy_us':[6], 'counted_dispatches_per_call':{'synthetic_reduce':1,'synthetic_apply':1},
       'measurement_kind':'synthetic_fixture_not_gpu_measurement'}
    manifest=root/'measurement.json';manifest.write_text(json.dumps(m))
    context=draft_context(manifest,ut)
    context.update(review_status='source_reviewed',semantic_name='synthetic two-stage RMSNorm',
                   semantics={'synthetic':True,'quantization':False},native_operation='aiter::rmsnorm')
    for name,shape,stride,access in [('out',[4,8],[8,1],'write'),('input',[4,8],[8,1],'read'),('weight',[8],[1],'read')]:
        context['arguments'].append({'name':name,'source':name,'kind':'tensor','shape':shape,'stride':stride,'dtype':'c10::BFloat16','access':access})
    context['arguments'].append({'name':'epsilon','source':'literal:1e-6','kind':'scalar','value':1e-6})
    context['citations']={'fixture':source_citation(ut,1,1),'inputs':source_citation(ut,2,5),
                          'launch':source_citation(ut,6,8),'reference':source_citation(ut,9,11)}
    def evidence(field, refs, derivation):
        context['field_evidence'][field]={'citations':refs,'derivation':derivation}
    evidence('semantic_name',['launch','reference'],'The synthetic UT represents complete RMSNorm in two stages.')
    evidence('native_operation',['reference'],'The reference is affine RMSNorm with separate BF16 output, matching aiter::rmsnorm semantics.')
    evidence('semantics.synthetic',['fixture'],'The UT explicitly identifies itself as a synthetic fixture.')
    evidence('semantics.quantization',['reference'],'The reference returns BF16 normalized values; no quantization scale or integer output.')
    for i,arg in enumerate(context['arguments']):
        prefix=f'arguments.{i}.'
        evidence(prefix+'binding',['inputs','launch','reference'],f'UT payload {arg["name"]} supplies this native argument; order follows the out/input/weight/epsilon signature.')
        if arg['kind']=='tensor':
            evidence(prefix+'access',['reference'],f'{arg["name"]}: {arg["access"]} in the synthetic RMSNorm contract.')
            evidence(prefix+'shape',['inputs'],f'The UT allocation explicitly declares {arg["shape"]}.')
            evidence(prefix+'stride',['inputs'],'Matrices use explicit empty_strided; the 1D ones allocation is dense with stride 1.')
            evidence(prefix+'dtype',['inputs'],'The UT explicitly allocates torch.bfloat16; c10::BFloat16 is its profiler spelling.')
        else:
            evidence(prefix+'value',['inputs'],'The UT epsilon is the literal 1e-6.')
    sidecar=root/'context.json';sidecar.write_text(json.dumps(context))
    arch=root/'arch.json';arch.write_text(json.dumps({'mem_bw_gbps':1000,'max_achievable_tflops':{'vector_bf16':10},'source':'synthetic test reference'}))
    subprocess.run([sys.executable,str(SCRIPTS/'tl_only.py'),'--measurement',str(manifest),'--ut-context',str(sidecar),
        '--tl-source',str(args.tl_source),'--tl-commit',args.tl_commit,'--arch',str(arch),'--output',str(root/'native-result')],check=True)
    result=json.loads((root/'native-result/rows.json').read_text())
    row=result['rows'][0]
    assert len(result['rows'])==1 and row['status']=='NATIVE'
    assert row['operation']=='aiter::rmsnorm' and row['measured_us']==6
    assert row['metadata_origin']=='source_reviewed_ut_metadata'
    assert len(row['context']['kernels'])==2
    assert all(k['call_stack_status']=='matched' for k in row['context']['kernels'])
    assert sha256(trace)==m['native_trace_sha256']
    print('PASS: native TL models one UT operation with both subkernels, stacks and six synthetic microseconds; original trace unchanged')


if __name__=='__main__': main()
