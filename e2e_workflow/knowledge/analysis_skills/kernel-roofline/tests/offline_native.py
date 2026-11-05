#!/usr/bin/env python3
"""Synthetic trace -> scope slicer -> current native TL -> packaged CSV row.

CPU-only integration test; synthetic times are not GPU performance evidence.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from trace_scope import slice_trace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tl-source',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    def event(cat,name,ts,dur,pid=1,tid=1,**extras):
        return dict(ph='X',cat=cat,name=name,ts=ts,dur=dur,pid=pid,tid=tid,**extras)
    events=[event('python_function','fixture.py(1): forward',0,110),
            event('user_annotation','krf.seam:0',10,90),
            event('cpu_op','aten::mm',12,68,args={'Input Dims':[[32,64],[64,128]],'Input type':['float','float']}),
            event('kernel','_krf_begin_marker',5,1,0,7),event('kernel','_krf_end_marker',95,1,0,7)]
    for index,(kernel,duration) in enumerate([('synthetic_main',5),('synthetic_reduce',10)]):
        corr=100+index; host_ts=20+index*10; gpu_ts=40+index*10
        events.extend([event('cuda_runtime','hipLaunchKernel',host_ts,1,args={'correlation':corr}),
            event('kernel',kernel,gpu_ts,duration,0,7,args={'correlation':corr,'stream':7}),
            dict(ph='s',cat='ac2g',name='ac2g',ts=host_ts,pid=1,tid=1,id=corr),
            dict(ph='f',cat='ac2g',name='ac2g',ts=gpu_ts,pid=0,tid=7,id=corr,bp='e')])
    selected,audit=slice_trace({'traceEvents':events},1)
    trace=args.output/'trace.json'; trace.write_text(json.dumps(selected))
    arch=args.output/'arch.json'; arch.write_text(json.dumps({'mem_bw_gbps':1000,
        'max_achievable_tflops':{'matrix_fp32':100,'matrix_tf32':100},'source':'synthetic test reference'}))
    measurement={'status':'OK','native_trace':str(trace.resolve()),'native_trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
        'task':'synthetic','case':'two_kernels','case_metadata':{'regime':'synthetic'},'execution':'eager',
        'measurement_kind':'synthetic_fixture_not_gpu_measurement'}
    manifest=args.output/'measurement.json'; manifest.write_text(json.dumps(measurement))
    out=args.output/'result'
    subprocess.run([sys.executable,str(SCRIPTS/'tl_only.py'),'--measurement',str(manifest),
        '--tl-source',str(args.tl_source),'--arch',str(arch),'--output',str(out)],check=True)
    result=json.loads((out/'rows.json').read_text())
    row=next(r for r in result['rows'] if r['operation']=='aten::mm')
    assert row['status']=='NATIVE'
    assert len(row['context']['kernels'])==2
    assert all(k['call_stack_status']=='matched' for k in row['context']['kernels'])
    assert row['measured_us']==15
    print('PASS: current native TL preserves one operation, two subkernels, call stacks and 15 synthetic GPU microseconds')
    return 0


if __name__=='__main__': raise SystemExit(main())
