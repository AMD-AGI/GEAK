#!/usr/bin/env python3
"""Run one verified UT/custom-adapter case with skill-owned measurement scope.

For rocprof, use the default marker-only mode under run_roofline.py. For TL, add
--capture, then pass measurement.json to tl_only.py. Both modes use the same
cache protocol, timed closure, per-call boundaries, and metadata.
"""
import argparse
import json
import os
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from workload_adapter import build_case
from measurement_runtime import measure


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter',type=Path,help='Module exposing build_case(task, case, arm)')
    p.add_argument('--harness',type=Path,help='Optional legacy UT loader; its timing/flush code is not used')
    p.add_argument('--ut-root',type=Path)
    p.add_argument('--task',required=True)
    p.add_argument('--case',required=True)
    p.add_argument('--regime',help='Override the declared phase for an explicit batch manifest')
    p.add_argument('--case-weight',type=float,help='Explicit deployment call weight; omitted uses UT metadata')
    p.add_argument('--arm',choices=['baseline','candidate'],default='baseline')
    p.add_argument('--runner',choices=['auto','invoke'],default='auto',help='invoke is historical/unverified scope')
    p.add_argument('--cache',choices=['cold-read','cold-write','warm'],default='cold-read')
    p.add_argument('--no-flush',action='store_true',help='Legacy spelling for --cache warm')
    p.add_argument('--llc-mib',type=float)
    p.add_argument('--evict-mib',type=float)
    p.add_argument('--execution',choices=['eager','graph'],default='eager')
    p.add_argument('--cpu-affinity',choices=['auto','local','unchanged'],default='auto')
    p.add_argument('--gpu-pci-bus-id')
    p.add_argument('--iters',type=int,default=5)
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--capture',action='store_true',help='Export full and selected native TL traces')
    p.add_argument('--output',type=Path,default=Path('measurement'))
    return p


def main():
    p=parser(); args=p.parse_args()
    if args.no_flush:
        args.cache='warm'
    if args.adapter and (args.harness or args.ut_root):
        p.error('choose --adapter or the UT loader arguments')
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        visibility={key:os.environ.get(key) for key in ('HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')}
        import torch
        case=build_case(args,torch)
        if any(os.environ.get(key)!=value for key,value in visibility.items()):
            raise ValueError('GPU_VISIBILITY_CHANGED_BY_LOADER; use device isolation or a loader that preserves selection')
        metadata=case['metadata'].setdefault('case_metadata',{})
        if args.regime:
            metadata['regime']=args.regime
        if args.case_weight is not None:
            if args.case_weight < 0:
                raise ValueError('case weight must be nonnegative')
            metadata['share_of_op_calls']=args.case_weight
        result=measure(case,args,torch)
        print(json.dumps(result,indent=2,default=str))
        return 0
    except Exception as error:
        failure={'status':'FAILED','task':args.task,'case':args.case,'arm':args.arm,
                 'error':f'{type(error).__name__}: {error}'}
        path=Path(os.environ.get('KRF_MEASUREMENT_MANIFEST',args.output/'measurement.json'))
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(failure,indent=2)+'\n')
        print(failure['error'],file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
