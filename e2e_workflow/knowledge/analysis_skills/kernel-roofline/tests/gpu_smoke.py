#!/usr/bin/env python3
"""Exercise real conditioning, marker capture and graph/eager timing on one GPU."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def build_case(task, case, arm):
    import torch
    x=torch.arange(4096,device='cuda',dtype=torch.float32)
    y=torch.ones_like(x)
    temp=torch.empty_like(x); output=torch.empty_like(x)
    def run():
        torch.add(x,y,out=temp)
        torch.mul(temp,2,out=output)
    def validate():
        run()
        torch.testing.assert_close(output,(x+y)*2)
        return True
    return {'run':run,'validate':validate,'metadata':{
        'operator_boundary':'add then multiply, two required device operations',
        'implementation_source':str(Path(__file__).resolve()),'case_metadata':{'regime':'smoke','share_of_op_calls':1}}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--llc-mib',type=float)
    args=p.parse_args()
    try:
        import torch
        available=torch.cuda.is_available()
    except ImportError:
        available=False
    if not available:
        print('SKIP: no usable PyTorch GPU runtime; no GPU validation performed')
        return 77
    runner=Path(__file__).resolve().parents[1]/'scripts/case_runner_template.py'
    results=[]
    for cache in ['warm','cold-read']:
        for execution in ['eager','graph']:
            output=args.output/f'{cache}-{execution}'
            command=[sys.executable,str(runner),'--adapter',str(Path(__file__).resolve()),'--task','smoke',
                '--case','two_stages','--arm','candidate','--cache',cache,'--execution',execution,
                '--iters','3','--warmup','3','--capture','--output',str(output)]
            if args.llc_mib:
                command+=['--llc-mib',str(args.llc_mib)]
            subprocess.run(command,check=True)
            record=json.loads((output/'measurement.json').read_text())
            assert record['status']=='OK' and record['correctness']=='passed'
            assert len(record['device_busy_us'])==3
            assert sum(record['counted_dispatches_per_call'].values())==2
            if cache=='cold-read': assert record['cache']['observed_dispatches']==3
            results.append({'cache':cache,'execution':execution,'status':'passed'})
    print(json.dumps(results,indent=2))
    return 0


if __name__=='__main__': raise SystemExit(main())
