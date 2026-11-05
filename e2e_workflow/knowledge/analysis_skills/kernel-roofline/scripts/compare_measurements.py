#!/usr/bin/env python3
"""Compare complete-closure device times, including cases with no TL model."""
import argparse
import json
import statistics
from pathlib import Path
from measurement_runtime import measurement_contract


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True,type=Path)
    p.add_argument('--candidate',required=True,type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    base=json.loads(args.baseline.read_text()); cand=json.loads(args.candidate.read_text())
    for key in ('task','case'):
        if base.get(key)!=cand.get(key):
            raise ValueError(f'different {key}')
    if any(m.get('status')!='OK' or not m.get('device_busy_us') for m in (base,cand)):
        raise ValueError('two successful captured measurements are required')
    if measurement_contract(base)!=measurement_contract(cand):
        raise ValueError('measurement conditions differ')
    b=statistics.median(base['device_busy_us']);c=statistics.median(cand['device_busy_us'])
    result={'task':base.get('task'),'case':base.get('case'),'baseline_us':b,'candidate_us':c,
            'ab_speedup':b/c if c>0 else None,'basis':'complete-closure device busy-time medians',
            'baseline_samples':base['device_busy_us'],'candidate_samples':cand['device_busy_us'],
            'correctness':{'baseline':base.get('correctness','unverified'),'candidate':cand.get('correctness','unverified')},
            'roofline':None,'note':'No reference or dispatch-floor score is invented.'}
    text=json.dumps(result,indent=2)+'\n';print(text,end='')
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    return 0


if __name__=='__main__':raise SystemExit(main())
