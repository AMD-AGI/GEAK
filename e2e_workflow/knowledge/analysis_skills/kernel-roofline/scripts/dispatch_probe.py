#!/usr/bin/env python3
"""Same-device empty-kernel timing diagnostic. Never used as a scoring bound."""
import argparse
import json
import statistics
from pathlib import Path
from measurement_runtime import CacheConditioner, affinity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--cache',choices=['warm','cold-read','cold-write'],default='cold-read')
    p.add_argument('--llc-mib',type=float)
    p.add_argument('--evict-mib',type=float)
    p.add_argument('--cpu-affinity',choices=['auto','local','unchanged'],default='auto')
    p.add_argument('--gpu-pci-bus-id')
    p.add_argument('--iters',type=int,default=20)
    args=p.parse_args()
    import torch
    import device_kernels as dk
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable')
    if args.iters<1:
        raise ValueError('iters must be positive')
    env=affinity(torch,args.cpu_affinity,args.gpu_pci_bus_id)
    cache=CacheConditioner(torch,args.cache,args.llc_mib,args.evict_mib); cache.verify()
    ptr=torch.empty(1,device='cuda')
    invoke=lambda:dk._krf_probe_noop[(1,)](ptr)
    invoke(); torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): invoke()
    result={'basis':'empirical diagnostic; not a lower bound or TL roof','environment':env,'cache':cache.metadata,'modes':{}}
    args.output.mkdir(parents=True,exist_ok=True)
    for mode,run in [('eager',invoke),('graph',graph.replay)]:
        pairs=[]
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(args.iters):
                cache()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record(); run(); end.record(); pairs.append((start,end))
            torch.cuda.synchronize()
        path=args.output/f'{mode}.json'; prof.export_chrome_trace(str(path))
        durations=[e['dur'] for e in json.loads(path.read_text())['traceEvents'] if e.get('cat')=='kernel' and '_krf_probe_noop' in e.get('name','')]
        if len(durations)!=args.iters:
            raise ValueError('probe dispatch count mismatch')
        result['modes'][mode]={'kernel_us':durations,'kernel_us_median':statistics.median(durations),
                              'event_us':[a.elapsed_time(b)*1000 for a,b in pairs],'calls_per_replay':1}
    (args.output/'probe.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
