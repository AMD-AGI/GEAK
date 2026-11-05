"""Shared GPU conditioning, affinity, and one-call measurement lifecycle."""
import ctypes
import hashlib
import json
import os
import platform
import statistics
from contextlib import nullcontext
from pathlib import Path


def measurement_contract(manifest):
    env=manifest.get('environment',{})
    cache=manifest.get('cache',{})
    return {'operator_boundary':manifest.get('operator_boundary'),'execution':manifest.get('execution'),
        'cache_policy':cache.get('policy'),'eviction_bytes':cache.get('eviction_bytes'),'llc_bytes':cache.get('llc_bytes'),
        'gpu_pci_bus_id':env.get('gpu_pci_bus_id'),'gpu_name':env.get('gpu_name'),'hostname':env.get('hostname'),
        'cpu_affinity':env.get('cpu_affinity'),'runtime_hip':env.get('runtime_hip'),'runtime_cuda':env.get('runtime_cuda')}


def parse_cpu_list(text):
    result=set()
    for part in text.strip().split(','):
        if not part:
            continue
        first, _, last=part.partition('-')
        result.update(range(int(first),int(last or first)+1))
    return result


def affinity(torch, policy='auto', pci_bus_id=None):
    before=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else []
    pci=pci_bus_id
    if not pci:
        props=torch.cuda.get_device_properties(torch.cuda.current_device())
        pci=getattr(props,'pci_bus_id',None)
        if not pci and getattr(torch.version,'hip',None):
            try:
                runtime=ctypes.CDLL('libamdhip64.so')
                buffer=ctypes.create_string_buffer(32)
                function=runtime.hipDeviceGetPCIBusId
                function.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int]
                function.restype=ctypes.c_int
                if function(buffer,len(buffer),torch.cuda.current_device())==0:
                    pci=buffer.value.decode()
            except (OSError,AttributeError):
                pass
    device=Path('/sys/bus/pci/devices')/str(pci).lower() if pci else None
    node=None
    chosen=before
    is_local=False
    if device and (device/'numa_node').exists():
        node=int((device/'numa_node').read_text())
        local=parse_cpu_list((device/'local_cpulist').read_text()) if (device/'local_cpulist').exists() else set()
        usable=sorted(local.intersection(before))
        if policy in ('auto','local') and node >= 0 and usable:
            os.sched_setaffinity(0,usable); chosen=usable
            is_local=True
        elif policy=='local':
            raise ValueError('no GPU-local CPU allowed by current affinity')
    elif policy=='local':
        raise ValueError('GPU PCI/NUMA mapping unavailable; supply --gpu-pci-bus-id')
    return {'gpu_pci_bus_id':pci,'gpu_numa_node':node,'cpu_affinity_before':before,
            'cpu_affinity':chosen,'affinity_policy':policy,
            'numa_status':'local' if is_local else 'unverified'}


class CacheConditioner:
    def __init__(self, torch, policy, llc_mib=None, evict_mib=None):
        self.torch=torch
        self.policy=policy
        self.metadata={'policy':policy,'verified':policy=='warm','eviction_bytes':0}
        self.src=self.dst=None
        if policy=='warm':
            return
        import device_kernels as dk
        self.dk=dk
        props=torch.cuda.get_device_properties(torch.cuda.current_device())
        arch=getattr(props,'gcnArchName','')
        llc=int(llc_mib*2**20) if llc_mib is not None else (256*2**20 if arch.startswith('gfx950') else getattr(props,'l2_cache_size',0))
        if llc <= 0:
            raise ValueError('cache size unknown: supply --llc-mib for the relevant last-level cache')
        size=int(evict_mib*2**20) if evict_mib is not None else 3*llc
        if size < 2*llc:
            raise ValueError('cold conditioning buffer must be at least twice the declared LLC')
        self.n=(size+3)//4
        self.blocks=(self.n+8191)//8192
        self.src=(torch.ones if policy=='cold-read' else torch.zeros)(self.n,dtype=torch.float32,device='cuda')
        self.dst=torch.full((self.blocks,),float('nan'),dtype=torch.float32,device='cuda')
        self.metadata.update({'eviction_bytes':self.n*4,'llc_bytes':llc,
            'llc_source':'explicit' if llc_mib is not None else 'gfx950_campaign_MALL_256MiB' if arch.startswith('gfx950') else 'device_property_l2',
            'conditioning_kernel':'_krf_read_evict' if policy=='cold-read' else '_krf_write_evict',
            'read_reduce_output_bytes':self.blocks*4 if policy=='cold-read' else 0,
            'writeback_caveat':policy=='cold-write'})

    def __call__(self):
        if self.policy=='cold-read':
            self.dk._krf_read_evict[(self.blocks,)](self.src,self.dst,self.n,BLOCK=8192)
        elif self.policy=='cold-write':
            self.dk._krf_write_evict[(self.blocks,)](self.src,self.n,BLOCK=8192)

    def verify(self):
        if self.policy=='warm':
            return
        torch=self.torch
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        # Compile outside the recorded conditioning duration.
        self(); torch.cuda.synchronize()
        if self.policy=='cold-read':
            actual=self.dst.cpu().tolist()
            expected=[min(8192,self.n-i*8192) for i in range(self.blocks)]
            if actual != expected:
                raise ValueError('CACHE_SELF_TEST_FAILED: read/reduction did not consume the buffer')
        else:
            if self.src.min().item()!=1 or self.src.max().item()!=1:
                raise ValueError('CACHE_SELF_TEST_FAILED: write buffer mismatch')
        begin.record(); self(); end.record(); end.synchronize()
        self.metadata.update({'verified':True,'probe_us':begin.elapsed_time(end)*1000,
            'evidence':'buffer operation checked; dispatch presence additionally checked in captured data',
            'residency_claim':'conditioning protocol; not a hardware-counter proof of every operand miss'})


def prime(run, conditioner, count, synchronize):
    for _ in range(count):
        conditioner(); run()
    synchronize()


def execute_calls(run, conditioner, count, begin_marker, end_marker, sync,
                  range_factory=None, event_pair=None):
    """Dependency-injected loop also exercised by CPU sequencing tests."""
    samples=[]
    for index in range(count):
        conditioner()
        begin_marker()
        sync()  # establish boundary even for adapters using other streams
        context=range_factory(f'krf.seam:{index}') if range_factory else nullcontext()
        with context:
            start,finish=event_pair() if event_pair else (None,None)
            if start:
                start.record()
            run()
            if finish:
                finish.record()
            sync()  # include all required streams before the end marker
        end_marker()
        if start:
            samples.append((start,finish))
    sync()
    return samples


def measure(case, args, torch):
    if args.iters <= 0 or args.warmup < 1:
        raise ValueError('iters must be positive and warmup >= 1')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable; no measured results produced')
    import device_kernels as dk
    environment=affinity(torch,args.cpu_affinity,args.gpu_pci_bus_id)
    environment.update({'hostname':platform.node(),'torch_version':torch.__version__,
        'runtime_hip':getattr(torch.version,'hip',None),'runtime_cuda':torch.version.cuda,
        'gpu_name':torch.cuda.get_device_name(),'visible_devices':os.environ.get('HIP_VISIBLE_DEVICES',os.environ.get('CUDA_VISIBLE_DEVICES'))})
    cache=CacheConditioner(torch,args.cache,args.llc_mib,args.evict_mib)
    cache.verify()
    if callable(case.get('validate')):
        if case['validate']() is False:
            raise ValueError('candidate correctness check failed')
        case['metadata']['correctness']='passed'
    run=case['run']
    prime(run,cache,args.warmup,torch.cuda.synchronize)
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    profiler=lambda:torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,with_stack=True)
    capture=None
    if args.execution=='graph':
        graph=torch.cuda.CUDAGraph()
        # TL can use the separate capture trace to recover operation metadata.
        if args.capture:
            capture=out/'capture'; capture.mkdir(exist_ok=True)
            with profiler() as capture_prof:
                with torch.cuda.graph(graph):
                    run()
            capture_prof.export_chrome_trace(str(capture/'graph_capture_rank_0.json'))
        else:
            with torch.cuda.graph(graph):
                run()
        run=graph.replay  # exactly one complete operation per replay
        prime(run,cache,2,torch.cuda.synchronize)
    scratch=torch.empty(1,dtype=torch.int32,device='cuda')
    # Compile only: pre-launching the same markers would create false windows.
    dk._krf_begin_marker.warmup(scratch,grid=(1,))
    dk._krf_end_marker.warmup(scratch,grid=(1,))
    begin=lambda:dk._krf_begin_marker[(1,)](scratch)
    end=lambda:dk._krf_end_marker[(1,)](scratch)
    manifest={**case['metadata'],'environment':environment,'cache':cache.metadata,
        'execution':args.execution,'calls_per_replay':1,'measured_calls':args.iters,
        'warmup_calls':args.warmup,'warmup_scope':'before_first_marker_same_process',
        'expected_dispatches':case.get('expected_dispatches'),'status':'STARTED',
        'capture_folder':str(capture) if capture else None}
    metadata_path=Path(os.environ.get('KRF_MEASUREMENT_MANIFEST',out/'measurement.json'))
    metadata_path.parent.mkdir(parents=True,exist_ok=True)
    metadata_path.write_text(json.dumps(manifest,indent=2,default=str)+'\n')
    context=profiler() if args.capture else nullcontext()
    with context as prof:
        samples=execute_calls(run,cache,args.iters,begin,end,torch.cuda.synchronize,
            torch.profiler.record_function,
            lambda:(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)))
    manifest['current_stream_event_us']=[start.elapsed_time(finish)*1000 for start,finish in samples]
    manifest['event_timing_basis']='current-stream interval; can include eager launch gaps; native TL retains device busy time'
    if args.capture:
        from trace_scope import slice_trace
        raw=out/'trace.raw.json'; prof.export_chrome_trace(str(raw))
        data=json.loads(raw.read_text())
        sliced,audit=slice_trace(data,args.iters,case.get('expected_dispatches'))
        if args.cache!='warm':
            seen=sum(1 for e in data['traceEvents'] if cache.metadata['conditioning_kernel'] in e.get('name','') and e.get('cat')=='kernel')
            if seen!=args.iters:
                raise ValueError('CACHE_DISPATCH_COUNT_MISMATCH')
            manifest['cache']['observed_dispatches']=seen
        (out/'trace.json').write_text(json.dumps(sliced))
        (out/'scope.json').write_text(json.dumps(audit,indent=2))
        manifest['device_busy_us']=[c['device_busy_us'] for c in audit['calls']]
        manifest['device_busy_us_median']=statistics.median(manifest['device_busy_us'])
        manifest['counted_dispatches_per_call']=audit['counted_dispatches_per_call']
        manifest['native_trace']=str(out/'trace.json')
        manifest['native_trace_sha256']=hashlib.sha256((out/'trace.json').read_bytes()).hexdigest()
    manifest['status']='OK'
    metadata_path.write_text(json.dumps(manifest,indent=2,default=str)+'\n')
    return manifest
