"""Slice measured GPU windows while retaining correlated native TL host context."""
from collections import Counter
from rocprof_window import BEGIN, END

GPU_CATEGORIES = {'kernel','gpu_memcpy','gpu_memset'}


def slice_trace(trace, expected_calls, expected_dispatches=None):
    events = trace['traceEvents']
    gpu = sorted(((i,e) for i,e in enumerate(events) if e.get('cat') in GPU_CATEGORIES), key=lambda item:item[1]['ts'])
    windows, current, begin_time = [], None, None
    selected, excluded = set(), []
    for index, event in gpu:
        name = event.get('name','')
        if BEGIN in name:
            if current is not None:
                raise ValueError('nested GPU markers')
            current = []
            begin_time=event['ts']+event.get('dur',0)
            excluded.append({'kernel':name,'us':event.get('dur',0),'reason':'begin_marker'})
        elif END in name:
            if current is None or not current:
                raise ValueError('missing/empty GPU measurement window')
            if any(e['ts'] < begin_time or e['ts']+e.get('dur',0) > event['ts'] for _,e in current):
                raise ValueError('GPU work crosses a measurement marker; join all operator streams')
            windows.append(current); current = None
            excluded.append({'kernel':name,'us':event.get('dur',0),'reason':'end_marker'})
        elif current is not None:
            if '_krf_' in name:
                raise ValueError('conditioning kernel inside seam window')
            current.append((index,event)); selected.add(index)
        else:
            excluded.append({'kernel':name,'us':event.get('dur',0),'reason':'outside_measured_call'})
    if current is not None or len(windows) != expected_calls:
        raise ValueError(f'DISPATCH_COUNT_MISMATCH: expected {expected_calls}, got {len(windows)}')
    counts = [Counter(e['name'] for _,e in w) for w in windows]
    expected = Counter(expected_dispatches) if expected_dispatches is not None else counts[0]
    if any(c != expected for c in counts):
        raise ValueError('DISPATCH_COUNT_MISMATCH: per-call device work changed')
    ranges = [e for e in events if e.get('cat')=='user_annotation' and e.get('name','').startswith('krf.seam:')]
    if len(ranges) != expected_calls:
        raise ValueError('missing measured host ranges')
    def in_range(e):
        return any(e.get('pid')==r.get('pid') and e.get('tid')==r.get('tid') and
                   e.get('ts',0) < r['ts']+r.get('dur',0) and e.get('ts',0)+e.get('dur',0) > r['ts'] for r in ranges)
    correlations = {events[i].get('args',{}).get('correlation') for i in selected} - {None}
    host_correlations={e.get('args',{}).get('correlation') for e in events if e.get('cat') in ('cuda_runtime','hip_runtime') and in_range(e)}-{None}
    if any(i not in selected and e.get('args',{}).get('correlation') in host_correlations for i,e in gpu):
        raise ValueError('required correlated GPU activity was outside the measurement window')
    kept = set(selected)
    for index,e in enumerate(events):
        cat=e.get('cat','')
        if e.get('ph')=='M' or (cat not in GPU_CATEGORIES and cat!='ac2g' and in_range(e)):
            kept.add(index)
        if cat in ('cuda_runtime','hip_runtime') and e.get('args',{}).get('correlation') in correlations:
            kept.add(index)
    # Keep both endpoints of every selected flow, including GPU-thread endpoints.
    def flow_key(e):
        return e.get('cat'),e.get('id'),e.get('scope')
    selected_flows=set()
    for e in events:
        if e.get('cat')!='ac2g':
            continue
        if in_range({**e,'dur':max(e.get('dur',0),.001)}) or any(
            e.get('pid')==events[i].get('pid') and e.get('tid')==events[i].get('tid') and
            events[i]['ts'] <= e.get('ts',-1) <= events[i]['ts']+events[i].get('dur',0) for i in selected):
            selected_flows.add(flow_key(e))
    kept.update(i for i,e in enumerate(events) if e.get('cat')=='ac2g' and flow_key(e) in selected_flows)
    per_call=[]
    for window in windows:
        intervals=sorted((e['ts'],e['ts']+e.get('dur',0)) for _,e in window)
        busy=0.; end=None
        for start,finish in intervals:
            busy += max(0.,finish-max(start,end if end is not None else start))
            end=max(finish,end if end is not None else finish)
        per_call.append({'device_busy_us':busy,'device_sum_us':sum(e.get('dur',0) for _,e in window),
                         'kernels':[{'kernel':e['name'],'us':e.get('dur',0),'stream':e.get('args',{}).get('stream')} for _,e in window]})
    return {**trace,'traceEvents':[e for i,e in enumerate(events) if i in kept]}, {
        'calls':per_call,'counted_dispatches_per_call':dict(expected),'excluded_kernels':excluded,
        'scope':'complete_operator_between_markers','host_range_count':len(ranges)}
