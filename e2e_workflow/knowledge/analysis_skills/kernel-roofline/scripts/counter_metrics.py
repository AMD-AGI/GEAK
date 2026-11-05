"""Counter-rate scoring and A/B attribution; independent of TL-native models."""
import math
from pathlib import Path


def positive(x):
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


def score(ai_display, perf_gflops, achieved_bw, peak_bw, peak_compute, integer_work=False):
    result = {'ai_display': ai_display, 'ai_derived': None, 'achieved_gflops': perf_gflops,
        'achieved_bw_gbps': achieved_bw, 'peak_bw_gbps': peak_bw, 'peak_compute_gflops': peak_compute,
        'efficiency': None, 'memory_efficiency': None, 'bound': None, 'reason': None,
        'ridge': peak_compute / peak_bw if positive(peak_compute) and positive(peak_bw) else None,
        'attainable_gflops': None, 'basis': 'counter_rates_from_report'}
    if positive(achieved_bw) and positive(peak_bw):
        result['memory_efficiency'] = achieved_bw / peak_bw
    if integer_work:
        result['reason'] = 'integer_work_has_no_compatible_flop_efficiency'
        return result
    # Prefer a same-report rate ratio over rounded display AI. These rates may
    # themselves be rounded; retain the source values and do not claim exact AI.
    if positive(achieved_bw) and positive(perf_gflops):
        ai = perf_gflops / achieved_bw
        result['ai_derived'] = ai
    else:
        ai = ai_display
    if not positive(peak_bw):
        result['reason'] = 'missing_bw_peak'
    elif not positive(peak_compute):
        result['reason'] = 'missing_compute_peak_memory_utilization_only'
    elif not positive(ai) or not positive(perf_gflops):
        result['reason'] = 'zero_or_rounded_work_memory_utilization_only'
    else:
        result['attainable_gflops'] = min(peak_compute, ai * peak_bw)
        result['bound'] = 'memory-bound' if ai < result['ridge'] else 'compute-bound'
        if result['bound'] == 'memory-bound' and positive(achieved_bw):
            result['efficiency'] = achieved_bw / peak_bw
            result['basis'] = 'achieved_bw_over_peak_bw'
        else:
            result['efficiency'] = perf_gflops / result['attainable_gflops']
        if result['efficiency'] > 1:
            result['reason'] = 'reference_exceeded_investigate_scope_model_timing_or_peak'
    return result


def attribute_change(baseline, candidate):
    """Both inputs use counts per matching call and their own measured time."""
    result = {}
    for label, field in [('time_ratio','measured_us'), ('bytes_ratio','bytes'), ('flops_ratio','flops')]:
        x, y = baseline.get(field), candidate.get(field)
        result[label] = y / x if positive(x) and isinstance(y, (int, float)) and y >= 0 else None
    t, b, f = result['time_ratio'], result['bytes_ratio'], result['flops_ratio']
    result['achieved_bw_ratio'] = b / t if b is not None and positive(t) else None
    result['ai_ratio'] = f / b if f is not None and positive(b) else None
    result['ab_speedup'] = 1 / t if positive(t) else None
    result['status'] = ('AI_SHIFT_NOT_REGRESSION' if t is not None and b is not None and 0 < t < 1 and b < t
                        else 'NO_ATTRIBUTION' if t is None or b is None else 'COMPARE_LATENCY_AND_WORK')
    return result


def case_rows(records, manifest, window):
    """Fold counter results with the selected counter-run timings, never TL time."""
    from fold import fold_kernels
    from measurement_runtime import measurement_contract
    if not manifest or not window:
        return {}
    time_source=window['passes'][0]
    audit=time_source['audit']
    merged=Path(time_source.get('file','')).name == 'pmc_perf.csv'
    selected=[x for x in audit if x['reason']=='counted']
    times={}
    complete=all(x.get('duration_us') is not None for x in selected)
    for x in selected:
        times[x['kernel']]=times.get(x['kernel'],0)+(x.get('duration_us') or 0)/window['expected_calls']
    total=sum(times.values()) if complete else None
    refs=sorted({key for r in records for key in r['references']})
    result={}
    for ref in refs:
        by_name={r['kernel']:r['references'].get(ref) for r in records}
        components=[{'kernel':name,'ms':time/1000} for name,time in times.items()]
        eff=lambda entry:(by_name.get(entry['kernel']) or {}).get('efficiency')
        roof,cov,unscored=fold_kernels(components,eff,total) if total and complete else (None,0,[])
        data=manifest.get('case_metadata',{})
        counted_bytes=counted_flops=0.
        counts_complete=complete
        suspect=False
        for name,time in times.items():
            metrics=by_name.get(name) or {}
            bw,perf=metrics.get('achieved_bw_gbps'),metrics.get('achieved_gflops')
            if bw is None or perf is None:
                counts_complete=False
            else:
                counted_bytes+=bw*time*1000
                counted_flops+=perf*time*1000
            suspect |= bool(metrics.get('peak_warning')) or (metrics.get('efficiency') or 0)>1
        result[ref]={'kernel':manifest.get('task','operator'),'case':manifest.get('case','case'),
            'measurement_contract':measurement_contract(manifest),
            'regime':data.get('regime','unspecified'),'arm':manifest.get('arm'),
            'case_weight':data.get('share_of_op_calls',1),'measured_us':total,
            'covered_measured_us':total*cov if total is not None else None,'roofline_us':roof,
            'kernel_coverage':cov,'unscored_kernels':unscored,'missing_policy':'drop','suspect':suspect,
            'bytes':counted_bytes if counts_complete else None,'flops':counted_flops if counts_complete else None,
            'basis':'rocprof '+ref,
            'timing_basis':('rocprof merged CSV' if merged else 'first retained counter pass') + ', serialized dispatch durations per call',
            'timing_source':time_source.get('file'),
            'counts_basis':'estimates from rounded native report rates times mean duration; raw pass counters retained separately',
            'unavailable_reason':'legacy_runner_scope_not_verified' if manifest.get('legacy_runner') else
                                 None if roof is not None else 'missing_counter_time_or_efficiency'}
    return result
