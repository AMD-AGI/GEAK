#!/usr/bin/env python3
"""Run verified upstream TraceLens on a measured trace; retain native artifacts."""
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from extract_tl_row import package_row
from peak_audit import audit_tl_arch
from measurement_runtime import measurement_contract
from ut_context import validate_context, enrich_eager_trace, event_arguments

UPSTREAM='https://github.com/AMD-AGI/TraceLens.git'


def verify_source(source, frozen_commit=None):
    def git(*args):
        return subprocess.check_output(['git','-C',str(source),*args],text=True).strip()
    head=git('rev-parse','HEAD')
    if git('status','--porcelain'):
        raise ValueError('TL source has local changes; use a clean upstream checkout')
    resolved=(frozen_commit or subprocess.check_output(['git','ls-remote',UPSTREAM,'refs/heads/main'],text=True).split()[0])
    if head != resolved:
        raise ValueError(f'TL checkout {head} differs from requested/main commit {resolved}; fetch and install that revision first')
    return {'upstream':UPSTREAM,'commit':head,'source':str(Path(source).resolve()),
            'selection':'frozen_comparison_commit' if frozen_commit else 'current_remote_main'}


def native_number(row, names):
    for name in names:
        try:
            value=float(row.get(name,''))
        except (ValueError,TypeError):
            continue
        if math.isfinite(value):
            return value
    return None


def missing_rows_reason(measurement, report_log):
    if measurement.get('execution') == 'graph' and 'Found 0 execution roots' in report_log:
        return 'native_graph_execution_roots_unrecognized; current TL requires supported framework annotations'
    if 'Detected GPU-only trace' in report_log:
        return 'native_cpu_operation_metadata_unavailable; device timing retained, no kernel-name model substitution'
    return 'no_native_operation_rows; inspect trace metadata and current TL mapping'


def verify_native_mapping(context, env, cwd):
    """Resolve the reviewed identity using the selected, unmodified TL registry."""
    code = '''import sys,json,inspect,hashlib
from pathlib import Path
from TraceLens.PerfModel.torch_op_mapping import op_to_perf_model_class_map
data=json.load(sys.stdin)
cls=op_to_perf_model_class_map.get(data['name'])
if cls is None: raise ValueError('reviewed UT identity has no registered native TL model: '+data['name'])
params=cls.get_param_details(data)
path=Path(inspect.getfile(cls))
print(json.dumps({'operation':data['name'],'model':cls.__module__+'.'+cls.__qualname__,
 'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'native_param_details':params}))
'''
    run = subprocess.run([sys.executable, '-c', code], env=env, cwd=cwd,
        input=json.dumps({'name':context['native_operation'],'args':event_arguments(context)}),
        capture_output=True, text=True, timeout=60)
    if run.returncode:
        raise ValueError('UT native mapping/schema check failed: ' + run.stderr[-3000:])
    return json.loads(run.stdout)


def verify_enriched_rows(rows, context, measurement):
    if len(rows) != 1 or rows[0]['operation'] != context['native_operation']:
        raise ValueError('UT-enriched native report did not preserve the single complete operation')
    row = rows[0]
    native = row['context']['native_row']
    calls = measurement['measured_calls']
    if native_number(native, ['operation_count']) != calls:
        raise ValueError('UT-enriched native operation count differs from measured calls')
    observed = {}
    for kernel in row['context']['kernels']:
        name = kernel['name']
        observed[name] = observed.get(name, 0) + kernel['detail']['count']
        if kernel['call_stack_status'] != 'matched':
            raise ValueError('UT-enriched native report has missing/ambiguous kernel call stacks')
    if observed != {name:n*calls for name,n in context['expected_dispatches_per_call'].items()}:
        raise ValueError('UT-enriched native kernel coverage differs from complete measured scope')


def export_native_rows(csv_path, measurement, tl, statistic='mean'):
    csv.field_size_limit(sys.maxsize)
    with Path(csv_path).open(newline='') as stream:
        native=list(csv.DictReader(stream))
    digest=hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()
    rows=[]
    for index,row in enumerate(native):
        context=package_row(row,source=csv_path,row_index=index,csv_sha256=digest,tl_commit=tl['commit'])
        roof=native_number(row,['Roofline Time (µs)','Roofline Time (µs)_first'])
        duration=native_number(row,[f'Kernel Time (µs)_{statistic}','Kernel Time (µs)'])
        native_pct=native_number(row,[f'Pct Roofline_{statistic}','Pct Roofline'])
        identity={key:row.get(key) for key in ('perf_params','process_name','process_label','thread_name','Input Strides','Concrete Inputs')}
        params_key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
        operation_count=native_number(row,['operation_count'])
        measured_calls=measurement.get('measured_calls',1)
        calls_per_seam=operation_count/measured_calls if operation_count is not None and measured_calls else None
        model=row.get('has_perf_model','').lower()=='true'
        reason=None
        if not model:
            reason='native_model_or_metadata_unavailable'
        elif roof is None:
            reason='native_roof_or_peak_unavailable'
        elif duration is None or duration <= 0:
            reason='native_timing_unavailable'
        if measurement.get('legacy_runner'):
            reason='legacy_runner_scope_not_verified'
        rows.append({'kernel':row['name'],'regime':measurement.get('case_metadata',{}).get('regime','unspecified'),
            'task':measurement.get('task'),'case':str(measurement.get('case','trace'))+':'+params_key,
            'operation':row['name'],'case_weight':measurement.get('case_metadata',{}).get('share_of_op_calls',1)*(calls_per_seam if calls_per_seam is not None else 1),
            'native_calls_per_measured_call':calls_per_seam,'fold_scope':'per native operation instance',
            'roofline_us':roof,'measured_us':duration,'native_pct_roofline':native_pct,
            'timing_statistic':statistic,'status':'UNAVAILABLE' if reason else 'NATIVE',
            'unavailable_reason':reason,'suspect':bool(native_pct is not None and native_pct>100),
            'basis':'TraceLens native operation; case rollup is separately derived',
            'context':context})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement',required=True,type=Path)
    p.add_argument('--tl-source',required=True,type=Path)
    p.add_argument('--tl-commit',help='Freeze the commit resolved at comparison start')
    p.add_argument('--arch',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--statistic',choices=['mean','median'],default='mean')
    p.add_argument('--rocprof-reference',type=Path,help='Optional peak-file comparison only; no counter run')
    p.add_argument('--ut-context',type=Path,help='Source-reviewed UT identity/arguments; enriched eager input is saved separately')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        if args.ut_context and any((args.output / name).exists() for name in ('native', 'ut-context.json', 'rows.json')):
            raise ValueError('UT-context analysis requires a fresh output directory')
        tl=verify_source(args.tl_source,args.tl_commit)
        measurement=json.loads(args.measurement.read_text())
        if measurement.get('status')!='OK' or not measurement.get('native_trace'):
            raise ValueError('successful --capture measurement is required')
        trace=Path(measurement['native_trace'])
        if hashlib.sha256(trace.read_bytes()).hexdigest()!=measurement['native_trace_sha256']:
            raise ValueError('trace changed since measurement')
        env=dict(os.environ,PYTHONPATH=str(args.tl_source.resolve()))
        imported=subprocess.check_output([sys.executable,'-c','import TraceLens; print(TraceLens.__file__)'],env=env,cwd=args.output,text=True).strip()
        if not Path(imported).resolve().is_relative_to(args.tl_source.resolve()):
            raise ValueError('selected Python imported a different TL checkout')
        ut_context = None
        metadata_origin = 'captured_trace'
        if args.ut_context:
            context = validate_context(json.loads(args.ut_context.read_text()), measurement, args.measurement)
            ut_context = {'context':context, 'source':str(args.ut_context.resolve()),
                          'sha256':hashlib.sha256(args.ut_context.read_bytes()).hexdigest(), 'status':'context_only'}
            if context.get('native_operation'):
                ut_context['native_mapping'] = verify_native_mapping(context, env, args.output)
                if measurement.get('execution') == 'eager':
                    augmented, count = enrich_eager_trace(json.loads(trace.read_text()), context, measurement)
                    trace = args.output.resolve()/'trace.ut-enriched.json'
                    if trace == Path(measurement['native_trace']).resolve():
                        raise ValueError('UT enrichment must not overwrite the captured trace')
                    trace.write_text(json.dumps(augmented))
                    ut_context.update({'status':'eager_metadata_enriched','added_ranges':count,
                        'analysis_trace':str(trace),'analysis_trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest()})
                    metadata_origin = 'source_reviewed_ut_metadata'
                else:
                    ut_context['reason'] = 'graph_requires_native_capture_replay_mapping; identity is retained as context'
            (args.output/'ut-context.json').write_text(json.dumps(ut_context,indent=2)+'\n')
        module='TraceLens.Reporting.generate_perf_report_pytorch'
        if measurement.get('execution')=='graph':
            module+='_inference'
        command=[sys.executable,'-m',module,'--profile_json_path',str(trace),'--gpu_arch_json_path',str(args.arch.resolve()),
                 '--output_csvs_dir',str(args.output.resolve()/'native'),'--enable_pseudo_ops','--include_call_stack','--disable_coll_analysis']
        if measurement.get('execution')=='graph':
            if not measurement.get('capture_folder'):
                raise ValueError('graph analysis needs its matching native capture trace')
            command+=['--capture_folder',measurement['capture_folder']]
        tl.update({'import_path':imported,'python':sys.executable,'report_command':command})
        result=subprocess.run(command,env=env,cwd=args.output,capture_output=True,text=True,timeout=3600)
        (args.output/'native_report.log').write_text(result.stdout+'\n'+result.stderr)
        if result.returncode:
            raise RuntimeError('native TL report failed; inspect native_report.log')
        csv_path=args.output/'native/unified_perf_summary.csv'
        rows=export_native_rows(csv_path,measurement,tl,args.statistic) if csv_path.is_file() else []
        if metadata_origin == 'source_reviewed_ut_metadata':
            verify_enriched_rows(rows, context, measurement)
        specs=[r['context']['native_row'].get('Compute Spec') for r in rows]
        arch=audit_tl_arch(args.arch,specs,args.rocprof_reference)
        for row in rows:
            row['measurement_contract']={**measurement_contract(measurement),'tl_commit':tl['commit'],'arch_sha256':arch['sha256']}
            row['metadata_origin'] = metadata_origin
            if ut_context:
                row['ut_context'] = ut_context
                if metadata_origin == 'source_reviewed_ut_metadata':
                    row['basis'] = 'TraceLens native model/report on source-reviewed UT-enriched metadata; captured device timing unchanged'
        result={'mode':'tl-only','engine':'TraceLens native','tl':tl,'arch':arch,'measurement':measurement,'rows':rows,
                'metadata_origin':metadata_origin,'ut_context':ut_context,
                'coverage':{'native_rows':len(rows),'scored_rows':sum(r['status']=='NATIVE' for r in rows)}}
        result['case_status']='UNAVAILABLE' if not rows or not result['coverage']['scored_rows'] else 'PARTIAL' if result['coverage']['scored_rows']!=len(rows) else 'NATIVE_ROWS_AVAILABLE'
        result['unavailable_cases']=[] if rows else [{'task':measurement.get('task'),'case':measurement.get('case'),
            'reason':missing_rows_reason(measurement,(args.output/'native_report.log').read_text()),
            'measured_device_busy_us':measurement.get('device_busy_us'),'roofline_us':None}]
        (args.output/'rows.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'case_status':result['case_status'],**result['coverage'],
                          'unavailable_cases':result['unavailable_cases']}))
        return 0
    except Exception as error:
        (args.output/'error.json').write_text(json.dumps({'error':str(error)},indent=2)+'\n')
        print(str(error),file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
