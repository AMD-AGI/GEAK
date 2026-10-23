#!/usr/bin/env python3
"""Summarize complete native-valid arms only after actual serving-rank source proof.

Launcher inventories are checked separately from runtime source observations. This
reader consumes retained evidence; it neither creates rank proof nor executes GPU work.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

BASIS = 'output_token_throughput'
CANDIDATE_PATHS = ['request_metrics.throughput.output.tokens_per_second',
                   'output_tok_per_s', 'output_tokens_per_second', BASIS]
CONTEXT = {
    'duration_seconds': 'request_metrics.throughput.duration_seconds',
    'requests_successful': 'num_requests_successful', 'requests_total': 'num_requests_total',
    'ttft_p50': 'request_metrics.latency.ttft.p50', 'ttft_p90': 'request_metrics.latency.ttft.p90',
    'itl_p50': 'request_metrics.latency.itl.p50', 'itl_p90': 'request_metrics.latency.itl.p90',
    'prefix_cache_hit_rate': 'request_metrics.cache.theoretical_cache_hit_rate',
    'conc': 'conc', 'tp': 'tp',
}


def value_at(document, path):
    value = document
    for key in path.split('.') if path else []:
        if isinstance(value, dict):
            value = value.get(key)
        elif isinstance(value, list) and key.isdigit():
            value = value[int(key)] if int(key) < len(value) else None
        else:
            return None
    return value


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def dig(document, paths):
    for path in paths:
        value = value_at(document, path)
        if number(value):
            return value
    return None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def local_file(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError('missing or nonlocal evidence file: ' + str(name))
    return path


def read_local(root, name):
    return json.loads(local_file(root, name).read_text())


def overlay_hashes(overlay):
    keep = {'.py', '.cuh', '.cu', '.h', '.cpp', '.cc', '.cxx', '.hip', '.s', '.mlir'}
    return {str(path.relative_to(overlay)): sha256(path)
            for path in Path(overlay).rglob('*') if path.is_file() and path.suffix.lower() in keep}


def hash_mapping(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        result = {}
        for row in value:
            if not isinstance(row, dict) or not isinstance(row.get('path'), str) or not row.get('sha256'):
                raise ValueError('malformed overlay inventory row')
            if row['path'] in result:
                raise ValueError('duplicate overlay inventory path')
            result[row['path']] = row['sha256']
        return result
    raise ValueError('overlay receipt hashes must be a mapping or path/sha256 list')


def validate_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError('stage settings must be a JSON object')
    receipt = settings.get('overlay_receipt') or {}
    native = settings.get('native_validation') or {}
    proof = settings.get('serving_source_proof') or {}
    if any(not isinstance(value, dict) for value in (receipt, native, proof)):
        raise ValueError('receipt, native validation and source proof settings must be objects')
    if not receipt.get('file') or not receipt.get('hashes_path') or not receipt.get('root_path'):
        raise ValueError('overlay_receipt.file, hashes_path and root_path are required')
    if not native.get('file') or not native.get('required_values'):
        raise ValueError('native_validation.file and required_values must identify actual native validity evidence')
    if not native.get('aggregate_required_values'):
        raise ValueError('native_validation.aggregate_required_values must identify profiling error/accounting gates')
    completion = native.get('completion_receipt') or {}
    if not isinstance(completion, dict) or not completion.get('file') or not completion.get('required_values'):
        raise ValueError('native_validation.completion_receipt must identify the actual client completion status')
    for required in (native['required_values'], native['aggregate_required_values'], completion['required_values']):
        if not isinstance(required, dict) or any(not isinstance(key, str) or not key for key in required):
            raise ValueError('native required_values must be nonempty dotted-path mappings')
    if not isinstance(native.get('duration_coverage'), dict):
        raise ValueError('native duration-coverage configuration is required')
    if not proof.get('file') or type(proof.get('world_size')) is not int or proof['world_size'] <= 0:
        raise ValueError('serving_source_proof.file and positive world_size are required')
    for name in (receipt['file'], native['file'], completion['file'], proof['file']):
        if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('receipt and proof files must remain inside each run directory')
    producer = proof.get('producer_file')
    if not isinstance(producer, str) or not Path(producer).is_file():
        raise ValueError('a reviewed serving_source_proof.producer_file must be prepared before replay')
    if sha256(producer) != proof.get('producer_sha256'):
        raise ValueError('serving source producer is missing its matching reviewed SHA-256')
    try:
        compile(Path(producer).read_bytes(), producer, 'exec')
    except SyntaxError as exc:
        raise ValueError('serving source producer does not compile: ' + str(exc)) from exc


def require_values(document, expected, label):
    if not isinstance(expected, dict) or any(not isinstance(path, str) for path in expected):
        raise ValueError('required_values must be a dotted-path mapping')
    for path, wanted in expected.items():
        actual = value_at(document, path)
        if actual != wanted or isinstance(actual, bool) != isinstance(wanted, bool):
            raise ValueError(f'{label}: {path} is {actual!r}, expected {wanted!r}')


def native_validation(run_dir, aggregate, settings, duration):
    config = settings['native_validation']
    tag = Path(run_dir).name
    native_file = config['file'].replace('{tag}', tag)
    native = read_local(run_dir, native_file)
    require_values(native, config['required_values'], native_file)
    require_values(aggregate, config['aggregate_required_values'], 'native aggregate')
    completion = config['completion_receipt']
    completion_file = completion['file'].replace('{tag}', tag)
    require_values(read_local(run_dir, completion_file), completion['required_values'], completion_file)
    coverage = config['duration_coverage']
    rows = value_at(native, coverage.get('path', 'metadata.metric_duration_coverage'))
    phase = coverage.get('phase', 'profiling')
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('malformed native duration coverage rows')
    rows = [r for r in rows if r.get(coverage.get('phase_field', 'phase_name')) == phase]
    if len(rows) != 1:
        raise ValueError('native duration evidence must uniquely name the measured phase')
    row = rows[0]
    actual_duration = row.get(coverage.get('duration_field', 'expected_duration_seconds'))
    threshold = row.get(coverage.get('threshold_field', 'required_ratio'))
    if not number(actual_duration) or actual_duration != duration:
        raise ValueError('native phase target differs from the requested complete duration')
    if not number(threshold) or not 0 < threshold <= 1:
        raise ValueError('native duration coverage has no valid required ratio')
    for key in coverage.get('ratio_fields', ['ttft_ratio', 'inter_token_latency_ratio']):
        ratio = row.get(key)
        if not number(ratio) or ratio < threshold:
            raise ValueError('native duration coverage failed: ' + key)
    measured = dig(aggregate, [CONTEXT['duration_seconds']])
    if measured is None or measured < duration:
        raise ValueError('aggregate duration is shorter than the requested arm')
    return {'ok': True, 'file': native_file, 'sha256': sha256(local_file(run_dir, native_file)),
            'completion_file': completion_file,
            'completion_sha256': sha256(local_file(run_dir, completion_file)),
            'phase': phase, 'target_seconds': actual_duration, 'measured_seconds': measured}


def serving_proof(run_dir, overlay, settings, applied):
    config = settings['serving_source_proof']
    document = read_local(run_dir, config['file'])
    if not isinstance(document, dict):
        raise ValueError('serving proof must be a JSON object')
    if document.get('contract') != 'serving_source_proof/v1' or document.get('run_tag') != Path(run_dir).name:
        raise ValueError('serving proof contract or run identity mismatch')
    if not isinstance(document.get('overlay_root'), str) or Path(document['overlay_root']).resolve() != Path(overlay).resolve():
        raise ValueError('serving proof source root differs from this arm overlay')
    ranks = document.get('ranks')
    if not isinstance(ranks, list) or len(ranks) != config['world_size'] or any(not isinstance(row, dict) for row in ranks):
        raise ValueError('serving proof is missing ranks')
    if {r.get('rank') for r in ranks} != set(range(config['world_size'])):
        raise ValueError('serving proof rank identities are incomplete or duplicated')
    expected = {r['target_callable']: r for r in applied.get('applied', [])}
    if not expected or len(expected) != len(applied.get('applied', [])):
        raise ValueError('applied targets do not define unique serving proof obligations')
    native_identities = {}
    for rank in ranks:
        if type(rank.get('rank')) is not int:
            raise ValueError('serving rank identifiers must be integers')
        if type(rank.get('pid')) is not int or rank['pid'] <= 0:
            raise ValueError('serving proof has no actual worker PID')
        if rank.get('post_lazy_load') is not True or rank.get('observers_removed') is not True:
            raise ValueError('serving proof must follow lazy loading and observer removal')
        log = rank.get('log_ref') or {}
        if not isinstance(log, dict):
            raise ValueError('malformed serving log reference')
        log_path = local_file(run_dir, log.get('path', ''))
        if not log_path.stat().st_size or sha256(log_path) != log.get('sha256'):
            raise ValueError('serving worker log reference is missing or changed')
        bindings = rank.get('bindings') or []
        if not isinstance(bindings, list) or any(not isinstance(row, dict) for row in bindings):
            raise ValueError('malformed serving binding records')
        by_target = {r.get('target_callable'): r for r in bindings}
        if len(by_target) != len(bindings) or set(by_target) != set(expected):
            raise ValueError('every serving rank must prove every applied target')
        for target, requirement in expected.items():
            binding = by_target[target]
            if not number(binding.get('invocations')) or binding['invocations'] <= 0:
                raise ValueError('serving target has no observed invocations: ' + target)
            hashes = requirement.get('bundled_source_hashes') or {}
            if not hashes or binding.get('source_hashes') != hashes:
                raise ValueError('serving source hashes differ from the accepted bundle: ' + target)
            if binding.get('resolved_file') not in hashes:
                raise ValueError('actual callable did not resolve to an accepted source file')
            leaves = binding.get('leaf_source_files') or []
            if not leaves or any(name not in hashes for name in leaves):
                raise ValueError('actual computation leaf source evidence is missing')
            for name, digest in hashes.items():
                if sha256(local_file(overlay, name)) != digest:
                    raise ValueError('accepted source changed after runtime proof')
            libraries = binding.get('native_libraries') or []
            if not isinstance(libraries, list) or any(not isinstance(row, dict) for row in libraries):
                raise ValueError('malformed native library records')
            if requirement.get('native_proof_required') and not libraries:
                raise ValueError('native target has no observed library proof')
            for library in libraries:
                if library.get('loaded') is not True or not library.get('namespace'):
                    raise ValueError('native library was not observed loaded')
                path = Path(library.get('path', ''))
                if not path.is_file() or sha256(path) != library.get('sha256'):
                    raise ValueError('observed native library is missing or changed')
                native_sources = {name: digest for name, digest in hashes.items()
                                  if Path(name).suffix.lower() in {'.cu', '.cuh', '.hip', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.s'}}
                library_hashes = library.get('source_hashes')
                if not isinstance(library_hashes, dict):
                    raise ValueError('malformed native source hash map')
                if not native_sources or any(library_hashes.get(name) != digest
                                             for name, digest in native_sources.items()):
                    raise ValueError('native library is not linked to the accepted source hashes')
                provenance = library.get('build_provenance_ref') or {}
                if not isinstance(provenance, dict):
                    raise ValueError('malformed native build provenance reference')
                if sha256(local_file(run_dir, provenance.get('path', ''))) != provenance.get('sha256'):
                    raise ValueError('native build provenance is missing or changed')
            if libraries:
                identity = sorted((item['namespace'], item['sha256']) for item in libraries)
                if target in native_identities and native_identities[target] != identity:
                    raise ValueError('serving ranks disagree on the actual native build')
                native_identities[target] = identity
    return {'ok': True, 'ranks': len(ranks), 'targets_per_rank': len(expected),
            'path': config['file'], 'sha256': sha256(local_file(run_dir, config['file'])),
            'scope': 'retained actual-worker evidence consistency; not independent GPU execution'}


def leg(run_dir, want, settings, duration, overlay=None, applied=None, candidate=True):
    run_dir = Path(run_dir)
    record = {'tag': run_dir.name, 'dir': str(run_dir), 'admissible': False}
    try:
        aggregate = read_local(run_dir, run_dir.name + '.json')
        if not isinstance(aggregate, dict):
            raise ValueError('native aggregate must be a JSON object')
        if aggregate.get('metric_basis', BASIS) != BASIS:
            raise ValueError('metric basis differs from output token throughput')
        throughput = dig(aggregate, CANDIDATE_PATHS)
        if throughput is None or throughput <= 0:
            raise ValueError('no finite positive output token throughput')
        record['throughput'] = throughput
        # Context metrics remain attached to every valid and invalid arm.
        for name, path in CONTEXT.items():
            value = dig(aggregate, [path])
            if value is not None:
                record[name] = value
        record['native_validation'] = native_validation(run_dir, aggregate, settings, duration)
        if candidate:
            receipt_config = settings['overlay_receipt']
            receipt = read_local(run_dir, receipt_config['file'])
            got = hash_mapping(value_at(receipt, receipt_config['hashes_path']))
            if not want or got != want:
                raise ValueError('launcher overlay inventory differs from the accepted overlay')
            root_value = value_at(receipt, receipt_config['root_path'])
            if not isinstance(root_value, str) or not Path(root_value).is_absolute():
                raise ValueError('launcher must record this arm overlay as an absolute path')
            arm_overlay = Path(root_value).resolve()
            if not arm_overlay.is_dir() or Path(root_value).is_symlink():
                raise ValueError('this arm overlay is absent')
            # Check only declared prelaunch inputs. Lazy native build products are
            # runtime outputs and must not redefine the accepted source inventory.
            for name, digest in got.items():
                if sha256(local_file(arm_overlay, name)) != digest:
                    raise ValueError('declared prelaunch source changed or disappeared: ' + name)
            record['overlay_inventory_matches'] = True
            record['overlay_root'] = str(arm_overlay)
            record['serving_source_proof'] = serving_proof(run_dir, arm_overlay, settings, applied)
        record['admissible'] = True
    except (OSError, ValueError, KeyError, TypeError) as exc:
        record['error'] = str(exc)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--validate-run', help='validate one completed arm without producing an A/B')
    parser.add_argument('--stock', action='store_true', help='single-arm validation of the unchanged control')
    parser.add_argument('--overlay')
    parser.add_argument('--applied')
    parser.add_argument('--reference-run')
    parser.add_argument('--historical-reference')
    parser.add_argument('--reference-key', default=CANDIDATE_PATHS[0])
    parser.add_argument('--out-reference')
    parser.add_argument('--out-summary')
    parser.add_argument('--noise-band', type=float, default=1.045)
    parser.add_argument('--expected-repeats', type=int, default=3)
    parser.add_argument('--duration-s', type=float, default=3600)
    parser.add_argument('--runs', nargs='+')
    args = parser.parse_args()
    settings = json.loads(Path(args.settings).read_text())
    try:
        validate_settings(settings)
    except ValueError as exc:
        print(json.dumps({'ok': False, 'status': 'blocked', 'reason': str(exc)}))
        return 78
    if args.preflight:
        print(json.dumps({'ok': True, 'scope': 'configuration only; runtime source proof is still required'}))
        return 0
    if args.validate_run:
        if not args.stock and not (args.overlay and args.applied):
            parser.error('candidate validation requires overlay and applied manifest')
        overlay = Path(args.overlay).resolve() if args.overlay else None
        applied = json.loads(Path(args.applied).read_text()) if args.applied else None
        record = leg(args.validate_run, (applied or {}).get('source_inventory', {}), settings,
                     args.duration_s, overlay, applied, candidate=not args.stock)
        print(json.dumps(record, indent=2))
        return 0 if record['admissible'] else 3
    if not all((args.overlay, args.applied, args.reference_run, args.out_reference, args.out_summary, args.runs)):
        parser.error('overlay, applied, reference-run, output paths and runs are required')
    if args.reference_key not in CANDIDATE_PATHS:
        parser.error('reference-key must identify a supported output-token-throughput field')
    overlay = Path(args.overlay).resolve()
    applied = json.loads(Path(args.applied).read_text())
    want = applied.get('source_inventory') or {}
    if applied.get('all_shadowed') is not True or not want:
        raise ValueError('accepted overlay has no passing binding check/source inventory')
    legs = [leg(path, want, settings, args.duration_s, overlay, applied) for path in args.runs]
    reference_path = Path(args.reference_run)
    stock = leg(reference_path.parent, {}, settings, args.duration_s, candidate=False)
    summary = {'metric_basis': BASIS, 'legs': legs, 'expected_repeats': args.expected_repeats,
               'repeats': sum(r['admissible'] for r in legs), 'fresh_stock': stock,
               'overlay_loaded_by_every_leg': bool(legs) and all(r.get('serving_source_proof', {}).get('ok') for r in legs)}
    independent = (len({str(Path(path).resolve()) for path in args.runs}) == len(args.runs)
                   and len({Path(path).name for path in args.runs}) == len(args.runs))
    summary['independent_arms'] = independent
    complete = (independent and len(legs) == args.expected_repeats
                and all(r['admissible'] for r in legs) and stock['admissible'])
    roots = [r.get('overlay_root') for r in legs]
    if len(set(roots)) != len(legs):
        complete = False
        summary['source_root_error'] = 'candidate arms must use distinct real overlay roots'
    if complete:
        values = [r['throughput'] for r in legs]
        summary[BASIS] = statistics.median(values)
        summary['legs_tok_s'] = values
        summary['spread_pct'] = (max(values) / min(values) - 1) * 100
    else:
        summary['error'] = ('every requested independent, complete native-valid arm and every '
                            'serving-rank proof is required; no duplicate or subset median')
    Path(args.out_summary).write_text(json.dumps(summary, indent=2) + '\n')
    if not complete:
        print(json.dumps(summary, indent=2))
        return 3
    historical = None
    if args.historical_reference:
        old = json.loads(Path(args.historical_reference).read_text())
        if old.get('metric_basis', BASIS) != BASIS:
            raise ValueError('historical reference metric basis mismatch')
        historical = dig(old, [args.reference_key] + CANDIDATE_PATHS)
        if historical is None or historical <= 0:
            raise ValueError('historical reference has no finite positive throughput')
    reference = {BASIS: stock['throughput'], 'metric_basis': BASIS, 'noise_band': args.noise_band,
        'provenance': {'method': 'reference-replay-ab', 'run': str(reference_path), 'key': args.reference_key,
                       'fresh_stock': True, 'historical_reference': args.historical_reference},
        'stock_context': stock, 'historical_output_token_throughput': historical,
        'stock_drift_pct': ((stock['throughput']/historical - 1)*100) if historical and historical > 0 else None}
    Path(args.out_reference).write_text(json.dumps(reference, indent=2) + '\n')
    print(json.dumps({'ok': True, 'reference': stock['throughput'], 'candidate_median': summary[BASIS],
                      'complete_arms': len(legs), 'serving_source_proof': True}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
