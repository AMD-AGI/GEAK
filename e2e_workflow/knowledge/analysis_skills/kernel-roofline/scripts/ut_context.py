#!/usr/bin/env python3
"""Source-backed UT identity and metadata bridge; never calculates a roofline."""
import argparse
import ast
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_evidence(path, role):
    path = Path(path).resolve()
    text = path.read_text()
    result = {'path': str(path), 'sha256': sha256(path), 'role': role}
    if path.suffix == '.py':
        tree = ast.parse(text)
        result['functions'] = [{
            'name': node.name, 'line': node.lineno,
            'source': ast.get_source_segment(text, node),
        } for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return result


def draft_context(measurement_path, ut_path):
    m = json.loads(Path(measurement_path).read_text())
    return {'schema_version': 1, 'review_status': 'needs_source_review',
        'binding': {**{k: m.get(k) for k in ('task', 'case', 'arm', 'implementation_sha256', 'native_trace_sha256')},
                    'measurement_sha256': sha256(measurement_path)},
        'scope': 'complete_measured_call', 'semantic_name': None, 'semantics': {},
        'citations': {}, 'field_evidence': {}, 'unknown_fields': [],
        'native_operation': None, 'arguments': [],
        'launch_frame': {'path':str(Path(ut_path).resolve()),'function':m.get('timed_closure')},
        'expected_dispatches_per_call': m.get('counted_dispatches_per_call'),
        'parameter_basis': 'runtime_snapshot' if m.get('input_metadata') else 'reviewed_source',
        'runtime_input_metadata': m.get('input_metadata'),
        'sources': [source_evidence(ut_path, 'ut'), source_evidence(m['implementation_source'], 'implementation')]}


def normalize_dtype(value):
    value = str(value).lower().split('::')[-1].split('.')[-1]
    return {'half': 'float16', 'float': 'float32', 'double': 'float64', 'long': 'int64'}.get(value, value)


def source_citation(path, first_line, last_line):
    """Capture exact source lines; a citation still requires semantic review."""
    path = Path(path).resolve()
    lines = path.read_text().splitlines()
    if not 1 <= first_line <= last_line <= len(lines):
        raise ValueError('invalid source citation range')
    return {'path':str(path), 'first_line':first_line, 'last_line':last_line,
            'quote':'\n'.join(lines[first_line-1:last_line])}


def validate_field_evidence(context, runtime=None):
    """Require actual UT-source citations for fields not mechanically observed.

    Hash/line checks verify provenance, not arbitrary Python/C++ semantics. The
    reviewer must establish that the cited case/branch implies the stated value.
    Native model defaults, a symbol, or the sidecar itself are never UT evidence.
    """
    if context.get('unknown_fields'):
        raise ValueError('UT_FIELDS_UNKNOWN: do not fill missing metadata with model defaults')
    allowed = {str(Path(s['path']).resolve()) for s in context['sources']
               if s['role'] in ('ut', 'implementation', 'implementation_body', 'case_metadata', 'reference')}
    citations = context.get('citations', {})
    for key, cite in citations.items():
        if str(Path(cite['path']).resolve()) not in allowed:
            raise ValueError(f'UT citation is outside the reviewed source chain: {key}')
        actual = source_citation(cite['path'], cite['first_line'], cite['last_line'])
        if not cite.get('quote') or cite['quote'] != actual['quote']:
            raise ValueError(f'UT citation text mismatch: {key}')
    def require(field):
        evidence = context.get('field_evidence', {}).get(field, {})
        refs = evidence.get('citations')
        if not isinstance(refs, list) or not refs or any(ref not in citations for ref in refs) or not evidence.get('derivation', '').strip():
            raise ValueError(f'UT_FIELD_EVIDENCE_MISSING: {field}; cite the actual UT source/case and explain the value')
    require('semantic_name')
    if context.get('native_operation'):
        require('native_operation')
    for key in context['semantics']:
        require('semantics.' + key)
    for index, arg in enumerate(context['arguments']):
        prefix = f'arguments.{index}.'
        require(prefix + 'binding')  # argument name, position and payload/constant source
        if arg['kind'] == 'tensor':
            require(prefix + 'access')
        observed = runtime is not None and not arg['source'].startswith('literal:')
        if not observed:
            for field in ('shape', 'stride', 'dtype') if arg['kind'] == 'tensor' else ('value',):
                require(prefix + field)
        for field in ('alias_group', 'storage_offset'):
            if field in arg and not observed:
                require(prefix + field)


def snapshot_inputs(inputs, torch):
    """Read tensor descriptors, scalar configuration and alias groups, not contents."""
    result, storage_groups = {}, {}
    def visit(value, path):
        if torch.is_tensor(value):
            storage = value.untyped_storage()._cdata
            group = storage_groups.setdefault(storage, len(storage_groups))
            result[path] = {'kind': 'tensor', 'shape': list(value.shape), 'stride': list(value.stride()),
                'dtype': str(value.dtype), 'storage_offset': value.storage_offset(), 'alias_group': group}
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, f'{path}.{key}' if path else str(key))
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(child, f'{path}.{index}' if path else str(index))
        elif value is None or isinstance(value, (str, int, float, bool)):
            result[path] = {'kind': 'scalar', 'value': value}
    visit(inputs, '')
    return result


def validate_context(context, measurement, measurement_path):
    if context.get('schema_version') != 1 or context.get('review_status') != 'source_reviewed':
        raise ValueError('UT context must be source_reviewed; a draft or symbol guess is not a model mapping')
    if context.get('scope') != 'complete_measured_call' or measurement.get('legacy_runner'):
        raise ValueError('UT metadata bridge requires a verified complete measured call')
    binding = context['binding']
    for key in ('task', 'case', 'arm', 'implementation_sha256', 'native_trace_sha256'):
        if binding.get(key) is None or binding[key] != measurement.get(key):
            raise ValueError(f'UT context binding mismatch: {key}')
    if binding.get('measurement_sha256') != sha256(measurement_path):
        raise ValueError('UT context binding mismatch: measurement hash')
    if not context.get('semantic_name') or not context.get('semantics'):
        raise ValueError('UT semantic identity and variant semantics are required')
    sources = context.get('sources', [])
    if not {'ut', 'implementation'} <= {s.get('role') for s in sources}:
        raise ValueError('UT and implementation source evidence are required')
    for source in sources:
        if sha256(source['path']) != source['sha256']:
            raise ValueError(f'UT source changed: {source["path"]}')
    if not any(s['role'] == 'implementation' and
               Path(s['path']).resolve() == Path(measurement['implementation_source']).resolve() and
               s['sha256'] == measurement['implementation_sha256'] for s in sources):
        raise ValueError('UT context cites a different measured implementation')
    ut_sha = measurement.get('ut_source_sha256')
    if ut_sha and not any(s['role'] == 'ut' and s['sha256'] == ut_sha for s in sources):
        raise ValueError('UT context cites a different input-generating UT')
    counts = context.get('expected_dispatches_per_call')
    if not counts or counts != measurement.get('counted_dispatches_per_call'):
        raise ValueError('UT expected dispatches differ from measured complete-call scope')
    runtime = measurement.get('input_metadata')
    if context.get('parameter_basis') not in ('runtime_snapshot', 'reviewed_source'):
        raise ValueError('parameter_basis must identify runtime_snapshot or reviewed_source')
    if context['parameter_basis'] == 'runtime_snapshot' and not runtime:
        raise ValueError('UT runtime input snapshot is absent')
    if not context.get('arguments'):
        raise ValueError('ordered UT argument metadata is required')
    for arg in context['arguments']:
        if not arg.get('name') or not arg.get('source'):
            raise ValueError('every argument needs a name and UT payload source')
        if arg.get('kind') == 'tensor':
            if arg.get('access') not in ('read', 'write', 'read_write') or not arg.get('dtype'):
                raise ValueError('tensor dtype and access role are required')
            if not isinstance(arg.get('shape'), list) or not isinstance(arg.get('stride'), list) or len(arg['shape']) != len(arg['stride']):
                raise ValueError('tensor shape and stride must have equal rank')
            if not all(type(n) is int and n >= 0 for n in arg['shape'] + arg['stride']):
                raise ValueError('invalid tensor shape or stride')
        elif arg.get('kind') != 'scalar' or 'value' not in arg:
            raise ValueError('argument must describe a tensor or scalar')
        literal = arg['source'].startswith('literal:')
        if literal and (arg['kind'] != 'scalar' or json.loads(arg['source'][8:]) != arg['value']):
            raise ValueError('literal source must agree with its scalar value')
        if runtime is not None and not literal:
            actual = runtime.get(arg['source'])
            if not actual or actual['kind'] != arg['kind']:
                raise ValueError(f'UT runtime argument missing: {arg["source"]}')
            fields = ('shape', 'stride') if arg['kind'] == 'tensor' else ('value',)
            if any(actual[k] != arg[k] for k in fields) or (arg['kind'] == 'tensor' and normalize_dtype(actual['dtype']) != normalize_dtype(arg['dtype'])):
                raise ValueError(f'UT runtime argument mismatch: {arg["source"]}')
            for field in ('alias_group', 'storage_offset'):
                if field in arg and arg[field] != actual.get(field):
                    raise ValueError(f'UT runtime argument mismatch: {arg["source"]}.{field}')
    validate_field_evidence(context, runtime)
    return context


def event_arguments(context):
    args = context['arguments']
    return {'Input Dims': [a['shape'] if a['kind'] == 'tensor' else [] for a in args],
        'Input Strides': [a['stride'] if a['kind'] == 'tensor' else [] for a in args],
        'Input type': [a['dtype'] if a['kind'] == 'tensor' else 'Scalar' for a in args],
        'Concrete Inputs': ['' if a['kind'] == 'tensor' else str(a['value']) for a in args],
        'KRF metadata origin': 'source-reviewed UT; added for analysis, not captured CPU operator',
        'Pseudo op': True}


def enrich_eager_trace(trace, context, measurement):
    """Add semantic CPU ranges; leave every original event, flow and time intact."""
    if measurement.get('execution') != 'eager':
        raise ValueError('UT context bridge supports eager only; graph needs native capture/replay mapping')
    events = trace['traceEvents']
    ranges = sorted([e for e in events if e.get('cat') == 'user_annotation' and e.get('name', '').startswith('krf.seam:')], key=lambda e: e['ts'])
    if len(ranges) != measurement['measured_calls']:
        raise ValueError('UT context has no complete per-call trace ranges')
    if any(a['ts'] + a['dur'] > b['ts'] for a, b in zip(ranges, ranges[1:])):
        raise ValueError('overlapping measured call ranges cannot be bridged')
    additions, owned = [], set()
    for r in ranges:
        inside = lambda e: r['ts'] <= e.get('ts', -1) and e.get('ts', -1) + e.get('dur', 0) <= r['ts'] + r['dur']
        cpu = [e for e in events if e.get('cat') == 'cpu_op' and inside(e)]
        selected = [(i, e) for i, e in enumerate(events) if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset') and inside(e)]
        if Counter(e['name'] for _, e in selected) != Counter(context['expected_dispatches_per_call']):
            raise ValueError('UT complete-call dispatch mismatch in trace')
        frame = context.get('launch_frame', {})
        pattern = re.compile(re.escape(frame.get('path', '')) + r'\(\d+\): ' + re.escape(frame.get('function') or '') + '$')
        anchors = [e for e in events if e.get('cat') == 'python_function' and inside(e) and
                   e.get('pid') == r['pid'] and e.get('tid') == r['tid'] and pattern.fullmatch(e['name'])]
        if len(anchors) != 1:
            raise ValueError('UT launch_frame must match exactly one captured Python call per measured range')
        anchor = anchors[0]
        runtime = [e for e in events if e.get('cat') in ('cuda_runtime', 'hip_runtime', 'cuda_driver') and
                   e.get('pid') == r.get('pid') and e.get('tid') == r.get('tid') and
                   anchor['ts'] <= e['ts'] and e['ts'] + e.get('dur', 0) <= anchor['ts'] + anchor['dur']]
        correlations = {e.get('args', {}).get('correlation') for e in runtime} - {None}
        if any(e.get('args', {}).get('correlation') not in correlations for _, e in selected):
            raise ValueError('missing same-thread runtime-to-device correlation; context-only analysis required')
        device_correlations = {e.get('args', {}).get('correlation') for _, e in selected}
        launches = [e for e in runtime if e.get('args', {}).get('correlation') in device_correlations]
        # Host-only empty/transpose/as_strided events own no device work. Keep
        # them, but never wrap an existing CPU op that owns a measured launch.
        for op in cpu:
            if any(op.get('pid') == e.get('pid') and op.get('tid') == e.get('tid') and
                   op['ts'] <= e['ts'] and e['ts'] + e.get('dur', 0) <= op['ts'] + op.get('dur', 0)
                   for e in launches):
                raise ValueError('existing CPU operators own measured GPU work: preserve native mapping, do not wrap or replace it')
        owned.update(i for i, _ in selected)
        additions.append({'ph': 'X', 'cat': 'cpu_op', 'name': context['native_operation'],
            'pid': r['pid'], 'tid': r['tid'], 'ts': anchor['ts'], 'dur': anchor['dur'], 'args': event_arguments(context)})
    if len(owned) != sum(e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset') for e in events):
        raise ValueError('device work outside UT call ranges; refuse partial model attribution')
    # Put the synthetic enclosing event before its equal-start Python frame.
    # Native TL's stable timestamp sort then preserves the whole launch subtree.
    return {**copy.deepcopy(trace), 'traceEvents': additions + copy.deepcopy(events),
            'krf_ut_context': {'origin': 'source_reviewed_ut_metadata', 'native_operation': context['native_operation']}}, len(additions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement', required=True, type=Path)
    parser.add_argument('--ut', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; keep reviewed evidence and use a fresh path')
    result = draft_context(args.measurement, args.ut)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Draft saved. Review UT semantics and ordered arguments before setting source_reviewed.')


if __name__ == '__main__':
    main()
