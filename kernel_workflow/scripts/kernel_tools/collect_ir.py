#!/usr/bin/env python3
"""Copy one compilation's IR and metadata without mixing kernels or configurations."""
import argparse
import json
import shutil
import sys
from pathlib import Path

from amd_artifacts import assembly_kernel_name


def _groups(cache, kernel_name=''):
    groups = {}
    for path in sorted(cache.rglob('*')):
        if (path.is_file() and path.suffix in ('.ttgir', '.llir', '.amdgcn')
                and (not kernel_name or kernel_name in path.stem)):
            groups.setdefault((path.parent, path.stem), {})[path.suffix] = path
    return groups


CONSTEXPR_LOG = '_constexprs.jsonl'   # written into the cache by dump_ir.sh's compile hook (--all)


def _constexpr_log(cache):
    """Rows the compile hook logged: {kernel, num_warps, num_stages, constexprs}. [] if none."""
    rows = []
    try:
        for line in (cache / CONSTEXPR_LOG).read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    except OSError:
        pass
    return rows


def _constexprs_for(rows, meta, stem):
    """The constexpr set logged for this compilation. A row whose `hash` equals the metadata's
    `hash` (the compiled kernel's cache hash) is an exact match. Otherwise rows are matched on
    kernel name + num_warps + num_stages. Returns (constexprs or None, candidates): one distinct
    set -> that set; several -> None plus the candidate sets."""
    name = meta.get('name') or stem
    if meta.get('hash'):
        exact = [r.get('constexprs') for r in rows if r.get('hash') == meta['hash']]
        if exact:
            return exact[0], exact[:1]
    sets = []
    for r in rows:
        if (r.get('kernel') == name and r.get('num_warps') == meta.get('num_warps')
                and r.get('num_stages') == meta.get('num_stages')
                and r.get('constexprs') not in sets):
            sets.append(r.get('constexprs'))
    return (sets[0] if len(sets) == 1 else None), sets


def collect_all(cache, dest, variant, kernel_name=''):
    """Every matching compilation into its own <dest>_c<i>/ (same file names as a single dump),
    plus <dest>/ir_index.json. For multi-case benchmarks, where one command compiles several
    configurations: each is dumped instead of refusing; pick one by the index (num_warps,
    num_stages, shared, and the constexprs dump_ir.sh's compile hook logged)."""
    groups = _groups(cache, kernel_name)
    if not groups:
        raise ValueError(f'no compilation matching {kernel_name!r}; cache retained at {cache}')
    index = []
    logged = _constexpr_log(cache)
    for k, key in enumerate(sorted(groups, key=lambda g: str(g[0] / g[1]))):
        sub = dest.parent / f'{dest.name}_c{k}'
        sub.mkdir(parents=True, exist_ok=True)
        _collect_one(cache, sub, variant, key, groups[key])
        meta = {}
        mp = key[0] / (key[1] + '.json')
        try:
            meta = json.loads(mp.read_text())
        except (OSError, ValueError):
            pass
        ce, cands = _constexprs_for(logged, meta, key[1])
        row = {'dir': sub.name, 'kernel': meta.get('name') or key[1],
               'num_warps': meta.get('num_warps'), 'num_stages': meta.get('num_stages'),
               'shared': meta.get('shared'), 'constexprs': ce,
               'compilation': str(key[0].relative_to(cache) / key[1])}
        if ce is None:
            row['constexprs_candidates'] = cands
            row['constexprs_note'] = ('not logged (no Triton compile hook ran)' if not logged else
                                      'several constexpr sets share this kernel/num_warps/num_stages and '
                                      'no logged compile hash matched; tell them apart by the .ttgir '
                                      'tensor shapes')
        index.append(row)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'ir_index.json').write_text(json.dumps(index, indent=2) + '\n')
    print(f'  {len(index)} compilation(s) -> {dest.parent}/{dest.name}_c*/ (index: {dest}/ir_index.json)')


def collect(cache, dest, variant, kernel_name=''):
    groups = _groups(cache, kernel_name)
    if len(groups) != 1:
        candidates = ', '.join(str(parent.relative_to(cache) / name) for parent, name in groups)
        raise ValueError(f'expected one compilation, found {len(groups)} matching {kernel_name!r}. '
                         'Use --kernel-name and a command pinned to one configuration. '
                         f'Candidates: {candidates or "none"}; cache retained at {cache}. '
                         'Or pass --all to dump every compilation into its own directory.')
    key, artifacts = next(iter(groups.items()))
    _collect_one(cache, dest, variant, key, artifacts)


def _collect_one(cache, dest, variant, key, artifacts):
    parent, stem = key
    missing = [ext for ext in ('.ttgir', '.llir', '.amdgcn')
               if ext not in artifacts or artifacts[ext].stat().st_size == 0]
    if missing:
        raise ValueError(f'incomplete compilation {stem}: missing/empty {", ".join(missing)}')
    metadata_path = parent / (stem + '.json')
    if not metadata_path.is_file():
        raise ValueError(f'incomplete compilation {stem}: missing metadata')
    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict) or not isinstance(metadata.get('name'), str) or not metadata['name'].strip():
        raise ValueError(f'invalid compilation metadata: {metadata_path}')
    actual_name = assembly_kernel_name(artifacts['.amdgcn'].read_text(), metadata['name'])
    manifest = {'schema_version': 1, 'kernel_name': (metadata or {}).get('name') or stem,
                'identity_status': 'verified' if actual_name else 'unavailable',
                'compilation': str(parent.relative_to(cache) / stem), 'artifacts': {},
                'metadata': None}
    for ext, source in artifacts.items():
        target = dest / (variant + ext)
        shutil.copy2(source, target)
        manifest['artifacts'][ext[1:]] = target.name
        print(f'  + {target} ({source.name})')
    if metadata is not None:
        target = dest / ('meta_' + metadata_path.name)
        shutil.copy2(metadata_path, target)
        manifest['metadata'] = target.name
    (dest / 'ir_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache', type=Path)
    p.add_argument('dest', type=Path)
    p.add_argument('variant')
    p.add_argument('--kernel-name', default='')
    p.add_argument('--all', action='store_true',
                   help='dump every matching compilation into <dest>_c<i>/ instead of requiring one')
    args = p.parse_args()
    try:
        (collect_all if args.all else collect)(args.cache, args.dest, args.variant, args.kernel_name)
    except (ValueError, OSError) as e:
        print(f'E_IR_SELECTION: {e}', file=sys.stderr)
        return 3
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
