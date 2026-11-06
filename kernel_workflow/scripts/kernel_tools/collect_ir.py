#!/usr/bin/env python3
"""Copy one compilation's IR and metadata without mixing kernels or configurations."""
import argparse
import json
import shutil
import sys
from pathlib import Path

from amd_artifacts import assembly_kernel_name


def collect(cache, dest, variant, kernel_name=''):
    groups = {}
    for path in sorted(cache.rglob('*')):
        if (path.is_file() and path.suffix in ('.ttgir', '.llir', '.amdgcn')
                and (not kernel_name or kernel_name in path.stem)):
            groups.setdefault((path.parent, path.stem), {})[path.suffix] = path
    if len(groups) != 1:
        candidates = ', '.join(str(parent.relative_to(cache) / name) for parent, name in groups)
        raise ValueError(f'expected one compilation, found {len(groups)} matching {kernel_name!r}. '
                         'Use --kernel-name and a command pinned to one configuration. '
                         f'Candidates: {candidates or "none"}; cache retained at {cache}')
    (parent, stem), artifacts = next(iter(groups.items()))
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
    args = p.parse_args()
    try:
        collect(args.cache, args.dest, args.variant, args.kernel_name)
    except (ValueError, OSError) as e:
        print(f'E_IR_SELECTION: {e}', file=sys.stderr)
        return 3
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
