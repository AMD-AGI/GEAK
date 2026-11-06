"""Validate kernel identities before joining AMD assembly and launch metadata."""
import json
import re
from pathlib import Path


def assembly_kernel_name(text, expected=None):
    """Return an observable kernel name, rejecting conflicts and multi-kernel input."""
    descriptors = re.findall(r'^\s*\.amdhsa_kernel\s+(\S+)', text, re.MULTILINE)
    # Only kernel-level names: AMDHSA argument records may also contain .name.
    metadata_names = []
    kernel_entries = 0
    for block in re.findall(r'^amdhsa\.kernels:[ \t]*\n((?:[ \t]+[^\n]*\n|\n)*)',
                            text + '\n', re.MULTILINE):
        names = re.findall(r'^([ \t]*(?:-[ \t]+)?)\.name:[ \t]*(\S+)', block, re.MULTILINE)
        entries = re.findall(r'^([ \t]*)-[ \t]+\.', block, re.MULTILINE)
        if entries:
            depth = min(len(indent) for indent in entries) + 2
            kernel_entries += sum(len(indent) + 2 == depth for indent in entries)
            metadata_names.extend(name for indent, name in names if len(indent) == depth)
    if len(descriptors) > 1 or kernel_entries > 1 or len(metadata_names) > 1:
        raise ValueError('resource analysis requires one kernel per assembly')
    names = {name.strip('\"\'') for name in descriptors + metadata_names}
    if len(names) > 1:
        raise ValueError('kernel identity mismatch within assembly')
    actual = next(iter(names), None)
    if expected is not None:
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError('invalid kernel name in compilation metadata')
        if actual is not None and actual != expected:
            raise ValueError(f'kernel identity mismatch: assembly={actual!r}, metadata={expected!r}')
    return actual


def inspect_artifact(path):
    """Read a manifest-bound artifact without trusting the manifest to rename its kernel.

    Files may be renamed; kernel symbols may not. A missing assembly identity is
    represented by None, so callers cannot attach launch/LDS data on a guess.
    """
    path = Path(path)
    text = path.read_text(errors='replace')
    actual = assembly_kernel_name(text)
    manifest = path.parent / 'ir_manifest.json'
    if not manifest.is_file():
        # Raw compiler caches often have a same-stem JSON beside the assembly.
        sidecar = path.with_suffix('.json')
        metadata = None
        if sidecar.is_file():
            data = json.loads(sidecar.read_text())
            if isinstance(data, dict) and 'name' in data:
                if not isinstance(data['name'], str) or not data['name'].strip():
                    raise ValueError('invalid kernel name in compilation metadata')
                assembly_kernel_name(text, data['name'])
                metadata = data if actual else None
        return actual, metadata, False
    data = json.loads(manifest.read_text())
    if not isinstance(data, dict) or not isinstance(data.get('artifacts'), dict):
        raise ValueError(f'invalid IR manifest: {manifest}')  # noqa: TRY004 -- invalid serialized input
    artifacts = data['artifacts']
    if path.name not in artifacts.values() and path.with_suffix('.amdgcn').name != artifacts.get('amdgcn'):
        return actual, None, False
    declared = data.get('kernel_name')
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError('invalid kernel name in IR manifest')
    actual = assembly_kernel_name(text, declared)
    filename = data.get('metadata')
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ValueError(f'invalid metadata path in IR manifest: {manifest}')
    metadata = json.loads((path.parent / filename).read_text())
    if not isinstance(metadata, dict) or metadata.get('name') != declared:
        raise ValueError('kernel identity mismatch between manifest and metadata')
    return actual, metadata if actual else None, True
