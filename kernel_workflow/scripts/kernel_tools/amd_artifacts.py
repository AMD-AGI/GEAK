"""Validate kernel identities before joining AMD assembly and launch metadata."""
import json
import re
from pathlib import Path


def _kernel_parts(lines):
    """Separate function bodies, descriptors and YAML records before selecting a kernel.

    LLVM may put descriptors after ALL functions. Never use descriptor positions as
    body boundaries in that layout. YAML argument names are not kernel identities.
    """
    text = "\n".join(lines)
    descriptors = {}
    for m in re.finditer(r"^\s*\.amdhsa_kernel\s+(\S+)[^\n]*\n.*?^\s*\.end_amdhsa_kernel[^\n]*",
                         text, re.MULTILINE | re.DOTALL):
        name = m.group(1)
        if name in descriptors:
            raise ValueError(f"duplicate kernel descriptor: {name}")
        descriptors[name] = m.group(0).splitlines()
    metadata = {}
    for block in re.findall(r"^amdhsa\.kernels:[ \t]*\n((?:[ \t]+[^\n]*\n|\n)*)", text + "\n", re.MULTILINE):
        entries = list(re.finditer(r"^([ \t]*)-[ \t]+\.", block, re.MULTILINE))
        if not entries:
            continue
        depth = min(len(m.group(1)) for m in entries)
        entries = [m for m in entries if len(m.group(1)) == depth]
        for i, m in enumerate(entries):
            entry = block[m.start():entries[i+1].start() if i+1 < len(entries) else len(block)]
            name_match = re.search(rf"^(?:[ \t]{{{depth+2}}}|[ \t]{{{depth}}}- )\.name:[ \t]*(\S+)", entry, re.MULTILINE)
            if name_match:
                name = name_match.group(1).strip("\"'")
                if name in metadata:
                    raise ValueError(f"duplicate kernel metadata: {name}")
                metadata[name] = entry.splitlines()
    # Remove descriptors and the module YAML tail from body discovery.
    body = re.sub(r"^\s*\.amdhsa_kernel\s+[^\n]*\n.*?^\s*\.end_amdhsa_kernel[^\n]*", "",
                  text, flags=re.MULTILINE | re.DOTALL).split(".amdgpu_metadata", 1)[0].splitlines()
    anchors = []
    for i, ln in enumerate(body):
        m = re.match(r"^\s*\.type\s+([\w.$@]+)\s*,\s*@function", ln)
        if m:
            anchors.append((m.group(1), i))
    if not anchors:
        for i, ln in enumerate(body):
            m = re.match(r"^\s*[0-9a-fA-F]+\s+<([\w.$@]+)>:\s*$", ln)
            if m and not m.group(1).startswith(".L"):
                anchors.append((m.group(1), i))
    if not anchors and descriptors:
        for i, ln in enumerate(body):
            m = re.match(r"^([\w.$@]+):", ln)
            if m and m.group(1) in descriptors:
                anchors.append((m.group(1), i))
    spans = [(n, i, anchors[k+1][1] if k+1 < len(anchors) else len(body))
             for k, (n, i) in enumerate(anchors)]
    if descriptors and metadata and set(descriptors) != set(metadata):
        raise ValueError("kernel identity mismatch between descriptors and module metadata")
    if metadata and spans and not set(metadata).issubset(n for n, _, _ in spans):
        raise ValueError("kernel identity mismatch between code and module metadata")
    if descriptors:
        spans = [item for item in spans if item[0] in descriptors]
    return body, spans, descriptors, metadata


def kernels_in_asm(lines):
    """Named function/kernel spans; indices refer to the original input lines."""
    _, spans, _, _ = _kernel_parts(lines)
    # Compatibility API: callers historically used this for enumeration only.
    names = [n for n, _, _ in spans]
    anchors = []
    for n in names:
        pattern = re.compile(r"(?:\.type\s+" + re.escape(n) + r"\s*,|<" + re.escape(n) + r">:|^" + re.escape(n) + r":)")
        anchors.append((n, next(i for i, ln in enumerate(lines) if pattern.search(ln))))
    return [(n, i, anchors[k+1][1] if k+1 < len(anchors) else len(lines))
            for k, (n, i) in enumerate(anchors)]


def select_kernel(lines, want=None):
    """Return a single body with ONLY its descriptor/YAML and the module target."""
    body, spans, descriptors, metadata = _kernel_parts(lines)
    if not spans:
        if want:
            raise SystemExit(f"--kernel {want!r}: this dump names no selectable kernel")
        assembly_kernel_name("\n".join(lines))  # refuse multi-kernel metadata-only input
        return lines, None
    selected = spans
    if want:
        selected = [s for s in spans if s[0] == want] or [s for s in spans if want in s[0]]
    if len(selected) != 1:
        raise SystemExit(f"select one kernel with --kernel; matches={len(selected)}, "
                         f"candidates: {', '.join(n for n, _, _ in spans)}")
    name, start, end = selected[0]
    target = [ln for ln in body[:start] if '.amdgcn_target' in ln]
    chosen = target + body[start:end] + descriptors.get(name, [])
    if name in metadata:
        chosen += ['.amdgpu_metadata', 'amdhsa.kernels:'] + metadata[name] + ['.end_amdgpu_metadata']
    assembly_kernel_name("\n".join(chosen), name)
    return chosen, name


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


def inspect_artifact(path, kernel=None):
    """Read a manifest-bound artifact without trusting the manifest to rename its kernel.

    Files may be renamed; kernel symbols may not. A missing assembly identity is
    represented by None, so callers cannot attach launch/LDS data on a guess.
    """
    path = Path(path)
    text = path.read_text(errors='replace')
    if kernel is not None:
        lines, selected = select_kernel(text.splitlines(), kernel)
        text = "\n".join(lines)
        actual = assembly_kernel_name(text, selected) or selected
    else:
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
                if actual and actual != data['name']:
                    raise ValueError('kernel identity mismatch between selected code and metadata')
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
    if actual and actual != declared:
        raise ValueError("kernel identity mismatch between selected code and manifest")
    actual = assembly_kernel_name(text, declared) or actual
    filename = data.get('metadata')
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ValueError(f'invalid metadata path in IR manifest: {manifest}')
    metadata = json.loads((path.parent / filename).read_text())
    if not isinstance(metadata, dict) or metadata.get('name') != declared:
        raise ValueError('kernel identity mismatch between manifest and metadata')
    return actual, metadata if actual else None, True
