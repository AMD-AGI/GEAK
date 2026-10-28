#!/usr/bin/env python3
"""Capture the environment behind one kernel-workflow validation.

The reusable learned card keeps ratios and principles. This file is the run-local lab notebook:
software versions/commits, GPU, workload/shape, measurement protocol, parity verdict and the exact
perf-knowledge decisions the planner acted on. Unknown values are written as `not_captured`, never
silently omitted or inferred from the requested authoring language.

Usage:

    printf '%s' "$METADATA_JSON" | python3 capture_validation_env.py \
      --output "$EVAL_DIR/validation_environment.yaml" \
      --workspace "$CANONICAL" --workflow-dir "$WORKFLOW_DIR" --gpu-ids 0 --metadata -
"""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_flydsl_compat import pin_task_stack  # noqa: E402  (sibling script, path-run)

NOT_CAPTURED = "not_captured"


def run(args, cwd=None):
    # A probe that cannot run is an unknown, not an error: this module also runs on
    # boxes with no ROCm and no git (CI), where `hipcc` or `git` simply is not there.
    # Letting OSError out turned "record what the environment was" into a crash.
    try:
        result = subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, check=False, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def git_probe(path):
    """Git identity for a path, or explicit unknowns when it is not in a checkout."""
    if not path:
        return {"root": NOT_CAPTURED, "commit": NOT_CAPTURED, "dirty": NOT_CAPTURED}
    root = run(["git", "-C", str(path), "rev-parse", "--show-toplevel"])
    if not root:
        return {"root": NOT_CAPTURED, "commit": NOT_CAPTURED, "dirty": NOT_CAPTURED}
    commit = run(["git", "-C", root, "rev-parse", "HEAD"]) or NOT_CAPTURED
    status = run(["git", "-C", root, "status", "--porcelain"])
    return {"root": root, "commit": commit, "dirty": bool(status)}


def package_probe(name):
    """Installed package version/module path without importing GPU code."""
    try:
        spec = importlib.util.find_spec(name)
        module_path = spec.origin if spec and spec.origin else NOT_CAPTURED
    except (ImportError, ModuleNotFoundError, ValueError):
        module_path = NOT_CAPTURED
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = NOT_CAPTURED
    return {"version": version, "module_path": module_path}


def rocm_probe():
    root = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    for rel in (".info/version", ".info/version-dev", "lib/.info/version"):
        path = root / rel
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return {"version": value, "root": str(root), "source": str(path)}
    version = run([str(root / "bin" / "hipcc"), "--version"])
    match = re.search(
        r"(?:HIP|ROCm)[^\n]*?(\d+\.\d+(?:\.\d+)?)", version, re.IGNORECASE,
    )
    return {
        "version": match.group(1) if match else NOT_CAPTURED,
        "root": str(root) if root.exists() else NOT_CAPTURED,
        "source": "hipcc --version" if version else NOT_CAPTURED,
    }


def package_git(package):
    module_path = package.get("module_path")
    if not module_path or module_path == NOT_CAPTURED:
        return {"root": NOT_CAPTURED, "commit": NOT_CAPTURED, "dirty": NOT_CAPTURED}
    return git_probe(Path(module_path).parent)


def flydsl_provider(module_path):
    if not module_path or module_path == NOT_CAPTURED:
        return NOT_CAPTURED
    normal = str(module_path).replace("\\", "/").lower()
    return "aiter_vendored_flydsl" if "/aiter/" in normal else "standalone_flydsl"


def decision_registry(perf_knowledge_dir):
    """IDs published by every catalogued decision/config source.

    Discovery is by the stable corpus layout rather than a GEMM filename list.  Adding a family under
    `corpus/decisions/` or `corpus/evidence/*_tuned_configs.yaml` must make its IDs resolvable without
    editing the workflow's validation code.
    """
    root = Path(str(perf_knowledge_dir or ""))
    corpus = root / "corpus"
    files = [
        *((path, r'^\s*- id:\s*["\']?([^"\'\s]+)')
          for path in sorted((corpus / "decisions").glob("*.yaml"))),
        *((path, r'^\s*- config_id:\s*["\']?([^"\'\s]+)')
          for path in sorted((corpus / "evidence").glob("*_tuned_configs.yaml"))),
    ]
    found = set()
    for path, pattern in files:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        found.update(re.findall(pattern, text, re.MULTILINE))
    return found


def build_manifest(metadata, workspace, workflow_dir, gpu_ids):
    # Probe the stack the BENCHMARK pins, not the box's ambient one. The oracle pins its roots
    # internally (geak_pythonpath.txt next to it), so a capture run from a plain shell records a
    # different stack than the one that produced the numbers in the same file. Measured: a GLM-5.2
    # MXFP4 MoE run whose oracle pins pyoverlay flydsl 0.3.4.1 + aiter_main@4faf8713 was recorded as
    # flydsl 0.1.5.dev504 + aiter@a6bb4993 -- the ambient /opt/venv and /sgl-workspace/aiter. A
    # provenance block naming the wrong commit is worse than `not_captured`: it reads as verified.
    stack_pinned = pin_task_stack([workspace])
    flydsl = package_probe("flydsl")
    aiter = package_probe("aiter")
    aiter_git = package_git(aiter)
    geak_git = git_probe(workflow_dir)
    workspace_git = git_probe(workspace)
    rocm = rocm_probe()

    device = str(metadata.get("device") or NOT_CAPTURED)
    gfx = str(metadata.get("gfx") or "")
    if not gfx:
        gfx = (re.search(r"gfx\d+", device, re.IGNORECASE) or [NOT_CAPTURED])[0]
    op_spec = metadata.get("op_spec") if isinstance(metadata.get("op_spec"), dict) else {}
    outcomes = metadata.get("decision_outcomes") or []
    registry = decision_registry(metadata.get("perf_knowledge_dir"))
    cited_ids = sorted({
        str(row.get("decision")) for row in outcomes
        if isinstance(row, dict) and row.get("decision")
    })

    manifest = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "kernel": metadata.get("kernel") or NOT_CAPTURED,
        "mode": metadata.get("mode") or NOT_CAPTURED,
        "requested_language": metadata.get("requested_language") or NOT_CAPTURED,
        "observed_language": metadata.get("observed_language") or NOT_CAPTURED,
        "flydsl_version": flydsl["version"],
        "rocm_version": rocm["version"],
        "aiter_commit": aiter_git["commit"],
        "geak_commit": geak_git["commit"],
        "workspace_commit": workspace_git["commit"],
        "provider": flydsl_provider(flydsl["module_path"]),
        "gpu": device,
        "gpu_ids": str(gpu_ids),
        "gfx": gfx,
        "op_spec": op_spec,
        "workload_spec_path": metadata.get("workload_spec_path") or NOT_CAPTURED,
        "measurement": metadata.get("measurement") or {},
        "validation": metadata.get("validation") or {},
        "decision_outcomes": outcomes,
        "decision_registry": {
            "perf_knowledge_dir": metadata.get("perf_knowledge_dir") or NOT_CAPTURED,
            "known_ids": len(registry),
            "cited_ids": cited_ids,
            "unresolved_refs": [ref for ref in cited_ids if ref not in registry],
        },
        "software_detail": {
            "stack_pinned": stack_pinned or NOT_CAPTURED,
            "flydsl_module": flydsl["module_path"],
            "aiter_version": aiter["version"],
            "aiter_module": aiter["module_path"],
            "aiter_root": aiter_git["root"],
            "aiter_dirty": aiter_git["dirty"],
            "geak_root": geak_git["root"],
            "geak_dirty": geak_git["dirty"],
            "workspace_root": workspace_git["root"],
            "workspace_dirty": workspace_git["dirty"],
            "rocm_root": rocm["root"],
            "rocm_version_source": rocm["source"],
        },
        "artifacts": {
            "report": metadata.get("report_path") or NOT_CAPTURED,
            "raw_logs": metadata.get("raw_logs") or NOT_CAPTURED,
        },
    }
    required = (
        "observed_language", "flydsl_version", "rocm_version", "aiter_commit",
        "geak_commit", "gpu", "gfx",
    )
    manifest["not_captured"] = [
        key for key in required if manifest.get(key) in (None, "", NOT_CAPTURED)
    ]
    return manifest


def dump_yaml(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except ImportError:
        # JSON is a YAML 1.2 subset, so the artifact remains a valid YAML document.
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--workflow-dir", required=True)
    parser.add_argument("--gpu-ids", default="")
    parser.add_argument("--metadata", default="-", help="JSON file, or - for stdin")
    args = parser.parse_args()

    if args.metadata == "-":
        metadata = json.load(sys.stdin)
    else:
        with open(args.metadata, encoding="utf-8") as handle:
            metadata = json.load(handle)
    if not isinstance(metadata, dict):
        parser.error("metadata must be a JSON object")

    output = Path(args.output)
    manifest = build_manifest(metadata, args.workspace, args.workflow_dir, args.gpu_ids)
    dump_yaml(manifest, output)
    print(json.dumps({
        "path": str(output),
        "not_captured": manifest["not_captured"],
        "decision_outcomes": len(manifest["decision_outcomes"]),
        "unresolved_decision_refs": manifest["decision_registry"]["unresolved_refs"],
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
