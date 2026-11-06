#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared product resolution and run hardware context; no GPU performance model.

Probe: hardware_context.py --output RUN/hardware_context.json
Replay: hardware_context.py --rocminfo-input captured.txt --output context.json
Check: hardware_context.py --check context.json [--live] [--reference FILE]

A native reference may carry FILE.hardware-context.json containing
{"hardware_context": <context captured with the reference>, "reference_sha256": ...}.
This sidecar attests device scope, never replaces native precision/model audits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "kernel_workflow/scripts/kernel_tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "kernel_workflow/scripts/kernel_tools"))
import _hwdata  # noqa: E402 (shared repository-local module)


def catalog():
    path = _hwdata.find("sku.json")
    if path is None:
        raise FileNotFoundError("shared SKU catalog unavailable")
    raw = path.read_bytes()
    return json.loads(raw)["skus"], hashlib.sha256(raw).hexdigest()


def products():
    """Legacy identity allowlist, derived from the shared registry."""
    rows, _ = catalog()
    return {r["product"]: (r["identity_target"], r["arch"], r["cus"])
            for r in rows.values() if r.get("identity_target")}


def build(identity):
    rows, digest = catalog()
    gfx = str(identity.get("gfx") or identity.get("device_gfx") or "").lower()
    name = str(identity.get("marketing_name") or identity.get("device_name") or "").strip()
    cu = identity.get("physical_cu_count")
    if not re.fullmatch(r"gfx[0-9a-f]+", gfx) or type(cu) is not int or cu <= 0:
        raise ValueError("hardware context needs an observed gfx and positive visible CU count")
    candidates = [(k, r) for k, r in rows.items() if r["product"] == name]
    key, row = candidates[0] if len(candidates) == 1 else (None, None)
    matched = bool(row and row["arch"] == gfx and row["cus"] == cu)
    status = "matched" if matched else "conflict" if row else "unknown"
    # These are resources exposed by one agent, not proof of whole-device scope.
    obs = {"gfx": gfx, "marketing_name": name, "physical_cu_count": cu}
    for field in ("visible_gpu_agents", "compute_partition", "memory_partition", "memory_bytes", "agents"):
        if identity.get(field) is not None:
            obs[field] = identity[field]
    for field in ("visible_gpu_agents", "memory_bytes"):
        if field in obs and (type(obs[field]) is not int or obs[field] <= 0):
            raise ValueError(f"{field} must be a positive integer observation")
    if (obs.get("compute_partition") not in (None, "SPX") or
            obs.get("memory_partition") not in (None, "NPS1")):
        matched, status = False, "conflict"
    sku = key.lower() if matched else "unknown"
    declared = identity.get("sku") or identity.get("device_sku")
    if declared and declared != "unknown" and str(declared).lower() != sku:
        raise ValueError("declared SKU conflicts with observed product/ISA/CU scope")
    return {"schema_version": 1, "observed": obs, "sku": sku,
            "status": status, "candidate_sku": key.lower() if key else None,
            "catalog_sha256": digest, "specification": row if matched else None,
            "analysis_role": "planning", "may_gate": False}


def validate(context):
    if not isinstance(context, dict) or context.get("schema_version") != 1:
        raise ValueError("unsupported hardware context schema")
    expected = build(context["observed"])
    for field in expected:
        if context.get(field) != expected[field]:
            raise ValueError(f"stale or inconsistent hardware context field: {field}")
    return context


def load(value):
    return validate(json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value)


def same_device_scope(left, right, *, allow_enrichment=False):
    """Pool members may differ in device ID/count; per-agent resources must agree."""
    left, right = validate(left), validate(right)
    fields = ("gfx", "marketing_name", "physical_cu_count", "compute_partition", "memory_partition", "memory_bytes")
    return (left["sku"] == right["sku"] and
            all((allow_enrichment and left["observed"].get(k) is None) or
                left["observed"].get(k) == right["observed"].get(k) for k in fields))


def planning_peaks(context):
    c = load(context)
    r = c["specification"]
    if c["status"] != "matched" or not r or r["geak_support"] != "supported":
        return None
    # Native TraceLens/rocprof references are never sourced from this function.
    return {"gfx": r["arch"], "sku": c["sku"], "product": c["sku"], "cu": c["observed"]["physical_cu_count"],
            "hbm_bw_bytes_s": (r.get("datasheet_hbm_tb_s") or r["peak_hbm_tb_s"]) * 1e12,
            "flops": {k: v * 1e12 for k, v in r["peak_tflops"].items()},
            "source": "sku_specification", "confidence": "high" if r["basis"] == "datasheet" else "low",
            "basis": r["basis"], "catalog_sha256": c["catalog_sha256"],
            "source_conflicts": r.get("source_conflicts", {}),
            "analysis_role": "planning", "may_gate": False}


def check_reference(context, reference):
    c = load(context)
    reference = Path(reference)
    binding = json.loads(Path(str(reference) + ".hardware-context.json").read_text())
    if c["status"] != "matched" or not same_device_scope(c, binding["hardware_context"]):
        raise ValueError("native reference hardware scope does not match this run")
    if binding["reference_sha256"] != hashlib.sha256(reference.read_bytes()).hexdigest():
        raise ValueError("native reference changed since its hardware binding")


def check_native_arch(context, path):
    c = load(context)
    data = json.loads(Path(path).read_text())
    if c["status"] != "matched" or str(data.get("name", "")).lower() != c["sku"]:
        raise ValueError("TraceLens arch name does not match the observed SKU; select an explicit native reference")
    # Preserve all native bandwidth, precision and calibration values verbatim.


def record_measurement(context, path):
    c = load(context)
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get("status") != "OK" or data.get("environment", {}).get("gpu_name") != c["observed"]["marketing_name"]:
        raise ValueError("successful native measurement on the observed device is required")
    bind_reference(c, path)


def bind_reference(context, path):
    """Producer-side provenance only; native consumers still audit the calibration."""
    c, path = load(context), Path(path)
    if c["status"] != "matched":
        raise ValueError("binding native references requires a matched product/resource scope")
    binding = {"hardware_context": c, "reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    sidecar = Path(str(path) + ".hardware-context.json")
    if sidecar.exists():
        check_reference(c, path)  # Never relabel a different or modified calibration.
        return
    sidecar.write_text(json.dumps(binding, indent=2) + "\n")


def probe(input_path=None):
    import gpu_identity
    text = Path(input_path).read_text() if input_path else gpu_identity._read_rocminfo(None)
    identity = gpu_identity.parse_rocminfo(text)
    agents, sizes = [], set()
    for block in re.split(r"^\s*Agent\s+\d+\s*$", text, flags=re.M):
        fields = gpu_identity._agent_fields(block.splitlines())
        if (fields.get("name"), fields.get("marketing name")) != (identity["gfx"], identity["marketing_name"]):
            continue
        agents.append({k: fields[k] for k in ("uuid", "bdfid", "chip id", "node") if k in fields})
        # Multiple GLOBAL pools expose the same memory; never add their sizes.
        pools = re.findall(r"Segment:\s*GLOBAL[^\n]*\n\s*Size:\s*(\d+)[^\n]*\bKB", block)
        if pools:
            sizes.add(max(map(int, pools)) * 1024)
    if len(sizes) > 1:
        raise ValueError("visible GPU agents have different memory capacity")
    if sizes:
        identity["memory_bytes"] = sizes.pop()
    if agents:
        identity["agents"] = agents
    return build(identity)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rocminfo-input")
    ap.add_argument("--output")
    ap.add_argument("--check")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--reference")
    ap.add_argument("--native-arch")
    ap.add_argument("--record-measurement")
    ap.add_argument("--bind-reference", help="producer only: bind a freshly collected native reference; requires --live")
    a = ap.parse_args()
    try:
        c = load(a.check) if a.check else probe(a.rocminfo_input)
        if a.live and not same_device_scope(c, probe(a.rocminfo_input)):
            raise ValueError("live GPU differs from the run hardware context")
        if a.reference:
            check_reference(c, a.reference)
        if a.native_arch:
            check_native_arch(c, a.native_arch)
        if a.record_measurement:
            if not a.live:
                raise ValueError("--record-measurement requires --live")
            record_measurement(c, a.record_measurement)
        if a.bind_reference:
            if not a.live:
                raise ValueError("--bind-reference requires --live on the calibration producer")
            bind_reference(c, a.bind_reference)
        result = json.dumps(c, indent=2) + "\n"
        if a.output:
            path = Path(a.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and not same_device_scope(c, load(path)):
                raise ValueError("existing run hardware context belongs to another device scope")
            temp = path.with_name(path.name + f".{os.getpid()}.tmp")
            temp.write_text(result)
            temp.replace(path)
        print(result, end="")
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as e:
        print(f"E_HARDWARE_CONTEXT: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
