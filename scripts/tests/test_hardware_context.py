# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise observed identity -> specification -> legacy/native consumer boundaries."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import hardware_context as hc  # noqa: E402 (shared repository-local module)


def identity(name="MI308X", cu=80, gfx="gfx942"):
    return {"gfx": gfx, "marketing_name": "AMD Instinct " + name, "physical_cu_count": cu}



@pytest.mark.parametrize("name,cu,sku,peak", [
    ("MI300X", 304, "mi300x", 1307.4), ("MI308X", 80, "mi308x", 344),
    ("MI300X", 80, "unknown", None), ("MI300X", 38, "unknown", None),
    ("MI300X VF", 304, "unknown", None),
])
def test_product_and_scope_reach_actual_resolver(name, cu, sku, peak):
    c = hc.build(identity(name, cu))
    assert c["sku"] == sku
    p = hc.planning_peaks(c)
    if peak is None:
        assert p is None
    else:
        assert p["cu"] == cu
        assert p["flops"]["fp16"] / 1e12 == peak
        assert p["sku"] == sku and p["may_gate"] is False
        if sku == "mi308x":
            assert p["confidence"] == "low" and p["basis"] == "orc-derived"


def test_same_arch_different_product_and_unknown_scope():
    p = hc.planning_peaks(hc.build(identity("MI350X", 256, "gfx950")))
    assert p["flops"]["fp16"] == 2300e12
    assert hc.planning_peaks(hc.build(identity("MI300X", 80))) is None


def test_stale_catalog_and_forged_peaks_are_not_consumed():
    c = hc.build(identity())
    c["catalog_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="catalog_sha256"):
        hc.load(c)
    c = hc.build(identity())
    c["specification"]["peak_tflops"]["fp16"] = 9999
    with pytest.raises(ValueError, match="specification"):
        hc.load(c)


def test_declared_sku_and_partition_conflicts_are_refused():
    with pytest.raises(ValueError, match="declared SKU"):
        hc.build(dict(identity(), sku="mi300x"))
    assert hc.build(dict(identity("MI300X", 304), compute_partition="CPX"))["status"] == "conflict"
    assert hc.build(dict(identity("MI300X", 304), memory_partition="NPS4"))["status"] == "conflict"


def test_cli_file_roundtrip_and_resume_conflict(tmp_path):
    raw = tmp_path / "rocminfo.txt"
    raw.write_text("Agent 1\n Name: gfx942\n Marketing Name: AMD Instinct MI308X\n Compute Unit: 80\n")
    output = tmp_path / "hardware_context.json"
    cmd = [sys.executable, str(ROOT / "scripts/hardware_context.py")]
    r = subprocess.run(cmd + ["--rocminfo-input", str(raw), "--output", str(output)], cwd="/tmp", capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert hc.load(output)["sku"] == "mi308x"
    r = subprocess.run(cmd + ["--check", str(output)], capture_output=True, text=True)
    assert r.returncode == 0 and json.loads(r.stdout) == hc.load(output)
    raw.write_text(raw.read_text().replace("MI308X", "MI300X").replace("80", "304"))
    r = subprocess.run(cmd + ["--rocminfo-input", str(raw), "--output", str(output)], capture_output=True, text=True)
    assert r.returncode == 2 and "another device scope" in r.stderr
    assert hc.load(output)["sku"] == "mi308x"


def test_pool_capacity_is_not_summed_or_guessed(tmp_path):
    p = tmp_path / "rocminfo.txt"
    p.write_text("Agent 1\n Name: gfx950\n Marketing Name: AMD Instinct MI355X\n Compute Unit: 256\n Uuid: GPU-test\n BDFID: 123\n" +
                 " Segment: GLOBAL; FLAGS: COARSE GRAINED\n Size: 4096(0x1000) KB\n" * 2 +
                 " Segment: GROUP\n Size: 160 KB\n")
    c = hc.probe(p)
    assert c["observed"]["memory_bytes"] == 4096 * 1024
    assert c["observed"]["agents"] == [{"uuid": "GPU-test", "bdfid": "123"}]


def test_native_reference_scope_and_hash_without_peak_rewrite(tmp_path):
    reference = tmp_path / "roofline.csv"
    reference.write_text("device,HBMBw,MFMABF16Flops\n0,4200,230000\n")
    c = hc.build(identity())
    sidecar = Path(str(reference) + ".hardware-context.json")
    sidecar.write_text(json.dumps({"hardware_context": c, "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}))
    before = reference.read_bytes()
    hc.check_reference(c, reference)
    assert reference.read_bytes() == before
    with pytest.raises(ValueError, match="scope"):
        hc.check_reference(hc.build(identity("MI300X", 304)), reference)
    reference.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        hc.check_reference(c, reference)
    with pytest.raises(ValueError, match="changed"):
        hc.bind_reference(c, reference)


def test_native_arch_uses_exact_product_not_isa(tmp_path):
    p = tmp_path / "arch.json"
    p.write_text(json.dumps({"name": "MI300X", "max_achievable_tflops": {"matrix_bf16": 708}}))
    original = p.read_bytes()
    hc.check_native_arch(hc.build(identity("MI300X", 304)), p)
    assert p.read_bytes() == original
    with pytest.raises(ValueError, match="TraceLens arch"):
        hc.check_native_arch(hc.build(identity()), p)


def test_workflow_contract_is_synchronized():
    subprocess.run([sys.executable, str(ROOT / "scripts/sync_hardware_context_workflow.py"), "--check"], check=True)


@pytest.mark.parametrize('bad', [True, 80.5, '80', -1])
def test_invalid_resource_types_are_not_silently_rounded(bad):
    with pytest.raises(ValueError):
        hc.build(identity(cu=bad))


@pytest.mark.parametrize('name,cu,extra,ok', [
    ('MI308X',80,[],True), ('MI308X',80,['--arch','gfx942'],True),
    ('MI308X',80,['--sku','MI300X'],False), ('MI308X',80,['--cus','304'],False),
    ('MI300X',80,['--arch','gfx942'],False), ('MI300X',38,[],False),
])
def test_local_calculator_uses_same_context(tmp_path, name, cu, extra, ok):
    p = tmp_path / 'hardware_context.json'
    p.write_text(json.dumps(hc.build(identity(name,cu))))
    env = dict(os.environ, GEAK_HARDWARE_CONTEXT=str(p))
    result = subprocess.run([sys.executable, str(ROOT / 'kernel_workflow/scripts/kernel_tools/calc_perf.py'),
        'roofline', '--M','1024','--N','1024','--K','1024','--json', *extra], env=env, capture_output=True, text=True)
    assert (result.returncode == 0) == ok, result.stderr
    if ok:
        data = json.loads(result.stdout)
        assert data['sku'] == 'MI308X' and data['may_gate'] is False
        assert data['hardware_context_sha256'] == hc.catalog()[1]
