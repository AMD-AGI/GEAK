# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "gpu_identity.py"
SPEC = importlib.util.spec_from_file_location("gpu_identity", SCRIPT)
gpu_identity = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gpu_identity)


def _rocminfo(gfx: str, product: str, cu: int = 64, *, count: int = 1) -> str:
    agents = [
        """
*******
Agent 1
*******
  Name:                    gfx000
  Marketing Name:          AMD Ryzen
  Compute Unit:            32
"""
    ]
    for index in range(count):
        agents.append(
            f"""
*******
Agent {index + 2}
*******
  Name:                    {gfx}
  Marketing Name:          {product}
  Compute Unit:            {cu}
"""
        )
    return "".join(agents)


def test_exact_r9700_product_is_structured() -> None:
    identity = gpu_identity.parse_rocminfo(
        _rocminfo("gfx1201", gpu_identity.R9700_MARKETING_NAME)
    )
    assert identity == {
        "gfx": "gfx1201",
        "marketing_name": "AMD Radeon AI PRO R9700",
        "target": "r9700",
        "physical_cu_count": 64,
        "visible_gpu_agents": 1,
    }


@pytest.mark.parametrize(
    "product",
    [
        "AMD Radeon PRO gfx1201",
        "AMD Radeon AI PRO R9700 Engineering Sample",
        "R9700",
        "",
    ],
)
def test_gfx1201_does_not_imply_r9700(product: str) -> None:
    identity = gpu_identity.parse_rocminfo(_rocminfo("gfx1201", product))
    assert identity["gfx"] == "gfx1201"
    assert identity["target"] == "unknown"


@pytest.mark.parametrize("target", ["mi350x", "mi355x"])
def test_exact_instinct_identity_selects_product(target: str) -> None:
    identity = gpu_identity.parse_rocminfo(
        _rocminfo("gfx950", f"AMD Instinct {target.upper()}", cu=256)
    )
    assert identity["gfx"] == "gfx950"
    assert identity["target"] == target
    assert identity["physical_cu_count"] == 256


@pytest.mark.parametrize("gfx, product", [
    ("gfx950", ""),
    ("gfx950", "AMD Instinct MI350X / MI355X"),
    ("gfx950", "AMD Instinct MI355X Engineering Sample"),
    ("gfx942", "AMD Instinct MI355X"),
    ("gfx942", "AMD Instinct MI300X"),
])
def test_ambiguous_or_mismatched_instinct_product_stays_unknown(gfx, product) -> None:
    assert gpu_identity.parse_rocminfo(_rocminfo(gfx, product))["target"] == "unknown"


def test_homogeneous_visible_agents_are_accepted() -> None:
    identity = gpu_identity.parse_rocminfo(
        _rocminfo("gfx1201", gpu_identity.R9700_MARKETING_NAME, count=2)
    )
    assert identity["target"] == "r9700"
    assert identity["visible_gpu_agents"] == 2


def test_generic_integrated_gpu_is_ignored_when_discrete_gpu_exists() -> None:
    text = _rocminfo("gfx1201", gpu_identity.R9700_MARKETING_NAME)
    text += """
*******
Agent 3
*******
  Name:                    gfx1036
  Marketing Name:          AMD Radeon Graphics
  Compute Unit:            2
"""
    identity = gpu_identity.parse_rocminfo(text)
    assert identity["gfx"] == "gfx1201"
    assert identity["target"] == "r9700"
    assert identity["visible_gpu_agents"] == 1


def test_generic_integrated_gpu_alone_remains_visible() -> None:
    identity = gpu_identity.parse_rocminfo(
        _rocminfo("gfx1036", "AMD Radeon Graphics", cu=2)
    )
    assert identity["gfx"] == "gfx1036"
    assert identity["target"] == "unknown"


def test_nested_isa_name_does_not_replace_agent_gfx() -> None:
    text = _rocminfo("gfx1201", gpu_identity.R9700_MARKETING_NAME)
    text += """
  Isa
    Name:                    amdgcn-amd-amdhsa--gfx1201
"""
    identity = gpu_identity.parse_rocminfo(text)
    assert identity["gfx"] == "gfx1201"
    assert identity["target"] == "r9700"


def test_mixed_visible_products_fail_closed() -> None:
    text = _rocminfo("gfx1201", gpu_identity.R9700_MARKETING_NAME)
    text += """
*******
Agent 3
*******
  Name:                    gfx1201
  Marketing Name:          Another gfx1201 Product
  Compute Unit:            64
"""
    with pytest.raises(gpu_identity.IdentityError, match="mixed identities"):
        gpu_identity.parse_rocminfo(text)


def test_integrated_gpu_filter_does_not_hide_unknown_discrete_product() -> None:
    text = _rocminfo("gfx1201", gpu_identity.R9700_MARKETING_NAME)
    text += """
*******
Agent 3
*******
  Name:                    gfx1201
  Marketing Name:          Another gfx1201 Product
  Compute Unit:            64
*******
Agent 4
*******
  Name:                    gfx1036
  Marketing Name:          AMD Radeon Graphics
  Compute Unit:            2
"""
    with pytest.raises(gpu_identity.IdentityError, match="mixed identities"):
        gpu_identity.parse_rocminfo(text)


def test_missing_gpu_agent_fails_closed() -> None:
    with pytest.raises(gpu_identity.IdentityError, match="no non-gfx000"):
        gpu_identity.parse_rocminfo(_rocminfo("gfx000", "CPU"))


def test_missing_physical_cu_count_fails_closed() -> None:
    text = _rocminfo("gfx1201", gpu_identity.R9700_MARKETING_NAME).replace(
        "  Compute Unit:            64\n", ""
    )
    with pytest.raises(gpu_identity.IdentityError, match="physical Compute Unit"):
        gpu_identity.parse_rocminfo(text)


@pytest.fixture
def forbid_live_probe(monkeypatch):
    def unexpected_probe(_path):
        pytest.fail("captured input attempted a live GPU probe")

    monkeypatch.setattr(gpu_identity, "_read_rocminfo", unexpected_probe)


def test_authenticated_capture_uses_raw_bytes_offline(tmp_path, capsys, forbid_live_probe) -> None:
    raw = _rocminfo("gfx950", "AMD Instinct MI350X", cu=256).replace("\n", "\r\n").encode()
    capture = tmp_path / "rocminfo capture.txt"
    capture.write_bytes(raw)

    assert gpu_identity.main([
        "--input", str(capture), "--input-sha256", hashlib.sha256(raw).hexdigest(),
    ]) == 0
    result = capsys.readouterr()
    assert result.err == ""
    assert json.loads(result.out) == {
        "gfx": "gfx950", "marketing_name": "AMD Instinct MI350X", "target": "mi350x",
        "physical_cu_count": 256, "visible_gpu_agents": 1,
    }


def test_capture_hash_is_checked_before_decode_or_parse(
    tmp_path, capsys, monkeypatch, forbid_live_probe,
) -> None:
    raw = b"invalid utf-8: \xff\r\n"
    capture = tmp_path / "rocminfo.txt"
    capture.write_bytes(raw)

    def unexpected_parse(_text):
        pytest.fail("unauthenticated captured bytes reached the parser")

    monkeypatch.setattr(gpu_identity, "parse_rocminfo", unexpected_parse)
    normalized_digest = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    assert gpu_identity.main([
        "--input", str(capture), "--input-sha256", normalized_digest,
    ]) == 2
    result = capsys.readouterr()
    assert result.out == ""
    assert "captured rocminfo SHA-256 differs" in result.err


@pytest.mark.parametrize("raw", [
    None, b"not rocminfo", b"invalid utf-8: \xff",
    _rocminfo("gfx950", "AMD Instinct MI350X", cu=0).encode(),
])
def test_bad_capture_does_not_fall_back_to_live_probe(
    raw, tmp_path, capsys, forbid_live_probe,
) -> None:
    capture = tmp_path / "rocminfo.txt"
    if raw is not None:
        capture.write_bytes(raw)
    assert gpu_identity.main([
        "--input", str(capture), "--input-sha256", hashlib.sha256(raw or b"").hexdigest(),
    ]) == 2
    result = capsys.readouterr()
    assert result.out == ""
    assert result.err.startswith("E_GPU_IDENTITY:")


@pytest.mark.parametrize("argv", [
    ["--input-sha256", "ab" * 32],
    ["--stdin", "--input-sha256", "ab" * 32],
    ["--rocminfo-bin", "/unused/rocminfo", "--input-sha256", "ab" * 32],
    ["--input", "/unused/capture", "--input-sha256", "AB" * 32],
    ["--input", "/unused/capture", "--input-sha256", "ab" * 31],
    ["--input", "/unused/capture", "--input-sha256", "z" * 64],
])
def test_capture_digest_requires_input_and_valid_sha256(argv, capsys, forbid_live_probe) -> None:
    with pytest.raises(SystemExit) as error:
        gpu_identity.main(argv)
    assert error.value.code == 2
    result = capsys.readouterr()
    assert result.out == ""
    assert "--input-sha256 requires --input and a lowercase 64-digit SHA-256" in result.err
