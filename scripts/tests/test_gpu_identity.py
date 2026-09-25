# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
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


def test_instinct_identity_stays_unknown_product() -> None:
    identity = gpu_identity.parse_rocminfo(
        _rocminfo("gfx950", "AMD Instinct MI355X", cu=256)
    )
    assert identity["gfx"] == "gfx950"
    assert identity["target"] == "unknown"
    assert identity["physical_cu_count"] == 256


def test_homogeneous_visible_agents_are_accepted() -> None:
    identity = gpu_identity.parse_rocminfo(
        _rocminfo("gfx1201", gpu_identity.R9700_MARKETING_NAME, count=2)
    )
    assert identity["target"] == "r9700"
    assert identity["visible_gpu_agents"] == 2


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


def test_missing_gpu_agent_fails_closed() -> None:
    with pytest.raises(gpu_identity.IdentityError, match="no non-gfx000"):
        gpu_identity.parse_rocminfo(_rocminfo("gfx000", "CPU"))


def test_missing_physical_cu_count_fails_closed() -> None:
    text = _rocminfo("gfx1201", gpu_identity.R9700_MARKETING_NAME).replace(
        "  Compute Unit:            64\n", ""
    )
    with pytest.raises(gpu_identity.IdentityError, match="physical Compute Unit"):
        gpu_identity.parse_rocminfo(text)
