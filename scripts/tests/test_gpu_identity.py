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
        "sku": "r9700",
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
    assert identity["sku"] == "unknown"


@pytest.mark.parametrize(
    ("gfx", "product", "cu", "sku"),
    [
        ("gfx950", "AMD Instinct MI355X", 256, "mi355x"),
        ("gfx950", "AMD Instinct MI350X", 256, "mi350x"),
        ("gfx942", "AMD Instinct MI300X", 304, "mi300x"),
        ("gfx942", "AMD Instinct MI325X", 304, "mi325x"),
        ("gfx942", "AMD Instinct MI308X", 80, "mi308x"),
    ],
)
def test_instinct_marketing_names_map_to_sku_not_target(
    gfx: str, product: str, cu: int, sku: str
) -> None:
    """`target` is the r9700|unknown product-pin contract and must not widen; the Instinct
    mapping lives in the separate `sku` field."""
    identity = gpu_identity.parse_rocminfo(_rocminfo(gfx, product, cu=cu))
    assert identity == {
        "gfx": gfx,
        "marketing_name": product,
        "target": "unknown",
        "sku": sku,
        "physical_cu_count": cu,
        "visible_gpu_agents": 1,
    }


@pytest.mark.parametrize(
    ("gfx", "product", "cu"),
    [
        # a partitioned device (CPX: one XCD per agent) is not the full product
        ("gfx942", "AMD Instinct MI300X", 38),
        ("gfx950", "AMD Instinct MI355X", 32),
        # the name must agree with the ISA
        ("gfx942", "AMD Instinct MI355X", 256),
        ("gfx950", "AMD Instinct MI300X", 304),
        # exact names only: no VF / suffix / prefix matching, no unlisted Instinct parts
        ("gfx942", "AMD Instinct MI300X VF", 304),
        ("gfx942", "Instinct MI300X", 304),
        ("gfx90a", "AMD Instinct MI250X", 110),
        ("gfx950", "", 256),
    ],
)
def test_instinct_identity_without_exact_product_stays_unknown(
    gfx: str, product: str, cu: int
) -> None:
    identity = gpu_identity.parse_rocminfo(_rocminfo(gfx, product, cu=cu))
    assert identity["gfx"] == gfx
    assert identity["target"] == "unknown"
    assert identity["sku"] == "unknown"
    assert identity["physical_cu_count"] == cu


def test_target_contract_is_unchanged() -> None:
    """Every identity ever emitted has target in exactly {r9700, unknown}."""
    for gfx, product, cu in (("gfx950", "AMD Instinct MI355X", 256),
                             ("gfx942", "AMD Instinct MI325X", 304),
                             ("gfx1201", gpu_identity.R9700_MARKETING_NAME, 64),
                             ("gfx1201", "Another gfx1201 Product", 64)):
        identity = gpu_identity.parse_rocminfo(_rocminfo(gfx, product, cu=cu))
        assert identity["target"] in {"r9700", "unknown"}
        assert identity["target"] == ("r9700" if identity["sku"] == "r9700" else "unknown")


def test_instinct_skus_agree_with_sku_table() -> None:
    """Every sku value names exactly one supported sku.json row with the same arch and
    full-device CU count (perf_knowledge/hardware/data/sku.json is the single SKU source)."""
    import json

    sku_json = SCRIPT.parents[1] / "perf_knowledge" / "hardware" / "data" / "sku.json"
    skus = json.loads(sku_json.read_text(encoding="utf-8"))["skus"]
    by_target = {
        row["identity_target"]: (name, row)
        for name, row in skus.items()
        if row.get("identity_target")
    }
    expected = {t for t, _, _ in gpu_identity.INSTINCT_PRODUCTS.values()} | {"r9700"}
    assert set(by_target) == expected
    for marketing, (target, gfx, cu) in gpu_identity.INSTINCT_PRODUCTS.items():
        name, row = by_target[target]
        assert row["arch"] == gfx, (marketing, name)
        assert row["cus"] == cu, (marketing, name)
        assert row["product"] == marketing, (marketing, name)
        assert row["geak_support"] == "supported", name
    assert by_target["r9700"][1]["product"] == gpu_identity.R9700_MARKETING_NAME
    assert gpu_identity.KNOWN_SKUS == expected | {"unknown"}


def test_homogeneous_instinct_agents_are_accepted() -> None:
    identity = gpu_identity.parse_rocminfo(
        _rocminfo("gfx950", "AMD Instinct MI355X", cu=256, count=8)
    )
    assert (identity["target"], identity["sku"]) == ("unknown", "mi355x")
    assert identity["visible_gpu_agents"] == 8


def test_mixed_partitioned_and_full_instinct_agents_fail_closed() -> None:
    text = _rocminfo("gfx942", "AMD Instinct MI300X", cu=304)
    text += """
*******
Agent 3
*******
  Name:                    gfx942
  Marketing Name:          AMD Instinct MI300X
  Compute Unit:            38
"""
    with pytest.raises(gpu_identity.IdentityError, match="mixed identities"):
        gpu_identity.parse_rocminfo(text)


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
