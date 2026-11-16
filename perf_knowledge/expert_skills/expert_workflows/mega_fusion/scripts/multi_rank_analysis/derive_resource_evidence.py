#!/usr/bin/env python3
"""Derive per-kernel resource and residency bounds from rocprofv3 dispatch trace."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


_LABELS = {
    "per_1x32_mx_quant": "pre_dispatch_quant",
    "megamoe_stage1": "stage1",
    "megamoe_stage2": "stage2",
    "ep_combine_intranode": "combine",
}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-trace", type=Path, required=True)
    parser.add_argument("--hardware-context", type=Path, required=True)
    parser.add_argument("--max-waves-per-cu", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    hardware = json.loads(args.hardware_context.read_text(encoding="utf-8"))
    lds_per_cu = int(hardware["local_memory_bytes_per_execution_unit"])
    selected = {}
    with args.kernel_trace.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for needle, label in _LABELS.items():
                if needle in row["Kernel_Name"] and label not in selected:
                    workgroup_size = (
                        int(row["Workgroup_Size_X"])
                        * int(row["Workgroup_Size_Y"])
                        * int(row["Workgroup_Size_Z"])
                    )
                    waves = (workgroup_size + 63) // 64
                    lds = int(row["LDS_Block_Size"])
                    lds_bound = lds_per_cu // lds if lds else 2**31 - 1
                    wave_bound = args.max_waves_per_cu // waves
                    selected[label] = {
                        "kernel_name": row["Kernel_Name"],
                        "workgroup_size": workgroup_size,
                        "waves_per_workgroup": waves,
                        "lds_bytes_per_workgroup": lds,
                        "scratch_bytes": int(row["Scratch_Size"]),
                        "vgpr_count": int(row["VGPR_Count"]),
                        "accum_vgpr_count": int(row["Accum_VGPR_Count"]),
                        "sgpr_count": int(row["SGPR_Count"]),
                        "grid_size": [
                            int(row["Grid_Size_X"]),
                            int(row["Grid_Size_Y"]),
                            int(row["Grid_Size_Z"]),
                        ],
                        "residency_upper_bound_from_lds_and_waves": min(
                            lds_bound, wave_bound
                        ),
                        "lds_residency_bound": lds_bound,
                        "wave_residency_bound": wave_bound,
                    }
    missing = sorted(set(_LABELS.values()) - set(selected))
    if missing:
        raise ValueError(f"kernel trace is missing resources for {missing}")
    _write(
        args.output,
        {
            "schema_version": "geak-resource-evidence-v1",
            "source": str(args.kernel_trace.resolve()),
            "lds_bytes_per_cu": lds_per_cu,
            "max_waves_per_cu": args.max_waves_per_cu,
            "kernels": selected,
            "scope_warning": (
                "Residency values are upper bounds from LDS and wave slots only; "
                "VGPR allocation and scheduler limits may reduce actual residency."
            ),
        },
    )
    print(f"GEAK_RESOURCE_EVIDENCE_JSON={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
