#!/usr/bin/env python3
"""Collect target-local launch, HBM, pairwise, and all-to-all ceilings."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def _run(binary: Path, preset: str, size: str | None, env: dict, output: Path) -> str:
    command = [str(binary), preset]
    if size:
        command.append(size)
    process = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env={**os.environ, **env},
        check=False,
    )
    text = process.stdout + process.stderr
    output.write_text(text, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"{preset} failed; see {output}")
    return text


def _write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    empty = _run(
        args.binary,
        "empty",
        None,
        {
            "NUM_GPU_DEVICES": "8",
            "NUM_ITERATIONS": "20",
            "NUM_WARMUPS": "5",
            "BATCHSIZES": "0,1,16,256",
            "GRIDSIZES": "1",
            "BLOCKSIZES": "256",
        },
        args.output_dir / "transferbench_empty.txt",
    )
    hbm = _run(
        args.binary,
        "hbm",
        "1G",
        {
            "GPU_INDICES": "0,1,2,3,4,5,6,7",
            "NUM_ITERATIONS": "20",
            "NUM_WARMUPS": "3",
            "BLOCKSIZES": "512",
            "ELEM_BYTES": "16",
            "NUM_SUB_EXECS": "256",
            "UNROLLS": "16",
            "TEMPORAL_MASK": "3",
            "SHOW_DETAILS": "0",
        },
        args.output_dir / "transferbench_hbm.txt",
    )
    a2a = _run(
        args.binary,
        "a2a",
        "128M",
        {
            "NUM_GPU_DEVICES": "8",
            "NUM_ITERATIONS": "20",
            "NUM_WARMUPS": "3",
            "A2A_DIRECT": "0",
            "A2A_LOCAL": "0",
            "A2A_MODE": "0",
            "MEM_TYPE": "2",
            "NUM_SUB_EXEC": "8",
            "USE_DMA_EXEC": "0",
        },
        args.output_dir / "transferbench_a2a.txt",
    )
    p2p = _run(
        args.binary,
        "p2p",
        "128M",
        {
            "NUM_GPU_DEVICES": "8",
            "NUM_ITERATIONS": "20",
            "NUM_WARMUPS": "3",
            "GPU_MEM_TYPE": "2",
        },
        args.output_dir / "transferbench_p2p.txt",
    )

    launch_values = [
        float(match.group(1))
        for match in re.finditer(
            r"^\s*256\s+1\s+256\s+0\s+\d+\s+\S+\s+(\S+)",
            empty,
            re.MULTILINE,
        )
    ]
    hbm_avg = [
        float(match.group(1))
        for match in re.finditer(
            r"^\s*│\s+0\s+\d+\s+│\s+\S+\s+(\S+)\s+\S+\s+│",
            hbm,
            re.MULTILINE,
        )
    ]
    a2a_match = re.search(
        r"Aggregate bandwidth \(CPU Timed\):\s+(\S+)\s+GB/s",
        a2a,
    )
    p2p_match = re.search(
        r"Averages \(During UniDir\):\s+\S+\s+\S+\s+\S+\s+(\S+)",
        p2p,
    )
    if (
        len(launch_values) != 8
        or len(hbm_avg) != 8
        or a2a_match is None
        or p2p_match is None
    ):
        raise RuntimeError("failed to parse one or more TransferBench outputs")

    payload = {
        "schema_version": "geak-hardware-measurements-v1",
        "tool": "TransferBench v1.69.01 develop:44dec25",
        "measurements": {
            "launch_overhead_us": max(launch_values),
            "device_memory_gbps": min(hbm_avg),
            "pairwise_interconnect_gbps": float(p2p_match.group(1)),
            "all_to_all_interconnect_gbps": float(a2a_match.group(1)),
        },
        "units": {
            "launch_overhead_us": "us/kernel, batch=256",
            "device_memory_gbps": "GB/s, minimum per-GPU average read bandwidth",
            "pairwise_interconnect_gbps": "GB/s, average unidirectional GPU-to-GPU",
            "all_to_all_interconnect_gbps": "GB/s, CPU-timed aggregate",
        },
        "raw_artifacts": [
            str((args.output_dir / name).resolve())
            for name in (
                "transferbench_empty.txt",
                "transferbench_hbm.txt",
                "transferbench_a2a.txt",
                "transferbench_p2p.txt",
            )
        ],
    }
    output = args.output_dir / "hardware_measurements.json"
    _write(output, payload)
    print(f"GEAK_HARDWARE_MEASUREMENTS_JSON={output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
