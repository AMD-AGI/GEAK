#!/usr/bin/env python3
"""Normalized AITER benchmark adapter for fp8 128x128 block-scale GEMM.

This adapter deliberately bypasses the heterogeneous top-level UT/benchmark CLIs while reusing the
same AITER callables and math contract. Every implementation receives one input set, one preallocated
final output, one correctness gate and one interleaved CUDA-event timing protocol. JIT/build, input
generation, final-output allocation and external preshuffle are outside timing; backend-internal
workspace/layout preparation remains inside and is recorded as such.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import statistics
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = 1
BLOCK_N = 128
BLOCK_K = 128
DEFAULT_CANDIDATES = (
    "triton", "triton_preshuffle", "ck", "ck_preshuffle",
    "cktile", "cktile_preshuffle", "asm",
)
SOURCE_PATHS = {
    "triton": "aiter/ops/triton/gemm/basic/gemm_a8w8_blockscale.py",
    "triton_preshuffle": "aiter/ops/triton/gemm/basic/gemm_a8w8_blockscale.py",
    "gluon": "aiter/ops/triton/gluon/gemm_a8w8_blockscale.py",
    "ck": "aiter/ops/gemm_op_a8w8.py",
    "ck_preshuffle": "aiter/ops/gemm_op_a8w8.py",
    "cktile": "aiter/ops/gemm_op_a8w8.py",
    "cktile_preshuffle": "aiter/ops/gemm_op_a8w8.py",
    "asm": "aiter/ops/gemm_op_a8w8.py",
}
LANGUAGES = {
    "triton": "triton", "triton_preshuffle": "triton", "gluon": "gluon",
    "ck": "ck", "ck_preshuffle": "ck", "cktile": "ck", "cktile_preshuffle": "ck",
    "asm": "asm",
}


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def content_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "not_captured"


def implementation_id(
    name: str,
    config: dict[str, Any],
    source_sha256: str,
    provenance: dict[str, Any] | None = None,
) -> str:
    payload = json.dumps(
        {
            "name": name, "config": config, "source_sha256": source_sha256,
            "provenance": provenance or {},
        },
        sort_keys=True, separators=(",", ":"),
    )
    return "impl_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False, timeout=10,
    )
    return result.stdout.strip() if result.returncode == 0 else "not_captured"


def git_dirty(path: Path):
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        capture_output=True, text=True, check=False, timeout=10,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else "not_captured"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        try:
            module = __import__(name)
            return str(getattr(module, "__version__", "not_captured"))
        except Exception:
            return "not_captured"


def remove_unittest_shadow_paths():
    """Task directories often contain `unittest.py`; exclude them before importing torch."""
    cwd = Path.cwd().resolve()
    sys.path[:] = [
        raw for raw in sys.path
        if not ((Path(raw).resolve() if raw else cwd) / "unittest.py").is_file()
    ]


def fp8_dtype_for_gfx(gfx: str) -> str:
    base = str(gfx).split(":", 1)[0]
    return (
        "fp8_e4m3fn_blockscale"
        if base in {"gfx950", "gfx1250", "gfx1200", "gfx1201"}
        else "fp8_e4m3fnuz_blockscale"
    )


def make_inputs(torch, get_fp8_dtypes, shuffle_weight, m: int, n: int, k: int, seed: int):
    if n % BLOCK_N or k % BLOCK_K:
        raise ValueError(f"N and K must be divisible by block scale {BLOCK_N}x{BLOCK_K}")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    _, fp8 = get_fp8_dtypes()
    x = (torch.rand((m, k), generator=generator, dtype=torch.float16, device="cuda") / 10).to(fp8)
    weight = (
        torch.rand((n, k), generator=generator, dtype=torch.float16, device="cuda") / 10
    ).to(fp8)
    x_scale = torch.rand((m, k // BLOCK_K), generator=generator, dtype=torch.float32, device="cuda")
    w_scale = torch.rand(
        (n // BLOCK_N, k // BLOCK_K), generator=generator, dtype=torch.float32, device="cuda"
    )
    weight_shuffled = shuffle_weight(weight, layout=(16, 16))
    weight_triton_shuffled = weight_shuffled.reshape(n // 16, k * 16)
    x_scale_shuffled = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
    return {
        "x": x,
        "weight": weight,
        "weight_shuffled": weight_shuffled,
        "weight_triton_shuffled": weight_triton_shuffled,
        "x_scale": x_scale,
        "x_scale_shuffled": x_scale_shuffled,
        "w_scale": w_scale,
    }


def reference(torch, tensors):
    x = tensors["x"]
    weight = tensors["weight"]
    m, k = x.shape
    n = weight.shape[0]
    x_scale = tensors["x_scale"].repeat_interleave(BLOCK_K, dim=1)[:m, :k]
    w_scale = tensors["w_scale"].repeat_interleave(BLOCK_N, dim=0)
    w_scale = w_scale.repeat_interleave(BLOCK_K, dim=1)[:n, :k]
    x_dequant = x.float() * x_scale
    w_dequant = weight.float() * w_scale
    return torch.nn.functional.linear(x_dequant, w_dequant).to(torch.bfloat16)


def build_candidates(torch, tensors, requested: list[str]):
    from aiter.ops.gemm_op_a8w8 import (
        gemm_a8w8_blockscale_bpreshuffle_asm,
        gemm_a8w8_blockscale_bpreshuffle_ck,
        gemm_a8w8_blockscale_bpreshuffle_cktile,
        gemm_a8w8_blockscale_ck,
        gemm_a8w8_blockscale_cktile,
    )
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
        gemm_a8w8_blockscale as triton_gemm,
        gemm_a8w8_blockscale_preshuffle as triton_preshuffle_gemm,
    )
    from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a8w8_blockscale import _get_config

    x, w = tensors["x"], tensors["weight"]
    m, k = x.shape
    n = w.shape[0]
    xs, ws = tensors["x_scale"], tensors["w_scale"]
    w_sh, w_triton_sh = tensors["weight_shuffled"], tensors["weight_triton_shuffled"]
    xs_sh = tensors["x_scale_shuffled"]
    outputs = {
        name: torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
        for name in requested
    }
    built: dict[str, tuple[Callable[[], Any], dict[str, Any]]] = {}

    if "triton" in requested:
        triton_config, triton_is_tuned = _get_config(m, n, k)
        built["triton"] = (
            lambda: triton_gemm(
                x, w, xs, ws, torch.bfloat16, outputs["triton"],
                config=copy.deepcopy(triton_config),
            ),
            {
                "layout": "unshuffled", "config": triton_config,
                "config_is_tuned": bool(triton_is_tuned), "identity_complete": True,
            },
        )
    if "triton_preshuffle" in requested:
        triton_ps_config, triton_ps_is_tuned = _get_config(m, n, k, True)
        built["triton_preshuffle"] = (
            lambda: triton_preshuffle_gemm(
                x, w_triton_sh, xs_sh, ws, torch.bfloat16, outputs["triton_preshuffle"],
                config=copy.deepcopy(triton_ps_config),
            ),
            {
                "layout": "preshuffled_16x16", "config": triton_ps_config,
                "config_is_tuned": bool(triton_ps_is_tuned), "identity_complete": True,
            },
        )
    if "gluon" in requested:
        from aiter.ops.triton.gluon.gemm_a8w8_blockscale import gemm_a8w8_blockscale as gluon_gemm
        built["gluon"] = (
            lambda: gluon_gemm(x, w, xs, ws, torch.bfloat16, outputs["gluon"]),
            {
                "layout": "unshuffled", "config": "aiter_selected",
                "identity_complete": False,
                "identity_gap": "resolved Gluon config is not exposed by the public wrapper",
            },
        )
    if "ck" in requested:
        built["ck"] = (
            lambda: gemm_a8w8_blockscale_ck(x, w, xs, ws, outputs["ck"]),
            {
                "layout": "unshuffled", "config": "default_compiled_instance",
                "identity_complete": True,
            },
        )
    if "ck_preshuffle" in requested:
        built["ck_preshuffle"] = (
            lambda: gemm_a8w8_blockscale_bpreshuffle_ck(
                x, w_sh, xs_sh, ws, outputs["ck_preshuffle"]
            ),
            {
                "layout": "preshuffled_16x16", "config": "default_compiled_instance",
                "identity_complete": True,
            },
        )
    if "cktile" in requested:
        built["cktile"] = (
            lambda: gemm_a8w8_blockscale_cktile(x, w, xs, ws, outputs["cktile"], False),
            {
                "layout": "unshuffled", "config": "default_compiled_instance",
                "identity_complete": True,
            },
        )
    if "cktile_preshuffle" in requested:
        built["cktile_preshuffle"] = (
            lambda: gemm_a8w8_blockscale_bpreshuffle_cktile(
                x, w_sh, xs_sh, ws, outputs["cktile_preshuffle"], True
            ),
            {
                "layout": "preshuffled_16x16",
                "config": "default_compiled_instance",
                "internal_prepare": "included_by_backend",
                "identity_complete": True,
            },
        )
    if "asm" in requested:
        zero_bias = torch.zeros(1, w.shape[0], dtype=torch.float32, device=x.device)
        built["asm"] = (
            lambda: gemm_a8w8_blockscale_bpreshuffle_asm(
                x, w_sh, outputs["asm"], xs_sh, ws, zero_bias_buf=zero_bias
            ),
            {
                "layout": "preshuffled_16x16", "config": "runtime_selected",
                "identity_complete": False,
                "identity_gap": "ASM kernelName/splitK selected at runtime and not captured",
            },
        )
    return built


def correctness(torch, actual, expected, atol: float, rtol: float):
    actual_f, expected_f = actual.float(), expected.float()
    diff = (actual_f - expected_f).abs()
    max_abs = float(diff.max().item())
    max_rel = float((diff / expected_f.abs().clamp_min(1e-5)).max().item())
    return {
        "pass": bool(torch.allclose(actual_f, expected_f, atol=atol, rtol=rtol)),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "atol": atol,
        "rtol": rtol,
    }


def time_interleaved(torch, candidates, warmup: int, repeats: int, samples: int):
    ready = {}
    errors = {}
    for name, (function, metadata) in candidates.items():
        try:
            for _ in range(warmup):
                function()
            torch.cuda.synchronize()
            ready[name] = (function, metadata)
        except Exception as exc:  # build/unsupported is data, not a suite crash
            errors[name] = {
                "status": "timing_failed", "phase": "warmup", "error": repr(exc),
            }

    timings = {name: [] for name in ready}
    names = list(ready)
    for sample in range(samples):
        order = names[sample % len(names):] + names[:sample % len(names)] if names else []
        for name in order:
            function, _ = ready[name]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                function()
            end.record()
            end.synchronize()
            timings[name].append(float(start.elapsed_time(end)) / repeats)
    return ready, timings, errors


def run(args):
    aiter_root = args.aiter_root.resolve()
    remove_unittest_shadow_paths()
    if str(aiter_root) not in sys.path:
        sys.path.insert(0, str(aiter_root))
    import torch
    from aiter.ops.shuffle import shuffle_weight
    from aiter.ops.triton.utils.types import get_fp8_dtypes

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm/CUDA device is required")
    requested = list(dict.fromkeys(args.candidate))
    if args.baseline not in requested:
        requested.insert(0, args.baseline)
    unknown = sorted(set(requested) - set(SOURCE_PATHS))
    if unknown:
        raise ValueError(f"unknown candidates: {unknown}")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    gfx = getattr(properties, "gcnArchName", "not_captured")
    errors = {}
    build_requested = list(requested)
    if "gluon" in build_requested and str(gfx).split(":", 1)[0] != "gfx950":
        errors["gluon"] = {
            "status": "unsupported",
            "phase": "precheck",
            "reason_code": "requires_gfx950",
            "error": f"Gluon block-scale GEMM requires gfx950/CDNA4, got {gfx}",
        }
        build_requested.remove("gluon")

    tensors = make_inputs(
        torch, get_fp8_dtypes, shuffle_weight, args.m, args.n, args.k, args.seed,
    )
    expected = reference(torch, tensors)
    candidates = build_candidates(torch, tensors, build_requested)

    # One untimed call supplies correctness and also makes first-call compilation explicit.
    correctness_rows = {}
    usable = {}
    for name, pair in candidates.items():
        try:
            actual = pair[0]()
            torch.cuda.synchronize()
            verdict = correctness(torch, actual, expected, args.atol, args.rtol)
            correctness_rows[name] = verdict
            if verdict["pass"]:
                usable[name] = pair
            else:
                errors[name] = {
                    "status": "correctness_failed",
                    "phase": "correctness",
                    "correctness": verdict,
                }
        except Exception as exc:
            errors[name] = {
                "status": "first_launch_failed", "phase": "first_launch", "error": repr(exc),
            }

    ready, timings, timing_errors = time_interleaved(
        torch, usable, args.warmup, args.repeats, args.samples,
    )
    errors.update(timing_errors)
    aiter_commit = git_commit(aiter_root)
    aiter_is_dirty = git_dirty(aiter_root)
    implementation_provenance = {
        "aiter_commit": aiter_commit,
        "aiter_dirty": aiter_is_dirty,
    }
    rows = []
    for name in requested:
        source_path = aiter_root / SOURCE_PATHS[name]
        source_hash = content_sha256(source_path)
        config = candidates.get(name, (None, {"config": "not_built"}))[1]
        if name in errors:
            rows.append({
                "implementation_id": implementation_id(
                    name, config, source_hash, implementation_provenance,
                ),
                "name": name,
                "language": LANGUAGES[name],
                "status": errors[name]["status"],
                "phase": errors[name].get("phase"),
                "reason_code": errors[name].get("reason_code", errors[name]["status"]),
                "error": errors[name].get("error"),
                "correctness": errors[name].get("correctness", correctness_rows.get(name)),
                "config": config,
                "source": SOURCE_PATHS[name],
                "source_sha256": source_hash,
            })
            continue
        values = timings[name]
        rows.append({
            "implementation_id": implementation_id(
                name, config, source_hash, implementation_provenance,
            ),
            "name": name,
            "language": LANGUAGES[name],
            "status": "measured",
            "correctness": correctness_rows[name],
            "config": config,
            "source": SOURCE_PATHS[name],
            "source_sha256": source_hash,
            "latency_ms": {
                "samples": values,
                "median": statistics.median(values),
                "p10": percentile(values, 0.10),
                "p90": percentile(values, 0.90),
            },
        })

    baseline = next((row for row in rows if row["name"] == args.baseline), None)
    baseline_ms = (
        baseline.get("latency_ms", {}).get("median") if baseline and baseline["status"] == "measured"
        else None
    )
    for row in rows:
        latency = row.get("latency_ms", {}).get("median")
        row["speedup_vs_baseline"] = (
            baseline_ms / latency if baseline_ms is not None and latency else None
        )

    fp8_dtype = fp8_dtype_for_gfx(gfx)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "seed": args.seed,
        "suite": "fp8_a8w8_blockscale",
        "math_contract": (
            "Y=(X*x_scale)@(W*w_scale).T; fp32 accumulation; bf16 output; 128x128 scaling"
        ),
        "known_gaps": [
            (
                "AITER flydsl_preshuffle_gemm_a8 uses per-token scaling and is not a valid "
                "implementation of this 128x128 block-scale contract; generated/custom FlyDSL "
                "implementations remain measured by kernel_workflow."
            ),
        ],
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "dtype": fp8_dtype,
        "baseline": args.baseline,
        "accounting": {
            "timed": "complete steady-state backend callable into a preallocated final output",
            "excluded": [
                "input generation", "reference", "final output allocation",
                "external preshuffle", "first-call JIT/build",
            ],
            "included": [
                "backend-internal workspace allocation",
                "backend-internal layout preparation",
                "split-K partial allocation and reduction",
            ],
            "timer": "CUDA/HIP events",
            "warmup": args.warmup,
            "repeats_per_sample": args.repeats,
            "samples": args.samples,
            "interleaved_candidates": True,
        },
        "environment": {
            "device": properties.name,
            "gfx": getattr(properties, "gcnArchName", "not_captured"),
            "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
            "aiter_version": package_version("aiter"),
            "aiter_commit": aiter_commit,
            "aiter_dirty": aiter_is_dirty,
            "flydsl_version": package_version("flydsl"),
        },
        "implementations": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aiter-root", type=Path, default=Path("/sgl-workspace/aiter"))
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--candidate", action="append", choices=sorted(SOURCE_PATHS),
                        default=None)
    parser.add_argument("--baseline", choices=sorted(SOURCE_PATHS), default="triton")
    parser.add_argument("--seed", type=int, default=1907)
    parser.add_argument("--run-id", help="stable ID for this physical benchmark run")
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    args.candidate = args.candidate or list(DEFAULT_CANDIDATES)
    args.run_id = args.run_id or uuid.uuid4().hex
    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
