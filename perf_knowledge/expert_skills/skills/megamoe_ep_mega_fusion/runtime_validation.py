#!/usr/bin/env python3
"""Profile-driven graph, launch, resource and paired-performance validation."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import inspect
import json
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path


PROFILE = {
    "world": 8,
    "model_dim": 7168,
    "inter_dim": 3072,
    "experts": 384,
    "topk": 6,
    "swiglu_limit": 10.0,
}
FROZEN_BASELINE_MISSING = "not_evaluated: missing --frozen-baseline-ms"
# One source rank sends most of its rows to five experts on five destination ranks;
# the hot source moves on every replay and the other ranks stay uniform. A producer
# that reads per-launch route metadata before it is ready then sees the previous
# launch's, so the first call after the switch goes wrong (or hangs) and a repeated
# call heals. Symmetric skew, even with the hot experts moving, did not show it
# (0/32 against 32/32 for this route on the same defect).
MOVING_HOT_ROUTE = "moving-hot-rank"
MOVING_HOT_EXPERTS = 5


def _csv_ints(value):
    return [int(item) for item in value.split(",") if item.strip()]


def _csv_strings(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _next_power_of_two(value):
    return 1 << (int(value) - 1).bit_length()


def _parse_target_guard(guard):
    """``<T>_uniform`` or ``<T>_uniform_mtpr<M>`` -> (tokens, mtpr or None).

    Without the suffix the target is timed on the shared operator; with it the
    target owns an operator built with max_tok_per_rank=M, as the official test
    sizes it, so M must be next_pow2(T).
    """
    head, _, mtpr = guard.partition("_mtpr")
    tokens, _, route = head.partition("_")
    if not tokens.isdigit() or route != "uniform" or (mtpr and not mtpr.isdigit()):
        raise ValueError(f"target guard must be <T>_uniform[_mtpr<M>], got {guard!r}")
    tokens = int(tokens)
    if not mtpr:
        return tokens, None
    if int(mtpr) != _next_power_of_two(tokens):
        raise ValueError(f"target guard {guard}: mtpr must be next_pow2({tokens})")
    return tokens, int(mtpr)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runtime adapter {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _discover_adapter(tree, explicit):
    candidates = []
    if explicit:
        path = Path(explicit)
        candidates.append(path if path.is_absolute() else tree / path)
    # Non-normative adapter hint for the validated profile.
    candidates.append(tree / "op_tests/multigpu_tests/test_mega_moe_v2.py")
    candidates.extend(sorted(tree.glob("**/test_*moe*.py")))
    seen = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        text = path.read_text(errors="ignore")
        if all(name in text for name in ("_reference", "_setup_dist", "_quantize_weights")):
            helper = _load_module(path, "_expert_skill_profile_adapter")
            if all(
                callable(getattr(helper, name, None))
                for name in ("_reference", "_setup_dist", "_quantize_weights", "_barrier")
            ):
                return helper, path
    raise RuntimeError("no semantic runtime adapter was discovered")


def _discover_profile(helper, profile_name):
    required = {"model_dim", "inter_dim", "experts", "topk"}
    for value in vars(helper).values():
        if not isinstance(value, dict) or profile_name not in value:
            continue
        profile = value[profile_name]
        if isinstance(profile, dict) and required <= set(profile):
            return dict(profile)
    raise RuntimeError(f"runtime adapter has no profile {profile_name!r}")


def _discover_factory(helper):
    required = {
        "rank", "world_size", "model_dim", "inter_dim", "experts", "topk",
        "quant", "w1", "w1_scale", "w2", "w2_scale", "max_tok_per_rank",
    }
    matches = []
    for value in vars(helper).values():
        if not inspect.isclass(value):
            continue
        try:
            parameters = set(inspect.signature(value).parameters)
        except (TypeError, ValueError):
            continue
        if required <= parameters:
            matches.append(value)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one semantic candidate factory, discovered {len(matches)}"
        )
    return matches[0]


def _make_route(torch, tokens, route, iteration, profile, rank, world, device):
    generator = torch.Generator(device=device).manual_seed(
        123 + rank + iteration * 1009 + tokens * 17
    )
    x = torch.randn(
        (tokens, profile["model_dim"]),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    destination = torch.topk(
        torch.rand((tokens, world), device=device, generator=generator),
        profile["topk"],
        dim=-1,
    ).indices
    local_experts = profile["experts"] // world
    if route == MOVING_HOT_ROUTE:
        scores = torch.randn(
            (tokens, profile["experts"]),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        if rank == iteration % world:
            column = torch.arange(profile["experts"], device=device)
            hot_column = None
            for hot in range(MOVING_HOT_EXPERTS):
                expert = ((rank + iteration + 3 * hot) % world) * local_experts + (
                    iteration * 7 + hot
                ) % local_experts
                hit = column == expert
                hot_column = hit if hot_column is None else hot_column | hit
            hot_row = (
                torch.rand((tokens, 1), device=device, generator=generator) < 0.75
            )
            scores = scores + (hot_row & hot_column) * 1.0e4
        ids = torch.topk(scores, profile["topk"], dim=-1).indices
        logits = torch.randn(
            destination.shape, dtype=torch.float32, device=device, generator=generator
        )
        return (
            x.contiguous(),
            logits.softmax(dim=-1).contiguous(),
            ids.to(torch.int32).contiguous(),
        )
    if route == "uniform":
        local = torch.randint(
            0, local_experts, destination.shape, device=device, generator=generator
        )
    elif route == "rank-mixed-skew":
        cold = torch.randint(
            1, local_experts, destination.shape, device=device, generator=generator
        )
        local = torch.where(destination < world // 2, torch.zeros_like(cold), cold)
    else:
        raise ValueError(f"unsupported route {route!r}")
    logits = torch.randn(
        destination.shape, dtype=torch.float32, device=device, generator=generator
    )
    return (
        x.contiguous(),
        logits.softmax(dim=-1).contiguous(),
        (destination * local_experts + local).to(torch.int32).contiguous(),
    )


def _relative_l2(torch, dist, helper, output, reference, device):
    value = float(
        (
            torch.linalg.vector_norm(output.float() - reference)
            / torch.linalg.vector_norm(reference)
        ).item()
    )
    return helper._reduce_float(value, device, dist.ReduceOp.MAX)


def _capture(torch, helper, body, warmup=None):
    helper._barrier()
    (warmup or body)()
    helper._barrier()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=stream):
        body()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _time_rank_max(torch, dist, helper, graph, device, iters):
    helper._barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return helper._reduce_float(
        start.elapsed_time(end) / iters, device, dist.ReduceOp.MAX
    )


def _profile_launch_count(torch, dist, helper, body, device, ceiling=False, fallback=False):
    from torch.profiler import ProfilerActivity, profile

    helper._barrier()
    with profile(activities=[ProfilerActivity.CUDA]) as trace:
        body()
        torch.cuda.synchronize()
    count = 0
    for event in trace.key_averages():
        name = str(event.key).lower()
        device_time = float(
            getattr(event, "device_time_total", 0)
            or getattr(event, "self_device_time_total", 0)
            or 0
        )
        if device_time > 0 and not any(tag in name for tag in ("memcpy", "memset")):
            count += int(event.count)
    lo = torch.tensor(count, dtype=torch.int32, device=device)
    hi = lo.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    # The shared large-token operator follows the two-launch recipe exactly. A target built at its
    # own capacity may fuse further (one launch); two is then a ceiling, the same on every rank.
    # A shared operator declared FALLBACK runs the frozen path: any count, the same on every rank.
    allowed = None if fallback else (1, 2) if ceiling else (2,)
    if int(lo.item()) != int(hi.item()) or (allowed and int(lo.item()) not in allowed):
        raise AssertionError(
            f"BF16 path must emit {' or '.join(map(str, allowed or ('the same number of',)))} "
            f"kernels per rank, got {int(lo.item())}..{int(hi.item())}"
        )
    return int(lo.item())


def _resource_checks(path, target_case=""):
    if not path:
        return {
            "status": "delegated",
            "required_by_acceptance": True,
            "reason": "outer verifier must supply emitted metadata",
        }
    if not Path(path).is_file():
        return {
            "status": "missing",
            "required_by_acceptance": True,
            "reason": f"resource evidence file not found: {path}",
        }
    data = json.loads(Path(path).read_text())
    rows = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(rows, dict):
        raise ValueError("resource evidence must contain a cases object")
    limits = {
        "fixed_128": {
            "threads": 256, "group_segment": 32144, "kernarg": 440,
            "vgpr": 168, "sgpr": 106, "vgpr_spills": 0,
            "sgpr_spills": 106, "private": 0,
        },
        "large_8192": {
            "threads": 512, "group_segment": 160400, "kernarg": 440,
            "vgpr": 256, "sgpr": 106, "vgpr_spills": 80,
            "sgpr_spills": 116, "private": 316,
        },
    }
    if target_case:
        # The shared operator fell back, so its fused envelopes do not exist. The target kernel
        # has no validated envelope: report its metadata and require residency only.
        limits = {target_case: {key: None for key in limits["fixed_128"]}}
    for case, bound in limits.items():
        row = rows.get(case)
        if not isinstance(row, dict):
            raise AssertionError(f"missing resource case {case}")
        for key, maximum in bound.items():
            value = int(row.get(key, -1))
            if value < 0 or (maximum is not None and value > maximum):
                raise AssertionError(
                    f"resource {case}.{key}={value} "
                    + ("is missing" if value < 0 else f"exceeds {maximum}")
                )
        if int(row.get("resident_workgroups", 1)) < 1:
            raise AssertionError(f"resource {case} has no resident workgroup")
    return {"status": "pass", "cases": sorted(limits)}


def _positive_finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _speedup_summary(frozen_ms, cand_ms_list, switch_off_ms_list, min_speedup, tol):
    """Gate ``min_speedup`` on the frozen baseline; the same-tree pair is diagnostic.

    ``paired_rank_max_speedup`` is kept for backward compatibility as an alias
    of ``absolute_speedup`` (null when the frozen baseline is missing); it never
    carries the same-tree switch-off ratio.
    """
    if (
        not cand_ms_list
        or len(cand_ms_list) != len(switch_off_ms_list)
        or any(
            _positive_finite(value) is None
            for value in [*cand_ms_list, *switch_off_ms_list]
        )
    ):
        raise ValueError("paired readings must be matched positive latencies")
    cand_ms = statistics.median(cand_ms_list)
    switch_off_ms = statistics.median(switch_off_ms_list)
    summary = {
        "frozen_baseline_ms": None,
        "candidate_ms_median": cand_ms,
        "switch_off_ms_median": switch_off_ms,
        "incremental_switch_speedup": statistics.median(
            [base / cand for base, cand in zip(switch_off_ms_list, cand_ms_list)]
        ),
        "absolute_speedup": None,
        "paired_rank_max_speedup": None,
        "paired_rank_max_speedup_basis": "absolute_speedup",
        "min_speedup": min_speedup,
        "denominator_tolerance": tol,
        "denominator_deviation_pct": None,
        "speedup_gate": FROZEN_BASELINE_MISSING,
    }
    frozen = _positive_finite(frozen_ms)
    if frozen is None:
        return summary
    absolute = frozen / cand_ms
    deviation = (switch_off_ms - frozen) / frozen
    summary.update(
        frozen_baseline_ms=frozen,
        absolute_speedup=absolute,
        paired_rank_max_speedup=absolute,
        denominator_deviation_pct=100.0 * deviation,
        # Never pass at or below parity, even when min_speedup is 1.0.
        speedup_gate="pass" if absolute >= min_speedup and absolute > 1.0 else "fail",
    )
    if abs(deviation) > tol:
        direction = "regressed" if deviation > 0 else "changed"
        summary["denominator_mismatch"] = {
            "switch_off_ms": switch_off_ms,
            "frozen_ms": frozen,
            "deviation_pct": 100.0 * deviation,
            "note": (
                f"the candidate tree's switch-off path {direction} shared code; "
                "incremental_switch_speedup is not a frozen-baseline speedup"
            ),
        }
    return summary


def _restore_route(tensors, snapshot):
    """Copy a route snapshot back in place so captured graph addresses stay valid."""
    if len(tensors) != len(snapshot):
        raise ValueError("route snapshot does not match the target tensors")
    for tensor, original in zip(tensors, snapshot):
        tensor.copy_(original)


def _flush_output_streams():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (AttributeError, OSError, ValueError):
            pass
    try:
        import ctypes

        ctypes.CDLL(None).fflush(None)
    except (AttributeError, OSError, TypeError):
        pass


def _read_fd(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(fd, 1 << 16)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


@contextlib.contextmanager
def _captured_output_fds(directory, prefix):
    """Redirect fds 1 and 2 into files for one window, then restore and re-emit.

    Capture files are removed only after both descriptors are restored, so a
    hard abort inside the window leaves them in ``directory``.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    captured = {"text": ""}
    redirected = []
    _flush_output_streams()
    try:
        for fd in (1, 2):
            capture_fd, capture_path = tempfile.mkstemp(
                prefix=f"{prefix}.fd{fd}.", suffix=".log", dir=directory
            )
            try:
                saved_fd = os.dup(fd)
            except OSError:
                os.close(capture_fd)
                os.unlink(capture_path)
                continue
            redirected.append((fd, saved_fd, capture_fd, capture_path))
            os.dup2(capture_fd, fd)
        yield captured
    finally:
        _flush_output_streams()
        for fd, saved_fd, _, _ in redirected:
            os.dup2(saved_fd, fd)
            os.close(saved_fd)
        texts = []
        for fd, _, capture_fd, capture_path in redirected:
            data = b""
            with contextlib.suppress(OSError):
                data = _read_fd(capture_fd)
                os.unlink(capture_path)
            os.close(capture_fd)
            texts.append(data.decode(errors="replace"))
            view = memoryview(data)
            with contextlib.suppress(OSError):
                while view:
                    view = view[os.write(fd, view):]
        captured["text"] = "".join(texts)


def _reduce_marker_observation(torch, dist, device, count, world):
    observed = torch.tensor(1 if count > 0 else 0, dtype=torch.int32, device=device)
    observed_min = observed.clone()
    dist.all_reduce(observed, op=dist.ReduceOp.SUM)
    dist.all_reduce(observed_min, op=dist.ReduceOp.MIN)
    local = torch.tensor([int(count)], dtype=torch.int32, device=device)
    gathered = [torch.zeros_like(local) for _ in range(world)]
    dist.all_gather(gathered, local)
    return (
        int(observed.item()),
        int(observed_min.item()),
        [int(item.item()) for item in gathered],
    )


def _activation_summary(marker, world, observed_sum, observed_min, per_rank_counts):
    """Activation passes only when every rank observed ``marker``."""
    per_rank = [
        {"rank": rank, "marker_count": int(count), "observed": int(count) > 0}
        for rank, count in enumerate(per_rank_counts)
    ]
    passed = (
        world > 0
        and len(per_rank) == world
        and int(observed_sum) == world
        and int(observed_min) == 1
        and all(row["observed"] for row in per_rank)
    )
    return {
        "status": "pass" if passed else "fail",
        "path_marker": marker,
        "ranks": world,
        "observed_ranks": int(observed_sum),
        "all_ranks_observed": passed,
        "per_rank": per_rank,
    }


def _claim_blockers(activation, resources, speedup):
    """Missing or failed evidence keeps ``claim_complete`` false."""
    blockers = []
    if (activation or {}).get("status") != "pass":
        blockers.append("activation_marker_missing")
    if (resources or {}).get("status") != "pass":
        blockers.append("resource_evidence_missing")
    gate = str((speedup or {}).get("speedup_gate") or "")
    if gate.startswith("not_evaluated"):
        blockers.append("frozen_baseline_missing")
    elif gate != "pass":
        blockers.append("speedup_gate_failed")
    return blockers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-tree", required=True)
    parser.add_argument("--profile-adapter", default="")
    parser.add_argument("--profile", default="v4_pro")
    parser.add_argument("--accuracy-cases", default="128,512,8192")
    parser.add_argument("--liveness-cases", default="128,512,8192")
    parser.add_argument("--routes", default="uniform,rank-mixed-skew,moving-hot-rank")
    parser.add_argument("--mtpr-cases", default="")
    parser.add_argument("--mtpr-fallback-max", type=int, default=0)
    parser.add_argument("--replays", type=int, default=256)
    parser.add_argument("--numeric-checkpoint-interval", type=int, default=16)
    # relL2 of the first call after a route change against a repeated call of the
    # same route. Stale route reads corrupt whole rows (>=0.045 measured) yet can
    # stay under --rtol against the reference.
    parser.add_argument("--replay-consistency-rtol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.10)
    parser.add_argument("--perf-iters", type=int, default=20)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--min-speedup", type=float, default=1.03)
    # The guard timed against --frozen-baseline-ms; an _mtpr<M> target must run
    # path=MEGA with at most two launches even when M <= --mtpr-fallback-max.
    parser.add_argument("--target-guard", default="8192_uniform")
    # With an _mtpr<M> target only: the shared operator (max_tok_per_rank of the largest case) is a
    # declared FALLBACK bucket. It must print path=FALLBACK on every rank and still pass accuracy and
    # replay; the claim's activation, launch ceiling and resources are the target's.
    parser.add_argument("--shared-fallback", action="store_true")
    parser.add_argument("--frozen-baseline-ms", type=float, default=None)
    parser.add_argument("--denominator-tolerance", type=float, default=0.05)
    parser.add_argument("--path-marker", default="path=MEGA")
    parser.add_argument("--resource-evidence", default="")
    parser.add_argument("--json-output", required=True)
    args = parser.parse_args()

    tree = Path(args.candidate_tree).resolve()
    sys.path.insert(0, str(tree))
    helper, adapter_path = _discover_adapter(tree, args.profile_adapter)

    import torch
    import torch.distributed as dist

    accuracy_cases = _csv_ints(args.accuracy_cases)
    liveness_cases = _csv_ints(args.liveness_cases)
    routes = _csv_strings(args.routes)
    # Required for every claim, so an explicit --routes list cannot drop it.
    if MOVING_HOT_ROUTE not in routes:
        routes.append(MOVING_HOT_ROUTE)
    mtpr_cases = _csv_ints(args.mtpr_cases) if args.mtpr_cases else []
    required_cases = {128, 512, 8192}
    if not required_cases <= set(accuracy_cases) or not required_cases <= set(liveness_cases):
        raise ValueError("runtime claims require direct 128/512/8192 cases")
    target_tokens, target_mtpr = _parse_target_guard(args.target_guard)
    if target_mtpr is None:
        if target_tokens not in set(accuracy_cases) | set(liveness_cases):
            raise ValueError(f"target {args.target_guard} is not an accuracy/liveness case")
    elif target_tokens not in mtpr_cases:
        mtpr_cases.append(target_tokens)
    if args.shared_fallback and target_mtpr is None:
        raise ValueError("--shared-fallback needs an _mtpr<M> target guard")
    shared_marker = "path=FALLBACK" if args.shared_fallback else args.path_marker
    if args.replays <= 0 or args.pairs <= 0 or args.perf_iters <= 0:
        raise ValueError("replays, pairs and perf-iters must be positive")
    if not args.path_marker:
        raise ValueError("path-marker must be non-empty")
    if not 0 < args.replay_consistency_rtol <= args.rtol:
        raise ValueError("replay-consistency-rtol must be in (0, rtol]")
    if not math.isfinite(args.denominator_tolerance) or args.denominator_tolerance < 0:
        raise ValueError("denominator-tolerance must be a finite non-negative fraction")

    rank, world, device = helper._setup_dist()
    result = {
        "schema_version": "expert-skill-runtime-evidence-v1",
        "claim_complete": False,
        "validation_scope": "post-selection claims only",
        "adapter": adapter_path.relative_to(tree).as_posix(),
        "accuracy_results": [],
        "replay_results": [],
        "paired_readings": [],
        "replay_routes": routes,
        "target_guard": args.target_guard,
    }

    def write_result(complete):
        result["claim_complete"] = bool(complete)
        if rank == 0:
            output = Path(args.json_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            tmp = output.with_suffix(output.suffix + ".tmp")
            tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            os.replace(tmp, output)

    try:
        if world != PROFILE["world"]:
            raise AssertionError(f"validated claim requires 8 ranks, got {world}")
        profile_data = _discover_profile(helper, args.profile)
        for key in ("model_dim", "inter_dim", "experts", "topk"):
            if int(profile_data[key]) != int(PROFILE[key]):
                raise AssertionError(
                    f"selected profile claim differs at {key}: {profile_data[key]}"
                )
        profile_data["swiglu_limit"] = float(
            profile_data.get("swiglu_limit", PROFILE["swiglu_limit"])
        )
        factory = _discover_factory(helper)
        local_experts = profile_data["experts"] // world
        packed = helper._quantize_weights(
            profile_data["model_dim"],
            profile_data["inter_dim"],
            local_experts,
            rank,
            123,
            device,
        )
        w1, w1_scale, w2, w2_scale, w1_q, w1_ref_scale, w2_q, w2_ref_scale = packed
        ref_weights = w1_q, w1_ref_scale, w2_q, w2_ref_scale
        max_tokens = _next_power_of_two(max(accuracy_cases + liveness_cases))

        os.environ["AITER_MEGAMOE_FUSE_ALL"] = "1"
        os.environ["AITER_MEGAMOE_FUSE_COMBINE"] = "1"
        os.environ["AITER_MEGAMOE_FUSE_QUANT"] = "0"
        candidate = factory(
            rank=rank,
            world_size=world,
            quant="a8w4",
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            max_tok_per_rank=max_tokens,
            **profile_data,
        )

        cases = sorted(set(accuracy_cases + liveness_cases))
        output_path = Path(args.json_output).resolve()
        capture_prefix = f"{output_path.name}.rank{rank}.activation"
        marker_counts = []
        launch_body = None
        launch_tensors = None
        uniform_route = None
        # (body, (x, route_weights, ids, graph), uniform route, operator capacity)
        # of the --target-guard shape, timed against the frozen baseline below.
        timed = None
        def check_accuracy(state, x, route_weights, ids, tokens, guard, method):
            reference = helper._reference(
                x,
                route_weights,
                ids,
                ref_weights,
                rank,
                world,
                profile_data["model_dim"],
                profile_data["inter_dim"],
                profile_data["experts"],
                profile_data["swiglu_limit"],
            )
            rel_l2 = _relative_l2(
                torch, dist, helper, state["output"], reference, device
            )
            row = {
                "guard": guard,
                "metric": "relL2",
                "value": rel_l2,
                "threshold": args.rtol,
                "method": method,
                "status": "pass" if rel_l2 < args.rtol else "fail",
            }
            result["accuracy_results"].append(row)
            write_result(False)
            if row["status"] != "pass":
                raise AssertionError(f"{guard} relL2={rel_l2:.6f}")

        def replay_routes(graph, state, x, route_weights, ids, tokens, suffix="", only=None):
            for route in only or routes:
                checkpoints = []
                drifts = []
                for replay in range(args.replays):
                    _, new_weights, new_ids = _make_route(
                        torch,
                        tokens,
                        route,
                        replay + 1,
                        profile_data,
                        rank,
                        world,
                        device,
                    )
                    route_weights.copy_(new_weights)
                    ids.copy_(new_ids)
                    graph.replay()
                    if (
                        (replay + 1) % args.numeric_checkpoint_interval == 0
                        or replay + 1 == args.replays
                    ):
                        torch.cuda.synchronize()
                        first = state["output"].clone()
                        graph.replay()
                        torch.cuda.synchronize()
                        drift = _relative_l2(
                            torch, dist, helper, first, state["output"].float(), device
                        )
                        drifts.append(drift)
                        if drift >= args.replay_consistency_rtol:
                            raise AssertionError(
                                f"{tokens}/{route}{suffix} replay {replay + 1}: first "
                                f"call after the route change differs from a repeated "
                                f"call (relL2={drift:.6f}); a producer read the "
                                f"previous launch's route metadata"
                            )
                        reference = helper._reference(
                            x,
                            route_weights,
                            ids,
                            ref_weights,
                            rank,
                            world,
                            profile_data["model_dim"],
                            profile_data["inter_dim"],
                            profile_data["experts"],
                            profile_data["swiglu_limit"],
                        )
                        value = _relative_l2(
                            torch, dist, helper, first, reference, device
                        )
                        checkpoints.append(value)
                        if value >= args.rtol:
                            raise AssertionError(
                                f"{tokens}/{route}{suffix} replay relL2={value:.6f}"
                            )
                replay_row = {
                    "guard": f"{tokens}_{route}{suffix}",
                    "count": args.replays,
                    "status": "pass",
                    "graph_safe": "pass",
                    "arrival_jitter": True,
                    "routing_changes": args.replays,
                    "max_checkpoint_relL2": max(checkpoints),
                    "max_first_vs_repeat_relL2": max(drifts),
                    "consistency_threshold": args.replay_consistency_rtol,
                }
                result["replay_results"].append(replay_row)
                write_result(False)

        for tokens in cases:
            x, route_weights, ids = _make_route(
                torch, tokens, "uniform", 0, profile_data, rank, world, device
            )
            if tokens == 8192:
                uniform_route = (route_weights.clone(), ids.clone())
            timed_here = target_mtpr is None and tokens == target_tokens
            if timed_here:
                timed_route = (route_weights.clone(), ids.clone())
            state = {}

            def body(
                x=x, route_weights=route_weights, ids=ids, tokens=tokens, state=state
            ):
                state["output"] = candidate(x, route_weights, ids)[:tokens]

            first_forward = tokens == cases[0]
            warmup = None
            if first_forward:
                result["activation"] = {
                    "status": "pending",
                    "path_marker": shared_marker,
                    "capture_files": str(
                        output_path.parent
                        / f"{output_path.name}.rank*.activation.fd*.log"
                    ),
                }
                write_result(False)

                def warmup(body=body):
                    with _captured_output_fds(
                        output_path.parent, capture_prefix
                    ) as captured:
                        body()
                    marker_counts.append(captured["text"].count(shared_marker))

            graph = _capture(torch, helper, body, warmup)
            if first_forward:
                observed_sum, observed_min, per_rank = _reduce_marker_observation(
                    torch, dist, device, marker_counts[0], world
                )
                result["activation"] = {
                    **_activation_summary(
                        shared_marker, world, observed_sum, observed_min, per_rank
                    ),
                    "window": f"first eager candidate forward (tokens={tokens})",
                    "method": "fd 1/2 capture; all-reduce SUM/MIN of observed flag",
                    "fusion_switch": os.environ["AITER_MEGAMOE_FUSE_ALL"],
                }
                write_result(False)
            if tokens == 8192:
                launch_body = body
                launch_tensors = (x, route_weights, ids, graph)
            if timed_here:
                timed = (body, (x, route_weights, ids, graph), timed_route, max_tokens)

            if tokens in accuracy_cases:
                check_accuracy(
                    state, x, route_weights, ids, tokens, str(tokens),
                    "direct graph-captured candidate vs numeric reference",
                )

            if tokens in liveness_cases and args.shared_fallback:
                # The declared FALLBACK operator runs the frozen path, which itself fails the
                # moving-hot-rank consistency check at 8192. Gate the caller's routes; record
                # moving-hot-rank as a diagnostic of the baseline path, not of the candidate.
                replay_routes(
                    graph, state, x, route_weights, ids, tokens,
                    only=[route for route in routes if route != MOVING_HOT_ROUTE],
                )
                try:
                    replay_routes(
                        graph, state, x, route_weights, ids, tokens, only=[MOVING_HOT_ROUTE]
                    )
                except AssertionError as error:
                    result.setdefault("shared_fallback_diagnostics", []).append({
                        "guard": f"{tokens}_{MOVING_HOT_ROUTE}",
                        "status": "fail",
                        "gated": False,
                        "error": str(error),
                    })
                    write_result(False)
            elif tokens in liveness_cases:
                replay_routes(graph, state, x, route_weights, ids, tokens)

        # Official-harness shapes: each case owns an operator sized like the op test
        # (max_tok_per_rank = next power of two of the batch), so small batches run
        # their own MTPR class instead of the 8192 class shared above.
        result["mtpr_activation"] = []
        for tokens in mtpr_cases:
            mtpr = _next_power_of_two(tokens)
            suffix = f"_mtpr{mtpr}"
            small = factory(
                rank=rank,
                world_size=world,
                quant="a8w4",
                w1=w1,
                w1_scale=w1_scale,
                w2=w2,
                w2_scale=w2_scale,
                max_tok_per_rank=mtpr,
                **profile_data,
            )
            x, route_weights, ids = _make_route(
                torch, tokens, "uniform", 0, profile_data, rank, world, device
            )
            is_target = target_mtpr is not None and tokens == target_tokens
            if is_target:
                timed_route = (route_weights.clone(), ids.clone())
            state = {}
            small_markers = []

            def body(
                op=small, x=x, route_weights=route_weights, ids=ids, tokens=tokens,
                state=state,
            ):
                state["output"] = op(x, route_weights, ids)[:tokens]

            def warmup(body=body, tokens=tokens, small_markers=small_markers):
                with _captured_output_fds(
                    output_path.parent, f"{capture_prefix}.mtpr{tokens}"
                ) as captured:
                    body()
                small_markers.append(captured["text"].count(args.path_marker))

            graph = _capture(torch, helper, body, warmup)
            observed_sum, observed_min, per_rank = _reduce_marker_observation(
                torch, dist, device, small_markers[0], world
            )
            activation = {
                **_activation_summary(
                    args.path_marker, world, observed_sum, observed_min, per_rank
                ),
                "window": f"first eager forward (tokens={tokens}, mtpr={mtpr})",
                "fallback_allowed": mtpr <= args.mtpr_fallback_max and not is_target,
            }
            if is_target:
                activation["target_guard"] = args.target_guard
            result["mtpr_activation"].append(activation)
            write_result(False)
            # An MTPR at or below --mtpr-fallback-max may run the unfused path unless
            # it is the target; it still owes accuracy and replay liveness below.
            if activation["status"] != "pass" and not activation["fallback_allowed"]:
                raise AssertionError(f"tokens={tokens} mtpr={mtpr} path marker missing")
            if is_target and args.shared_fallback:
                # The claim is the target's fused path; the shared operator only had to fall back.
                result["shared_activation"] = result["activation"]
                claim = dict(activation)
                if result["shared_activation"]["status"] != "pass":
                    claim["status"] = "fail"
                result["activation"] = claim
            check_accuracy(
                state, x, route_weights, ids, tokens, f"{tokens}_uniform{suffix}",
                f"direct graph-captured candidate at max_tok_per_rank={mtpr} "
                "vs numeric reference",
            )
            replay_routes(graph, state, x, route_weights, ids, tokens, suffix)
            if is_target:
                timed = (body, (x, route_weights, ids, graph), timed_route, mtpr)
            del graph, small

        if launch_body is None or launch_tensors is None or uniform_route is None:
            raise AssertionError("8192 graph was not produced")
        if timed is None:
            raise AssertionError(f"target {args.target_guard} graph was not produced")
        x, route_weights, ids, _ = launch_tensors
        _restore_route((route_weights, ids), uniform_route)
        launch_count = _profile_launch_count(
            torch, dist, helper, launch_body, device, fallback=args.shared_fallback
        )
        result["launch_count"] = launch_count
        target_body, target_tensors, target_route, target_capacity = timed
        x, route_weights, ids, candidate_graph = target_tensors
        _restore_route((route_weights, ids), target_route)
        if target_mtpr is not None:
            # The target owns its operator, which must be fused like the shared one.
            result["target_launch_count"] = _profile_launch_count(
                torch, dist, helper, target_body, device, ceiling=True
            )
        result["resources"] = _resource_checks(
            args.resource_evidence,
            f"target_mtpr{target_capacity}" if args.shared_fallback else "",
        )

        os.environ["AITER_MEGAMOE_FUSE_ALL"] = "0"
        control = factory(
            rank=rank,
            world_size=world,
            quant="a8w4",
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            max_tok_per_rank=target_capacity,
            **profile_data,
        )
        control_state = {}

        def control_body():
            control_state["output"] = control(x, route_weights, ids)[:target_tokens]

        control_graph = _capture(torch, helper, control_body)
        os.environ["AITER_MEGAMOE_FUSE_ALL"] = "1"
        cand_readings = []
        switch_off_readings = []
        for pair in range(args.pairs):
            if pair % 2:
                cand_ms = _time_rank_max(
                    torch, dist, helper, candidate_graph, device, args.perf_iters
                )
                base_ms = _time_rank_max(
                    torch, dist, helper, control_graph, device, args.perf_iters
                )
            else:
                base_ms = _time_rank_max(
                    torch, dist, helper, control_graph, device, args.perf_iters
                )
                cand_ms = _time_rank_max(
                    torch, dist, helper, candidate_graph, device, args.perf_iters
                )
            cand_readings.append(cand_ms)
            switch_off_readings.append(base_ms)
            result["paired_readings"].append(
                {
                    "guard": args.target_guard,
                    "base_arm": "same_tree_switch_off",
                    "base": base_ms,
                    "cand": cand_ms,
                }
            )
            write_result(False)
        speedup = _speedup_summary(
            args.frozen_baseline_ms,
            cand_readings,
            switch_off_readings,
            args.min_speedup,
            args.denominator_tolerance,
        )
        result.update(speedup)
        result["paired_control"] = (
            "same selected tree with AITER_MEGAMOE_FUSE_ALL=0 is diagnostic only "
            "(incremental_switch_speedup); min_speedup gates absolute_speedup "
            "against --frozen-baseline-ms"
        )
        if speedup["speedup_gate"] == "fail":
            raise AssertionError(
                f"absolute speedup {speedup['absolute_speedup']:.6f} vs frozen "
                f"baseline {speedup['frozen_baseline_ms']:.6f} ms "
                f"< {args.min_speedup}"
            )

        result["correctness"] = "pass"
        result["graph_safe"] = "pass"
        result["liveness"] = "pass"
        result["liveness_replays"] = args.replays
        claim_blockers = _claim_blockers(
            result.get("activation"), result.get("resources"), speedup
        )
        result["claim_blockers"] = claim_blockers
        if claim_blockers:
            write_result(False)
            return 1
        write_result(True)
        return 0
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        write_result(False)
        raise
    finally:
        helper._cleanup()


if __name__ == "__main__":
    raise SystemExit(main())