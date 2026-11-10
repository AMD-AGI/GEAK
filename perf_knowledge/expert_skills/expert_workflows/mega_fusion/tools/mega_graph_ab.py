#!/usr/bin/env python3
"""Fast paired CUDA-graph A/B for MegaMoE V2 EP8 at any tokens/rank.

Each arm is ``NAME=TREE[#KEY=VAL[#KEY=VAL...]]``: a source tree (an aiter checkout
that contains ``op_tests/multigpu_tests/bench_mega_moe_v2.py``) plus environment
switches applied only to that arm. The first arm is the base of every paired
delta. Values may contain commas (``#SWITCH=a,b``). Example::

    mega_graph_ab.py --tokens 64 --mtpr 64 \
        --arm base=$FROZEN_TREE \
        --arm cand=$CAND_TREE#AITER_MEGAMOE_FUSE_ALL=1 \
        --passes 3 --json-output ab.json

Measurement contract:
  1. ONE OPERATOR PER PROCESS: every leg (one arm, one pass) is its own
     ``torchrun`` job; MegaMoEV2 instances are never mixed in one process.
  2. GRAPH TIMING: the leg captures one call in a CUDA graph, warms it, then
     times ``--rounds`` x ``--iters`` replays; each round is the rank-max of the
     per-rank mean. The leg value is the median round.
  3. PAIRING + ROTATION: each pass runs every arm once, rotating the arm order
     per pass; the paired delta of an arm is (arm - base) in the same pass.
     Report the median paired delta and how many passes favour the arm
     (``n_faster``). Graph timing noise is about +/-1 us; a claim of a gain needs
     at least three passes with a consistent sign.
  4. OUTPUT CHECK: every leg saves its per-rank output; non-base legs report
     relL2 and bit-exactness against the base arm's output from an earlier pass
     (same inputs: inputs are seeded by rank). This is a quick screen, not the
     acceptance accuracy test.
  5. ATTRIBUTION: ``[megamoe] path=...`` markers are counted per leg and
     reported; ``--expect-path ARM=PATH`` voids a leg whose markers differ.
  6. RETRY: a leg with rc!=0, no result line, or a marker violation is retried
     ``--attempts`` times; legs that never succeed are listed in
     ``dropped_legs``. The JSON document is rewritten after every leg.
  7. ISOLATION: every arm gets its own FLYDSL_RUNTIME_CACHE_DIR outside every
     tree (``--jit-root``), and PYTHONPATH starts with the arm's tree. The leg
     asserts the operator module was imported from that tree.

``--fake-cmd`` replaces the torchrun leg for CPU-only tests: the command runs
with ARM_NAME/ARM_TREE/AB_PASS in its environment and must print the result
line itself.
"""
import argparse
import json
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time

RESULT_TAG = "MEGA_AB_RESULT "
PATH_MARKER_RE = re.compile(r"\[megamoe\]\s+path=(\S+)")
ROUTES = ("uniform", "rank-mixed-skew", "rank-balanced-hot", "hot-rank0")


# --------------------------------------------------------------------------- leg (inside torchrun)

def _leg(args):
    import torch
    import torch.distributed as dist

    tree = os.path.realpath(args.tree)
    sys.path.insert(0, os.path.join(tree, "op_tests", "multigpu_tests"))
    import bench_mega_moe_v2 as B  # noqa: E402
    import aiter.ops.flydsl.kernels.mega_moe as mm  # noqa: E402

    if not os.path.realpath(mm.__file__).startswith(tree + os.sep):
        raise RuntimeError(f"operator imported from {mm.__file__}, not from {tree}")
    rank, world, device = B.setup_dist()
    local_experts = B.EXPERTS // world
    x, rw, ids = B.make_inputs(args.tokens, rank, world, B.MODEL_DIM, B.EXPERTS, B.TOPK,
                               args.route, args.hot_bias, device)
    w1, w1s, w2, w2s = B.make_weights(local_experts, B.MODEL_DIM, B.INTER_DIM, rank, device)
    mega = mm.MegaMoEV2(rank=rank, world_size=world, model_dim=B.MODEL_DIM,
                        inter_dim=B.INTER_DIM, experts=B.EXPERTS, topk=B.TOPK, quant="a8w4",
                        w1=w1, w1_scale=w1s, w2=w2, w2_scale=w2s,
                        max_tok_per_rank=args.mtpr, swiglu_limit=B.SWIGLU_LIMIT)
    holder = {}

    def body():
        holder["o"] = mega(x, rw, ids)

    graph = B.capture(body)
    for _ in range(2):
        B.time_graph(graph, args.iters, device)
    rounds = [B.time_graph(graph, args.iters, device)[1] for _ in range(args.rounds)]

    out = holder["o"][: args.tokens].float().cpu()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(out, os.path.join(args.out_dir, f"out_rank{rank}.pt"))
    rel, exact, finite = -1.0, 1, int(torch.isfinite(out).all().item())
    ref_path = os.path.join(args.ref_dir, f"out_rank{rank}.pt") if args.ref_dir else ""
    has_ref = int(bool(ref_path) and os.path.exists(ref_path))
    if has_ref:
        ref = torch.load(ref_path)
        rel = ((out - ref).norm() / ref.norm().clamp_min(1e-30)).item()
        exact = int(torch.equal(out, ref))
    t = torch.tensor([rel, -exact, -finite, -has_ref], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    B.barrier()
    if rank == 0:
        doc = {"rounds_ms": rounds, "median_ms": statistics.median(rounds),
               "finite": bool(-t[2].item() > 0)}
        if -t[3].item() > 0:
            doc.update(relL2_vs_base=float(t[0].item()), bitexact_vs_base=bool(-t[1].item() > 0))
        print(RESULT_TAG + json.dumps(doc), flush=True)


# --------------------------------------------------------------------------- parent

def parse_arm(text):
    if "=" not in text:
        raise ValueError(f"arm {text!r} must be NAME=TREE[#KEY=VAL...]")
    name, rest = text.split("=", 1)
    tree, *items = rest.split("#")
    env = {}
    for item in filter(None, (p.strip() for p in items)):
        if "=" not in item:
            raise ValueError(f"arm {name}: env item {item!r} must be KEY=VAL")
        k, v = item.split("=", 1)
        env[k] = v
    if not name or not tree:
        raise ValueError(f"arm {text!r} must be NAME=TREE[#KEY=VAL...]")
    return {"name": name, "tree": os.path.abspath(tree), "env": env}


def rotation(arms, pass_idx):
    k = pass_idx % len(arms)
    return arms[k:] + arms[:k]


def parse_result(text):
    for line in reversed(text.splitlines()):
        pos = line.find(RESULT_TAG)
        if pos >= 0:
            return json.loads(line[pos + len(RESULT_TAG):])
    return None


def summarize(arms, legs):
    """Per-arm medians and per-pass paired deltas (us) against the first arm."""
    base = arms[0]["name"]
    by = {}
    for leg in legs:
        by.setdefault(leg["pass"], {})[leg["arm"]] = leg["median_ms"]
    out = {}
    for arm in arms:
        name = arm["name"]
        vals = [p[name] for p in by.values() if name in p]
        entry = {"n": len(vals), "median_ms": statistics.median(vals) if vals else None}
        if name != base:
            deltas = [(p[name] - p[base]) * 1000.0 for p in by.values() if name in p and base in p]
            entry.update(
                paired_deltas_us=[round(d, 3) for d in deltas],
                median_delta_us=round(statistics.median(deltas), 3) if deltas else None,
                n_pairs=len(deltas),
                n_faster=sum(d < 0 for d in deltas),
            )
            if vals and out[base]["median_ms"]:
                entry["speedup_vs_base"] = round(out[base]["median_ms"] / entry["median_ms"], 4)
        out[name] = entry
    return out


def _run(cmd, env, timeout):
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    try:
        text, _ = proc.communicate(timeout=timeout)
        return proc.returncode, text
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        text, _ = proc.communicate()
        return "timeout", text


def run(args):
    arms = [parse_arm(a) for a in args.arm]
    if len(arms) < 2:
        raise SystemExit("need at least two --arm (the first is the base)")
    names = [a["name"] for a in arms]
    if len(set(names)) != len(names):
        raise SystemExit("arm names must be unique")
    expect = dict(e.split("=", 1) for e in args.expect_path)
    for arm in arms:
        if os.path.realpath(args.jit_root).startswith(os.path.realpath(arm["tree"]) + os.sep):
            raise SystemExit(f"--jit-root must be outside every tree ({arm['tree']})")
    os.makedirs(args.work_dir, exist_ok=True)
    doc = {"tool": "mega_graph_ab", "tokens": args.tokens, "mtpr": args.mtpr, "route": args.route,
           "iters": args.iters, "rounds": args.rounds, "passes": args.passes,
           "arms": arms, "legs": [], "dropped_legs": [], "summary": {}}
    if args.frozen_baseline_ms:
        doc["frozen_baseline_ms"] = args.frozen_baseline_ms

    def flush():
        doc["summary"] = summarize(arms, doc["legs"])
        if args.frozen_baseline_ms:
            for name, entry in doc["summary"].items():
                if entry["median_ms"]:
                    entry["speedup_vs_frozen"] = round(args.frozen_baseline_ms / entry["median_ms"], 4)
        if args.json_output:
            tmp = args.json_output + ".tmp"
            with open(tmp, "w") as f:
                json.dump(doc, f, indent=1)
            os.replace(tmp, args.json_output)

    base_dir = os.path.join(args.work_dir, names[0])
    for p in range(args.passes):
        for arm in rotation(arms, p):
            env = dict(os.environ)
            env.update(MORI_SHMEM_HEAP_SIZE=env.get("MORI_SHMEM_HEAP_SIZE", "40G"),
                       MORI_SOCKET_IFNAME=env.get("MORI_SOCKET_IFNAME", "lo"),
                       FLYDSL_RUNTIME_CACHE_DIR=os.path.join(args.jit_root, arm["name"]),
                       PYTHONPATH=arm["tree"] + os.pathsep + env.get("PYTHONPATH", ""),
                       ARM_NAME=arm["name"], ARM_TREE=arm["tree"], AB_PASS=str(p))
            env.update(arm["env"])
            out_dir = os.path.join(args.work_dir, arm["name"])
            if args.fake_cmd:
                cmd = shlex.split(args.fake_cmd)
            else:
                cmd = ["torchrun", "--standalone", f"--nproc_per_node={args.nproc}",
                       os.path.abspath(__file__), "--leg", "--tree", arm["tree"],
                       "--tokens", str(args.tokens), "--mtpr", str(args.mtpr),
                       "--route", args.route, "--hot-bias", str(args.hot_bias),
                       "--iters", str(args.iters), "--rounds", str(args.rounds),
                       "--out-dir", out_dir,
                       "--ref-dir", base_dir if arm["name"] != names[0] else ""]
            attempts, leg = [], None
            for a in range(args.attempts):
                rc, text = _run(cmd, env, args.leg_timeout)
                with open(os.path.join(args.work_dir, f"{arm['name']}_p{p}_a{a}.log"), "w") as f:
                    f.write(text)
                res = parse_result(text) if rc == 0 else None
                paths = PATH_MARKER_RE.findall(text)
                err = None
                if rc != 0:
                    err = f"rc={rc}"
                elif res is None:
                    err = "no result line"
                elif arm["name"] in expect and (not paths or set(paths) != {expect[arm["name"]]}):
                    err = f"path markers {sorted(set(paths))} != {expect[arm['name']]}"
                if err is None:
                    res.update(arm=arm["name"], **{"pass": p}, attempt=a,
                               path_markers={k: paths.count(k) for k in sorted(set(paths))})
                    doc["legs"].append(res)
                    leg = res
                    break
                attempts.append({"attempt": a, "error": err, "tail": text[-1500:]})
                time.sleep(args.retry_sleep)
            else:
                doc["dropped_legs"].append({"arm": arm["name"], "pass": p, "attempts": attempts})
            flush()
            msg = f"{leg['median_ms']:.4f} ms" if leg else "DROPPED"
            print(f"[mega_graph_ab] pass {p} {arm['name']:>16s}: {msg}", flush=True)
    flush()
    for name, entry in doc["summary"].items():
        print(f"[mega_graph_ab] {name}: " + json.dumps(entry), flush=True)
    return 1 if doc["dropped_legs"] else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leg", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--tree", help=argparse.SUPPRESS)
    ap.add_argument("--out-dir", help=argparse.SUPPRESS)
    ap.add_argument("--ref-dir", default="", help=argparse.SUPPRESS)
    ap.add_argument("--arm", action="append", default=[], help="NAME=TREE[#KEY=VAL...]; first is base")
    ap.add_argument("--tokens", type=int, default=64, help="tokens per rank in the call")
    ap.add_argument("--mtpr", type=int, default=0, help="max_tok_per_rank (default: --tokens)")
    ap.add_argument("--route", choices=ROUTES, default="uniform")
    ap.add_argument("--hot-bias", type=float, default=0.6)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--retry-sleep", type=float, default=5.0)
    ap.add_argument("--leg-timeout", type=float, default=1800.0)
    ap.add_argument("--expect-path", action="append", default=[], help="ARM=PATH, e.g. cand=MEGA")
    ap.add_argument("--frozen-baseline-ms", type=float, default=0.0)
    ap.add_argument("--work-dir", default="")
    ap.add_argument("--jit-root", default="")
    ap.add_argument("--json-output", default="")
    ap.add_argument("--fake-cmd", default="")
    args = ap.parse_args(argv)
    args.mtpr = args.mtpr or args.tokens
    if args.leg:
        _leg(args)
        return 0
    stamp = f"mega_graph_ab_{os.getpid()}_{int(time.time())}"
    args.work_dir = os.path.abspath(args.work_dir or os.path.join("/tmp", stamp))
    args.jit_root = os.path.abspath(args.jit_root or os.path.join("/tmp", stamp + "_jit"))
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
