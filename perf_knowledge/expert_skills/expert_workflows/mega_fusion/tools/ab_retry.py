#!/usr/bin/env python3
"""Portable retrying interleaved A/B runner for MegaMoE V2 EP8.

Runtime paths come only from the environment. The runner emits attribution
diagnostics and binds both marker count and expected path value.

Measurement contract:
  1. RETRY: any attempt with rc!=0, a parse failure, or a marker-count violation
     is retried up to --attempts times with --retry-sleep seconds between tries.
     A leg counts ONLY if an attempt succeeded.
  2. DROPPED LEGS ARE VISIBLE: a leg that exhausts its attempts is appended to
     doc["dropped_legs"] with every attempt's rc and error tail. Never silently absent.
  3. MARKER GATE: --expect-markers N asserts N '[megamoe] path=...' lines per leg
     (one per rank); --expect-path / arm.expect_paths asserts their exact values.
     A FUSED arm that prints eight SCATTERED markers is VOID, not a valid count.
     Guards listed in arm.fallback_guards record markers without asserting them.
  4. INCREMENTAL: the full aggregate doc (records + pairs + dropped legs +
     claim_complete) is rewritten after EVERY leg, so a kill at any point leaves a
     complete, readable claim rather than a fragment.
  5. ROTATION: --rotate changes arm-within-block position to avoid systematic
     first-arm/cache bias.
  6. KNOWN POST-RESULT TEARDOWN: rc 1/-6 is accepted only when a complete
     [RESULT] precedes MORI's exact CheckStatusValid assertion. Every other nonzero
     exit remains VOID, and the accepted rc is retained in the record.

This file does NOT modify ab_runner.py, the bench, or the COMMANDMENT.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

RESULT_RE = re.compile(
    r"\[RESULT\].*?tokens=(\d+).*?mega_e2e=([0-9.]+)/([0-9.]+)ms.*?"
    r"stage1=([0-9.]+)/([0-9.]+)ms stage2_combine=([0-9.]+)/([0-9.]+)ms"
)
MARKER_RE = re.compile(r"^.*?\[megamoe\]\s+path=\S+.*$", re.M)
PATH_MARKER_RE = re.compile(r"\[megamoe\]\s+path=(\S+)")
MORI_POST_RESULT_TEARDOWN_RE = re.compile(
    r"mori/(?:include|src)/mori/shmem/[^\n:]+:\d+:.*?"
    r"ShmemStates::CheckStatusValid\(\): Assertion [`']false['`] failed\."
)
UNRELATED_FATAL_RE = re.compile(
    r"hip failed with out of memory|out of memory|memory access fault|"
    r"segmentation fault|device-side assert|"
    r"(?:Runtime|Value|Type|Key|Index|Assertion|ZeroDivision)Error:",
    re.IGNORECASE,
)
PYTHON_EXCEPTION_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)):"
)
UNRELATED_SIGNAL_RE = re.compile(r"\bSignal (?!6\b)\d+\b", re.IGNORECASE)

GUARDS = {
    "8192_uniform": ("8192", "uniform"),
    "8192_rank-mixed-skew": ("8192", "rank-mixed-skew"),
    "512_uniform": ("512", "uniform"),
    "512_rank-mixed-skew": ("512", "rank-mixed-skew"),
}
# Any other guard id is `<tokens>_<route>[_mtpr<M>]`; without the suffix the bench
# keeps its own default max-tokens-per-rank.
GUARD_RE = re.compile(r"^(\d+)_([A-Za-z][A-Za-z0-9-]*?)(?:_mtpr(\d+))?$")


def parse_guard(guard):
    """Return (tokens, route, mtpr-or-None) for a guard id."""
    if guard in GUARDS:
        return (*GUARDS[guard], None)
    m = GUARD_RE.match(guard)
    if not m:
        raise ValueError(f"unknown guard id {guard!r}: expected <tokens>_<route>[_mtpr<M>]")
    return m.group(1), m.group(2), m.group(3)


def pool_free_gib():
    """Per-card free VRAM in GiB, read from sysfs (works without rocm-smi and
    without seeing the foreign container's PIDs)."""
    import glob
    out = []
    for u in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_used")):
        t = u.replace("vram_used", "vram_total")
        try:
            used = int(open(u).read().strip())
            tot = int(open(t).read().strip())
        except Exception:
            continue
        out.append(round((tot - used) / 1073741824.0, 1))
    return out


def wait_for_pool(min_free_gib, wait_s, poll_s=15):
    """Block until EVERY card has min_free_gib free, or give up. Returns (ok, sample).

    A foreign process may be invisible in this container's PID namespace while
    still holding VRAM. Gate every collective launch on per-card free memory
    instead of discovering contention from a distributed OOM.
    """
    if not min_free_gib:
        return True, pool_free_gib()
    t0 = time.time()
    while True:
        f = pool_free_gib()
        if f and min(f) >= min_free_gib:
            return True, f
        if time.time() - t0 >= wait_s:
            return False, f
        print(f"[ab_retry] pool not ready (min free {min(f) if f else '?'} GiB < "
              f"{min_free_gib}); waiting {poll_s}s", flush=True)
        time.sleep(poll_s)


def _attempt(tree, env_extra, guard, iters, logdir, tag, attempt, fake_cmd, timeout):
    tokens, route, mtpr = parse_guard(guard)
    env = dict(os.environ)
    # Runtime roots are supplied by the caller/environment. Never bake a checkout
    # or cache path into this repository.
    mori = os.environ.get("AB_RETRY_MORI_ROOT") or os.environ.get("MORI_ROOT", "")
    python_paths = [tree]
    if mori:
        python_paths.extend([mori, os.path.join(mori, "python")])
    if env.get("PYTHONPATH"):
        python_paths.extend(p for p in env["PYTHONPATH"].split(os.pathsep) if p)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(python_paths))
    jit_dir = os.environ.get("AB_RETRY_JIT_DIR") or os.environ.get("AITER_JIT_DIR")
    if jit_dir:
        env["AITER_JIT_DIR"] = jit_dir
    elif not fake_cmd:
        return 2, (
            "configuration error: set AB_RETRY_JIT_DIR or AITER_JIT_DIR to a "
            "standalone writable cache outside the candidate tree\n"
        ), 0.0
    if jit_dir and not fake_cmd:
        tree_real = os.path.realpath(tree)
        jit_real = os.path.realpath(jit_dir)
        if os.path.commonpath([tree_real, jit_real]) == tree_real:
            return 2, (
                "configuration error: AITER JIT cache must be outside the candidate tree\n"
            ), 0.0
    env["MORI_SOCKET_IFNAME"] = os.environ.get("AB_RETRY_SOCKET_IFNAME", "lo")
    env["MORI_SHMEM_HEAP_SIZE"] = os.environ.get("AB_RETRY_SHMEM_HEAP", "40G")
    env.update({k: str(v) for k, v in env_extra.items()})
    if fake_cmd:
        cmd = ["bash", "-c", fake_cmd]
        env["FAKE_GUARD"] = guard
        env["FAKE_TOKENS"] = tokens
        env["FAKE_ROUTE"] = route
        env["FAKE_MTPR"] = mtpr or ""
        env["FAKE_ATTEMPT"] = str(attempt)
        env["FAKE_TAG"] = tag
    else:
        cmd = [
            "torchrun", "--standalone", "--nproc_per_node=8",
            "op_tests/multigpu_tests/bench_mega_moe_v2.py",
            "--tokens", tokens, "--route", route, "--iters", str(iters), "--mega-only",
        ]
        if mtpr:
            cmd += ["--mtpr", mtpr]
    t0 = time.time()
    try:
        # Keep stdout/stderr in one pipe so `[RESULT]` -> teardown ordering is real.
        # Concatenating separately captured streams always places stdout first and can
        # falsely turn a pre-result stderr failure into a post-result failure.
        p = subprocess.run(
            cmd, cwd=tree, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=timeout,
        )
        rc, out = p.returncode, p.stdout
    except subprocess.TimeoutExpired as e:
        rc = 124
        chunks = []
        for chunk in (e.stdout, e.stderr):
            if isinstance(chunk, bytes):
                chunks.append(chunk.decode("utf-8", "replace"))
            elif chunk:
                chunks.append(str(chunk))
        out = "".join(chunks)
    if logdir:
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, f"{tag}.a{attempt}.log"), "w") as f:
            f.write(out)
    return rc, out, round(time.time() - t0, 1)


def run_leg(tree, env_extra, guard, iters, logdir, tag, attempts, retry_sleep,
            expect_markers, fake_cmd, timeout, min_free_gib=0, pool_wait_s=0,
            expect_paths=None):
    """Return (record_or_None, dropped_or_None). A leg counts only if an attempt succeeded."""
    tries = []
    mtpr = parse_guard(guard)[2]
    for k in range(1, attempts + 1):
        ok, sample = wait_for_pool(min_free_gib, pool_wait_s)
        if not ok:
            tries.append({"attempt": k, "rc": None, "wall_s": 0,
                          "void": f"pool not ready (free GiB {sample})", "tail": ""})
            print(f"[ab_retry] {tag} attempt {k}/{attempts} SKIPPED: pool not ready "
                  f"{sample}", flush=True)
            continue
        rc, out, wall = _attempt(tree, env_extra, guard, iters, logdir, tag, k,
                                 fake_cmd, timeout)
        markers = sorted(set(MARKER_RE.findall(out)))
        paths = PATH_MARKER_RE.findall(out)
        m = RESULT_RE.search(out)
        teardown = MORI_POST_RESULT_TEARDOWN_RE.search(out)
        exception_types = set(PYTHON_EXCEPTION_RE.findall(out))
        accepted_teardown = bool(
            rc in (1, -6)
            and m
            and teardown
            and m.end() < teardown.start()
            and not UNRELATED_FATAL_RE.search(out)
            and exception_types.issubset({"ChildFailedError"})
            and not UNRELATED_SIGNAL_RE.search(out)
        )
        void = None
        if rc != 0 and not accepted_teardown:
            void = f"rc={rc}"
        elif not m:
            void = "no [RESULT] line"
        elif mtpr and not re.search(rf"\bmtpr={mtpr}\b", m.group(0)):
            void = f"[RESULT] line does not report mtpr={mtpr}"
        elif expect_markers is not None and len(paths) != expect_markers:
            void = f"marker count {len(paths)} != {expect_markers}"
        expected_paths = ({str(expect_paths)} if isinstance(expect_paths, str)
                          else {str(v) for v in (expect_paths or [])})
        actual_paths = set(paths)
        if void is None and expected_paths and actual_paths != expected_paths:
            void = (f"path markers {sorted(actual_paths)} != expected "
                    f"{sorted(expected_paths)}")
        tries.append({"attempt": k, "rc": rc, "wall_s": wall, "void": void,
                      "tail": out[-800:] if void else ""})
        if void is None:
            rec = {
                "guard": guard, "tree": tree,
                "env": {k2: str(v) for k2, v in env_extra.items()},
                "rc": rc, "accepted_post_result_teardown": accepted_teardown,
                "wall_s": wall, "attempt": k, "attempts_used": k,
                "e2e_mean_ms": float(m.group(2)), "e2e_max_ms": float(m.group(3)),
                "stage1_mean_ms": float(m.group(4)), "stage1_max_ms": float(m.group(5)),
                "stage2_combine_mean_ms": float(m.group(6)),
                "stage2_combine_max_ms": float(m.group(7)),
                "markers": markers,
                "n_path_markers": len(paths),
                "path_marker_values": sorted(set(paths)),
            }
            rec["residual_max_ms"] = round(
                rec["e2e_max_ms"] - rec["stage1_max_ms"] - rec["stage2_combine_max_ms"], 4)
            return rec, None
        print(f"[ab_retry] {tag} attempt {k}/{attempts} VOID ({void}); "
              f"{'retrying after %ds' % retry_sleep if k < attempts else 'DROPPING LEG'}",
              flush=True)
        if k < attempts:
            time.sleep(retry_sleep)
    return None, {"guard": guard, "tag": tag, "tree": tree,
                  "env": {k2: str(v) for k2, v in env_extra.items()},
                  "attempts": tries, "reason": tries[-1]["void"]}


def pct_pairs(records, base_arm, cand_arm, field="e2e_max_ms"):
    b = [r for r in records if r["arm"] == base_arm and r.get(field)]
    c = [r for r in records if r["arm"] == cand_arm and r.get(field)]
    return [round((b[i][field] - c[i][field]) / b[i][field] * 100, 4)
            for i in range(min(len(b), len(c)))]


def _med(xs):
    return round(statistics.median(xs), 4) if xs else None


def kernel_time_block(g, base, cand):
    """The numbers the attribution gate needs, next to the e2e numbers it must not be
    confused with.

    A launch-structure change needs mechanism attribution next to its operator result.
    End-to-end and separately captured stage timers can disagree under fusion; that
    disagreement must be visible, but the sum is diagnostic rather than a generally
    valid fused-kernel score: the
    timers may come from separate graphs and different rank maxima. This block reports:

      stage1_ms / stage2_combine_ms   the two per-kernel timers, rank-max, median over pairs
      kernel_sum_ms                   their sum -- for a stage1+stage2 fusion this is the
                                      `replaced_sum_us` side of the comparison
      residual_ms                     e2e minus the two timers, i.e. the launch gaps

    It deliberately does NOT decide which kernel is "the changed one": only the patch
    author knows that. Fill verify's `attribution.changed_us` / `replaced_sum_us` only
    when the collection makes them genuinely comparable on the same timeline/rank.
    Otherwise retain the operator e2e score and report this block as a mechanism caveat.
    """
    out = {}
    for arm in (base, cand):
        rs = [r for r in g if r["arm"] == arm]
        s1 = _med([r["stage1_max_ms"] for r in rs])
        s2 = _med([r["stage2_combine_max_ms"] for r in rs])
        out[arm] = {
            "stage1_ms": s1, "stage2_combine_ms": s2,
            "kernel_sum_ms": round(s1 + s2, 4) if s1 is not None and s2 is not None else None,
            "residual_ms": _med([r["residual_max_ms"] for r in rs]),
            "n": len(rs),
        }
    b, c = out[base].get("kernel_sum_ms"), out[cand].get("kernel_sum_ms")
    # b or c is None on every write before both arms have produced a leg -- the doc is
    # rewritten after EVERY leg on purpose, so this function must survive a half-filled
    # record set rather than take the whole run down with it.
    out["kernel_sum_pct"] = round((b - c) / b * 100, 4) if b and c else None
    out["_note"] = ("kernel_sum_pct is positive when the separately reported kernel timers are "
                    "smaller. A disagreement with e2e requires mechanism investigation; this sum "
                    "is not a promotion metric unless both arms were measured on one comparable "
                    "timeline and rank.")
    return out


def build_doc(records, dropped, plan, complete):
    names = [x["name"] for x in plan["arms"]]
    base = plan.get("base_arm", names[0])
    doc = {"records": records, "dropped_legs": dropped,
           "n_dropped": len(dropped), "claim_complete": bool(complete),
           "base_arm": base, "arms": names,
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "pool_free_gib_at_write": pool_free_gib()}
    for guard in sorted({g for g, _ in plan["sequence"]}):
        g = [r for r in records if r["guard"] == guard]
        gd = {"n_legs": len(g), "n_dropped": len([d for d in dropped if d["guard"] == guard])}
        for n in names:
            if n == base:
                continue
            pr = pct_pairs(g, base, n)
            braw = [r["e2e_max_ms"] for r in g if r["arm"] == base]
            craw = [r["e2e_max_ms"] for r in g if r["arm"] == n]
            gd[f"{base}_vs_{n}"] = {
                "e2e_pairs": pr, "n": len(pr),
                "median": round(statistics.median(pr), 4) if pr else None,
                "n_pos": sum(1 for x in pr if x > 0),
                "base_raw": braw, "cand_raw": craw,
                "complete_sep": bool(braw and craw and min(braw) > max(craw)),
                "s2c_pairs": pct_pairs(g, base, n, "stage2_combine_max_ms"),
                "s1_pairs": pct_pairs(g, base, n, "stage1_max_ms"),
                "markers": sorted({v for r in g if r["arm"] == n
                                   for v in r.get("path_marker_values", [])}),
                "kernel_time": kernel_time_block(g, base, n),
            }
        doc.setdefault("by_guard", {})[guard] = gd
    return doc


def rotate_sequence(guards, arms, blocks):
    """Rotate arm-within-block position per block (anti position-artifact)."""
    seq = []
    for g in guards:
        for b in range(blocks):
            order = arms[b % len(arms):] + arms[:b % len(arms)]
            for a in order:
                seq.append([g, a])
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--logdir", default="")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--retry-sleep", type=int, default=15)
    ap.add_argument("--expect-markers", type=int, default=None)
    ap.add_argument("--expect-path", action="append", default=[],
                    help="exact allowed [megamoe] path value; repeat for multiple")
    ap.add_argument("--fake-cmd", default="",
                    help="self-test only: run this shell command instead of the bench")
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--min-free-gib", type=float, default=0,
                    help="require this much free VRAM on EVERY card before an attempt")
    ap.add_argument("--pool-wait-s", type=int, default=0,
                    help="bounded wait for the pool-readiness gate, per attempt")
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    arms = {x["name"]: x for x in plan["arms"]}
    iters = plan.get("iters", 10)
    if "sequence" not in plan:
        plan["sequence"] = rotate_sequence(plan["guards"],
                                           [x["name"] for x in plan["arms"]],
                                           plan.get("blocks", 2))
    for guard in {g for g, _ in plan["sequence"]}:
        parse_guard(guard)
    records, dropped = [], []
    n = len(plan["sequence"])
    # write an (empty) complete-shaped doc up front: a kill before leg 1 still leaves a readable file
    json.dump(build_doc(records, dropped, plan, False), open(a.out, "w"), indent=1)
    for i, (guard, armname) in enumerate(plan["sequence"]):
        arm = arms[armname]
        tag = f"{i:03d}_{guard}_{armname}"
        print(f"[ab_retry] {tag} ({i+1}/{n}) ...", flush=True)
        em = arm.get("expect_markers", a.expect_markers)
        ep = arm.get("expect_paths", a.expect_path)
        if guard in arm.get("fallback_guards", []):
            # The unfused path is allowed here: path markers are recorded, not required.
            em, ep = None, []
        rec, drop = run_leg(arm["tree"], arm.get("env", {}), guard, iters, a.logdir,
                            tag, a.attempts, a.retry_sleep, em, a.fake_cmd, a.timeout,
                            a.min_free_gib, a.pool_wait_s, ep)
        if rec is not None:
            rec["arm"] = armname
            rec["tag"] = tag
            records.append(rec)
            print(f"[ab_retry] {tag} OK e2e_max={rec['e2e_max_ms']} "
                  f"markers={rec['n_path_markers']}{rec['path_marker_values']} "
                  f"attempts={rec['attempts_used']}", flush=True)
        else:
            drop["arm"] = armname
            dropped.append(drop)
            print(f"[ab_retry] {tag} DROPPED after {a.attempts} attempts: {drop['reason']}",
                  flush=True)
        # refresh the FULL aggregate after every leg
        json.dump(build_doc(records, dropped, plan, i == n - 1), open(a.out, "w"), indent=1)
    print(f"[ab_retry] wrote {a.out}  legs={len(records)} dropped={len(dropped)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
