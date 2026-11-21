#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Kernel profile of an OGA model on a hip-ep package, split into the prefill
pass and the decode passes, with the source file of each kernel.

It runs rocprofv3 --kernel-trace around one model_benchmark run (--prompt_file
of L tokens, -g G -r 1 -w 1) under the serving lock, and writes heads.json.
The trace holds two generations (warmup and timed), each a prefill plus `gen`
decode forwards: model_benchmark runs one decode forward per generated token.
The forwards run back to back, so idle gaps cannot separate them. Instead the
earliest kernel launched exactly 2 x (gen + 1) times marks where each forward
starts (reduce_sum_i64_kernel on Qwen3.8-27B and Qwen3-0.6B); the split is refused
when a kernel of the forward body runs before it. The last gen + 1 forwards
are the timed generation: its first is the prefill, the rest decode. Kernels
are traced to their source files in HIPEP_SRC when it is set.

    python3 kprof.py record --model M --pkg P --out DIR [--len 1024 --gen 32]
"""
import argparse
import collections
import csv
import glob
import os
import re
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import hipep_env  # noqa: E402

KERNELS_DIR = os.path.join("lib", "Runtime", "Kernels")


def short_name(name):
    """Demangled kernel name without `void`, namespaces' anonymity or arguments."""
    n = name.replace("(anonymous namespace)::", "")
    n = re.sub(r"^void ", "", n)
    depth, out = 0, []
    for ch in n:            # cut at the argument list, not at a '(' inside <...>
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0:
            break
        out.append(ch)
    return "".join(out).strip()


def base_name(name):
    """Function name only: no namespace, template arguments or argument list."""
    n = short_name(name)
    n = n.split("<", 1)[0]
    return n.rsplit("::", 1)[-1]


def _pass(kernels):
    """kernels: [(start, end, name)] -> a pass dict."""
    return {"start": kernels[0][0], "end": max(e for _, e, _ in kernels),
            "kernels": [(n, e - s) for s, e, n in kernels],
            "busy_ns": sum(e - s for s, e, _ in kernels)}


def phases(rows, gen, n_gens=2):
    """The prefill and decode forwards of the last generation, split at a per-forward marker."""
    ks = sorted((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"])
                for r in rows)
    if not ks:
        raise ValueError("empty trace")
    names = [base_name(n) for _, _, n in ks]
    counts = collections.Counter(names)
    for per_gen in (gen + 1, gen):
        marks = [n for n in dict.fromkeys(names) if counts[n] == n_gens * per_gen]
        if marks:
            break
    else:
        top = ", ".join(f"{n}x{c}" for n, c in counts.most_common(6))
        raise ValueError(f"no kernel is launched {n_gens} x ({gen} + 1) or {n_gens} x {gen} "
                         f"times ({len(ks)} launches; most common: {top})")
    starts = [i for i, n in enumerate(names) if n == marks[0]]
    # The marker must open a forward: only one-off setup (and the HIP runtime's own copy and
    # fill kernels) may run before its first launch. A kernel of the forward body there means
    # the marker sits mid-forward and every window would straddle two forwards.
    body = {n for n in names[:starts[0]] if not n.startswith("__amd_rocclr_")} & set(names[starts[0]:])
    if body:
        raise ValueError(f"{marks[0]} does not open a forward: {', '.join(sorted(body)[:3])} "
                         f"run before its first launch and again after")
    bounds = starts + [len(ks)]
    fw = [_pass(ks[a:b]) for a, b in zip(bounds, bounds[1:])][-per_gen:]
    if fw[0]["busy_ns"] < max(p["busy_ns"] for p in fw):
        raise ValueError(f"the first forward of the timed generation is not its largest "
                         f"(marker {marks[0]}), so the prefill cannot be told apart")
    return {"prefill": fw[0], "decode": fw[1:], "marker": marks[0]}


def table(passes, top=15, sources=None):
    """Kernel shares over a list of passes, by base name (template variants together)."""
    ms, calls = collections.Counter(), collections.Counter()
    for p in passes:
        for name, dur in p["kernels"]:
            ms[base_name(name)] += dur / 1e6
            calls[base_name(name)] += 1
    busy = sum(ms.values()) or 1.0
    by_base = [{"base": b, "calls": calls[b], "ms": round(v, 4), "share": v / busy,
                **({"source": sources.find(b)} if sources else {})} for b, v in ms.most_common(top)]
    return {"busy_ms": round(busy, 4), "by_base": by_base}


class SourceIndex:
    """Find the hip-ep kernel source that defines a kernel."""

    def __init__(self, src_root):
        self.root = src_root
        self.files = []
        for ext in ("hip", "h", "cpp", "cuh"):
            self.files += glob.glob(os.path.join(src_root, KERNELS_DIR, "**", f"*.{ext}"),
                                    recursive=True)
        self.files.sort()
        self._text = {}

    def _read(self, path):
        if path not in self._text:
            with open(path, errors="replace") as f:
                self._text[path] = f.read()
        return self._text[path]

    def find(self, base):
        """The file declaring `base` as a __global__ kernel, else the one naming it most."""
        word = re.compile(rf"\b{re.escape(base)}\b")
        decl = re.compile(rf"__global__[^;{{]*?\b{re.escape(base)}\b", re.DOTALL)
        best, best_n = None, 0
        for path in self.files:
            text = self._read(path)
            if decl.search(text):
                return os.path.relpath(path, self.root)
            n = len(word.findall(text))
            if n > best_n:
                best, best_n = path, n
        return os.path.relpath(best, self.root) if best else None


def summarize(trace_csv, src_root, gen, top=15):
    with open(trace_csv) as f:
        rows = list(csv.DictReader(f))
    ph = phases(rows, gen)
    idx = SourceIndex(src_root) if src_root else None
    pre = table([ph["prefill"]], top, idx)
    pre["span_ms"] = round((ph["prefill"]["end"] - ph["prefill"]["start"]) / 1e6, 4)
    dec = table(ph["decode"], top, idx) if ph["decode"] else None
    if dec:
        dec["steps"] = len(ph["decode"])
        dec["busy_ms_per_step"] = round(dec["busy_ms"] / len(ph["decode"]), 4)
        dec["median_step_busy_ms"] = round(
            statistics.median(p["busy_ns"] for p in ph["decode"]) / 1e6, 4)
    return {"trace": os.path.abspath(trace_csv), "marker": ph["marker"],
            "phases": {"prefill": pre, "decode": dec}}


def _traces(out_dir):
    return glob.glob(os.path.join(out_dir, "**", "*kernel_trace.csv"), recursive=True)


def record(env, model, out_dir, length, gen, timeout=3600):
    os.makedirs(out_dir, exist_ok=True)
    for old in _traces(out_dir):  # a rerun must not read an earlier run's trace
        os.remove(old)
    exe = os.path.join(env.pkg, "bin", "model_benchmark")
    prompt = bench.prompt_file(model, length, out_dir)
    app = hipep_env.pkg_argv(env, exe, ["-i", model, "--prompt_file", prompt, "-g", str(gen),
                                        "-r", "1", "-w", "1", "-b", "1"])
    cmd = [sys.executable, os.path.join(env.therock, "bin", "rocprofv3"), "--kernel-trace",
           "--output-format", "csv", "-d", out_dir, "-o", "run", "--", *app]
    home = hipep_env.pkg_home(out_dir, env.pkg)
    extra = {"ROCM_PATH": env.therock}
    with hipep_env.serving_lock(env):
        p = subprocess.run(cmd, cwd=env.lib, env=hipep_env.pkg_env(env, home, extra),
                           capture_output=True, text=True, timeout=timeout, check=False)
    with open(os.path.join(out_dir, "record.log"), "w") as f:
        f.write(p.stdout + p.stderr)
    traces = _traces(out_dir)
    if p.returncode != 0 or len(traces) != 1:
        raise RuntimeError(f"rocprofv3 exited {p.returncode}, {len(traces)} traces; "
                           f"log: {out_dir}/record.log")
    return traces[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--model", required=True)
    r.add_argument("--pkg", default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--len", type=int, default=1024)
    r.add_argument("--gen", type=int, default=32)
    a = ap.parse_args()

    out = os.path.join(a.out, "heads.json")
    if os.path.exists(out):
        os.remove(out)  # a failed run must not leave an earlier profile behind
    env = hipep_env.Env.from_environ(a.pkg)
    trace = record(env, a.model, a.out, a.len, a.gen)
    heads = {"model": a.model, "pkg": env.pkg, "len": a.len, "gen": a.gen,
             "kernel_lib_md5": hipep_env.md5(hipep_env.kernel_lib(env.pkg)),
             **summarize(trace, os.environ.get("HIPEP_SRC"), a.gen)}
    hipep_env.write_json(out, heads)
    for ph in ("prefill", "decode"):
        t = heads["phases"][ph]
        if not t:
            continue
        print(f"[kprof] {ph}: busy {t['busy_ms']:.1f} ms" +
              (f" over {t['steps']} steps" if ph == "decode" else ""))
        for k in t["by_base"][:6]:
            print(f"    {100 * k['share']:5.1f}%  {k['ms']:9.2f} ms  {k['base']}  ({k.get('source')})")
    print(f"[kprof] wrote {out}")


if __name__ == "__main__":
    main()
