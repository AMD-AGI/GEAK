#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Refresh hip-ep's autotune tables for one model and rebuild the kernel library
with them. This is the tuning lane: the configs come from hip-ep's own sweep
drivers, read off their output by hip-ep's own update_lut.py; nothing here judges
which config is fastest.

Each op refreshes its table, hip/autotune/<op>/lut/<arch>.{json,fb}, for the model:
  - matmul_nbits: the model's 4-bit shapes (hip-ep's extract_shapes.py on the
    decoder graph; an export's mtp.onnx is not part of a plain forward), on the
    prefill and padded-row paths with extended entries. A row is replaced only by
    a config at least MIN_GAIN faster in the same sweep;
  - gqa: the model's attention geometry (genai_config.json), prefill and decode,
    when hip-ep's sweep grid has that geometry.
Every op sweeps on a private copy of the kernel sources at the release tag plus
--overlay, so the sweep measures the code that will ship and the shared hip-ep tree
is never touched, with a driver built by the release's compile command for the op's
kernel, under the serving-GPU lock. The output overlay is --overlay plus every
changed table, and build_kernels.sh builds the library from it. The integrator
passes the kernel lane's overlay, so the tables are re-swept on the changed kernel. lut_refresh.json
records, per op, whether its table changed or why it was skipped. The hip-ep
checkout and build dir are HIPEP_SRC and HIPEP_BUILD, the ones build_kernels.sh
reads, so the tables are swept on the tree the library is built from.

    python3 lut_refresh.py --model M --out DIR --arch gfx1151 [--ops matmul_nbits,gqa] [--overlay IN]
"""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hipep_env  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
K = hipep_env.K
# matmul_nbits is swept at these prefill M; a new config replaces a row's only when it is
# at least MIN_GAIN faster in the same sweep.
SWEEP_M = "64,128,256,512,1024,2048,4096"
MIN_GAIN = "0.03"


def _run(cmd, log, env=None, cwd=None):
    with open(log, "a") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"{os.path.basename(cmd[0])} exited {p.returncode}; log: {log}")


def _compile(compiler, flags, inc, path, obj, log):
    _run([compiler, "-c", *flags, "-x", "hip", *inc, path, "-o", obj], log)
    return obj


def _link(compiler, kflags, objs, exe, log):
    offload = next(f for f in kflags if f.startswith("--offload-arch="))
    _run([compiler, "--hip-link", offload, *objs, "-o", exe], log)
    return exe


def extract_shapes(root, model, out_dir, log):
    """hip-ep's extract_shapes.py over the decoder graph alone. Returns the CSV and its row count."""
    mdir = os.path.join(out_dir, "models", os.path.basename(os.path.normpath(model)))
    os.makedirs(mdir, exist_ok=True)
    link = os.path.join(mdir, "model.onnx")
    if not os.path.lexists(link):
        os.symlink(os.path.join(os.path.abspath(model), "model.onnx"), link)
    csv = os.path.join(out_dir, "shapes.csv")
    _run([sys.executable, os.path.join(root, hipep_env.autotune("matmul_nbits"), "scripts",
                                       "extract_shapes.py"),
          "--models", os.path.join(out_dir, "models"), "--out", csv, "--bits", "4"], log)
    # The vocab projection (N >= 65536) is left out, as hip-ep's own sweep scope
    # does: tens of minutes per point, for an M inference reaches only on a
    # full-sequence-logits path.
    sweep_csv, rows = os.path.join(out_dir, "shapes_sweep.csv"), 0
    with open(csv) as f, open(sweep_csv, "w") as g:
        header = f.readline()
        g.write(header)
        n_col = header.strip().split(",").index("N")
        for line in f:
            if line.strip() and int(line.split(",")[n_col]) < 65536:
                g.write(line)
                rows += 1
    return sweep_csv, rows


def geometry(model):
    """(query heads, KV heads, head size) of the model's attention, from genai_config.json."""
    with open(os.path.join(model, "genai_config.json")) as f:
        d = json.load(f)["model"]["decoder"]
    return d["num_attention_heads"], d["num_key_value_heads"], d["head_size"]


def gqa_grid(root):
    """The (H, KV, d) geometries hip-ep's gqa update_lut.py can sweep (its GEOMETRIES)."""
    path = os.path.join(root, hipep_env.autotune("gqa"), "scripts", "update_lut.py")
    spec = importlib.util.spec_from_file_location("gqa_update_lut", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {tuple(g[:3]) for g in mod.GEOMETRIES}


class MatMulNBits:
    name = "matmul_nbits"

    @staticmethod
    def arch(gfx):
        return gfx

    @staticmethod
    def plan(root, model, out, log):
        """(measure args, build args), or None and why the op is skipped."""
        shapes, rows = extract_shapes(root, model, out, log)
        if not rows:
            return None, "the model has no 4-bit MatMulNBits shape to sweep"
        return (["--shapes", shapes, "--m", SWEEP_M, "--sweep-args", "--extended", "--no-decode",
                 "--no-dp4a"],
                ["--merge-existing", "--phase", "Prefill", "--min-gain", MIN_GAIN,
                 "--report", os.path.join(out, "matmul_nbits_diff.md")]), ""

    @staticmethod
    def build_sweep(root, build, out, therock, log):
        compiler, kflags = hipep_env.compile_flags(
            hipep_env.product_command(build, "hip/matmul_nbits_kernel.hip.o"))
        offload = next(f for f in kflags if f.startswith("--offload-arch="))
        at = os.path.join(root, hipep_env.autotune("matmul_nbits"))
        inc = [f"-I{os.path.join(root, K, 'include')}", f"-I{os.path.join(root, K, 'hip')}",
               f"-I{at}", f"-I{os.path.join(build, 'schemas')}",
               f"-I{os.path.join(build, '_deps', 'flatbuffers-src', 'include')}",
               "-isystem", os.path.join(therock, "include")]
        host = ["-O3", "-std=c++17", "-w", offload, "-D__HIP_PLATFORM_AMD__=1", "-D__HIP_ROCclr__=1"]
        objs = [_compile(compiler, flags, inc, path, os.path.join(out, f"mn_{name}.o"), log)
                for name, flags, path in (
                    ("kernel", kflags, os.path.join(root, K, "hip", "matmul_nbits_kernel.hip")),
                    ("autotune", host, os.path.join(at, "matmul_nbits_autotune.cpp")),
                    ("sweep", host, os.path.join(at, "tools", "matmul_nbits_autotune_sweep.cpp")),
                    ("empty_lut", host, os.path.join(at, "tools", "empty_lut_data.cpp")))]
        return _link(compiler, kflags, objs, os.path.join(out, "mn_sweep"), log)


class Gqa:
    name = "gqa"

    @staticmethod
    def arch(gfx):
        return gfx[len("gfx"):]  # its update_lut.py takes the arch without the prefix

    @staticmethod
    def plan(root, model, out, log):
        geo = geometry(model)
        g = ":".join(str(x) for x in geo)
        if geo not in gqa_grid(root):
            return None, f"geometry {g} (heads:KV heads:head size) is not in hip-ep's gqa sweep grid"
        data = os.path.join(out, "gqa_data")
        return (["--geometry", g, "--data", data],
                ["--merge-existing", "--geometry", g, "--data", data]), ""

    @staticmethod
    def build_sweep(root, build, out, therock, log):
        # The sweep #includes the kernel source, so it is the one TU (hip-ep's tools/Makefile).
        compiler, kflags = hipep_env.compile_flags(hipep_env.product_command(build, "hip/gqa_kernel.hip.o"))
        inc = [f"-I{os.path.join(root, K, 'include')}", f"-I{os.path.join(root, K, 'hip')}",
               "-isystem", os.path.join(therock, "include")]
        src = os.path.join(root, hipep_env.autotune("gqa"), "tools", "gqa_autotune_sweep.cpp")
        obj = _compile(compiler, [*kflags, "-w"], inc, src, os.path.join(out, "gqa_sweep.o"), log)
        return _link(compiler, kflags, [obj], os.path.join(out, "gqa_sweep"), log)


OPS = {op.name: op for op in (MatMulNBits, Gqa)}


def clear_outputs(out):
    """What the tuner reads must come from this run, never from an earlier one in the same dir."""
    for d in ("overlay", "kbuild"):
        if os.path.isdir(os.path.join(out, d)):
            shutil.rmtree(os.path.join(out, d))
    for f in ["lut_refresh.json"] + [f"{op}_diff.md" for op in OPS]:
        if os.path.exists(os.path.join(out, f)):
            os.remove(os.path.join(out, f))


def private_tree(src, overlay, out_dir):
    """lib/Runtime/Kernels/{hip,include} at the release tag, with the overlay applied."""
    root = os.path.join(out_dir, "src")
    for sub in ("hip", "include"):
        dst = os.path.join(root, K, sub)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
    hipep_env.export_tree(src, root, [f"{K}/hip", f"{K}/include"])
    if overlay:
        shutil.copytree(overlay, root, dirs_exist_ok=True)
    return root


def write_overlay(overlay_in, root, luts, overlay_out):
    """The overlay to build: the input overlay (a kernel lane's edits) plus each changed table,
    given as its path under root without the extension."""
    if overlay_in:
        shutil.copytree(overlay_in, overlay_out)
    for lut in luts:
        dst = os.path.join(overlay_out, os.path.relpath(os.path.dirname(lut), root))
        os.makedirs(dst, exist_ok=True)
        for ext in ("json", "fb"):
            shutil.copy2(f"{lut}.{ext}", dst)


def _table(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _update_lut(root, op):
    return [sys.executable, os.path.join(root, hipep_env.autotune(op.name), "scripts", "update_lut.py")]


def _ops(s):
    ops = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in ops if x not in OPS]
    if not ops or bad:
        raise argparse.ArgumentTypeError(f"expected ops from {sorted(OPS)}, got {s!r}")
    return ops


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", required=True, help="the device's gfx name; hip-ep loads that table")
    ap.add_argument("--ops", type=_ops, default="matmul_nbits,gqa")
    ap.add_argument("--overlay", default=None, help="overlay the sweeps and the build start from")
    a = ap.parse_args()
    src, build = hipep_env.require("HIPEP_SRC"), hipep_env.require("HIPEP_BUILD")
    env = hipep_env.Env.from_environ()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    clear_outputs(out)
    log = os.path.join(out, "lut_refresh.log")
    root = private_tree(src, a.overlay, out)

    result, planned = {}, []
    for name in a.ops:
        op = OPS[name]
        lut = os.path.join(root, hipep_env.autotune(name), "lut", a.arch)  # <lut>.json and <lut>.fb
        before = _table(f"{lut}.json")
        # hip-ep builds in only the tables it ships, so a table for another arch would change nothing.
        args, reason = (None, f"hip-ep ships no {name} table for {a.arch}") if before is None else \
            op.plan(root, a.model, out, log)
        if args is None:
            result[name] = {"changed": False, "skipped": reason}
            continue
        planned.append((op, args, lut, before, op.build_sweep(root, build, out, env.therock, log)))

    run_env = {k: v for k, v in os.environ.items() if not k.startswith("HIPDNN_")}
    run_env["LD_LIBRARY_PATH"] = os.path.join(env.therock, "lib")
    with hipep_env.serving_lock(env):
        for op, (measure, _), _, _, sweep in planned:
            _run(_update_lut(root, op) + ["measure", "--arch", op.arch(a.arch), "--sweep", sweep, *measure],
                 log, env=run_env)

    changed = []
    for op, (_, build_args), lut, before, _ in planned:
        upd = _update_lut(root, op)
        _run(upd + ["build", "--arch", op.arch(a.arch), *build_args], log)
        _run(upd + ["compile", "--arch", op.arch(a.arch), "--flatc", os.path.join(build, "bin", "flatc")], log)
        result[op.name] = {"changed": _table(f"{lut}.json") != before, "skipped": ""}
        if result[op.name]["changed"]:
            changed.append(lut)

    if a.overlay or changed:
        overlay = os.path.join(out, "overlay")
        write_overlay(a.overlay, root, changed, overlay)
        _run(["bash", os.path.join(HERE, "build_kernels.sh"), overlay, os.path.join(out, "kbuild")], log)
    hipep_env.write_json(os.path.join(out, "lut_refresh.json"), {"arch": a.arch, "ops": result})
    for name, r in result.items():
        print(f"[lut_refresh] {name}: {'changed' if r['changed'] else 'unchanged'} {r['skipped']}".rstrip())
    print(f"[lut_refresh] wrote {out}/lut_refresh.json; kernel library in {out}/kbuild "
          "when it differs from the release")


if __name__ == "__main__":
    main()
