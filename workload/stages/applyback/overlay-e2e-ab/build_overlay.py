#!/usr/bin/env python3
"""Layer every cold-re-verified kernel into **one** reversible overlay, and verify that it
actually takes effect.

Two things worth stating plainly:

* **The overlays must be chain-seeded, not mounted side by side.** `PYTHONPATH=a:b` only
  imports the first sitecustomize on sys.path, and the second overlay directory silently
  has no effect — you end up with an overlay that installed only one kernel, with nothing
  to show for it. So each added kernel passes `--from` the previous overlay.
* **How a kernel binds back is not guessed here.** The extraction skill already recorded
  `{kind, target|module, impl_module, impl_attr, file}` in meta.json["candidate_bind"] —
  the same descriptor it uses for its own candidate leg. This copies it, replacing only
  `file` with the optimized kernel_src.

Usage:
    python3 build_overlay.py --optimized <optimized.json> --tasks-dir <dir> \
                             --overlay <out_dir> [--manifest <out.json>]
"""
import argparse, json, os, shutil, subprocess, sys


def find_overlay_setup(geak_root, tasks_dir):
    """Prefer the copy shipped with the task directory (same version as at extraction
    time), falling back to the one in the repository."""
    cands = [os.path.join(tasks_dir, "overlay_setup.py"),
             os.path.join(geak_root, "e2e_workflow", "scripts", "overlay_setup.py")]
    for t in sorted(os.listdir(tasks_dir)) if os.path.isdir(tasks_dir) else []:
        cands.insert(0, os.path.join(tasks_dir, t, "overlay_setup.py"))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def add_entry(tool, overlay, base, bind, src):
    if bind.get("kind") == "rebind":
        args = ["add-rebind", "--target", bind["target"],
                "--impl-module", bind["impl_module"], "--impl-attr", bind["impl_attr"],
                "--impl-file", src]
    else:
        args = ["add-module", "--module", bind["module"], "--patched-file", src]
    cmd = [sys.executable, tool] + args + ["--overlay", overlay]
    if base:
        cmd += ["--from", base]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimized", required=True)
    ap.add_argument("--tasks-dir", required=True)
    ap.add_argument("--overlay", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--geak-root", default=os.environ.get("GEAK_ROOT", ""))
    args = ap.parse_args()

    opt = json.load(open(args.optimized))
    kept = [r for r in opt.get("kernels", []) if r.get("verdict") == "pass"]
    out = {"applied": [], "skipped": [], "overlay": os.path.abspath(args.overlay)}

    if not kept:
        out["skipped"].append({"reason": "no kernel from the optimize stage passed cold "
                                         "re-verification"})
        json.dump(out, open(args.manifest or "/dev/stdout", "w"), indent=1, ensure_ascii=False)
        return 3

    tool = find_overlay_setup(args.geak_root, args.tasks_dir)
    if not tool:
        print("cannot find overlay_setup.py (neither in the task directories nor under "
              "e2e_workflow/scripts)", file=sys.stderr)
        return 2

    shutil.rmtree(args.overlay, ignore_errors=True)
    base = ""
    for r in kept:
        k = r["kernel"]
        task = os.path.join(args.tasks_dir, k)
        try:
            meta = json.load(open(os.path.join(task, "meta.json")))
        except (OSError, ValueError) as e:
            out["skipped"].append({"kernel": k, "reason": "cannot read meta.json: %s" % e})
            continue
        bind = meta.get("candidate_bind") or {}
        if not bind:
            out["skipped"].append({"kernel": k, "reason":
                "meta.json has no candidate_bind — the extraction skill did not record how "
                "to bind this back, and guessing is not an option; skipped"})
            continue
        # The optimized source: prefer best_kernel_src (re-verification has already
        # restored it into kernel_src), otherwise whatever kernel_src currently holds.
        ksrc = os.path.join(task, "kernel_src")
        files = sorted(os.listdir(ksrc)) if os.path.isdir(ksrc) else []
        if not files:
            out["skipped"].append({"kernel": k, "reason": "kernel_src/ is empty"})
            continue
        src = os.path.join(ksrc, files[0])
        rc, log = add_entry(tool, args.overlay, base, bind, src)
        if rc != 0:
            out["skipped"].append({"kernel": k,
                                   "reason": "overlay install failed: %s" % log[-400:]})
            continue
        base = args.overlay          # the next kernel seeds from this overlay
        out["applied"].append({"kernel": k, "bind_kind": bind.get("kind"),
                               "target": bind.get("target") or bind.get("module"),
                               "impl_file": src,
                               "verified_speedup": r.get("verified_speedup")})

    # --- Verify the overlay really shadows. Measuring without this check means "no gain"
    #     may simply be an overlay that was never loaded.
    checked = []
    env = dict(os.environ)
    env["PYTHONPATH"] = args.overlay + os.pathsep + env.get("PYTHONPATH", "")
    for a in out["applied"]:
        mod = (a["target"] or "").split(":")[0]
        if not mod:
            continue
        r = subprocess.run([sys.executable, tool, "check", "--module", mod],
                           capture_output=True, text=True, env=env)
        where = (r.stdout or "").strip().splitlines()[-1] if r.stdout.strip() else ""
        inside = args.overlay in where
        checked.append({"module": mod, "resolves_to": where, "shadowed": inside})
        if not inside:
            a["warning"] = ("the overlay does not shadow %s — it still resolves to %s. "
                            "Any e2e measured in this state is meaningless."
                            % (mod, where or "<unknown>"))
    out["checks"] = checked
    out["all_shadowed"] = bool(checked) and all(c["shadowed"] for c in checked)

    with open(args.manifest or os.path.join(args.overlay, "applied.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("installed %d kernels, skipped %d; overlay=%s shadow check=%s"
          % (len(out["applied"]), len(out["skipped"]), args.overlay,
             "passed" if out["all_shadowed"] else "**FAILED**"))
    return 0 if out["applied"] else 3


if __name__ == "__main__":
    sys.exit(main())
