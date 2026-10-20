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


def read_check(out):
    """(file, verdict) from `overlay_setup.py check`, which prints a "<module> -> <file>"
    line followed by one of OVERLAY_ACTIVE / INJECTED / INSTALL."""
    where, verdict = "", ""
    for line in out.splitlines():
        line = line.strip()
        if " -> " in line:
            where = line.split(" -> ", 1)[1]
        elif line in ("OVERLAY_ACTIVE", "INJECTED") or line.startswith("INSTALL"):
            verdict = line.split()[0]
    return where, verdict


def editable_src(ksrc_dir):
    """The one editable file in kernel_src/, chosen by what it is.

    Sorting the directory and taking the first entry picks `__pycache__` as soon as one
    attempt has run -- a directory, which the overlay then installs as the kernel."""
    if not os.path.isdir(ksrc_dir):
        return None
    names = [n for n in sorted(os.listdir(ksrc_dir))
             if os.path.isfile(os.path.join(ksrc_dir, n)) and not n.startswith("__")]
    py = [n for n in names if n.endswith(".py")]
    pick = (py or names or [None])[0]
    return os.path.join(ksrc_dir, pick) if pick else None


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
        # The optimized source. candidate_bind["file"] is authoritative when it names a
        # file that exists in the task: the extraction skill wrote it knowing how this
        # particular leaf binds, and for a leaf that is HIP C++ it points at a delivery
        # module rather than at kernel_src. Overriding it with editable_src() installs the
        # .py of a kernel whose edit lives in a .h -- an overlay that shadows the
        # dispatcher with an identical dispatcher, leaves the image's prebuilt .so serving
        # every token, and still passes the shadow check below. The e2e that follows would
        # be measuring the baseline against itself.
        ksrc = os.path.join(task, "kernel_src")
        named = bind.get("file")
        src, via = None, ""
        if named:
            for cand in (os.path.join(task, named), os.path.join(ksrc, named)):
                if os.path.isfile(cand):
                    src, via = cand, "candidate_bind.file"
                    break
            if not src:
                out["skipped"].append({"kernel": k, "reason":
                    "candidate_bind names file %r and it is not in the task directory; "
                    "installing a different file would bind something the extraction "
                    "skill did not describe" % named})
                continue
        if not src:
            src, via = editable_src(ksrc), "kernel_src (candidate_bind named no file)"
        if not src:
            out["skipped"].append({"kernel": k, "reason": "kernel_src/ holds no editable file"})
            continue
        rc, log = add_entry(tool, args.overlay, base, bind, src)
        if rc != 0:
            out["skipped"].append({"kernel": k,
                                   "reason": "overlay install failed: %s" % log[-400:]})
            continue
        base = args.overlay          # the next kernel seeds from this overlay
        entry = {"kernel": k, "bind_kind": bind.get("kind"),
                 "target": bind.get("target") or bind.get("module"),
                 "impl_file": src, "impl_file_from": via,
                 "verified_speedup": r.get("verified_speedup")}
        # A kernel spread over several files is delivered through one module. Record what
        # that module is expected to pull in, so a reader can tell at a glance whether the
        # .cu/.h actually travelled with the .py.
        if os.path.isdir(ksrc):
            entry["kernel_src_files"] = sorted(
                n for n in os.listdir(ksrc)
                if os.path.isfile(os.path.join(ksrc, n))
                and not n.startswith("__") and n != "source_manifest.json")
            if len(entry["kernel_src_files"]) > 1 and via != "candidate_bind.file":
                entry["warning"] = (
                    "kernel_src holds %d files but the overlay installs only %s, and "
                    "candidate_bind named none. If the optimized edit is in one of the "
                    "others it is not in this overlay."
                    % (len(entry["kernel_src_files"]), os.path.basename(src)))
        out["applied"].append(entry)

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
        where, verdict = read_check(r.stdout or "")
        # Trust the tool's own verdict rather than looking for the overlay path in its
        # output. `check` prints "<module> -> <file>" and then a separate verdict line, so
        # taking the last line as the path judged a correctly installed overlay to be
        # missing -- it fails closed, which aborts a run that was in fact fine.
        inside = verdict in ("OVERLAY_ACTIVE", "INJECTED")
        checked.append({"module": mod, "resolves_to": where, "verdict": verdict,
                        "shadowed": inside})
        if not inside:
            a["warning"] = ("the overlay does not shadow %s — it resolves to %s (%s). "
                            "Any e2e measured in this state is meaningless."
                            % (mod, where or "<unknown>", verdict or "no verdict"))
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
