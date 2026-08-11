#!/usr/bin/env python3
"""Local experience store for the kernel workflow — the machine-produced, code-carrying KB.

This is the concrete v1 of the warm-start experience store described in the KB plan
(Part 4). It is deliberately self-contained and dependency-light (stdlib + PyYAML) so a
lane agent can call it over Bash with no orchestration.

Two knowledge sources must not be confused (see the plan, Part 4.0):
  * perf_knowledge/ + learned cards  — human methodology, injected as an index.
  * kb_artifacts/ (THIS store)       — machine-produced run outcomes that CARRY the diff.

On-disk layout (rooted at --root, default <repo>/kb_artifacts):

    <root>/<gfx>/<kernel_class>/<slug>/<exp_id>/
        meta.yaml     # identity + metric + prose pointers
        patch.diff    # the cumulative winning diff (verbatim copy)
        report.md     # optional: the tech_lead report copied for prose (strategy/recipe/lessons)

    slug = <kernel_name>__<language>__<gfx>     # deterministic, identical on read + write

Subcommands:
    write    Store one measured win. Applies the KernelForge write gate
             (missing_arch / no_improvement / empty_diff) and NEVER raises — any
             failure prints {"written": false, "reason": ...} and exits 0 so the
             calling run degrades instead of crashing.
    resolve  Enumerate solutions for a slug, keep the SAME gfx only, rank by speedup,
             return the top-N, and mirror every candidate's prose into
             <refs-dir>/ (kb_references) so a rejected warm start is still visible.

All speedups are only comparable within one GPU arch, so resolve drops cross-arch
candidates outright rather than down-weighting them.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time

try:
    import yaml
except Exception:  # pragma: no cover - yaml ships in this env; degrade to json-only meta
    yaml = None


# --------------------------------------------------------------------------- #
# Identity helpers — read and write MUST derive the slug identically, never via
# an LLM, or a run can never find its own lineage.
# --------------------------------------------------------------------------- #
def _safe(seg: str) -> str:
    """Slug-safe a path segment: keep [A-Za-z0-9._-], collapse the rest to '-'."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(seg or "").strip())
    s = s.strip("-.") or "x"
    return s[:80]


def _norm_gfx(gfx: str) -> str:
    m = re.search(r"gfx\d+", str(gfx or ""), re.IGNORECASE)
    return m.group(0).lower() if m else ""


def make_slug(kernel_name: str, language: str, gfx: str) -> str:
    return f"{_safe(kernel_name)}__{_safe(language)}__{_norm_gfx(gfx) or 'unknown'}"


def _read_meta(meta_path: str):
    try:
        with open(meta_path, "r") as f:
            text = f.read()
        if yaml is not None:
            return yaml.safe_load(text) or {}
        return json.loads(text)
    except Exception:
        return None


def _dump_meta(meta: dict) -> str:
    if yaml is not None:
        return yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)
    return json.dumps(meta, indent=2, ensure_ascii=False)


def _atomic_write(path: str, data: str):
    """Same-directory temp file -> fsync -> os.replace -> dir fsync (crash-safe)."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".swap")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
    try:
        dirfd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    except OSError:
        pass


def _impl_signature(patch_text: str) -> str:
    return "sha256:" + hashlib.sha256(patch_text.encode("utf-8", "replace")).hexdigest()[:32]


# --------------------------------------------------------------------------- #
# write
# --------------------------------------------------------------------------- #
def cmd_write(a) -> dict:
    gfx = _norm_gfx(a.gfx)
    if not gfx:
        return {"written": False, "reason": "missing_arch"}

    try:
        speedup = float(a.speedup)
    except (TypeError, ValueError):
        return {"written": False, "reason": "invalid_speedup"}
    if not (speedup > 1.0):  # covers NaN, <=1.0
        return {"written": False, "reason": "no_improvement"}

    patch_text = ""
    if a.patch and os.path.isfile(a.patch):
        try:
            with open(a.patch, "r", errors="replace") as f:
                patch_text = f.read()
        except OSError:
            patch_text = ""
    if not patch_text.strip():
        return {"written": False, "reason": "empty_diff"}

    kernel_class = a.kernel_class or "unknown"
    slug = make_slug(a.kernel_name, a.language, gfx)
    exp_id = time.strftime("%Y%m%d_%H%M%S") + "_" + hashlib.sha1(
        (slug + patch_text[:256] + str(time.time())).encode("utf-8", "replace")
    ).hexdigest()[:6]
    out_dir = os.path.join(a.root, gfx, _safe(kernel_class), slug, exp_id)

    baseline_ms = None
    try:
        baseline_ms = float(a.baseline_wall_ms)
    except (TypeError, ValueError):
        baseline_ms = None
    wall_ms = (baseline_ms / speedup) if (baseline_ms and speedup > 0) else None

    meta = {
        "layer": "artifact",
        "lifecycle": "candidate",           # earns 'active' only via independent reproduction (plan Part 2.5)
        "gfx": gfx,
        "platforms": [gfx],
        "kernel_class": kernel_class,
        "kernel_name": a.kernel_name,
        "language": a.language,
        "metric": {
            "speedup": round(speedup, 6),
            "wall_ms": round(wall_ms, 6) if wall_ms is not None else None,
            "baseline_wall_ms": round(baseline_ms, 6) if baseline_ms is not None else None,
            "gpu_arch": gfx,
        },
        "impl_signature": _impl_signature(patch_text),
        "verified_on": time.strftime("%Y-%m-%d"),
        "verified_stack": {},               # filled by a later stack-aware pass (plan Part 2.1)
        "source_eval_dir": a.eval_dir or "",
        "patch_content": "patch.diff",
    }

    # Prose (strategy / recipe / lessons). v1 copies the tech_lead report verbatim as the prose
    # body and lifts its first non-empty line as the one-sentence strategy.
    strategy = ""
    report_copied = None
    if a.report and os.path.isfile(a.report):
        try:
            with open(a.report, "r", errors="replace") as f:
                report_text = f.read()
            for line in report_text.splitlines():
                s = line.strip().lstrip("# ").strip()
                if s:
                    strategy = s[:300]
                    break
            report_copied = report_text
        except OSError:
            pass
    if a.strategy:
        strategy = a.strategy[:300]
    meta["strategy"] = strategy

    try:
        _atomic_write(os.path.join(out_dir, "patch.diff"), patch_text)
        _atomic_write(os.path.join(out_dir, "meta.yaml"), _dump_meta(meta))
        if report_copied is not None:
            _atomic_write(os.path.join(out_dir, "report.md"), report_copied)
    except OSError as e:
        return {"written": False, "reason": "io_error: " + str(e)[:120]}

    return {
        "written": True,
        "reason": "ok",
        "slug": slug,
        "exp_id": exp_id,
        "dir": out_dir,
        "speedup": round(speedup, 4),
    }


# --------------------------------------------------------------------------- #
# resolve
# --------------------------------------------------------------------------- #
def _iter_solutions(root: str, gfx: str, slug: str):
    """Yield (meta, exp_dir) for every solution matching (gfx, slug), any kernel_class."""
    base = os.path.join(root, gfx)
    if not os.path.isdir(base):
        return
    for kernel_class in sorted(os.listdir(base)):
        slug_dir = os.path.join(base, kernel_class, slug)
        if not os.path.isdir(slug_dir):
            continue
        for exp_id in sorted(os.listdir(slug_dir)):
            exp_dir = os.path.join(slug_dir, exp_id)
            meta_path = os.path.join(exp_dir, "meta.yaml")
            if not os.path.isfile(meta_path):
                meta_path = os.path.join(exp_dir, "meta.json")
            meta = _read_meta(meta_path)
            if isinstance(meta, dict):
                yield meta, exp_dir


def _speedup_of(meta: dict) -> float:
    try:
        return float((meta.get("metric") or {}).get("speedup"))
    except (TypeError, ValueError):
        return 0.0


def cmd_resolve(a) -> dict:
    gfx = _norm_gfx(a.gfx)
    if not gfx:
        return {"read_reason": "missing_arch", "candidates": []}

    slug = make_slug(a.kernel_name, a.language, gfx)
    root = a.root
    if not os.path.isdir(os.path.join(root, gfx)):
        return {"read_reason": "kernel_page_not_found", "slug": slug, "candidates": []}

    found = list(_iter_solutions(root, gfx, slug))
    # Same-arch is already guaranteed by the <gfx> path segment; the metric's gpu_arch is a
    # second belt-and-braces guard against a mislabeled entry.
    found = [(m, d) for (m, d) in found
             if _norm_gfx((m.get("metric") or {}).get("gpu_arch") or m.get("gfx") or gfx) == gfx]
    if not found:
        return {"read_reason": "no_same_arch", "slug": slug, "candidates": []}

    found.sort(key=lambda md: _speedup_of(md[0]), reverse=True)
    top = found[: max(1, int(a.top_n or 3))]

    # Mirror EVERY selected candidate's prose into kb_references/ up front, so a rejected warm
    # start is still visible after the fact (plan Part 4.6). The verify loop later rewrites
    # index.md statuses; here we seed it with status "read".
    refs_dir = a.refs_dir
    set_hash = hashlib.sha1(("|".join(d for _, d in top)).encode("utf-8", "replace")).hexdigest()[:7]
    set_dir = os.path.join(refs_dir, "sets", set_hash)
    candidates = []
    index_lines = [f"# Warm-start references — slug `{slug}` (gfx {gfx})", ""]
    for rank, (meta, exp_dir) in enumerate(top, start=1):
        patch_path = os.path.join(exp_dir, "patch.diff")
        speedup = _speedup_of(meta)
        ref_name = f"reference_{rank:02d}.md"
        prose_path = os.path.join(set_dir, ref_name)
        try:
            report_path = os.path.join(exp_dir, "report.md")
            body = ""
            if os.path.isfile(report_path):
                with open(report_path, "r", errors="replace") as f:
                    body = f.read()
            prose = (
                f"# Reference {rank:02d} — {slug}\n\n"
                f"- speedup: {speedup:.4f}x\n"
                f"- strategy: {meta.get('strategy', '')}\n"
                f"- source: {meta.get('source_eval_dir', '')}\n"
                f"- verified_on: {meta.get('verified_on', '')}\n\n"
                f"---\n\n{body}\n"
            )
            _atomic_write(prose_path, prose)
        except OSError:
            prose_path = ""
        candidates.append({
            "rank": rank,
            "slug": slug,
            "exp_dir": exp_dir,
            "speedup": round(speedup, 4),
            "arch": gfx,
            "patch_path": patch_path,
            "prose_path": prose_path,
            "strategy": meta.get("strategy", ""),
            "status": "read",
        })
        index_lines.append(
            f"- Rank {rank}: `{prose_path}` | speedup {speedup:.4f}x | "
            f"patch `{patch_path}` | status `read`"
        )
    try:
        _atomic_write(os.path.join(refs_dir, "index.md"), "\n".join(index_lines) + "\n")
    except OSError:
        pass

    return {"read_reason": "read", "slug": slug, "candidates": candidates}


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("write", help="store one measured win")
    w.add_argument("--root", required=True)
    w.add_argument("--kernel-name", dest="kernel_name", required=True)
    w.add_argument("--language", required=True)
    w.add_argument("--gfx", required=True)
    w.add_argument("--kernel-class", dest="kernel_class", default="unknown")
    w.add_argument("--speedup", required=True)
    w.add_argument("--baseline-wall-ms", dest="baseline_wall_ms", default=None)
    w.add_argument("--patch", default="")
    w.add_argument("--eval-dir", dest="eval_dir", default="")
    w.add_argument("--report", default="")
    w.add_argument("--strategy", default="")

    r = sub.add_parser("resolve", help="enumerate + rank top-N solutions for a slug")
    r.add_argument("--root", required=True)
    r.add_argument("--kernel-name", dest="kernel_name", required=True)
    r.add_argument("--language", required=True)
    r.add_argument("--gfx", required=True)
    r.add_argument("--top-n", dest="top_n", type=int, default=3)
    r.add_argument("--refs-dir", dest="refs_dir", required=True)

    a = p.parse_args(argv)
    try:
        if a.cmd == "write":
            out = cmd_write(a)
        elif a.cmd == "resolve":
            out = cmd_resolve(a)
        else:  # pragma: no cover
            out = {"error": "unknown command"}
    except Exception as e:  # never crash the caller
        out = ({"written": False, "reason": "exception: " + str(e)[:160]}
               if a.cmd == "write"
               else {"read_reason": "exception: " + str(e)[:160], "candidates": []})
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
