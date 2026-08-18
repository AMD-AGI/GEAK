#!/usr/bin/env python3
"""Measurement-driven promotion: candidate -> active (plan Part 2.5 — the ONLY upgrade channel).

The single rule of the whole lifecycle: a KB claim earns `lifecycle: active` **only** by being
reproduced on-box, never by being written or scraped. This tool is that rule made executable. It scans
the machine-produced artifact store (`kb_artifacts/<gfx>/<kernel_class>/<slug>/<exp_id>/meta.yaml`,
written by kernel_workflow/scripts/experience_store.py) and promotes a slug when the SAME win has been
reproduced by >= --min-repro *independent* runs (distinct `source_eval_dir`).

On promotion it, for every `candidate` meta.yaml in that slug:
  * flips  lifecycle: candidate -> active
  * refreshes  verified_on  to the newest date seen in the group
  * stamps  verified_stack  from --stack (if given), recording the stack the reproduction ran on

Dry-run by default (prints what WOULD change); pass --write to persist. Nothing here ever deletes,
downgrades, or touches non-candidate cards — it is strictly the promotion half of the state machine
(demotion to stale/archived is driven by upstream drift, see _ingest_web.py / the monthly review).

Usage:
  _promote.py --root ../kb_artifacts                       # dry-run report
  _promote.py --root ../kb_artifacts --write --stack '{"rocm":"7.1","aiter":"a6bb4993"}'
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_ROOT = os.path.join(REPO, "kb_artifacts")

try:
    import yaml
except Exception:
    yaml = None


def _read_meta(path):
    try:
        text = open(path, encoding="utf-8").read()
    except OSError:
        return None
    if yaml is not None:
        try:
            d = yaml.safe_load(text)
            return d if isinstance(d, dict) else None
        except Exception:
            return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _dump_meta(meta):
    if yaml is not None:
        return yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)
    return json.dumps(meta, indent=2, ensure_ascii=False)


def iter_slugs(root):
    """Yield (gfx, kernel_class, slug, [(meta, meta_path), ...]) for every slug dir under root."""
    if not os.path.isdir(root):
        return
    for gfx in sorted(os.listdir(root)):
        gdir = os.path.join(root, gfx)
        if not os.path.isdir(gdir):
            continue
        for kc in sorted(os.listdir(gdir)):
            kdir = os.path.join(gdir, kc)
            if not os.path.isdir(kdir):
                continue
            for slug in sorted(os.listdir(kdir)):
                sdir = os.path.join(kdir, slug)
                if not os.path.isdir(sdir):
                    continue
                sols = []
                for exp_id in sorted(os.listdir(sdir)):
                    mp = os.path.join(sdir, exp_id, "meta.yaml")
                    if not os.path.isfile(mp):
                        mp = os.path.join(sdir, exp_id, "meta.json")
                    if not os.path.isfile(mp):
                        continue
                    m = _read_meta(mp)
                    if m is not None:
                        sols.append((m, mp))
                if sols:
                    yield gfx, kc, slug, sols


def _independent_reproductions(sols):
    """Count distinct runs that produced a real win — distinct source_eval_dir (fallback: exp dir)."""
    seen = set()
    for meta, mp in sols:
        sp = ((meta.get("metric") or {}).get("speedup")) if isinstance(meta.get("metric"), dict) else None
        try:
            if sp is None or float(sp) <= 1.0:
                continue
        except (TypeError, ValueError):
            continue
        key = meta.get("source_eval_dir") or os.path.dirname(mp)
        seen.add(key)
    return len(seen)


def _newest_verified_on(sols):
    dates = [str(m.get("verified_on")) for m, _ in sols if m.get("verified_on")]
    return max(dates) if dates else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="kb_artifacts root (default: <repo>/kb_artifacts)")
    ap.add_argument("--min-repro", type=int, default=2, help="independent reproductions required (default 2)")
    ap.add_argument("--stack", default="", help="JSON to stamp into verified_stack on promotion")
    ap.add_argument("--write", action="store_true", help="persist changes (default: dry-run)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    stack = {}
    if a.stack:
        try:
            stack = json.loads(a.stack)
        except Exception:
            print(f"_promote: --stack is not valid JSON: {a.stack!r}", file=sys.stderr)
            return 2

    promoted, examined, changed_files = [], 0, 0
    for gfx, kc, slug, sols in iter_slugs(a.root):
        examined += 1
        repro = _independent_reproductions(sols)
        candidates = [(m, mp) for m, mp in sols if str(m.get("lifecycle")) == "candidate"]
        if repro < a.min_repro or not candidates:
            continue
        vo = _newest_verified_on(sols)
        promoted.append({"gfx": gfx, "kernel_class": kc, "slug": slug,
                         "reproductions": repro, "verified_on": vo, "n_cards": len(candidates)})
        for meta, mp in candidates:
            meta["lifecycle"] = "active"
            if vo:
                meta["verified_on"] = vo
            if stack:
                meta["verified_stack"] = stack
            if a.write:
                try:
                    tmp = mp + ".tmp"
                    open(tmp, "w", encoding="utf-8").write(_dump_meta(meta))
                    os.replace(tmp, mp)
                    changed_files += 1
                except OSError as e:
                    print(f"_promote: could not write {mp}: {e}", file=sys.stderr)

    if a.json:
        print(json.dumps({"examined_slugs": examined, "promoted": promoted,
                          "written": changed_files if a.write else 0, "dry_run": not a.write},
                         ensure_ascii=False, indent=2))
        return 0

    print(f"# _promote — scanned {examined} slug(s) under {a.root}")
    if not promoted:
        print(f"# nothing to promote (need >= {a.min_repro} independent reproductions per slug).")
        return 0
    for p in promoted:
        print(f"[promote] {p['gfx']}/{p['kernel_class']}/{p['slug']}: "
              f"{p['reproductions']} reproductions -> active"
              f"{' (verified_on ' + p['verified_on'] + ')' if p['verified_on'] else ''}"
              f" [{p['n_cards']} card(s)]")
    print(f"\n{'WROTE ' + str(changed_files) + ' file(s).' if a.write else 'DRY-RUN — re-run with --write to persist.'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
