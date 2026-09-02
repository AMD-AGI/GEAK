#!/usr/bin/env python3
"""Local experience store for the kernel workflow — the machine-produced KB that carries the diff.

Self-contained, stdlib + PyYAML only, so a lane agent can call it over Bash. On-disk layout
(rooted at --root, default <repo>/kb_artifacts):

    <root>/<gfx>/<kernel_class>/<slug>/<exp_id>/
        meta.yaml     # identity + metric + prose pointers
        patch.diff    # the winning diff (verbatim copy)
        report.md     # optional tech_lead report, copied for prose
        artifact/     # carrier=tuned_artifact only: the tuned tables themselves

Two CARRIERS, one ranking. `carrier: patch` is a diff, the original and still the default. A
`carrier: tuned_artifact` entry is a tuned config table produced by the e2e tuning track: the win is
data a library reads plus the env var that binds it, and no diff can express it (the file is deployed
into an installed package, structurally outside any git tree). Both are ranked by the SAME isolated
speedup on the same page, which is the whole reason tuning writes here — the e2e store gates on the
Director's serving verdict, so a 3.29x tuned kernel in a run whose e2e came back flat was, until this
carrier existed, discarded entirely. `resolve` serves ONE carrier per call and defaults to `patch`, so
a caller that cannot install a tuned table never sees one.

    slug = <canon(kernel_name)>__<language>__<gfx>   # deterministic; read and write derive it identically

Numeric precision is NOT in the slug and NOT in the remote key. It is a FILTER: written into
`meta.upstream` (and `value.upstream` remotely), applied only when a reader passes `--precision`,
and an entry that states none is never excluded. That asymmetry is deliberate — a tuned table is
dtype-specialized and `bench_key` only flags comparability rather than partitioning the ranking, so
without the filter one dtype's entries can take every top-N slot on a page a reader of the other
dtype is looking at. Keying on it instead would have moved every existing entry's address on a
store with no delete.

Subcommands:
    write      Store one measured win behind the gate (missing_arch / no_improvement / empty_diff,
               or no_artifact / unreadable_artifact on the tuned carrier).
    resolve    Rank the top-N same-gfx solutions for a slug and mirror their prose into <refs-dir>,
               for ONE --carrier (default patch), optionally narrowed to one --precision.
    remap      Rewrite a stored patch's paths onto the calling workspace's layout, or refuse and say why.
    languages  Which languages a kernel has a page in — the store, not a task_type guess, decides.
    backfill-content
               Bring imported entries up to the current content shape (dry-run unless --apply).
    export-remote
               Render entries as KB Store candidates (one JSON line each); uploads nothing.
    sync-local Bring the on-disk KB store level with the tree, keyed by canonical id. Idempotent
               and non-destructive, so the lane runs it before every read: it is what lets the
               READ address by key without losing the backlog only the tree ever held.
    resolve-remote
               `resolve`, but addressed by canonical id against a KB store (kb/store_local.py).
    write-remote
               `write`, landing the same result in the local store AND under its key.
    attest / attest-remote
               Count one attempt to USE a stored entry (validated | failed | not_reproduced), so a
               later curation pass can retire what nobody can reproduce. Moves no speedup, no rank.

Speedups only compare within one GPU arch, so resolve drops cross-arch entries outright. Neither
command ever raises: on failure it prints a JSON reason and exits 0 so the caller degrades.

resolve serves a CURATED top-N, not the raw speedup order: entries the curation retired
(`retained: false`) are never offered, near-ties below `--min-speedup` are not worth a verify slot,
and only one entry per `direction:` is ranked (same-idea runners-up ride along as `alternates`,
since they verify or fail together). A speedup only means something against its own
`metric.bench_key`, so each candidate carries one plus a `comparable` flag against rank 1.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time

# The shared KB plane lives at the repo root as the `kb` package, not beside this file. Executed as
# a CLI from an arbitrary cwd, so the root is derived from __file__ and never from the environment.
# Moved to the FRONT even when already present: this script's own directory is sys.path[0], it
# holds an unrelated `kb.py` (the learned-card CLI), and that shadows the `kb` PACKAGE for anyone
# who has the repo root on PYTHONPATH — every import below then fails with "kb is not a package".
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
while _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

from kb.attest import OUTCOMES as _OUTCOMES
from kb.attest import RETIRE_THRESHOLD as _RETIRE_THRESHOLD
from kb.curate import collapse_by_direction, demote_hinted
from kb.ladder import publish
from kb.plane import open_plane, read_planes
from kb.store_local import CHAMPION_METRIC, SAFE_COMPONENT_CHARS

try:
    import yaml
except Exception:  # yaml ships in this env; degrade to json-only meta
    yaml = None


# Identity: read and write MUST derive the slug identically (never via an LLM) or a run
# can never find its own lineage.
def _safe(seg: str) -> str:
    """Slug-safe a path segment: keep what the remote plane's path validator accepts (see
    kb.store_local.SAFE_COMPONENT_CHARS), collapse the rest to '-'. The '-' and the 80-char cap are
    load-bearing, not stylistic: a stored artifact name is hashed into its content signature, so
    respelling either re-addresses every tuned entry already in the store."""
    s = re.sub("[^%s]+" % SAFE_COMPONENT_CHARS, "-", str(seg or "").strip())
    s = s.strip("-.") or "x"
    return s[:80]


def _norm_gfx(gfx: str) -> str:
    m = re.search(r"gfx\d+", str(gfx or ""), re.IGNORECASE)
    return m.group(0).lower() if m else ""


# Numeric precision is a FILTER, never a dimension — see the remote-export note on why the address
# cannot carry it. These two functions are the whole of the folding rule, kept together so read and
# write cannot disagree about what counts as the same dtype.
_PRECISION_ALIASES = {"float8": "fp8", "f8": "fp8", "float16": "fp16", "f16": "fp16",
                      "half": "fp16", "bfloat16": "bf16", "bf": "bf16",
                      "float32": "fp32", "f32": "fp32", "float": "fp32"}


def _norm_precision(value) -> str:
    """Fold a dtype spelling to a comparable token: `FP8-w8a8` and `fp8_w8a8` are one thing.

    Only the LEADING token is aliased; a quantization scheme suffix (`_w8a8`, `_w4a16`) is kept
    verbatim, because calling two of those interchangeable would offer a reader a table its runtime
    cannot use — the failure this filter exists to prevent.
    """
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    if not text:
        return ""
    head, sep, rest = text.partition("_")
    return _PRECISION_ALIASES.get(head, head) + sep + rest


def _precision_matches(want: str, have: str) -> bool:
    """Whether an entry recorded at `have` may be offered to a caller asking for `want`.

    Unstated on EITHER side is a match: the entire recovered backlog predates this field, so
    excluding what states no precision would empty every page in the store. A bare base dtype
    matches its refinements in BOTH directions — `fp8` is the coarser statement about the same
    thing an `fp8_w8a8` reader wants, and vice versa. On the token boundary, not a raw prefix, so
    `fp8` cannot swallow `fp8x` or `fp16`.
    """
    if not want or not have:
        return True
    return want == have or want.startswith(have + "_") or have.startswith(want + "_")


def _precision_of(meta) -> str:
    upstream = (meta or {}).get("upstream")
    return _norm_precision((upstream or {}).get("precision") if isinstance(upstream, dict) else "")


def _upstream_of(a) -> dict:
    """The serving context this win was measured in — recorded, never keyed.

    The remote-export note below says why none of this may become an identity dimension (a dimension
    the READ side cannot recompute is a permanent silent 404) and that it rides in `value.upstream`
    instead — but nothing ever wrote that field, so precision was absent from the RECORD, not just
    from the address, and the e2e lane's records could not be joined back to the kernel entries they
    produced. Stored verbatim; `_norm_precision` folds only at COMPARISON time, so a later reader
    that disagrees with today's folding rule still has the original string.
    """
    out = {}
    for key, attr in (("precision", "precision"), ("framework", "serving_framework"),
                      ("framework_version", "serving_framework_version")):
        value = str(getattr(a, attr, "") or "").strip()[:80]
        if value:
            out[key] = value
    return out


# One kernel is named differently per layout: `fused_moe_kernel` (kernel dir), `fused_moe_kernel_task`
# (e2e head extraction), `triton_fused_moe_kernel.py` (language in the filename). Canonicalizing on
# BOTH sides is what lets a head run find, and extend, its own lineage instead of forking a new page.
_NAME_PREFIXES = ("triton_", "hip_", "ck_", "cuda_", "torch_")
_NAME_SUFFIXES = (".py", ".hip", ".cu", ".cpp", "_task")


def canon_name(kernel_name: str) -> str:
    """Basename, no language prefix, no task/extension suffix. Case kept for readability; matching
    is case-insensitive via _match_key()."""
    s = os.path.basename(str(kernel_name or "").strip().rstrip("/"))
    changed = True
    while changed:
        changed = False
        for p in _NAME_PREFIXES:
            if len(s) > len(p) and s.lower().startswith(p):
                s, changed = s[len(p):], True
        for suf in _NAME_SUFFIXES:
            if len(s) > len(suf) and s.lower().endswith(suf):
                s, changed = s[: -len(suf)], True
    return s or str(kernel_name or "")


def _match_key(kernel_name: str) -> str:
    """Comparison key for slug matching: canonical name, case- and separator-insensitive."""
    return re.sub(r"[^a-z0-9]+", "", canon_name(kernel_name).lower())


def make_slug(kernel_name: str, language: str, gfx: str) -> str:
    return f"{_safe(canon_name(kernel_name))}__{_safe(language)}__{_norm_gfx(gfx) or 'unknown'}"


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


def _atomic_write(path: str, data):
    """Crash-safe: same-dir temp -> fsync -> os.replace -> dir fsync.

    `bytes` goes out unencoded, for a tuned table that is not text and must not be re-encoded on
    the way in.
    """
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".swap")
    try:
        with os.fdopen(fd, "wb" if isinstance(data, bytes) else "w") as f:
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


def content_signature(patch_text: str) -> str:
    """Path-INSENSITIVE identity of a diff: added/removed code lines only, no headers or paths.

    A warm-started run re-emits the patch it adopted as its own `git diff`, from a different
    workspace with different path prefixes — byte-different, same code. Without this the store keeps
    re-importing its own output as a fresh 'win'; with it, that re-measurement is a REPRODUCTION of
    the entry it came from, which is what promotes candidate -> active.
    """
    body = []
    for line in (patch_text or "").splitlines():
        if line.startswith(("+++", "---", "diff ", "index ", "@@", "new file", "deleted file",
                            "similarity ", "rename ", "old mode", "new mode", "Binary files")):
            continue
        if line[:1] in ("+", "-"):
            s = re.sub(r"\s+", " ", line[1:]).strip()
            if s:
                body.append(line[0] + s)
    if not body:
        return ""
    return "csha:" + hashlib.sha256("\n".join(body).encode("utf-8", "replace")).hexdigest()[:32]


# `patch` is everything the kernel lane has ever written; `tuned_artifact` is the e2e tuning track's
# carrier. Kept as a closed set so a typo in --carrier is refused rather than silently minting a third
# kind of entry that nothing knows how to serve.
CARRIERS = ("patch", "tuned_artifact")

# One tuned op can deploy several files (a config table plus the cache-invalidation companion the
# runtime keys on). They travel together or the entry is not adoptable.
_MAX_ARTIFACTS = 24


def _expand_artifact_paths(paths):
    """Flatten the caller's list to actual files, in a stable order.

    A tuner hands back whatever shape its output took — `--artifact <one table>` from one op, `<the
    dir the tuner filled>` from the next — so a directory is expanded here rather than making the
    caller hang its win/no-win outcome on an `os.path.isdir`. Sorted, so the same set of files signs
    to the same value on two boxes whose readdir order differs.
    """
    files = []
    for raw in (paths or []):
        path = str(raw or "").strip()
        if not path:
            continue
        if os.path.isfile(path):
            files.append(path)
        elif os.path.isdir(path):
            for dirpath, dirnames, filenames in os.walk(path):
                dirnames.sort()
                # Byproducts of running the tuner, not the tuning result.
                dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git")]
                files.extend(os.path.join(dirpath, f) for f in sorted(filenames))
                if len(files) >= _MAX_ARTIFACTS:
                    break
        # The cap is the whole list's, not one directory's: breaking out of the walk while the outer
        # loop kept adding let a caller passing several paths overshoot it. _artifact_sources re-caps,
        # so this was never wrong downstream — it just meant the number here was not the number.
        if len(files) >= _MAX_ARTIFACTS:
            break
    return files[:_MAX_ARTIFACTS]


def _artifact_sources(paths):
    """[(stored_name, source_path)] for every readable artifact, deduped by stored name.

    Names are collapsed to what `kb.store_local.safe_rel_path()` accepts BEFORE anything is hashed or
    copied, because the same names go on to address the file remotely. A shape-derived filename
    (`fused_moe_M=64,N=...json` is the normal aiter shape) contains characters that make the remote
    plane's path validator raise, and it raises while building the whole upload map — so one unlucky
    table would abort the write for every other file in the record.
    """
    out, seen = [], set()
    for path in _expand_artifact_paths(paths):
        name = _safe(os.path.basename(path)).lstrip(".") or "artifact"
        base, i = name, 2
        while name in seen:                       # two dirs, same basename: keep both, say which
            stem, dot, ext = base.partition(".")
            name = "%s_%d%s%s" % (stem, i, dot, ext)
            i += 1
        seen.add(name)
        out.append((name, path))
        if len(out) >= _MAX_ARTIFACTS:
            break
    return out


def artifact_signature(paths) -> str:
    """Content identity of a tuned artifact set — the `carrier: tuned_artifact` analogue of
    content_signature().

    Same job, same consequence: re-tuning a shape usually reproduces the same table, and without a
    signature the store re-imports its own output as a fresh win instead of counting a reproduction
    (which is what promotes candidate -> active). Hashed over the BYTES plus the stored basename,
    sorted, because a tuned table is binary-ish CSV/JSON whose line-level diff carries no meaning —
    unlike a patch, where paths must be ignored and code lines must not be.
    """
    items = sorted(paths)
    if not items:
        return ""
    # Say what is wrong. The two callers build this list independently, one of them from a stored
    # name list, and a bare path list unpacks as "too many values" three frames down inside a hash
    # loop — an error that names neither the argument nor the caller.
    bad = next((i for i in items if not (isinstance(i, tuple) and len(i) == 2)), None)
    if bad is not None:
        raise TypeError("artifact_signature wants (stored_name, path) pairs, got %r" % (bad,))
    h = hashlib.sha256()
    for name, path in items:
        try:
            with open(path, "rb") as f:
                blob = f.read()
        except OSError:
            return ""
        h.update(name.encode("utf-8", "replace"))
        h.update(b"\0")
        h.update(hashlib.sha256(blob).digest())
    return "asha:" + h.hexdigest()[:32]


def bench_key(metric_kind: str, case_names) -> str:
    """Identity of the MEASUREMENT a speedup came from; two speedups compare only when it matches.
    Order-insensitive. The `b2:` namespace is deliberate — imported entries carry opaque `b:` keys
    from whatever harness produced them, which must never be read as comparable to ours."""
    cases = sorted(c for c in (case_names or []) if c)
    if not cases and not metric_kind:
        return ""
    raw = f"{str(metric_kind or 'unknown')}|{','.join(cases)}"
    return "b2:" + hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:12]


# --- report prose ------------------------------------------------------------------------------
# The two sections worth reading first. Heading text varies wildly across the imported backlog
# (`## What didn't work (dead-ends — do not re-fund)`, `(confirmed dead ends)`, ... 20+ suffixes
# over 248 reports), so match the stem only and tolerate the typographic apostrophe.
_SEC_DEAD_ENDS = re.compile(r"^(#{2,3})[^\n]*what\s+didn.?t\s+work[^\n]*$", re.I | re.M)
_SEC_KEY_OPTS = re.compile(r"^(#{2,3})[^\n]*key\s+optimizations[^\n]*$", re.I | re.M)
# Structured dead-ends the tech_lead emits alongside the prose. Absent => we keep the prose only,
# rather than regex-guessing structure out of bullets/tables/paragraphs and inventing empty fields.
_DEAD_ENDS_BLOCK = re.compile(
    r"<!--\s*dead-ends:yaml\s*-->\s*```(?:ya?ml)?\n(.*?)```", re.S | re.I)


def _split_section(text: str, pattern):
    """(heading+body, text_without_it) for the first match, else ('', text). The body ends at the
    next heading of the same or shallower level, so a `###` subsection stays with its parent."""
    m = pattern.search(text or "")
    if not m:
        return "", text
    level = len(m.group(1))
    tail = text[m.end():]
    nxt = re.search(r"^#{1,%d} " % level, tail, re.M)
    end = m.end() + (nxt.start() if nxt else len(tail))
    return text[m.start():end].rstrip() + "\n", text[:m.start()] + text[end:]


def reorder_report(text: str) -> str:
    """Hoist 'Key optimizations' and "What didn't work" above everything else. Nothing is dropped —
    an agent with room still reads the whole report, one that is tight on context reads the two
    sections that change what it does. Returns the text untouched when neither is present."""
    if not text:
        return text
    key, rest = _split_section(text, _SEC_KEY_OPTS)
    dead, rest = _split_section(rest, _SEC_DEAD_ENDS)
    if not key and not dead:
        return text
    return "".join(s for s in (key, dead) if s) + "\n---\n\n" + rest.lstrip("\n")


def dead_ends_md(text: str) -> str:
    """The "What didn't work" body verbatim, minus any machine-readable block (that is parsed
    separately). Kept as text: the 248 imported reports write it as bullets, markdown tables and
    plain paragraphs, and no regex turns all three into honest structure."""
    sec, _ = _split_section(text or "", _SEC_DEAD_ENDS)
    if not sec:
        return ""
    body = sec.split("\n", 1)[1] if "\n" in sec else ""
    return _DEAD_ENDS_BLOCK.sub("", body).strip()


def parse_dead_ends(text: str):
    """The tech_lead's machine-readable dead-end list, or []. Each entry keeps whatever keys the
    report supplied (idea / measured / mechanism); a malformed block is dropped, never patched up."""
    m = _DEAD_ENDS_BLOCK.search(text or "")
    if not m or yaml is None:
        return []
    try:
        data = yaml.safe_load(m.group(1))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [{str(k): v for k, v in d.items()} for d in data
            if isinstance(d, dict) and str(d.get("idea") or "").strip()]


def _techniques(meta: dict):
    """The curated one-line summaries of what the patch actually does. Every imported entry has
    them and until now nothing read them — they are the densest thing in the store."""
    t = (meta or {}).get("techniques")
    if not isinstance(t, list):
        return []
    return [str(x).strip() for x in t if str(x).strip()]


def _techniques_md(items) -> str:
    if not items:
        return ""
    return "- techniques:\n" + "".join(f"    * {i}\n" for i in items)


def _stack_str(meta: dict) -> str:
    st = (meta or {}).get("verified_stack")
    if not isinstance(st, dict) or not st:
        return "unrecorded"
    return ", ".join(f"{k} {v}" for k, v in sorted(st.items()))


def _alternates_md(alts) -> str:
    """Same-direction runners-up. They were collapsed out of the ranking because they verify or fail
    together, but their techniques are exactly where they differ from rank 1 — so list those."""
    if not alts:
        return "- same-direction alternates: 0\n"
    lines = [f"- same-direction alternates: {len(alts)}\n"]
    for alt in alts:
        techs = "; ".join(alt.get("techniques") or []) or "no techniques recorded"
        lines.append(f"    * {alt['speedup']:.4f}x — {techs}\n")
    return "".join(lines)


def _prose_body(meta: dict, body: str) -> str:
    """The report, with the two load-bearing sections hoisted. meta's dead-ends copy is a FALLBACK
    for an entry whose report.md is gone — pasting it next to the report would just duplicate it."""
    if (body or "").strip():
        return reorder_report(body)
    out = []
    for d in (meta.get("dead_ends") or []):
        if isinstance(d, dict):
            bits = [str(d.get(k)) for k in ("measured", "mechanism") if d.get(k)]
            out.append(f"- {d.get('idea')}" + (f" — {' — '.join(bits)}" if bits else ""))
    md = str(meta.get("dead_ends_md") or "").strip()
    if not out and not md:
        return "(no report recorded for this entry)"
    head = "## What didn't work (from meta; report.md not available)\n\n"
    return head + ("\n".join(out) + "\n\n" if out else "") + md + "\n"


def _rocm_version() -> str:
    try:
        with open("/opt/rocm/.info/version", "r", errors="replace") as f:
            return f.read().strip().splitlines()[0].strip()
    except (OSError, IndexError):
        return ""


def detect_stack(language: str) -> dict:
    """WHAT the speedup was measured on. This runs in the same container as the kernel, so every
    value is observed, not inferred; anything unobservable is left out rather than guessed."""
    out = {}
    if str(language or "").lower() == "triton":
        for mod in ("triton", "torch"):
            try:
                out[mod] = str(__import__(mod).__version__)
            except Exception:
                pass
    rocm = _rocm_version()
    if rocm:
        out["rocm"] = rocm
    return out


def _find_by_content(root: str, gfx: str, slug: str, csig: str):
    """(meta, exp_dir) of the entry on this page holding the same code, or None. Hashes patch.diff
    for entries written before the signature was recorded (the imported backlog)."""
    for meta, exp_dir in _iter_solutions(root, gfx, slug):
        known = meta.get("content_signature")
        if not known:
            try:
                with open(os.path.join(exp_dir, "patch.diff"), "r", errors="replace") as f:
                    known = content_signature(f.read())
            except OSError:
                known = ""
        if known and known == csig:
            return meta, exp_dir
    return None


def _record_reproduction(dup, csig: str, speedup: float, a) -> dict:
    """Count a re-measurement onto the entry that already holds this code. Two of them promote
    candidate -> active. The original's metric is NOT overwritten: it was measured on its own bench."""
    meta, exp_dir = dup
    try:
        reps = int(meta.get("reproductions") or 1) + 1
    except (TypeError, ValueError):
        reps = 2
    meta["reproductions"] = reps
    meta["content_signature"] = csig
    if reps >= 2:
        meta["lifecycle"] = "active"
    try:
        _atomic_write(os.path.join(exp_dir, "meta.yaml"), _dump_meta(meta))
    except OSError as e:
        return {"written": False, "reason": "io_error: " + str(e)[:120]}
    return {
        "written": False,
        "reason": "duplicate_impl",
        "slug": make_slug(a.kernel_name, a.language, _norm_gfx(a.gfx)),
        "dir": exp_dir,
        "speedup": round(speedup, 4),
        "reproduced": os.path.basename(exp_dir),
        "reproductions": reps,
        "lifecycle": meta["lifecycle"],
    }


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

    carrier = str(getattr(a, "carrier", "") or "patch")
    if carrier not in CARRIERS:
        return {"written": False, "reason": "unknown_carrier"}

    patch_text = ""
    if a.patch and os.path.isfile(a.patch):
        try:
            with open(a.patch, "r", errors="replace") as f:
                patch_text = f.read()
        except OSError:
            patch_text = ""

    # What must exist for the entry to mean anything differs by carrier, and only by carrier: a patch
    # entry with no diff is empty, and a tuned-artifact entry with no table is empty. A tuning win MAY
    # also carry a diff (the routing/dispatch half), and when it does it rides along as prose-adjacent
    # evidence — it is not what makes the entry adoptable, so it is not what gates the write.
    artifacts = _artifact_sources(getattr(a, "artifact", None))
    if carrier == "patch":
        if not patch_text.strip():
            return {"written": False, "reason": "empty_diff"}
    elif not artifacts:
        return {"written": False, "reason": "no_artifact"}

    kernel_class = a.kernel_class or "unknown"
    case_names = [c.strip() for c in (a.case_names or "").split(",") if c.strip()]
    slug = make_slug(a.kernel_name, a.language, gfx)

    # A re-measurement of code the store already holds is a REPRODUCTION, not a new entry.
    csig = (content_signature(patch_text) if carrier == "patch"
            else artifact_signature(artifacts))
    if carrier != "patch" and not csig:
        return {"written": False, "reason": "unreadable_artifact"}
    dup = _find_by_content(a.root, gfx, slug, csig) if csig else None
    if dup:
        return _record_reproduction(dup, csig, speedup, a)

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
        "lifecycle": "candidate",           # earns 'active' only via independent reproduction
        # Absent on every entry written before the tuning track existed, and readers must treat a
        # missing carrier as `patch` — the whole imported backlog is diffs.
        "carrier": carrier,
        "gfx": gfx,
        "kernel_class": kernel_class,
        "kernel_name": a.kernel_name,
        "language": a.language,
        "metric": {
            "speedup": round(speedup, 6),
            "wall_ms": round(wall_ms, 6) if wall_ms is not None else None,
            "baseline_wall_ms": round(baseline_ms, 6) if baseline_ms is not None else None,
            "gpu_arch": gfx,
            # What the speedup was measured against; resolve compares candidates only within one
            # bench_key. Empty when the caller does not supply them.
            "metric_kind": a.metric_kind or "",
            "bench_key": bench_key(a.metric_kind, case_names),
            "case_names": case_names,
        },
        # The optimization IDEA, not the impl: resolve ranks at most one entry per direction.
        "direction": (a.direction or "")[:120],
        "content_signature": csig,
        "reproductions": 1,
        # exp_dir of the warm-start entry this was built on — tells a later curation pass "the same
        # idea, one round further" from "an independent second discovery".
        "derived_from": a.parent or "",
        "verified_on": time.strftime("%Y-%m-%d"),
        # Observed here, in the container that took the measurement — a speedup with no stack behind
        # it cannot be compared to anything later.
        "verified_stack": detect_stack(a.language),
        "source_eval_dir": a.eval_dir or "",
    }
    upstream = _upstream_of(a)
    if upstream:
        # Omitted entirely when the caller states none, so an entry written by an unmodified caller
        # is byte-identical to what it was before this field existed.
        meta["upstream"] = upstream
    if carrier == "tuned_artifact":
        # The three things that make a tuned table usable and that a diff would have carried
        # implicitly: which files, what binds them, and what silently ignores them if skipped.
        meta["artifact_files"] = [name for name, _ in artifacts]
        # The STORED name is sanitized, because a tuned table's name is derived from its shape
        # (`E=8,N=1024,device_name=AMD Instinct MI355X.json`) and those characters make the remote
        # plane's safe_rel_path() raise. But the runtime finds the table only under that exact name
        # — install it as `E-8-N-1024-...` and it is silently ignored, which looks like a tuning
        # loss rather than a filing error. So the destination name is carried explicitly.
        meta["artifact_names"] = {name: os.path.basename(src) for name, src in artifacts}
        if getattr(a, "apply_env", ""):
            meta["apply_env"] = str(a.apply_env)[:400]
        if getattr(a, "cache_invalidation", ""):
            meta["cache_invalidation"] = str(a.cache_invalidation)[:400]
        if getattr(a, "tuner", ""):
            meta["tuner"] = str(a.tuner)[:80]
        # A tuning win reaches production through deploy.sh, not `git apply`. Saying so in the entry
        # is what stops a warm start from trying the wrong installation route and calling it a failure.
        meta["apply_route"] = "deploy_bundle"

    # Copy the tech_lead report verbatim as prose; lift its first non-empty line as the strategy,
    # and keep its dead-ends so the next run on this kernel does not re-fund a closed direction.
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
            structured = parse_dead_ends(report_text)
            prose = dead_ends_md(report_text)
            if structured:
                meta["dead_ends"] = structured
            if prose:
                meta["dead_ends_md"] = prose
        except OSError:
            pass
    if a.strategy:
        strategy = a.strategy[:300]
    meta["strategy"] = strategy

    try:
        # A tuning win's optional routing diff still lands as patch.diff — it is real evidence — but
        # an empty one is not written at all, so `os.path.isfile(patch.diff)` stays a truthful test
        # of "there is a diff here" for every reader, old and new.
        if patch_text.strip():
            _atomic_write(os.path.join(out_dir, "patch.diff"), patch_text)
        for name, src in artifacts:
            with open(src, "rb") as f:
                blob = f.read()
            _atomic_write(os.path.join(out_dir, "artifact", name), blob)
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
        "carrier": carrier,
        "artifacts": [name for name, _ in artifacts],
        "speedup": round(speedup, 4),
    }


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


def _list_pages(root: str, gfx: str):
    """Yield (slug, match_key, language) for every page under <root>/<gfx>/<kernel_class>/.
    The slug splits from the RIGHT: a kernel name may itself contain '__' (e.g. `_w8a8__v2`)."""
    base = os.path.join(root, gfx)
    if not os.path.isdir(base):
        return []
    out = {}
    for kernel_class in sorted(os.listdir(base)):
        kc_dir = os.path.join(base, kernel_class)
        if not os.path.isdir(kc_dir):
            continue
        for slug in sorted(os.listdir(kc_dir)):
            if slug in out or not os.path.isdir(os.path.join(kc_dir, slug)):
                continue
            parts = str(slug).rsplit("__", 2)
            name, lang = (parts[0], parts[1]) if len(parts) == 3 else (slug, "")
            out[slug] = (slug, _match_key(name), lang.lower())
    return [out[s] for s in sorted(out)]


def resolve_slug(root: str, gfx: str, kernel_name: str, language: str, match: str = "fuzzy"):
    """Find the kernel page for (kernel_name, language) on this arch, most-specific tier first:
      exact       the canonical slug is on disk;
      normalized  same canonical name up to case/separators (`wvsplitk` -> `wvSplitK`);
      fuzzy       one canonical name contains the other, unambiguously — this is what turns an e2e
                  op_kind (`fused_moe`) into the `fused_moe_kernel` page.
    Returns (slug_or_'', tier, info); info carries the pages NOT served, so a surprising match shows
    up in the log instead of silently steering the run.
    """
    want_slug = make_slug(kernel_name, language, gfx)
    pages = _list_pages(root, gfx)
    info = {"other_language_pages": [], "ambiguous": []}
    if any(s == want_slug for s, _k, _lg in pages):
        return want_slug, "exact", info

    want_key, want_lang = _match_key(kernel_name), str(language or "").strip().lower()
    info["other_language_pages"] = [s for s, k, lg in pages if k == want_key and lg != want_lang]
    if match == "exact":
        return "", "none", info

    same_key = [s for s, k, lg in pages if k == want_key and lg == want_lang]
    if same_key:
        return same_key[0], "normalized", info
    if match != "fuzzy" or len(want_key) < 6:
        return "", "none", info

    # Containment, closest name first: `fused_moe` prefers `fused_moe_kernel` over
    # `fused_moe_kernel_gptq_awq`. Two equally-close pages are AMBIGUOUS -> serve neither.
    cands = []
    for s, k, lg in pages:
        if len(k) < 6 or not (want_key in k or k in want_key):
            continue
        if lg != want_lang:
            info["other_language_pages"].append(s)
        else:
            cands.append((abs(len(k) - len(want_key)), s))
    if not cands:
        return "", "none", info
    cands.sort()
    if len(cands) > 1 and cands[0][0] == cands[1][0]:
        info["ambiguous"] = [s for d, s in cands if d == cands[0][0]]
        return "", "ambiguous", info
    return cands[0][1], "fuzzy", info


# ---------------------------------------------------------------------------------------------
# Path remapping. A stored patch was produced in the workspace that won it — an arena checkout
# (`source/triton_fused_moe_kernel.py`, `csrc/...`) — while an e2e head run edits an extracted
# subtree (`kernel_src/.../fused_moe_kernel.py`). Same code, different prefix AND different
# basename, so no `-p<N>` strip depth reaches the file: without rewriting the paths, every warm
# start on the head path fails to apply and the KB is dead weight there.
def _diff_targets(patch_text: str):
    """Paths the diff touches: {path: is_new_file}. '' if the diff renames (not remappable)."""
    targets, pending_new = {}, False
    for line in (patch_text or "").splitlines():
        if line.startswith("rename from ") or line.startswith("rename to "):
            return None
        if line.startswith("new file mode"):
            pending_new = True
        elif line.startswith("--- "):
            pending_new = pending_new or line[4:].strip() in ("/dev/null", "a//dev/null")
        elif line.startswith("+++ "):
            p = line[4:].strip().split("\t")[0]
            if p != "/dev/null":
                targets[re.sub(r"^b/", "", p)] = pending_new
            pending_new = False
    return targets


def _match_path(target: str, editable):
    """Best editable path for one patch target, most-specific tier first: identical path, then one
    path is the tail of the other, then same basename, then same basename modulo the language
    prefix/extension (`triton_fused_moe_kernel.py` -> `fused_moe_kernel.py`). A tier with two
    equally good hits is ambiguous -> no mapping, rather than a guess that verify pays to reject."""
    t_base = os.path.basename(target)
    tiers = (
        [e for e in editable if e == target],
        [e for e in editable if e.endswith("/" + target) or target.endswith("/" + e)],
        [e for e in editable if os.path.basename(e) == t_base],
        [e for e in editable if _match_key(os.path.basename(e)) == _match_key(t_base)
         and os.path.splitext(e)[1] == os.path.splitext(target)[1]],
    )
    for tier in tiers:
        if len(set(tier)) == 1:
            return tier[0]
        if tier:
            return ""
    return ""


def _rewrite_paths(patch_text: str, mapping: dict) -> str:
    """Rewrite the a//b/ path on every header line; hunks are copied through untouched."""
    out = []
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            for old, new in mapping.items():
                line = line.replace(f"a/{old} ", f"a/{new} ").replace(f"b/{old}", f"b/{new}")
        elif line.startswith("--- a/") or line.startswith("+++ b/"):
            head, path = line[:6], line[6:].split("\t")[0]
            if path in mapping:
                line = head + mapping[path]
        out.append(line)
    return "\n".join(out) + "\n"


def _drop_sections(patch_text: str, drop: set) -> str:
    """Remove whole `diff --git` sections for the given target paths."""
    out, keep = [], True
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            tail = line.split(" b/", 1)
            keep = not (len(tail) == 2 and tail[1].strip() in drop)
        if keep:
            out.append(line)
    return "\n".join(out) + "\n"


# A patch that also touches a non-source file this workspace lacks (a .gitignore line, a README
# note) is still a perfectly good kernel patch. Refusing the whole thing over it wastes the entry;
# the section is dropped and named in `dropped` so the decision is visible.
_SOURCE_EXTS = {".py", ".hip", ".cu", ".cuh", ".cpp", ".cc", ".hpp", ".h", ".c", ".jinja",
                ".s", ".asm", ".json", ".yaml", ".yml", ".sh", ".mk", ".txt"}


def cmd_remap(a) -> dict:
    """Rewrite a stored patch's paths onto THIS workspace's layout, or refuse and say why."""
    try:
        with open(a.patch, "r", errors="replace") as f:
            patch_text = f.read()
    except OSError as e:
        return {"remapped": False, "reason": "unreadable_patch: " + str(e)[:80]}

    editable = [p.strip().lstrip("./") for p in (a.editable or "").split(",") if p.strip()]
    if not editable and a.workspace:
        editable = _walk_workspace(a.workspace)
    if not editable:
        return {"remapped": False, "reason": "no_editable_set"}

    targets = _diff_targets(patch_text)
    if targets is None:
        return {"remapped": False, "reason": "rename_not_supported"}
    if not targets:
        return {"remapped": False, "reason": "no_paths_in_patch"}

    mapping, unmapped, new_files = {}, [], []
    for target, is_new in sorted(targets.items()):
        if is_new:
            new_files.append(target)
            continue
        hit = _match_path(target, editable)
        if hit and hit != target:
            mapping[target] = hit
        elif not hit:
            unmapped.append(target)
    # A file the patch CREATES has nothing to match, so it follows the layout shift its edited
    # siblings underwent. No shift (every edited path already fits here) => this workspace has the
    # patch's own layout and the new file belongs exactly where the patch puts it.
    for target in new_files:
        host = next(iter(mapping.values()), None)
        if not host and not any(not targets[t] for t in targets):
            unmapped.append(target)          # a patch of ONLY new files has nothing to anchor to
        elif host and os.path.dirname(host) != os.path.dirname(target):
            mapping[target] = os.path.join(os.path.dirname(host), os.path.basename(target))

    dropped = [p for p in unmapped if os.path.splitext(p)[1] not in _SOURCE_EXTS]
    unmapped = [p for p in unmapped if p not in dropped]
    # All-or-nothing on SOURCE files: applying the mapped half of a patch leaves the workspace
    # inconsistent, and verify would pay a full on-box run to discover that.
    if unmapped:
        return {"remapped": False, "reason": "unmapped_paths", "unmapped": unmapped,
                "dropped": dropped, "mapped": mapping}
    if not mapping and not dropped:
        return {"remapped": False, "reason": "no_change_needed", "mapped": {}}
    text = _drop_sections(patch_text, set(dropped)) if dropped else patch_text
    try:
        _atomic_write(a.out, _rewrite_paths(text, mapping))
    except OSError as e:
        return {"remapped": False, "reason": "io_error: " + str(e)[:80]}
    return {"remapped": True, "reason": "ok", "out": a.out, "mapped": mapping, "dropped": dropped}


_SKIP_DIRS = {".git", "__pycache__", "node_modules", "build", ".venv", "exp"}


def _walk_workspace(workspace: str, cap: int = 20000):
    """Every file under the workspace, repo-relative, as a fallback editable set. Deliberately NOT
    filtered by extension: a whitelist made real targets invisible (a `.cpp.jinja` template that
    exists at the patch's exact path) and refused a patch that would have applied verbatim."""
    out = []
    for dirpath, dirnames, filenames in os.walk(workspace):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            out.append(os.path.relpath(os.path.join(dirpath, fn), workspace))
            if len(out) >= cap:
                return out
    return out


def _speedup_of(meta: dict) -> float:
    try:
        return float((meta.get("metric") or {}).get("speedup"))
    except (TypeError, ValueError):
        return 0.0


def _is_retired(meta: dict) -> bool:
    """The curation's own verdict, as written into meta.yaml by the pass that built the store."""
    return meta.get("retained") is False or bool(meta.get("retired_reason"))


def _local_attestations(meta: dict) -> dict:
    """This entry's attestation ledger, or {} when nobody has ever tried it.

    Empty rather than a zeroed ledger because `remote_value` drops empty values, and a record
    that has never been recalled should carry no ledger at all — four zeroes and no ledger mean
    the same thing to a reader, and the shorter one does not imply somebody looked.
    """
    from kb.attest import BUCKETS, attestations_of
    ledger = attestations_of(meta if isinstance(meta, dict) else {})
    counted = any(ledger[k] for k in ("recalls",) + BUCKETS)
    return ledger if counted else {}


def _rank_key(md):
    """Recorded speedup, then reproductions, then exp_id for determinism."""
    meta, exp_dir = md
    try:
        reps = int(meta.get("reproductions") or 0)
    except (TypeError, ValueError):
        reps = 0
    return (-_speedup_of(meta), -reps, os.path.basename(exp_dir))


def _track_record_md(meta) -> str:
    """One line on what happened the last times this patch was adopted, or nothing at all.

    Omitted entirely for an untried entry rather than printed as "0 attempts": the reader is an
    agent about to spend a verify slot, and a line that says nothing still costs it a decision.
    """
    ledger = _local_attestations(meta)
    if not ledger:
        return ""
    hint = _retire_hint_of(meta)
    return ("- track record: adopted %d time(s) — %d reproduced a win, %d did not win, %d would "
            "not run%s\n" % (ledger["recalls"], ledger["validations"], ledger["failures"],
                             ledger["not_reproduced"], " (**%s**)" % hint if hint else ""))


def _render_references(refs_dir: str, address: str, summary: str, views):
    """Mirror the offered candidates' prose into `refs_dir` and index it, one prose path per view.

    Written up front, before any verdict, so a warm start that is later rejected stays auditable.
    Both planes render the same page — a reference reads the same whether the entry came out of a
    directory or from behind a KB Store key — so only `address` and each view's `origin` line
    differ between them. A page that cannot be written is reported as "" rather than failing the
    read: the patch is still adoptable without its prose.
    """
    views = list(views)
    key = "|".join(v["key"] for v in views).encode("utf-8", "replace")
    set_dir = os.path.join(refs_dir, "sets", hashlib.sha256(key).hexdigest()[:7])
    top_bench = views[0]["bench_key"] if views else ""
    index_lines = [
        f"# Warm-start references — {address}", "", summary,
        f"Speedups compare only within one bench key; rank 1's is `{top_bench or 'none'}`.", "",
    ]
    paths = []
    for rank, v in enumerate(views, start=1):
        meta = v["meta"]
        prose_path = os.path.join(set_dir, f"reference_{rank:02d}.md")
        try:
            body = ""
            if os.path.isfile(v["report_path"]):
                with open(v["report_path"], "r", errors="replace") as f:
                    body = f.read()
            _atomic_write(prose_path, (
                f"# Reference {rank:02d} — {address}\n\n"
                f"- speedup: {v['speedup']:.4f}x ({v['metric_kind'] or 'unknown metric'}, "
                f"bench `{v['bench_key'] or 'none'}`)\n"
                f"- direction: {v['direction'] or 'unlabeled'}\n"
                + _techniques_md(_techniques(meta))
                + f"- strategy: {meta.get('strategy', '')}\n"
                + v["origin"]
                + f"- verified_on: {meta.get('verified_on', '')}\n"
                f"- verified_stack: {_stack_str(meta)}\n"
                + _track_record_md(meta)
                + _alternates_md(v["alts"])
                + f"\n---\n\n{_prose_body(meta, body)}\n"
            ))
        except OSError:
            prose_path = ""
        paths.append(prose_path)
        index_lines.append(
            f"- Rank {rank}: `{prose_path}` | speedup {v['speedup']:.4f}x | direction "
            f"`{v['direction'] or 'unlabeled'}` | bench `{v['bench_key'] or 'none'}` | "
            f"patch `{v['patch_path']}` | {len(v['alts'])} alternate(s) | status `read`"
        )
    try:
        _atomic_write(os.path.join(refs_dir, "index.md"), "\n".join(index_lines) + "\n")
    except OSError:
        pass
    return paths


def _candidate(rank: int, v: dict, gfx: str, prose_path: str, top_bench: str) -> dict:
    """The candidate record both planes hand the lane. Extra keys ride in `v['extra']`."""
    return dict({
        "rank": rank,
        "exp_dir": v["exp_dir"],
        "speedup": round(v["speedup"], 4),
        "arch": gfx,
        "patch_path": v["patch_path"],
        "prose_path": prose_path,
        "strategy": str(v["meta"].get("strategy") or ""),
        "direction": v["direction"],
        "techniques": _techniques(v["meta"]),
        "bench_key": v["bench_key"],
        "metric_kind": v["metric_kind"],
        # `patch` for every entry the kernel lane has ever written, so a reader that ignores this
        # key behaves exactly as it did before carriers existed.
        "carrier": v.get("carrier", "patch"),
        # Empty for a patch entry. Non-empty means: install these, export apply_env, run the cache
        # invalidation, restart the server — `git apply` does nothing for this candidate.
        "artifact_paths": v.get("artifact_paths") or [],
        "artifact_names": v.get("artifact_names") or {},
        "apply_env": v.get("apply_env", ""),
        "cache_invalidation": v.get("cache_invalidation", ""),
        # False = ranked against rank 1 on a DIFFERENT case set, so their ordering is a prior only.
        # Adoption is decided by this run's own measurement either way.
        "comparable": bool(v["bench_key"]) and v["bench_key"] == top_bench,
        "alternates": v["alts"],
        # What happened the last times somebody actually adopted this patch, as opposed to the
        # speedup its own writer measured once. An entry offered at rank 1 that three lanes have
        # since failed to reproduce should not read identically to an untried one, and before this
        # it did. A hint DEMOTES (see kb/curate.py:demote_hinted); only retraction hides a record.
        "validations": _local_attestations(v["meta"]).get("validations", 0),
        "recalls": _local_attestations(v["meta"]).get("recalls", 0),
        "retire_hint": _retire_hint_of(v["meta"]),
        "status": "read",
    }, **v["extra"])


def _retire_hint_of(meta) -> str:
    from kb.attest import retire_hint
    return retire_hint(meta if isinstance(meta, dict) else {})


def cmd_resolve(a) -> dict:
    gfx = _norm_gfx(a.gfx)
    if not gfx:
        return {"read_reason": "missing_arch", "candidates": []}

    root = a.root
    requested_slug = make_slug(a.kernel_name, a.language, gfx)
    if not os.path.isdir(os.path.join(root, gfx)):
        return {"read_reason": "kernel_page_not_found", "slug": requested_slug, "candidates": []}

    slug, match_tier, match_info = resolve_slug(root, gfx, a.kernel_name, a.language, a.match)
    base_out = {
        "slug": slug or requested_slug,
        "requested_slug": requested_slug,
        "match_tier": match_tier,
        "other_language_pages": sorted(set(match_info["other_language_pages"])),
        "ambiguous_pages": match_info["ambiguous"],
        "candidates": [],
    }
    if not slug:
        reason = ("ambiguous_kernel_page" if match_tier == "ambiguous"
                  else "no_page_for_language" if match_info["other_language_pages"]
                  else "kernel_page_not_found")
        return dict(base_out, read_reason=reason)

    # The <gfx> path segment already guarantees same-arch; re-check metric.gpu_arch to catch a mislabeled entry.
    found = [(m, d) for (m, d) in _iter_solutions(root, gfx, slug)
             if _norm_gfx((m.get("metric") or {}).get("gpu_arch") or m.get("gfx") or gfx) == gfx]
    if not found:
        return dict(base_out, read_reason="no_same_arch")

    # ONE carrier per call, `patch` unless asked otherwise. A caller that can `git apply` a diff
    # generally cannot install a tuned table (it needs the deploy bundle and a server restart), and
    # the two are not substitutable, so mixing them in one ranked list would offer a kernel lane a
    # candidate it has no way to adopt. Entries written before carriers existed have no field and
    # are diffs.
    want_carrier = str(getattr(a, "carrier", "") or "patch")
    of_carrier = [(m, d) for (m, d) in found if str(m.get("carrier") or "patch") == want_carrier]
    other_carrier_n = len(found) - len(of_carrier)
    if not of_carrier:
        return dict(base_out, read_reason="no_such_carrier", carrier=want_carrier,
                    other_carriers=other_carrier_n)
    found = of_carrier

    # Precision, on the same footing as carrier and for the same reason: not a dimension of the
    # address (the reader cannot always recompute it), but a hard fact about whether an entry is
    # usable here. A tuned table is dtype-specialized, and `bench_key` does NOT partition the
    # ranking — it only sets a `comparable` flag — so without this an fp8 page's top-N can be taken
    # entirely by bf16 entries that then lose their direction slot to nothing.
    #
    # Off by default. Omitting --precision reproduces the previous behaviour exactly, and an entry
    # that states no precision is never excluded (see _precision_matches).
    want_precision = _norm_precision(getattr(a, "precision", ""))
    other_precision_n = 0
    if want_precision:
        of_precision = [(m, d) for (m, d) in found
                        if _precision_matches(want_precision, _precision_of(m))]
        other_precision_n = len(found) - len(of_precision)
        if not of_precision:
            return dict(base_out, read_reason="no_such_precision", carrier=want_carrier,
                        precision=want_precision, other_precisions=other_precision_n)
        found = of_precision

    # --- curation gate: what this page may OFFER, before any ranking -------------------------
    total = len(found)
    servable = found if a.include_retired else [(m, d) for (m, d) in found if not _is_retired(m)]
    retired_n = total - len(servable)
    try:
        min_speedup = float(a.min_speedup)
    except (TypeError, ValueError):
        min_speedup = 1.0
    above = [(m, d) for (m, d) in servable if _speedup_of(m) >= min_speedup]
    below_n = len(servable) - len(above)
    stats = {"total": total, "retired": retired_n, "below_min_speedup": below_n,
             "min_speedup": min_speedup, "carrier": want_carrier,
             "other_carriers": other_carrier_n,
             "precision": want_precision, "other_precisions": other_precision_n}
    if not above:
        return dict(base_out, filtered=stats,
                    read_reason="all_retired" if not servable else "below_min_speedup")

    ordered = demote_hinted(sorted(above, key=_rank_key), lambda md: _retire_hint_of(md[0]))
    stats["demoted_by_hint"] = sum(1 for md in ordered if _retire_hint_of(md[0]))
    top, alternates, collapsed = collapse_by_direction(
        ordered, lambda md: md[0].get("direction"), lambda md: md[1], a.top_n)
    stats["same_direction_collapsed"] = collapsed

    views = []
    for (meta, exp_dir), alt_of in zip(top, alternates):
        metric = meta.get("metric") or {}
        views.append({
            "key": exp_dir,
            "meta": meta,
            "exp_dir": exp_dir,
            "patch_path": os.path.join(exp_dir, "patch.diff"),
            "report_path": os.path.join(exp_dir, "report.md"),
            "carrier": str(meta.get("carrier") or "patch"),
            # Absolute, because the caller installs these from wherever it happens to be running.
            "artifact_paths": [os.path.join(exp_dir, "artifact", n)
                               for n in (meta.get("artifact_files") or [])],
            # stored name -> the name it must be installed under; see the write path.
            "artifact_names": dict(meta.get("artifact_names") or {}),
            "apply_env": str(meta.get("apply_env") or ""),
            "cache_invalidation": str(meta.get("cache_invalidation") or ""),
            "speedup": _speedup_of(meta),
            "direction": str(meta.get("direction") or ""),
            "bench_key": str(metric.get("bench_key") or ""),
            "metric_kind": str(metric.get("metric_kind") or ""),
            "origin": f"- source: {meta.get('source_eval_dir', '')}\n",
            "alts": [{
                "exp_dir": d,
                "patch_path": os.path.join(d, "patch.diff"),
                "speedup": round(_speedup_of(m), 4),
                "bench_key": str((m.get("metric") or {}).get("bench_key") or ""),
                "techniques": _techniques(m),
            } for (m, d) in alt_of],
            "extra": {"slug": slug},
        })

    summary = (f"Matched `{requested_slug}` -> `{slug}` ({match_tier}). {len(top)} direction(s) "
               f"offered from {total} recorded run(s): {retired_n} retired by curation, "
               f"{below_n} below {min_speedup:g}x, {collapsed} same-direction re-discoveries "
               f"moved to `alternates`.")
    prose = _render_references(a.refs_dir, f"slug `{slug}` (gfx {gfx})", summary, views)
    candidates = [_candidate(rank, v, gfx, p, views[0]["bench_key"])
                  for rank, (v, p) in enumerate(zip(views, prose), start=1)]
    return dict(base_out, read_reason="read", candidates=candidates, filtered=stats)


def cmd_languages(a) -> dict:
    """Which languages this kernel actually has a page in. A caller that guesses `triton` for a
    kernel the store keeps under `hip`/`ck` gets read_reason=empty and silently loses its history,
    so let the store answer instead of a task_type mapping that cannot tell hip from ck."""
    gfx = _norm_gfx(a.gfx)
    if not gfx:
        return {"languages": [], "reason": "missing_arch"}
    pages = _list_pages(a.root, gfx)
    want = _match_key(a.kernel_name)
    langs = sorted({lg for _s, k, lg in pages if k == want and lg})
    if langs:
        return {"gfx": gfx, "languages": langs, "match_tier": "exact", "reason": "ok"}
    near = sorted({lg for _s, k, lg in pages if lg and (want in k or k in want)})
    if near:
        return {"gfx": gfx, "languages": near, "match_tier": "fuzzy", "reason": "ok"}
    return {"gfx": gfx, "languages": [], "match_tier": "none", "reason": "no_page"}


# Stacks the imported backlog was measured on, recovered from the campaign's own eval dirs
# (`analysis.json` / `codebase_context.md` device strings). Marked as recovered, not observed —
# a later reader must be able to tell a backfilled stack from one detect_stack() saw first-hand.
_BACKFILL_STACK = {
    # rocm is on all three, not just the two that compile against it directly: the whole campaign
    # ran in one container image, and rocm is the version the remote identity is keyed on, so a
    # triton entry without it exports to a different address than the hip entry beside it.
    "triton": {"triton": "3.6.0", "torch": "2.11.0", "rocm": "7.2"},
    "hip": {"rocm": "7.2"},
    "ck": {"rocm": "7.2"},
}


def _backfill_one(meta: dict, exp_dir: str, stacks: dict):
    """Fields to add/drop for one entry, as (new_meta, changes) — or (meta, {}) when already done."""
    out = dict(meta)
    changes = {"add": [], "drop": [], "fix": []}

    if not str(out.get("dead_ends_md") or "").strip():
        try:
            with open(os.path.join(exp_dir, "report.md"), "r", errors="replace") as f:
                report = f.read()
        except OSError:
            report = ""
        prose = dead_ends_md(report)
        if prose:
            out["dead_ends_md"] = prose
            changes["add"].append("dead_ends_md")

    # Fill per KEY, not per dict: an early backfill gave triton entries {triton, torch} and no
    # rocm, which is exactly the key the remote identity is derived from. Values already present
    # are never overwritten — an observed stack always outranks a recovered one.
    st = out.get("verified_stack")
    st = dict(st) if isinstance(st, dict) else {}
    known = stacks.get(str(out.get("language") or "").lower()) or {}
    added = [k for k in known if not str(st.get(k) or "").strip()]
    if added:
        st.update({k: known[k] for k in added})
        st.setdefault("recorded_by", "campaign20_backfill")
        out["verified_stack"] = st
        changes["add"].append("verified_stack:" + ",".join(sorted(added)))

    # impl_signature is a different hash under a name nothing reads: _find_by_content() falls back
    # to re-hashing patch.diff for all 248 entries on every resolve. Recompute under the real name.
    if not out.get("content_signature"):
        try:
            with open(os.path.join(exp_dir, "patch.diff"), "r", errors="replace") as f:
                csig = content_signature(f.read())
        except OSError:
            csig = ""
        if csig:
            out["content_signature"] = csig
            changes["fix"].append("content_signature")
    if "impl_signature" in out and out.get("content_signature"):
        out.pop("impl_signature")
        changes["drop"].append("impl_signature")

    # Never read, and each is either constant or a duplicate of a field right next to it.
    for dead in ("layer", "platforms", "patch_content"):
        if dead in out:
            out.pop(dead)
            changes["drop"].append(dead)

    if not (changes["add"] or changes["drop"] or changes["fix"]):
        return meta, {}
    return out, changes


def cmd_backfill_content(a) -> dict:
    """Bring the imported backlog up to the current content shape. Dry-run by default; only ever
    adds the fields named above — retained / direction / techniques / metric are never touched."""
    root = a.root
    if not os.path.isdir(root):
        return {"ok": False, "reason": "no_such_root: " + root}
    stacks = dict(_BACKFILL_STACK)
    scanned = changed = failed = 0
    for dirpath, _dirs, files in os.walk(root):
        if "meta.yaml" not in files:
            continue
        scanned += 1
        meta_path = os.path.join(dirpath, "meta.yaml")
        meta = _read_meta(meta_path)
        if not isinstance(meta, dict):
            failed += 1
            continue
        new_meta, changes = _backfill_one(meta, dirpath, stacks)
        if not changes:
            continue
        changed += 1
        print(json.dumps({"dir": dirpath, **changes}, ensure_ascii=False))
        if a.apply:
            try:
                _atomic_write(meta_path, _dump_meta(new_meta))
            except OSError as e:
                failed += 1
                print(json.dumps({"dir": dirpath, "error": str(e)[:120]}))
    return {"ok": True, "applied": bool(a.apply), "scanned": scanned,
            "changed": changed, "failed": failed}


# --- remote KB export -------------------------------------------------------------------------
# Record shape mirrors KernelForge's (knowledge/kernel_identity.py and
# rewrite_by_flydsl/{identity,agent_kb,record_store}.py @ baabdae); the ADDRESS does not, and
# kb/identity.py owns it for both workflows and says why. Read and write must both go through it:
# the store finds nothing if the two sides disagree by one segment, and there is no error to notice
# — a mistyped dimension just reads as a cold start.
#
# The scheme is `geak:`, not `kernel:`, because our credential is scoped to `geak` identities and
# 403s on both `kernel:` and `inference:`. That scheme is client-defined and exact-lookup only, so
# every dimension has to be something the READ side can recompute from what it already knows;
# nothing may be derived from run-local state. Two consequences worth having in view here:
#
#   * the serving framework (vllm / sglang), its version and the numeric precision are NOT
#     dimensions, even though an e2e run knows all three. kernel_lane.js does not — it has no
#     upstream awareness at all, and pass-through from e2e forwards only `target_language`. A
#     dimension the reader cannot reconstruct is a permanent silent 404. They ride in
#     `value.upstream` instead, where a client can filter on them (`resolve`/`resolve-remote
#     --precision`); precision is additionally already spelled into most kernel names
#     (`fused_moe_int4_w4a16`, `_w8a8_triton_block_scaled_mm`) so keying on it would double-encode
#     and split those pages. Being a filter rather than a dimension is also what makes it safe on a
#     store with no delete: a caller that cannot state its precision loses filtering, not its whole
#     history, where a wrong key segment would have lost the page.
#   * every write publishes to BOTH rungs of kernel_canonical_ids(). The service does no prefix
#     aggregation, so the version-agnostic page exists only because we put records there.
REMOTE_PRODUCER = "geak"
REMOTE_ARTIFACT_KIND = "rewrite"        # upstream ARTIFACT_KIND for a recipe bundle

try:
    from kb import identity as _kbid
except ImportError:                     # resolve/write stay usable; only the remote pair needs it
    _kbid = None

REMOTE_SCHEME = "geak"
REMOTE_DOMAIN = "kernel"
REMOTE_FRAMEWORK = "rocm"
REMOTE_UNKNOWN_VERSION = "unspecified"


def _identity_module():
    if _kbid is None:
        raise RuntimeError("kb_identity_unavailable: kb/identity.py must be importable from the repo root")
    return _kbid


def remote_segment(value, fallback: str) -> str:
    """Fold a free-form value into one identity dimension. Delegates so there is one folding rule."""
    return _identity_module().segment(value, fallback)


def remote_gpu(gfx: str, override: str = "") -> str:
    """The compile target (`gfx950`), NOT the product model, and the LEADING dimension.

    Upstream keys this on the marketing name (`mi355x`) and leaves gfx out of the identity. We
    diverge on both counts. gfx is what every producer and consumer on our side already holds — off
    the box, out of meta.yaml, out of the e2e harness — whereas the product model exists only as a
    lookup table someone has to keep current, and an unmapped arch would file half a kernel's
    history under a name nothing looks up. It leads because an arch mismatch is the quietest way to
    waste a round: a gfx942-tuned patch compiles clean on gfx950 and is merely slower, where a
    wrong kernel_name or language at least fails to apply.
    """
    return remote_segment(override or _norm_gfx(gfx), fallback="unknown")


def remote_framework_version(meta: dict, override: str = "") -> str:
    """The ROCm version this entry was measured on, cut to `<major>.<minor>` for the address.

    Coarse on purpose, and now additionally droppable: it is the last segment precisely because
    7.2 -> 7.3 usually keeps a patch applicable, so the second rung of the ladder is the one that
    keeps 20 kernels warm through an image upgrade instead of cold-starting all of them at once.

    Never guessed: no rocm key exports as `unspecified`. That entry still gets both rungs, so the
    version-agnostic page sees it even though its exact page is one nobody will construct.
    """
    stack = meta.get("verified_stack")
    raw = str(override or "").strip()
    if not raw:
        raw = str((stack or {}).get("rocm") or "").strip() if isinstance(stack, dict) else ""
    return _identity_module()._short_version(raw)


def remote_identity(meta: dict, producer: str = REMOTE_PRODUCER, gpu: str = "",
                    version: str = "") -> dict:
    """The four dimensions of the address.

    `producer` is accepted and recorded but is no longer a dimension: it has one value here, the
    service stamps it on every record, and artifacts are already partitioned under
    `kb/<producer>/<session_id>/`. `version` overrides only the key, never the record — a box whose
    ROCm this script cannot detect would otherwise file at `:unspecified` and split one kernel's
    history in two while `value.verified_stack` keeps saying, correctly, that nothing was observed.
    """
    return _identity_module().kernel_identity(
        gfx=remote_gpu(meta.get("gfx") or (meta.get("metric") or {}).get("gpu_arch") or "", gpu),
        # canon_name for the same reason make_slug uses it: the name is layout-derived, so an e2e
        # head extraction calls this kernel `<name>_task` and the standalone lane calls it `<name>`.
        # The slug plane has folded that since it existed; this one did not, so a head published to
        # an address no reader constructs — and a miss here is a 404 nobody can tell from "never
        # recorded". e2e_store.py addresses the kernel page with the profiler's (already canonical)
        # symbol, so folding here is also what makes that cross-reference a live link.
        kernel_name=canon_name(meta.get("kernel_name")),
        backend=meta.get("language"),
        rocm_version=remote_framework_version(meta, version),
    )


def remote_canonical_ids(identity: dict):
    """Every address this entry is published at, most specific first. Never a subset."""
    return _identity_module().kernel_canonical_ids(identity)


def remote_canonical_id(identity: dict) -> str:
    """The exact address — rung 1. The one a fresh write is fingerprinted from."""
    return remote_canonical_ids(identity)[0]


def _remote_digest(meta: dict, exp_dir: str) -> str:
    """The port fingerprint that names the candidate. Reuses content_signature, which already
    dedups this store by patch content, so re-exporting one entry updates one candidate upstream
    instead of piling on a new one per run."""
    sig = str(meta.get("content_signature") or "")
    if sig:
        return sig.split(":", 1)[-1]
    if str(meta.get("carrier") or "patch") != "patch":
        # (stored_name, path), the same shape cmd_write signs through _artifact_sources — and the
        # same names, since `artifact_files` holds what that pass already collapsed them to. The
        # name is hashed, so re-deriving it here from the basename would be a second spelling of
        # one thing and would digest the record differently on the export path than on the write
        # path: not a raise, a duplicate candidate upstream.
        names = meta.get("artifact_files") or []
        items = [(n, os.path.join(exp_dir, "artifact", n)) for n in names]
        return artifact_signature(items).split(":", 1)[-1]
    try:
        with open(os.path.join(exp_dir, "patch.diff"), "r", errors="replace") as f:
            return content_signature(f.read()).split(":", 1)[-1]
    except OSError:
        return ""


def remote_session_id(canonical_id: str, kernel_name: str, digest: str) -> str:
    """`<producer>-<name>-<identity fp>-<port digest>`, upstream's shape.

    Pass the EXACT rung: the id is reused verbatim on the coarser one, which is what lets the two
    share uploaded artifacts instead of duplicating a 240KB patch. Fingerprinting each rung on its
    own would give one measurement two unrelated ids and stop the coarse page from being a
    reproduction of the exact one.
    """
    return _identity_module().session_id(canonical_id, kernel_name, digest, REMOTE_PRODUCER)


def _sha256_file(path: str):
    h, size = hashlib.sha256(), 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def remote_value(meta: dict, digest: str = "") -> dict:
    """The producer-owned half of the record. Upstream treats `value` as opaque — no schema to
    satisfy — so this is our own meta.yaml minus what the identity already carries.

    bench_key and metric_kind are not optional here even though nothing upstream reads them:
    get_top_sessions ranks purely on the `speedup` number we ourselves declare, so a `b:` entry and
    a `b2:` one get ordered against each other as if they were comparable. The reader has to filter
    on these client-side, and it cannot do that if we did not send them."""
    metric = meta.get("metric") or {}
    value = {
        "direction": str(meta.get("direction") or ""),
        "techniques": _techniques(meta),
        "strategy": str(meta.get("strategy") or ""),
        "metric": {
            "speedup": metric.get("speedup"),
            "wall_ms": metric.get("wall_ms"),
            "baseline_wall_ms": metric.get("baseline_wall_ms"),
            "metric_kind": str(metric.get("metric_kind") or ""),
            "bench_key": str(metric.get("bench_key") or ""),
            "case_names": list(metric.get("case_names") or []),
        },
        "verified_stack": meta.get("verified_stack") if isinstance(meta.get("verified_stack"), dict) else {},
        # The dimensions the ADDRESS deliberately does not carry (see the export note above). They
        # are sent so a client can filter on them, which is the whole reason they were excluded from
        # the key rather than simply dropped — `resolve-remote --precision` is that client.
        "upstream": meta.get("upstream") if isinstance(meta.get("upstream"), dict) else {},
        "verified_on": str(meta.get("verified_on") or ""),
        "measured_by": str(meta.get("measured_by") or ""),
        "reproductions": meta.get("reproductions"),
        # NOT the same thing as `reproductions`, and the two are easy to conflate into one wrong
        # number. `reproductions` counts how many times this lane WROTE the same patch again — a
        # measure of how often the optimizer rediscovers an idea, produced entirely by the writer.
        # `attestations` counts what happened when somebody READ this record and took it to a box:
        # recalls / validations / failures / not_reproduced, in the vocabulary kb/attest.py defines
        # and both lanes share. A record can be rediscovered five times and never once survive a
        # recall, and only the second number says so.
        "attestations": _local_attestations(meta),
        "lifecycle": str(meta.get("lifecycle") or ""),
        "retained": meta.get("retained"),
        # The same two fields the e2e records carry, so one reader can ask "should I believe this"
        # of either lane without knowing which one wrote it. Derived, not invented: `active` is
        # earned here only by independent reproduction (see the write path), so it already IS the
        # validation flag — it was just spelled in a vocabulary nothing outside this file knew.
        # The basis is named for what actually produced the number rather than mapped onto the
        # e2e taxonomy: a kernel's speedup comes from its own isolated bench harness, and calling
        # that a `hot_ab` would claim a serving-level A/B that never ran.
        "validated": str(meta.get("lifecycle") or "") == "active",
        "validation_basis": "kernel_bench",
        # The same digest the session id is built from, so a reader that dedups against its own
        # store and the address it was filed under can never disagree about what this patch is.
        "content_signature": ("csha:" + digest) if digest else str(meta.get("content_signature") or ""),
        "artifacts": {"patch": "patch.diff", "report": "report.md"},
    }
    carrier = str(meta.get("carrier") or "patch")
    if carrier != "patch":
        # A remote reader ranks on `speedup` alone and would otherwise pull patch.diff, find nothing
        # installable, and score this as a broken record rather than a differently-shaped one.
        value["carrier"] = carrier
        value["apply_route"] = str(meta.get("apply_route") or "")
        value["artifacts"] = {"report": "report.md",
                              **{n: "artifact/" + n for n in (meta.get("artifact_files") or [])}}
        # Both halves travel: the STORED names are how the bytes are addressed under the session,
        # and artifact_names is how a reader learns what to install each one AS. A reader that gets
        # one without the other can locate the file or name it, not both.
        value["artifact_files"] = list(meta.get("artifact_files") or [])
        if meta.get("artifact_names"):
            value["artifact_names"] = dict(meta["artifact_names"])
        for k in ("apply_env", "cache_invalidation", "tuner"):
            if meta.get(k):
                value[k] = str(meta[k])
    dead = meta.get("dead_ends")
    if isinstance(dead, list) and dead:
        value["dead_ends"] = dead
    # dead_ends_md is deliberately NOT sent: it runs to tens of KB and report.md already carries it
    # verbatim as an artifact. Structured dead ends are small enough to ride in the record.
    return {k: v for k, v in value.items() if v not in ("", None, [], {})}


def remote_records(meta: dict, exp_dir: str, producer: str = REMOTE_PRODUCER, gpu: str = "",
                   version: str = ""):
    """One measurement as upload-ready candidates — one per rung, most specific first.

    All rungs carry the same session id, the same knowledge and the same files. They are not
    variants of a result; they are one result filed at every address a reader might construct. That
    is why the caller must publish all of them or none: a coarse page fed by only some runs ranks
    worse than an empty one, because a reader cannot tell a thin page from a complete one.

    `rung` is stamped on each record so an uploader can skip re-transferring artifacts for rungs
    after the first — remotely the bytes are shared via the session id, and re-PUTting them would
    only burn the presign window.
    """
    identity = remote_identity(meta, producer, gpu, version)
    cids = remote_canonical_ids(identity)
    digest = _remote_digest(meta, exp_dir)
    sid = remote_session_id(cids[0], identity["kernel_name"], digest)
    speedup = _speedup_of(meta)
    files = []
    # patch.diff is absent on a tuned-artifact entry that shipped no routing fix, and the artifacts
    # are absent on every patch entry — the isfile() check below is what makes one loop serve both.
    names = ["patch.diff", "report.md"]
    names += [os.path.join("artifact", n) for n in (meta.get("artifact_files") or [])]
    for name in names:
        path = os.path.join(exp_dir, name)
        if not os.path.isfile(path):
            continue
        file_sha, size = _sha256_file(path)
        files.append({"path": name.replace(os.sep, "/"), "local_path": path,
                      "kind": REMOTE_ARTIFACT_KIND, "sha256": file_sha, "size": size})
    # The knowledge document upstream's own writer produces: four keys, everything else under
    # `value`. `speedup` sits at the top because that is the ranking key the service reads — it
    # only honours a flat top-level `knowledge.<name>` scalar and rejects a nested path with a 400.
    knowledge = {
        "producer": remote_segment(producer, REMOTE_PRODUCER),
        "speedup": round(speedup, 4) if speedup else None,
        "identity": identity,
        "value": remote_value(meta, digest),
    }
    return [{
        "canonical_id": cid,
        "session_id": sid,
        "exp_dir": exp_dir,
        "rung": rung,
        "knowledge": knowledge,
        "files": files,
        # Upstream's own gate: a candidate is always recorded, the pointer moves only on a real win.
        # Evaluated per rung, since each address keeps its own champion pointer.
        "champion_eligible": speedup > 1.0,
        "champion": False,
    } for rung, cid in enumerate(cids)]


def remote_record(meta: dict, exp_dir: str, producer: str = REMOTE_PRODUCER, gpu: str = "",
                  version: str = "") -> dict:
    """The exact-rung record alone. Kept for callers that only want the address, never for writing —
    writing one rung and not the other is the failure mode remote_records() exists to prevent."""
    return remote_records(meta, exp_dir, producer, gpu, version)[0]


def _int_or(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _carrying_remote_state(store, rec: dict, local_reproduced: bool):
    """`rec` with everything the store's existing copy of this session has EARNED moved onto it:
    its attestation ledger, its reproduction count, and its retirement. Returns (rec, report).

    Every field `remote_value` sends is derived from the LOCAL meta.yaml, and the session id is a
    digest of the patch's content — path-independent by design, so the same patch measured again
    from a fresh `--root` lands on the same session and `mode="replace"` rewrites the document
    whole. Without this, that rewrite silently resets the record's ledger to zero, drops its
    reproduction count back to 1, and resurrects a tombstone `curate-remote` wrote. The new
    document looks perfectly well-formed, which is what makes the loss invisible.

    Called per plane and per rung, never once for all of them: the planes drift, and each ladder
    rung keeps its own copy of the session, so each has to be asked what it already holds.
    """
    from kb.attest import VALIDATED, attestations_of, carry_attestations, recent_verdicts
    from kb.retract import is_retired, retracted_document

    report = {"carried_attestations": False, "retracted": False, "unretired": False}
    try:
        previous = store.get_session(rec["canonical_id"], rec["session_id"])
    except Exception:
        previous = None                          # an unreadable copy is treated as no copy
    previous_value = previous.get("value") if isinstance(previous, dict) else None
    if not isinstance(previous_value, dict):
        return rec, report

    knowledge = dict(rec["knowledge"])
    fresh_value = dict(knowledge.get("value") or {})
    value = carry_attestations(previous_value, dict(fresh_value))
    # Reported as "this write would otherwise have lost a ledger", not merely "a ledger is present":
    # a local meta.yaml can carry its own counts, and those were never at risk.
    report["carried_attestations"] = value.get("attestations") != fresh_value.get("attestations")

    # Monotone, and it counts BOXES. The local number only ever knows what this tree has seen, so
    # a rewrite from a clean tree carries 1; the remote number is the cross-box total and must not
    # go backwards. When the remote already held this session but the local write filed a fresh
    # entry rather than a `duplicate_impl`, this is a rediscovery the remote had not heard about,
    # so it is worth one — otherwise the local write already counted it (see _record_reproduction).
    previous_reps = _int_or(previous_value.get("reproductions"), 0)
    reps = max(_int_or(value.get("reproductions"), 1),
               previous_reps + (0 if local_reproduced else 1))
    value["reproductions"] = reps
    # The same threshold _record_reproduction applies on disk, applied to the cross-box count.
    if reps >= 2:
        value["lifecycle"] = "active"
        value["validated"] = True
    report["reproductions"] = reps
    knowledge["value"] = value

    if not is_retired(previous_value):
        return dict(rec, knowledge=knowledge), report
    reason = str(previous_value.get("retired_reason") or "").strip() \
        or "retracted; no reason recorded"
    # A different bar to lift than e2e's, and it has to be. e2e un-retires when the incoming write
    # is itself `validated`, which for that lane means a hot A/B strictly stronger than the gate its
    # writes pass. This lane's write gate is only "beat 1.0x on the producer's own bench" — the very
    # claim the retraction distrusted — so lifting on the write would let any re-run undo curation,
    # which is the hazard being fixed here, just slower. Nor can it be the reproduction count: a
    # retraction is normally written onto a record that already has one, so "one more write" would
    # clear every tombstone on its next visit.
    # What DOES clear it is the one signal a retraction is built from the absence of: somebody read
    # this record, took it to a box, and reproduced the win. Same window `should_retire` reads, so
    # the two cannot disagree about whether the record is currently believed.
    if VALIDATED in recent_verdicts(attestations_of(value)):
        # Cleared, not merely annotated. The incoming document is derived from a meta.yaml that may
        # ITSELF say `retained: false` — the sync path re-files curated entries, so it usually does
        # — and a record carrying both a tombstone flag and an `unretired_at` is a record no reader
        # can act on: `is_retired` still hides it, while the lift claims it is believed again.
        for gone in ("retained", "retired_reason", "retracted_at", "retracted_by"):
            value.pop(gone, None)
        value.update({"unretired_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "unretired_from_reason": reason})
        # No attestation is counted for the reprieve, unlike e2e's. There, the write itself is the
        # new evidence and has to enter the ledger or the next sweep re-retracts on the unchanged
        # negatives. Here the evidence is a validated recall that is ALREADY in the window, so the
        # veto already holds — and counting a recall nobody performed would break the one invariant
        # the ledger has, that `recalls` is attempts on hardware.
        withdrawn = previous_value.get("withdrawn_scores")
        if isinstance(withdrawn, dict) and withdrawn:
            value["withdrawn_scores"] = withdrawn
        knowledge["value"] = value
        report["unretired"] = True
        return dict(rec, knowledge=knowledge), report
    # Still retracted. Re-applied through retracted_document rather than by copying the flag,
    # because a tombstone is not one field: retraction also zeroes the top-level ranking scalar,
    # and `retained: False` on a document still carrying a real speedup is inert against every
    # reader that ranks on the scalar (see kb/retract.py).
    # The scores it was ORIGINALLY taken back for are moved across first, so retracted_document's
    # setdefault leaves them alone: a reviewer judging the retraction needs the number that was
    # retracted, not whatever this rewrite happened to measure onto the tombstone.
    withdrawn = previous_value.get("withdrawn_scores")
    if isinstance(withdrawn, dict) and withdrawn:
        value["withdrawn_scores"] = withdrawn
    report["retracted"] = True
    tombstone = retracted_document(knowledge, reason, (CHAMPION_METRIC,),
                                   actor=str(previous_value.get("retracted_by") or ""))
    # `retracted_document` stamps the moment it runs, which is right the first time and wrong every
    # time after: this record was taken back once, and re-stamping it on each rewrite would date the
    # retraction to whenever somebody last touched the patch. The lane now re-files entries on every
    # warm start (`sync-local`), so an un-pinned timestamp would read as "curated seconds ago"
    # forever, which is exactly what an auditor would use to decide the retraction is still current.
    when = str(previous_value.get("retracted_at") or "").strip()
    if when:
        tombstone["value"]["retracted_at"] = when
    return dict(rec, knowledge=tombstone), report


def _carrying_plane(store, recs, local_reproduced: bool):
    """`recs` rebuilt against one plane, plus the merged report. See _carrying_remote_state."""
    out, merged = [], {"carried_attestations": False, "retracted": False, "unretired": False}
    for rec in recs:
        carried, report = _carrying_remote_state(store, rec, local_reproduced)
        out.append(carried)
        for key in ("carried_attestations", "retracted", "unretired"):
            merged[key] = merged[key] or report[key]
        if "reproductions" in report:
            merged["reproductions"] = max(merged.get("reproductions", 0), report["reproductions"])
    return out, merged


def _tree_entries(root: str, want_gfx: str, want_name: str, include_retired: bool):
    """Every publishable entry in the slug tree as (exp_dir, meta), plus what was passed over.

    Shared by the two readers that take the tree and speak canonical ids back — `export-remote`,
    which renders them for review, and `sync-local`, which files them straight into the store.
    The two must not drift about what counts as an entry: the second exists to make the tree
    redundant for READING, and it can only do that if it carries across everything the first
    would have offered.
    """
    scanned, skipped = 0, {"retired": 0, "no_patch": 0, "unreadable": 0, "filtered": 0}
    entries = []
    for dirpath, _dirs, files in sorted(os.walk(root)):
        if "meta.yaml" not in files:
            continue
        scanned += 1
        meta = _read_meta(os.path.join(dirpath, "meta.yaml"))
        if not isinstance(meta, dict):
            skipped["unreadable"] += 1
            continue
        gfx = _norm_gfx(meta.get("gfx") or (meta.get("metric") or {}).get("gpu_arch") or "")
        if (want_gfx and gfx != want_gfx) or (want_name and _match_key(meta.get("kernel_name")) != want_name):
            skipped["filtered"] += 1
            continue
        # Retired entries are dominated duplicates, not negative knowledge, and the service ranks on
        # the speedup we declare — offering them would put a retired win in someone's top-N.
        if _is_retired(meta) and not include_retired:
            skipped["retired"] += 1
            continue
        # An entry with nothing installable in it is not a knowledge record, whatever its speedup
        # says. Which file that is depends on the carrier: a diff for `patch`, the tuned tables for
        # `tuned_artifact` (whose patch.diff is optional and often absent).
        if str(meta.get("carrier") or "patch") == "patch":
            installable = os.path.isfile(os.path.join(dirpath, "patch.diff"))
        else:
            installable = any(os.path.isfile(os.path.join(dirpath, "artifact", n))
                              for n in (meta.get("artifact_files") or []))
        if not installable:
            skipped["no_patch"] += 1
            continue
        entries.append((dirpath, meta))
    return entries, scanned, skipped


def cmd_export_remote(a) -> dict:
    """Render this store as KB Store candidates, one JSON line each, champion pre-decided.

    Nothing is uploaded here — this only produces what to upload, so the mapping is reviewable and
    diffable before anything leaves the machine. kb/remote_upload.py consumes the output.
    """
    root = a.root
    if not os.path.isdir(root):
        return {"ok": False, "reason": "no_such_root: " + root}
    entries, scanned, skipped = _tree_entries(
        root, _norm_gfx(a.gfx) if a.gfx else "", _match_key(a.kernel_name) if a.kernel_name else "",
        bool(a.include_retired))
    records = []
    for dirpath, meta in entries:
        records.extend(remote_records(meta, dirpath, a.producer, a.gpu))

    # One champion per identity, upstream's rule: must beat 1.0x, and highest wins. Ties break on
    # session id so two runs of this exporter promote the same candidate.
    best = {}
    for rec in records:
        if not rec["champion_eligible"]:
            continue
        cur = best.get(rec["canonical_id"])
        key = (rec["knowledge"]["speedup"] or 0.0, rec["session_id"])
        if cur is None or key > cur[0]:
            best[rec["canonical_id"]] = (key, rec)
    for _key, rec in best.values():
        rec["champion"] = True

    # Byte-identical patches under one identity are ONE candidate upstream, so a collision here is
    # the dedup working. Which of them we send still matters: the record is written with
    # mode=replace, so keeping the lower of two measurements of the same patch would publish a
    # speedup we have already beaten. Highest wins, exp_dir breaks ties, and the dropped rows are
    # named in the summary rather than vanishing.
    by_id = {}
    dropped = []
    for rec in records:
        ident = (rec["canonical_id"], rec["session_id"])
        cur = by_id.get(ident)
        if cur is None:
            by_id[ident] = rec
            continue
        ranked = sorted((cur, rec),
                        key=lambda r: (-(r["knowledge"]["speedup"] or 0.0), r["exp_dir"]))
        by_id[ident] = ranked[0]
        dropped.append(ranked[1]["exp_dir"])

    emitted = 0
    out = open(a.out, "w") if a.out else None
    try:
        for rec in records:
            if by_id.get((rec["canonical_id"], rec["session_id"])) is not rec:
                continue
            line = json.dumps(rec, ensure_ascii=False)
            (out.write(line + "\n") if out else print(line))
            emitted += 1
    finally:
        if out:
            out.close()
    # `emitted` counts records, not measurements: each entry is published at every rung of its
    # ladder, so the honest headline is both numbers. `sessions` is how many distinct measurements
    # went out; emitted/sessions should equal the ladder depth for a healthy export.
    return {"ok": True, "scanned": scanned, "emitted": emitted,
            "sessions": len({r["session_id"] for r in records}),
            "identities": len({r["canonical_id"] for r in records}),
            "exact_identities": len({r["canonical_id"] for r in records if r["rung"] == 0}),
            "champions": len(best),
            "deduped": len(dropped), "deduped_dirs": sorted(dropped),
            "skipped": skipped, "out": a.out or "-"}


def cmd_sync_local(a) -> dict:
    """File the slug tree's entries into the on-disk KB store, keyed by canonical id.

    The convergence step. Writes on this lane have gone to both address schemes for a while
    (`write-remote --plane both` files the tree, then the store, then the service), so the store
    already holds everything measured since that became the default. What it has never held is the
    BACKLOG: every entry older than that, plus anything a curation pass imported into the tree by
    hand. Until that is carried across, moving the READ onto the store would be a silent downgrade
    rather than a change of address — same kernel, thinner page, and no way for the reader to tell.

    Safe to run before every read, which is how the lane runs it. The session id is a digest of the
    patch's content, so an entry that is already filed lands back on its own session, and the write
    goes through the same `_carrying_remote_state` a re-measurement does: the store's attestation
    ledger, its retraction and its cross-box reproduction count all survive being re-synced.
    `local_reproduced` is TRUE here for that same reason — a sync is a copy of a measurement that
    was already counted, not a new box reporting one, and a counter that ticked once per read would
    make the tree's oldest entries look like its most reproduced.

    Retirement travels, and has to be APPLIED rather than copied. `remote_value` already carries
    `retained: false` across, but a flag alone is inert: ranking reads the top-level scalar, so a
    tombstone that arrived still declaring 1.9x would sit at the head of the page for every reader
    that has not been taught the flag (kb/retract.py). `retracted_document` is what makes the three
    parts — flag, zeroed scalar, no champion pointer — travel together.

    The one thing it will not do is overrule a LIFT. A record the store retired and then un-retired
    (`unretired_at`, written when a validated recall reproduced the win after the retraction) keeps
    its reprieve: the tree entry that was curated away is the very record the recall vindicated, and
    re-imposing that verdict on every read would silently undo the lift once per warm start. An
    already-retracted record is likewise left to `_carrying_plane`, which re-applies the store's own
    tombstone — going through `retracted_document` again would only churn `retracted_at`.
    """
    from kb.retract import is_retired, retracted_document
    root = a.root
    if not os.path.isdir(root):
        return {"ok": False, "reason": "no_such_root: " + root}
    a.plane = "local"                       # a sync is a local catch-up; it never talks to anyone
    store, _also, why = open_plane(a, CHAMPION_METRIC, 1.0, create=True)
    if store is None:
        return {"ok": False, "reason": why or "no_store"}
    entries, scanned, skipped = _tree_entries(
        root, _norm_gfx(a.gfx) if a.gfx else "", _match_key(a.kernel_name) if a.kernel_name else "",
        include_retired=True)

    synced, fresh, retired, promoted, errors = [], 0, 0, [], []
    for exp_dir, meta in entries:
        # One bad entry must not stop the catch-up. The read that follows is better off with the
        # rest of the backlog than with none of it, and an entry that cannot be filed is named here
        # rather than leaving a page thin for a reason nothing recorded. Broad on purpose: hashing
        # the payload, deriving the identity and writing it all fail differently, and none of them
        # is worth losing the other entries over.
        try:
            recs = remote_records(meta, exp_dir, a.producer, a.gpu)
            files = {f["path"]: f["local_path"] for f in recs[0]["files"]}
            previous = store.get_session(recs[0]["canonical_id"], recs[0]["session_id"])
            held = previous.get("value") if isinstance(previous, dict) else None
            held = held if isinstance(held, dict) else {}
            already = isinstance(previous, dict)
            plane_recs, _carried = _carrying_plane(store, recs, local_reproduced=True)
            # See the docstring: apply the tree's curation, unless the store has already ruled on
            # this session — either by holding its own tombstone, or by lifting one.
            retire_now = (_is_retired(meta) and not is_retired(held)
                          and not held.get("unretired_at"))
            if retire_now:
                reason = (str(meta.get("retired_reason") or "").strip()
                          or "retired by the local curation")
                plane_recs = [dict(r, knowledge=retracted_document(
                    r["knowledge"], reason, (CHAMPION_METRIC,),
                    actor=str(meta.get("retracted_by") or ""))) for r in plane_recs]
            elif _is_retired(meta) and not is_retired(held):
                # The lift again, from the other side. `remote_value` copies `retained: false` out
                # of the meta verbatim, so a document derived from a curated tree entry arrives
                # carrying the flag even on the path that decided not to retract — and the flag
                # alone is enough for `is_retired` to hide the record the recall just vindicated.
                # Dropping it here is what makes "the store's own state wins" true of the WHOLE
                # document and not only of the ranking scalar.
                plane_recs = [dict(r, knowledge=dict(r["knowledge"], value={
                    k: v for k, v in (r["knowledge"].get("value") or {}).items()
                    if k not in ("retained", "retired_reason")})) for r in plane_recs]
            written, moved, error = publish(store, plane_recs, files,
                                            lambda rec: rec["knowledge"].get("speedup"),
                                            promote=not retire_now)
        except Exception as e:
            errors.append({"exp_dir": exp_dir, "reason": "%s: %s" % (type(e).__name__, str(e)[:160])})
            continue
        if error:
            errors.append({"exp_dir": exp_dir, "reason": error})
            continue
        retired += 1 if retire_now else 0
        fresh += 0 if already else 1
        synced.extend(written)
        promoted.extend(moved)
    return {"ok": not errors, "store": store.root, "scanned": scanned, "entries": len(entries),
            # `entries` counts measurements, `synced` the addresses they were filed at — one per
            # ladder rung, so the ratio is the ladder depth for a healthy sync.
            "synced": len(synced), "identities": len(set(synced)),
            "new_sessions": fresh, "retired": retired,
            "champions": len(set(promoted)), "skipped": skipped,
            "errors": errors}


def _value_as_meta(value: dict, gfx: str) -> dict:
    """Read a record's `value` back as a meta.

    `remote_value()` produced it FROM a meta, minus what the identity already carries, so the
    prose helpers below (`_techniques_md`, `_alternates_md`, `_prose_body`) work on it unchanged
    and a remote-sourced reference reads exactly like a local one.
    """
    meta = dict(value or {})
    metric = dict(meta.get("metric") or {})
    metric.setdefault("gpu_arch", gfx)
    meta["metric"] = metric
    return meta


def _store_ladder(a, gfx: str):
    """The addresses to try, most specific first, each paired with the tier it represents.

    framework_version is the one dimension a reader can get wrong without noticing: the store is
    keyed on the ROCm an entry was measured on, this box may be on another, and a bare miss looks
    exactly like a cold start. The ladder is the answer — but only because the WRITER publishes the
    version-agnostic rung too. Nothing here derives a page that was never written; each rung is a
    real address that a `write-remote` on this box would also have filled.

    An explicit --canonical-id is taken as given and gets no ladder. A caller that names an address
    is usually auditing one page, and silently widening the read would misreport which page
    answered.
    """
    if a.canonical_id:
        return [(a.canonical_id, "exact")]
    meta = {"kernel_name": a.kernel_name, "language": a.language,
            "verified_stack": detect_stack(a.language)}
    identity = remote_identity(meta, a.producer, remote_gpu(gfx, getattr(a, "gpu", "")),
                               getattr(a, "framework_version", ""))
    return list(zip(remote_canonical_ids(identity), ("exact", "any_version")))


def _legacy_name_ladder(a, gfx: str):
    """The rungs a pre-canon_name writer used, READ ONLY: `<name>_task`, from a head extraction.

    Those records cannot be moved — no delete, no search — so the spelling they were written with is
    the only way back to them, and without this rung the write-side fix would strand them for good.
    Tried after the whole canonical ladder, so a legacy page can rescue but never shadow. NOT in
    `_store_ladder`: retract-remote and attest-remote WRITE to every rung it returns.
    """
    if a.canonical_id:
        return []
    meta = {"kernel_name": a.kernel_name, "language": a.language,
            "verified_stack": detect_stack(a.language)}
    identity = remote_identity(meta, a.producer, remote_gpu(gfx, getattr(a, "gpu", "")),
                               getattr(a, "framework_version", ""))
    identity["kernel_name"] += "_task"      # the fold is idempotent, so appending re-spells it
    return list(zip(remote_canonical_ids(identity), ("legacy_name", "legacy_name_any_version")))


def cmd_retract_remote(a) -> dict:
    """Take back a key-addressed kernel record. The counterpart to `write-remote`.

    The service has no delete, so this rewrites the session in place: `retained: false`, a reason,
    the ranking scalar zeroed, and the identity's champion re-pointed at the best survivor. See
    kb/retract.py for why all three are needed and why any two of them is worse than none.

    Both rungs are visited, because `write-remote` filled both with the SAME session id. Retracting
    only the exact rung leaves the record live on the version-agnostic page, which is the page a box
    on a different ROCm reads — i.e. it would survive exactly where it is least verifiable.

    `--canonical-id` addresses one page only, matching `resolve-remote`'s rule: a caller that names
    an address is auditing it, and quietly widening a WRITE beyond what was asked for is not a
    behaviour this command should have.
    """
    from kb.retract import retract_session, retraction_ok
    gfx = _norm_gfx(a.gfx)
    if not gfx and not a.canonical_id:
        return {"retracted": False, "reason": "missing_arch"}
    store, mirror, why = open_plane(a, CHAMPION_METRIC, 1.0)
    planes = [p for p in (store, mirror) if p is not None]
    if not planes:
        return {"retracted": False, "reason": why}
    out = {"applied": bool(a.apply), "session_id": a.session_id, "reason": a.reason,
           "plane_note": why, "pages": []}
    for cid, tier in _store_ladder(a, gfx):
        for plane in planes:
            report = retract_session(plane, cid, a.session_id, a.reason, CHAMPION_METRIC,
                                     actor=str(getattr(a, "measured_by", "") or ""),
                                     scan=int(a.scan), apply=bool(a.apply))
            out["pages"].append(dict(report, tier=tier))
    out["retracted"] = retraction_ok(out["pages"], a.apply)
    return out


def cmd_curate_remote(a) -> dict:
    """Scan a kernel page, apply the shared retire policy, retract what has earned it.

    The kernel-lane counterpart of `e2e_store.py curate`, and deliberately the same shape: the
    judgement comes from `kb.attest.should_retire` so the two lanes cannot drift into different
    bars, and the execution is `retract-remote`'s, on every rung, because one session id addresses
    both. DRY RUN BY DEFAULT — this acts on a whole page, and the store has no delete.

    Judges from the FINEST rung only. The version-agnostic page holds records from other stack
    versions, and retiring one of those from this box's page would convict it of a ledger it does
    not own.
    """
    from kb.attest import RETIRE_THRESHOLD, attestations_of, retire_hint, should_retire
    from kb.retract import is_retired, retract_session, retraction_ok
    threshold = max(1, int(getattr(a, "threshold", RETIRE_THRESHOLD) or RETIRE_THRESHOLD))
    out = {"applied": bool(a.apply), "threshold": threshold, "scanned": 0, "already_retired": 0,
           "kept": 0, "candidates": [], "pages": [], "ok": True, "error": ""}
    gfx = _norm_gfx(a.gfx)
    if not gfx and not a.canonical_id:
        return dict(out, ok=False, error="missing_arch")
    ladder = _store_ladder(a, gfx)
    store, mirror, why = open_plane(a, CHAMPION_METRIC, 1.0)
    planes = [p for p in (store, mirror) if p is not None]
    if not planes:
        return dict(out, ok=False, error=why)
    try:
        found = planes[0].candidates(ladder[0][0], limit=max(1, int(a.scan)))
    except Exception as e:
        return dict(out, ok=False,
                    error="read_failed: %s: %s" % (type(e).__name__, str(e)[:160]))
    out["scanned"] = len(found)
    for candidate in found:
        value = candidate.value if isinstance(candidate.value, dict) else {}
        if is_retired(value):
            out["already_retired"] += 1
            continue
        reason = should_retire(value, threshold=threshold)
        if not reason:
            out["kept"] += 1
            continue
        out["candidates"].append({
            "session_id": candidate.session_id, "reason": reason,
            "direction": str(value.get("direction") or ""),
            "carrier": str(value.get("carrier") or "patch"),
            "speedup": candidate.speedup or 0.0,
            "is_champion": bool(candidate.is_champion),
            "retire_hint": retire_hint(value),
            "attestations": attestations_of(value)})
    for entry in out["candidates"]:
        for cid, tier in ladder:
            for plane in planes:
                report = retract_session(plane, cid, entry["session_id"], entry["reason"],
                                         CHAMPION_METRIC,
                                         actor=str(getattr(a, "measured_by", "")
                                                   or "experience_store curate"),
                                         scan=int(a.scan), apply=bool(a.apply))
                out["pages"].append(dict(report, tier=tier))
    out["ok"] = retraction_ok(out["pages"], a.apply) if out["candidates"] else True
    return out


def _attest_evidence(a) -> dict:
    """The one-line record of what this box saw, shared by both attest paths."""
    evidence = {}
    for key, raw in (("measured_speedup", getattr(a, "measured_speedup", None)),
                     ("note", getattr(a, "note", "")),
                     ("canonical_id", getattr(a, "canonical_id", ""))):
        if raw not in (None, ""):
            evidence[key] = raw
    # Argparse hands the ratio over as a string. Stored as one it would land in meta.yaml quoted,
    # and every later reader comparing it against a speedup would be comparing str to float.
    if "measured_speedup" in evidence:
        try:
            evidence["measured_speedup"] = float(evidence["measured_speedup"])
        except (TypeError, ValueError):
            evidence.pop("measured_speedup")
    return evidence


def cmd_attest(a) -> dict:
    """Count one attempt to actually USE a stored entry, straight into its meta.yaml.

    The local plane has no session ids and no service — an entry IS a directory — so this is a
    read-modify-atomic-write of the same file the write path owns, using the same arithmetic
    kb/attest.py applies remotely. Sharing the arithmetic and not the transport is deliberate: the
    counters have to mean the same thing on both planes or a curation pass cannot compare them,
    but a local store should not need a KB plane to record that a patch did not apply.

    Like the remote one, this moves nothing: the speedup meta declares is left exactly as it was,
    and the entry keeps its rank. A patch that failed to apply on one workspace is a fact about
    that workspace as much as about the patch.

    The KEYED record is the truth. `meta.yaml` lives in a checkout that is created empty and
    deleted with the run, so a ledger kept only there is invisible to every other box. Pass
    `--session-id` (as `resolve-remote` reports it) and this records the same verdict on both, and
    says which planes took it. Without a reachable plane the count still lands on disk and
    `ledger_scope` reads `local_only` — a real degradation, named rather than hidden, because a
    curation pass must not read an uncorroborated local counter as agreement between boxes.
    """
    from kb.attest import record_attestation, retire_hint
    exp_dir = str(getattr(a, "exp_dir", "") or "")
    meta_path = os.path.join(exp_dir, "meta.yaml")
    meta = _read_meta(meta_path)
    session_id = str(getattr(a, "session_id", "") or "")
    # A keyed candidate is served out of a hydrated refs dir with no meta.yaml of its own. That is
    # a missing MIRROR, not a missing record, so it must not swallow the verdict.
    if not meta and not session_id:
        return {"attested": False, "reason": "no_meta", "exp_dir": exp_dir}
    out = {"attested": bool(a.apply), "applied": bool(a.apply), "exp_dir": exp_dir,
           "outcome": a.outcome, "ledger_scope": "local_only"}
    if meta:
        try:
            updated = record_attestation(dict(meta), a.outcome,
                                         actor=str(getattr(a, "measured_by", "") or ""),
                                         evidence=_attest_evidence(a))
        except Exception as e:
            return {"attested": False, "reason": "bad_outcome: " + str(e)[:120], "exp_dir": exp_dir}
        out["attestations"] = updated["attestations"]
        out["retire_hint"] = retire_hint(updated)
        if a.apply:
            try:
                _atomic_write(meta_path, _dump_meta(updated))
            except OSError as e:
                out.update({"attested": False, "reason": "write_failed: " + str(e)[:120]})
    else:
        out["ledger_scope"] = "keyed_only"
    if not session_id:
        return out
    keyed = cmd_attest_remote(a)
    out["keyed"] = keyed
    if keyed.get("attested"):
        out.update({"ledger_scope": "keyed", "retire_hint": keyed.get("retire_hint") or ""})
    else:
        out["reason"] = "keyed_ledger_missed: " + str(keyed.get("reason") or "")[:120]
    return out


def cmd_attest_remote(a) -> dict:
    """The same count, against a key-addressed record on either plane.

    Walks BOTH rungs for the same reason `retract-remote` does: `write-remote` filled them with one
    session id, and a box on a different ROCm reads the version-agnostic rung — leaving it with a
    stale ledger hides the failures from exactly the readers most likely to hit them.
    """
    from kb.attest import attest_session, attestation_ok, retire_hint
    gfx = _norm_gfx(a.gfx)
    if not gfx and not a.canonical_id:
        return {"attested": False, "reason": "missing_arch"}
    store, mirror, why = open_plane(a, CHAMPION_METRIC, 1.0)
    planes = [p for p in (store, mirror) if p is not None]
    if not planes:
        return {"attested": False, "reason": why}
    out = {"applied": bool(a.apply), "session_id": a.session_id, "outcome": a.outcome,
           "plane_note": why, "pages": []}
    for cid, tier in _store_ladder(a, gfx):
        for plane in planes:
            report = attest_session(plane, cid, a.session_id, a.outcome,
                                    actor=str(getattr(a, "measured_by", "") or ""),
                                    evidence=_attest_evidence(a), apply=bool(a.apply))
            out["pages"].append(dict(report, tier=tier))
    out["attested"] = attestation_ok(out["pages"], a.apply)
    hints = [p.get("retire_hint") for p in out["pages"] if p.get("retire_hint")]
    out["retire_hint"] = hints[0] if hints else ""
    return out


def _store_near_misses(store, cid: str):
    """Identities differing from `cid` only in framework_version, newest-looking last.

    A third tier below the ladder, and only reachable on a store that predates double-writing —
    once every write fills the version-agnostic rung, that rung answers first and this never runs.
    Kept because the alternative for such a store is a cold start on a kernel that has history.
    """
    parts = cid.split(":")
    if len(parts) != 7:
        return []
    return sorted(other for other in store.identities()
                  if (lambda s: len(s) == 7 and s[:6] == parts[:6] and s[6] != parts[6])
                  (other.split(":")))


def cmd_resolve_remote(a) -> dict:
    """Rank the top-N candidates under one canonical id and mirror their prose, like `resolve`.

    Same output shape as `resolve` on purpose: the lane's schema, verify gate and adopt step do not
    change when the KB moves behind a key. What differs is where curation happens. The local store
    is curated on disk (`retained: false`, one entry per direction); the KB Store ranks on nothing
    but the `speedup` a producer declared, so the direction collapse and the bench-key comparability
    check have to be redone here, client-side, against the records it hands back.
    """
    gfx = _norm_gfx(a.gfx)
    if not gfx and not a.canonical_id:
        return {"read_reason": "missing_arch", "candidates": []}
    # Reading takes ONE plane at a time, never two merged: that would need a comparability rule
    # across planes that nothing here has. `both` therefore picks, and picks the service first —
    # `open_plane` would have handed back the local store, letting a stale mirror shadow the shared
    # one without saying so. See kb/plane.py:read_planes.
    planes, why = read_planes(a, CHAMPION_METRIC)
    if not planes:
        return {"read_reason": why.split(":", 1)[0], "reason": why, "candidates": []}
    store = planes[0][0]

    ladder = _store_ladder(a, gfx) + _legacy_name_ladder(a, gfx)
    cid, match_tier = ladder[0]
    segs = cid.split(":")
    requested_slug = make_slug(a.kernel_name or (segs[3] if len(segs) > 3 else ""),
                               a.language or (segs[4] if len(segs) > 4 else ""), gfx)
    base_out = {"slug": requested_slug, "requested_slug": requested_slug, "canonical_id": cid,
                "match_tier": match_tier, "tried": [c for c, _t in ladder],
                "other_language_pages": [], "ambiguous_pages": [], "candidates": [],
                # Which plane ANSWERED, which `--plane both` alone does not tell you.
                "read_plane": ""}

    # Descend the ladder, then the pre-ladder near misses. Stopping at the first rung that holds
    # anything is deliberate: a coarser page is a superset only if every writer double-wrote, and
    # `tried` records the descent so a thin answer can be told apart from a lucky one.
    # Retracted records are dropped as the page is read, not after the rung is chosen. The local
    # `resolve` has filtered on `_is_retired` since it existed; this path did not, and reported a
    # hardcoded `"retired": 0` while serving them — so a record someone had explicitly taken back
    # came straight back out of the service. Filtering here rather than below also means a rung
    # whose every entry has been retracted correctly reads as EMPTY and the ladder descends, instead
    # of stopping on a page that turns out to have nothing to offer.
    # Carrier and precision are filtered here for the same reason as retraction: a rung holding
    # nothing this caller can install must read as EMPTY so the ladder descends, rather than stopping
    # on a page with nothing to offer. Records written before carriers existed have no field and are
    # diffs; precision filters only when asked, and an entry that states none is kept — the whole
    # backlog predates the field.
    # `--include-retired` keeps them instead, for an audit: the local `resolve` has had that escape
    # hatch since it existed, and without it here there is no way to ask the service what curation
    # took back. It also changes the DESCENT — a rung of nothing but tombstones now stops the ladder
    # rather than reading empty — which is exactly what an auditor wants and exactly why it is a
    # flag and not the default.
    want_carrier = str(getattr(a, "carrier", "") or "patch")
    want_precision = _norm_precision(getattr(a, "precision", ""))
    include_retired = bool(getattr(a, "include_retired", False))
    other_carrier = [0]
    other_precision = [0]
    # What the page handed back before any of these filters ran, which is not the same as what the
    # page HOLDS: the service ignores the `limit` argument and pages `--scan` rows (kb/store_remote
    # .py:candidates), so a busy identity can be read through a keyhole with nothing saying so.
    scanned = [0]

    def live(canonical_id):
        rows = store.candidates(canonical_id, limit=0)
        scanned[0] = len(rows)
        retired_n = sum(1 for c in rows if _is_retired(c.value))
        kept = rows if include_retired else [c for c in rows if not _is_retired(c.value)]
        of_carrier = [c for c in kept if str((c.value or {}).get("carrier") or "patch") == want_carrier]
        other_carrier[0] = len(kept) - len(of_carrier)
        if not want_precision:
            return of_carrier, retired_n
        of_precision = [c for c in of_carrier
                        if _precision_matches(want_precision, _precision_of(c.value))]
        other_precision[0] = len(of_carrier) - len(of_precision)
        return of_precision, retired_n

    # The WHOLE descent is redone on the next plane — ladder, then near misses — rather than
    # stopping at the first page either plane happens to hold: a thin remote page must not shadow
    # the hand-curated local tree, and a coarse rung on one plane must not shadow an exact rung on
    # the other. `live` reads `store` from this scope, so rebinding it here re-points the closure.
    found, retired, read_plane = [], 0, ""
    for store, read_plane in planes:
        for cid, match_tier in ladder:
            found, retired = live(cid)
            if found:
                break
        if found:
            break
        near = _store_near_misses(store, ladder[0][0])
        for other in near:
            found, retired = live(other)
            if found:
                cid, match_tier = other, "other_version"
                break
        base_out.update({"other_language_pages": near,
                         "tried": [c for c, _t in ladder] + near})
        if found:
            break
    if not found:
        return dict(base_out, read_reason="kernel_page_not_found")
    base_out.update({"canonical_id": cid, "match_tier": match_tier, "read_plane": read_plane})

    try:
        min_speedup = float(a.min_speedup)
    except (TypeError, ValueError):
        min_speedup = 1.0
    above = [c for c in found if (c.speedup or 0.0) >= min_speedup]
    # `total` counts what the page held, `retired` how many of those were taken back — so the two
    # still sum to the page size even though `found` is already the survivors. Under
    # `--include-retired` the tombstones ARE in `found`, so adding them again would double-count;
    # `retired` still reports them, it just no longer names a set that was removed.
    # `scan_saturated` is the honest caveat on all of the above: it says the numbers describe as
    # much of the page as was fetched. Only the service truncates — LocalKBStore.candidates with
    # limit=0 returns everything — so the local plane reports no limit rather than a false one.
    scan_limit = max(1, int(getattr(a, "scan", 25) or 25)) if read_plane == "remote" else 0
    stats = {"total": len(found) + (0 if include_retired else retired), "retired": retired,
             "include_retired": include_retired,
             "below_min_speedup": len(found) - len(above), "min_speedup": min_speedup,
             "carrier": want_carrier, "other_carriers": other_carrier[0],
             "precision": want_precision, "other_precisions": other_precision[0],
             "scanned": scanned[0], "scan_limit": scan_limit,
             "scan_saturated": bool(scan_limit and scanned[0] >= scan_limit)}
    if not above:
        return dict(base_out, filtered=stats, read_reason="below_min_speedup")

    # `above` is already speedup-ordered by the store.
    ordered = demote_hinted(above, lambda c: _retire_hint_of(c.value))
    stats["demoted_by_hint"] = sum(1 for c in ordered if _retire_hint_of(c.value))
    top, alternates, collapsed = collapse_by_direction(
        ordered, lambda c: c.value.get("direction"), lambda c: c.session_id, a.top_n)
    stats["same_direction_collapsed"] = collapsed

    cache_dir = a.cache_dir or os.path.join(os.path.dirname(os.path.abspath(a.refs_dir)), "kb_cache")
    views = []
    for c, alt_of in zip(top, alternates):
        meta = _value_as_meta(c.value, gfx)
        metric = meta.get("metric") or {}
        # Only now do artifact bytes move: the ranking above read knowledge documents alone.
        bundle = store.materialize(cid, c, cache_dir)
        views.append({
            "key": c.session_id,
            "meta": meta,
            "exp_dir": bundle,
            "patch_path": os.path.join(bundle, "files", "patch.diff"),
            "report_path": os.path.join(bundle, "files", "report.md"),
            "carrier": str(meta.get("carrier") or "patch"),
            # materialize() lays the record's files out under `files/`, mirroring the paths
            # remote_records() uploaded them under — so `artifact/<stored name>` round-trips.
            "artifact_paths": [os.path.join(bundle, "files", "artifact", n)
                               for n in (meta.get("artifact_files") or [])],
            "artifact_names": dict(meta.get("artifact_names") or {}),
            "apply_env": str(meta.get("apply_env") or ""),
            "cache_invalidation": str(meta.get("cache_invalidation") or ""),
            "speedup": c.speedup or 0.0,
            "direction": str(meta.get("direction") or ""),
            "bench_key": str(metric.get("bench_key") or ""),
            "metric_kind": str(metric.get("metric_kind") or ""),
            "origin": f"- session: {c.session_id}{' (champion)' if c.is_champion else ''}\n",
            # Alternates are materialized too. They are same-direction runners-up, so there are few
            # of them, and a candidate listed with a path that resolves to nothing is worse than not
            # listing it: the next reader cannot tell a missing file from a broken export.
            "alts": [{
                "session_id": alt.session_id,
                "patch_path": os.path.join(store.materialize(cid, alt, cache_dir),
                                           "files", "patch.diff"),
                "speedup": round(alt.speedup or 0.0, 4),
                "bench_key": str((alt.value.get("metric") or {}).get("bench_key") or ""),
                "techniques": _techniques(alt.value),
            } for alt in alt_of],
            "extra": {"slug": requested_slug, "canonical_id": cid,
                      "session_id": c.session_id, "is_champion": c.is_champion},
        })

    summary = (
        f"{len(top)} direction(s) offered from {stats['total']} recorded candidate(s): "
        f"{stats['below_min_speedup']} below {min_speedup:g}x, "
        f"{collapsed} same-direction re-discoveries moved to `alternates`."
        + ({"any_version": f" Served from `{cid}` — the version-agnostic page; nothing was"
                           " recorded under this box's own ROCm.",
            "other_version": f" Served from `{cid}` — a DIFFERENT stack version, and not even the"
                             " version-agnostic page had it."}.get(match_tier, "")))
    prose = _render_references(a.refs_dir, f"`{cid}`", summary, views)
    candidates = [_candidate(rank, v, gfx, p, views[0]["bench_key"])
                  for rank, (v, p) in enumerate(zip(views, prose), start=1)]
    return dict(base_out, read_reason="read", candidates=candidates, filtered=stats)


def cmd_write_remote(a) -> dict:
    """Store one measured win in BOTH planes, under the same gates as `write`.

    The local entry stays the source of truth — curation, reproductions and dead ends all live
    there — and the KB record is derived from it, so the two cannot drift into disagreeing about
    what was measured. What lands under the key depends on the patch, not on the caller:

      * a patch the identity has not seen APPENDS a session, because the session id is a digest of
        the patch; the champion pointer then moves only if it beat 1.0x and the incumbent.
      * the same patch measured again REPLACES that one session in place. It is a reproduction,
        not a second candidate, which is exactly what the local plane already calls it.
    """
    local = cmd_write(a)
    store, also, why = open_plane(a, CHAMPION_METRIC, 1.0, create=True)
    if store is None:
        return dict(local, remote={"written": False, "reason": why})

    exp_dir = local.get("dir") or local.get("reproduced") or ""
    meta = _read_meta(os.path.join(exp_dir, "meta.yaml")) if exp_dir else None
    if not isinstance(meta, dict):
        # No local entry means a gate rejected it (no_improvement / empty_diff / duplicate with an
        # unreadable target). Nothing measured, nothing to publish.
        return dict(local, remote={"written": False,
                                   "reason": local.get("reason") or "no_local_entry"})

    recs = remote_records(meta, exp_dir, a.producer,
                          remote_gpu(_norm_gfx(a.gfx), getattr(a, "gpu", "")),
                          getattr(a, "framework_version", ""))
    files = {f["path"]: f["local_path"] for f in recs[0]["files"]}
    # Asked BEFORE the write: a session that already exists is this same patch measured again, and
    # the caller deserves to know its result replaced one rather than adding one.
    replaced = store.get_session(recs[0]["canonical_id"], recs[0]["session_id"]) is not None
    # `duplicate_impl` means the LOCAL tree already held this patch, so cmd_write already counted
    # this measurement as a reproduction; a fresh local entry has not been counted anywhere yet.
    local_reproduced = str(local.get("reason") or "") == "duplicate_impl"
    # Each plane is asked separately what it already holds, because the two drift — the mirror can
    # carry a ledger, or a retraction, that the primary has never seen.
    plane_recs, carried = _carrying_plane(store, recs, local_reproduced)
    written, promoted, error = publish(store, plane_recs, files,
                                       lambda rec: rec["knowledge"].get("speedup"))
    if error:                                    # a KB write must not fail a measured result
        return dict(local, remote={"written": False, "partial": written, "reason": error})
    out = {
        "written": True, "canonical_id": recs[0]["canonical_id"],
        "canonical_ids": written, "session_id": recs[0]["session_id"],
        "speedup": recs[0]["knowledge"].get("speedup"), "champion": bool(promoted),
        "champion_of": promoted, "files": sorted(files), "store": store.root,
        # true = this measurement landed on a session that already existed, i.e. the same patch.
        "replaced": replaced,
        # What the existing copy had earned and this write kept rather than overwrote. `retracted`
        # is the one to read: it says the result landed on a tombstone and stayed a tombstone.
        **carried,
    }
    if also is not None:
        # The second plane never gates the first. It reports its own outcome so an unreachable
        # service is visible as a failed mirror rather than as a silent one.
        mirror_recs, mirror_carried = _carrying_plane(also, recs, local_reproduced)
        mirrored, mirror_promoted, mirror_error = publish(
            also, mirror_recs, files, lambda rec: rec["knowledge"].get("speedup"))
        out["mirror"] = {"written": not mirror_error, "store": also.root,
                         "canonical_ids": mirrored, "champion_of": mirror_promoted,
                         "reason": mirror_error or "", **mirror_carried}
    elif why:
        out["mirror"] = {"written": False, "reason": why}
    return dict(local, remote=out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_write_args(w):
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
        # Curation inputs. Without --direction an entry can never be grouped with its own
        # re-discoveries; without the bench fields its speedup compares to nothing.
        w.add_argument("--direction", default="")
        w.add_argument("--metric-kind", dest="metric_kind", default="")
        w.add_argument("--case-names", dest="case_names", default="")
        w.add_argument("--parent", default="",
                       help="exp_dir of the warm-start entry this win was built on")
        # The tuning carrier. Default `patch` keeps every existing caller byte-identical.
        w.add_argument("--carrier", choices=CARRIERS, default="patch",
                       help="what this entry ships: a diff (default) or tuned config tables")
        w.add_argument("--artifact", action="append", default=[],
                       help="carrier=tuned_artifact: a tuned table to store (repeatable)")
        w.add_argument("--apply-env", dest="apply_env", default="",
                       help="KEY=VAL ... the artifact needs to bind; without it the table is inert")
        w.add_argument("--cache-invalidation", dest="cache_invalidation", default="",
                       help="what must run post-install or the new rows are silently ignored")
        w.add_argument("--tuner", default="", help="which tuner produced it (gradlib, ckProfiler, ...)")
        # Serving context: recorded in `upstream`, never in the address. Optional everywhere, so a
        # caller that does not know its precision writes exactly what it wrote before.
        w.add_argument("--precision", default="",
                       help="numeric precision this was measured at (fp8, fp8_w8a8, bf16, ...); "
                            "recorded for filtering, NOT part of the key")
        w.add_argument("--serving-framework", dest="serving_framework", default="",
                       help="vllm | sglang — recorded alongside precision, never keyed")
        w.add_argument("--serving-framework-version", dest="serving_framework_version", default="",
                       help="the SERVING framework's version, not ROCm's (--framework-version)")
        return w

    add_write_args(sub.add_parser("write", help="store one measured win"))

    r = sub.add_parser("resolve", help="enumerate + rank top-N solutions for a slug")
    r.add_argument("--root", required=True)
    r.add_argument("--kernel-name", dest="kernel_name", required=True)
    r.add_argument("--language", required=True)
    r.add_argument("--gfx", required=True)
    r.add_argument("--top-n", dest="top_n", type=int, default=3, help="max DIRECTIONS to offer")
    r.add_argument("--refs-dir", dest="refs_dir", required=True)
    r.add_argument("--match", choices=("exact", "normalized", "fuzzy"), default="fuzzy",
                   help="how hard to try to map the caller's kernel name onto a page (default fuzzy)")
    r.add_argument("--min-speedup", dest="min_speedup", type=float, default=1.05,
                   help="never spend an on-box verify on a recorded win below this (default 1.05)")
    r.add_argument("--include-retired", dest="include_retired", action="store_true",
                   help="also offer entries the curation retired (audit/debug only)")
    r.add_argument("--carrier", choices=CARRIERS, default="patch",
                   help="which carrier to offer; one per call (default patch)")
    r.add_argument("--precision", default="",
                   help="only offer entries measured at this precision; entries that state none "
                        "are always offered. Omit to filter on nothing (the default)")

    lg = sub.add_parser("languages", help="which languages this kernel has a page in")
    lg.add_argument("--root", required=True)
    lg.add_argument("--kernel-name", dest="kernel_name", required=True)
    lg.add_argument("--gfx", required=True)

    bf = sub.add_parser("backfill-content", help="bring imported entries up to the current shape")
    bf.add_argument("--root", required=True)
    bf.add_argument("--apply", action="store_true", help="write; without it, only report the diff")

    xr = sub.add_parser("export-remote", help="render entries as KB Store candidates (JSON lines)")
    xr.add_argument("--root", required=True)
    xr.add_argument("--gfx", default="", help="only this arch (default: every arch in the store)")
    xr.add_argument("--kernel-name", dest="kernel_name", default="", help="only this kernel")
    xr.add_argument("--producer", default=REMOTE_PRODUCER,
                    help="the system that owns this candidate stream and its champion pointer")
    xr.add_argument("--gpu", default="", help="override the gfx dimension; default is the entry's own gfx")
    xr.add_argument("--include-retired", dest="include_retired", action="store_true",
                    help="also export entries the curation retired (they would rank as live wins)")
    xr.add_argument("--out", default="", help="write JSON lines here instead of stdout")

    # No `--plane`: a sync is a local catch-up by definition, and offering `remote` here would
    # invite pushing a whole tree at a service that cannot delete what it accepts.
    sl = sub.add_parser(
        "sync-local", help="file the slug tree's entries into the on-disk KB store, keyed by "
                           "canonical id; idempotent, keeps what the store already earned")
    sl.set_defaults(plane="local", scan=25)
    sl.add_argument("--root", required=True, help="the slug tree to read")
    sl.add_argument("--store", required=True, help="on-disk KB store root to bring level")
    sl.add_argument("--gfx", default="", help="only this arch (default: every arch in the tree)")
    sl.add_argument("--kernel-name", dest="kernel_name", default="", help="only this kernel")
    sl.add_argument("--producer", default=REMOTE_PRODUCER)
    sl.add_argument("--gpu", default="", help="override the gfx dimension; default is the entry's own")

    # The key-addressed pair. Same gates, same output shapes as resolve/write — only the plane
    # the records live on changes, so the lane can be pointed at either.
    def add_plane_args(w):
        # `both` writes locally and mirrors to the service; reads always take exactly one plane.
        w.add_argument("--plane", choices=("local", "remote", "both"), default="local",
                       help="local dir, the KB Store service (GEAK_KB_STORE_URL/GEAK_KB_STORE_TOKEN, "
                            "or the un-prefixed KB_STORE_URL/KB_STORE_TOKEN), or both")
        w.add_argument("--scan", type=int, default=25,
                       help="remote only: candidates hydrated before curation (page cap is 200)")
        return w

    rr = add_plane_args(sub.add_parser("resolve-remote",
                                       help="rank top-N candidates under one canonical id"))
    rr.add_argument("--store", default="", help="on-disk KB store root (--plane local/both)")
    rr.add_argument("--canonical-id", dest="canonical_id", default="",
                    help="the key to read; derived from kernel/language/gfx when omitted")
    rr.add_argument("--kernel-name", dest="kernel_name", default="")
    rr.add_argument("--language", default="")
    rr.add_argument("--gfx", default="")
    rr.add_argument("--producer", default=REMOTE_PRODUCER)
    rr.add_argument("--gpu", default="", help="override the gfx dimension; default is --gfx")
    rr.add_argument("--framework-version", dest="framework_version", default="",
                    help="rocm <major>.<minor>; default is detected on this box")
    rr.add_argument("--top-n", dest="top_n", type=int, default=3, help="max DIRECTIONS to offer")
    rr.add_argument("--refs-dir", dest="refs_dir", required=True)
    rr.add_argument("--cache-dir", dest="cache_dir", default="",
                    help="where selected candidates are materialized (default <refs-dir>/../kb_cache)")
    rr.add_argument("--min-speedup", dest="min_speedup", type=float, default=1.05)
    rr.add_argument("--include-retired", dest="include_retired", action="store_true",
                    help="also offer entries the curation retired, and let a rung of nothing but "
                         "tombstones stop the ladder instead of reading empty (audit/debug only)")
    rr.add_argument("--carrier", choices=CARRIERS, default="patch",
                    help="which carrier to offer; one per call (default patch)")
    rr.add_argument("--precision", default="",
                    help="only offer entries measured at this precision; entries that state none "
                         "are always offered. A rung holding only other dtypes reads as empty and "
                         "the ladder descends. Omit to filter on nothing (the default)")

    wr = add_plane_args(add_write_args(
        sub.add_parser("write-remote", help="store one win in the local store AND under its key")))
    wr.add_argument("--store", default="", help="on-disk KB store root (--plane local/both)")
    wr.add_argument("--producer", default=REMOTE_PRODUCER)
    wr.add_argument("--gpu", default="", help="override the gfx dimension; default is --gfx")
    wr.add_argument("--framework-version", dest="framework_version", default="",
                    help="rocm <major>.<minor> for the key; default is the measured stack")

    tr = add_plane_args(sub.add_parser(
        "retract-remote", help="take back a written record: retained=false, score zeroed, champion "
                               "re-pointed (there is no delete — this is a rewrite)"))
    tr.add_argument("--store", default="", help="on-disk KB store root (--plane local/both)")
    tr.add_argument("--canonical-id", dest="canonical_id", default="",
                    help="retract on THIS page only; omit to walk both rungs of the ladder")
    tr.add_argument("--session-id", dest="session_id", required=True,
                    help="the session to retract, from the write-remote output")
    tr.add_argument("--reason", required=True,
                    help="why the record is wrong; it is all a future reader has to judge by")
    tr.add_argument("--kernel-name", dest="kernel_name", default="")
    tr.add_argument("--language", default="")
    tr.add_argument("--gfx", default="")
    tr.add_argument("--producer", default=REMOTE_PRODUCER)
    tr.add_argument("--gpu", default="", help="override the gfx dimension; default is --gfx")
    tr.add_argument("--framework-version", dest="framework_version", default="")
    tr.add_argument("--measured-by", dest="measured_by", default="", help="who is retracting it")
    tr.add_argument("--apply", action="store_true", help="actually rewrite; default is a dry run")

    at = add_plane_args(sub.add_parser(
        "attest", help="count one attempt to USE a stored entry; changes no speedup, no rank"))
    at.add_argument("--exp-dir", dest="exp_dir", default="",
                    help="the entry that was tried, as `resolve` reports it; omit when the "
                         "candidate came keyed and has no local mirror")
    at.add_argument("--outcome", required=True, choices=_OUTCOMES,
                    help="validated = reproduced a win; failed = applied but did not win; "
                         "not_reproduced = would not apply or would not build; inapplicable = "
                         "does not fit THIS workspace")
    at.add_argument("--session-id", dest="session_id", default="",
                    help="the keyed record this entry came from, as resolve-remote reports it. "
                         "The keyed ledger is the truth; without this the count is local_only")
    at.add_argument("--store", default="", help="on-disk KB store root (--plane local/both)")
    at.add_argument("--canonical-id", dest="canonical_id", default="")
    at.add_argument("--kernel-name", dest="kernel_name", default="")
    at.add_argument("--language", default="")
    at.add_argument("--gfx", default="")
    at.add_argument("--producer", default=REMOTE_PRODUCER)
    at.add_argument("--gpu", default="", help="override the gfx dimension; default is --gfx")
    at.add_argument("--framework-version", dest="framework_version", default="")
    at.add_argument("--measured-speedup", dest="measured_speedup", default=None,
                    help="what it did here, for the history entry")
    at.add_argument("--note", default="", help="one line a future reader can act on")
    at.add_argument("--measured-by", dest="measured_by", default="", help="who tried it")
    at.add_argument("--apply", action="store_true", help="actually record it; default is a dry run")

    ar = add_plane_args(sub.add_parser(
        "attest-remote", help="the same count, against a key-addressed record on either plane"))
    ar.add_argument("--store", default="", help="on-disk KB store root (--plane local/both)")
    ar.add_argument("--canonical-id", dest="canonical_id", default="",
                    help="attest on THIS page only; omit to walk both rungs of the ladder")
    ar.add_argument("--session-id", dest="session_id", required=True,
                    help="the session that was tried, from the write-remote output")
    ar.add_argument("--outcome", required=True, choices=_OUTCOMES)
    ar.add_argument("--kernel-name", dest="kernel_name", default="")
    ar.add_argument("--language", default="")
    ar.add_argument("--gfx", default="")
    ar.add_argument("--producer", default=REMOTE_PRODUCER)
    ar.add_argument("--gpu", default="", help="override the gfx dimension; default is --gfx")
    ar.add_argument("--framework-version", dest="framework_version", default="")
    ar.add_argument("--measured-speedup", dest="measured_speedup", default=None)
    ar.add_argument("--note", default="")
    ar.add_argument("--measured-by", dest="measured_by", default="", help="who tried it")
    ar.add_argument("--apply", action="store_true", help="actually record it; default is a dry run")

    cu = add_plane_args(sub.add_parser(
        "curate-remote", help="scan a page, apply the retire policy, retract what has earned it"))
    cu.add_argument("--store", default="", help="on-disk KB store root (--plane local/both)")
    cu.add_argument("--canonical-id", dest="canonical_id", default="")
    cu.add_argument("--kernel-name", dest="kernel_name", default="")
    cu.add_argument("--language", default="")
    cu.add_argument("--gfx", default="")
    cu.add_argument("--producer", default=REMOTE_PRODUCER)
    cu.add_argument("--gpu", default="", help="override the gfx dimension; default is --gfx")
    cu.add_argument("--framework-version", dest="framework_version", default="")
    cu.add_argument("--threshold", type=int, default=_RETIRE_THRESHOLD,
                    help="negative attempts, with no win ever, that make a record a candidate")
    cu.add_argument("--measured-by", dest="measured_by", default="", help="who is curating")
    cu.add_argument("--apply", action="store_true", help="actually rewrite; default is a dry run")

    m = sub.add_parser("remap", help="rewrite a stored patch's paths onto this workspace's layout")
    m.add_argument("--patch", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--editable", default="", help="comma-separated workspace-relative editable paths")
    m.add_argument("--workspace", default="", help="scanned for source files when --editable is empty")

    a = p.parse_args(argv)
    try:
        if a.cmd == "write":
            out = cmd_write(a)
        elif a.cmd == "resolve":
            out = cmd_resolve(a)
        elif a.cmd == "remap":
            out = cmd_remap(a)
        elif a.cmd == "languages":
            out = cmd_languages(a)
        elif a.cmd == "backfill-content":
            out = cmd_backfill_content(a)
        elif a.cmd == "export-remote":
            out = cmd_export_remote(a)
        elif a.cmd == "sync-local":
            out = cmd_sync_local(a)
        elif a.cmd == "resolve-remote":
            out = cmd_resolve_remote(a)
        elif a.cmd == "write-remote":
            out = cmd_write_remote(a)
        elif a.cmd == "retract-remote":
            out = cmd_retract_remote(a)
        elif a.cmd == "attest":
            out = cmd_attest(a)
        elif a.cmd == "attest-remote":
            out = cmd_attest_remote(a)
        elif a.cmd == "curate-remote":
            out = cmd_curate_remote(a)
        else:  # pragma: no cover
            out = {"error": "unknown command"}
    except Exception as e:  # never crash the caller
        err = "exception: " + str(e)[:160]
        out = ({"written": False, "reason": err} if a.cmd in ("write", "write-remote")
               else {"retracted": False, "reason": err} if a.cmd == "retract-remote"
               else {"ok": False, "error": err} if a.cmd in ("curate-remote", "sync-local")
               else {"attested": False, "reason": err} if a.cmd in ("attest", "attest-remote")
               else {"remapped": False, "reason": err} if a.cmd == "remap"
               else {"read_reason": err, "candidates": []})
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
