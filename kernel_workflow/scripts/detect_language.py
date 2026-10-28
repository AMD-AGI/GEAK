#!/usr/bin/env python3
"""Decide which authoring language a kernel is written in, from the source itself.

Why this exists: the lane's `TARGET_LANGUAGE` is a REQUEST, not an observation. It defaults to
`triton` and is only ever set by the caller, so on an optimize run -- where the language is whatever
the existing file happens to be -- it stays `triton` no matter what the kernel is. Feeding that into
a learned card labels every optimize-mode finding `triton`, which is worse than the `unknown` it
replaces: a wrong label is trusted, an absent one is not.

So the language is read off the source. Two rules make the answer trustworthy:

  * Never default. Ambiguous or unrecognised returns `null` with the reason, because the caller can
    handle "I don't know" and cannot handle a confident wrong answer.
  * Report the evidence. Every verdict carries the file, line and matched text that produced it, so a
    reviewer can check it without rerunning anything.

Usage:
    python3 detect_language.py PATH [PATH ...] [--entry KERNEL_SYMBOL] [--json]

PATH is a file or a directory (walked). --entry names the kernel symbol under optimization; files
defining it are weighted above the rest, which is what separates the kernel from its test harness.

Exit status: 0 when a language was decided, 1 when it was not. The caller is expected to branch on
that rather than parse prose.
"""
import argparse
import json
import os
import re
import sys

# (language, pattern, weight, declares). Two independent axes, and conflating them breaks detection
# in both directions:
#
#   `declares`  -- does this marker mean the file IS written in the language, as opposed to merely
#                  referring to it? An import, a kernel decorator, an include of the language's own
#                  headers, `__global__`: yes. A namespace reference such as `ck_tile::`: no, a HIP
#                  kernel may call one CK utility without being a CK kernel.
#   `weight`    -- how much the match contributes when ranking languages that both declare.
#
# Measured on aiter, both mistakes are real. Treating `ck_tile::` as a declaration reported five
# plainly-HIP `.cu` files (15 `__global__` apiece) as `ck`, and one as `cutlass_port`. Then treating
# `__global__` as a mere mention made HIP undetectable in all 33.
MARKERS = [
    # --- Python DSLs ---------------------------------------------------------------------------
    ("flydsl", r"^\s*(?:from|import)\s+flydsl\b", 12, True),
    ("flydsl", r"@\s*flyc\.kernel\b", 12, True),
    ("flydsl", r"^\s*from\s+[\w.]*\bflydsl\b[\w.]*\s+import", 10, True),
    ("flydsl", r"\bflydsl\.(?:expr|compiler|runtime|autotune)\b", 5, False),

    # Gluon is a Triton dialect: its files import triton, so it must SUPPRESS triton rather than
    # merely outscore it -- see PARENT below.
    ("gluon", r"triton\.experimental\.gluon\b", 12, True),
    ("gluon", r"@\s*gluon\.jit\b", 12, True),
    ("gluon", r"\bgl\.(?:alloc_shared|warp_id|thread_id)\b", 5, False),

    ("triton", r"^\s*(?:from|import)\s+triton\b", 12, True),
    ("triton", r"@\s*triton\.(?:jit|autotune)\b", 12, True),
    # A thin wrapper that imports the real kernel out of a triton subpackage IS the entry point of a
    # Triton kernel; `aiter.ops.triton._triton_kernels.*` is the shape this takes in practice.
    ("triton", r"^\s*from\s+[\w.]*\btriton\b[\w.]*\s+import", 10, True),
    ("triton", r"\btl\.(?:load|store|dot|program_id|arange)\b", 5, False),

    ("tilelang", r"^\s*(?:from|import)\s+tilelang\b", 12, True),
    ("tilelang", r"@\s*tilelang\.jit\b", 12, True),

    ("mojo", r"^\s*fn\s+\w+.*\bcapturing\b", 10, True),

    # --- C++ / HIP families -------------------------------------------------------------------
    ("ck", r"#include\s*[<\"]ck[/_]", 12, True),
    ("ck", r"\bck_tile::", 5, False),
    ("ck", r"\bck::tensor_operation\b", 5, False),

    ("rocwmma", r"#include\s*[<\"]rocwmma/", 12, True),
    ("rocwmma", r"\brocwmma::", 5, False),

    ("hipkittens", r"#include\s*[<\"](?:hip)?kittens", 12, True),
    ("hipkittens", r"\bkittens::", 5, False),

    ("cutlass_port", r"#include\s*[<\"]cutlass/", 12, True),
    ("cutlass_port", r"\bcutlass::", 5, False),

    ("hip", r"\b__global__\b", 10, True),
    # Not anchored at the start of the include path: a host-side HIP file commonly reaches the
    # runtime through a wrapper such as `ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h`.
    ("hip", r"#include\s*[<\"][\w/]*\bhip\b[\w/]*[/.]", 10, True),
    ("hip", r"\bhipLaunchKernelGGL\b", 10, True),
    ("hip", r"\b__launch_bounds__\b", 5, False),

    # --- assembly -----------------------------------------------------------------------------
    # Inline asm inside a HIP kernel is not an asm kernel, so mnemonics only corroborate; the kernel
    # descriptor directive, or the file extension, is what settles it.
    ("asm", r"\.amdhsa_kernel\b", 12, True),
    ("asm", r"\bv_mfma_\w+", 3, False),
    ("asm", r"\bs_waitcnt\b|\bds_read_b\d+\b|\bbuffer_load_dword", 3, False),
]
EXT_WEIGHT = 12

# A language whose files legitimately carry another language's markers, so the parent's score is not
# evidence of the parent. Suppression requires a DECISIVE child marker: "this file includes a CK
# header" outranks its HIP-ness, while "this file mentions ck_tile:: once" does not.
PARENT = {
    "gluon": "triton",
    "ck": "hip",
    "rocwmma": "hip",
    "hipkittens": "hip",
    "cutlass_port": "hip",
}

# Extension alone settles the languages that have one. Everything else shares .py or .cpp with a
# sibling and has to be decided by content.
BY_EXT = {
    ".s": "asm", ".asm": "asm",
    ".mojo": "mojo", ".🔥": "mojo",
}

SOURCE_EXT = {".py", ".cpp", ".cc", ".cxx", ".c", ".cu", ".hip", ".h", ".hpp", ".cuh",
              ".s", ".asm", ".mojo", ".🔥"}
SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".venv", "build", "dist"}
# A file that drives, builds or exercises a kernel is not the kernel. Its vote is halved rather than
# dropped: `pa_gluon_aot/` is one Gluon kernel plus a prebuild and a warmup script that both import
# triton, and counting those two equally makes the directory look like Triton.
HARNESS_HINT = re.compile(
    r"(?:^|[/_])(?:test|bench|benchmark|harness|conftest|setup|warmup|prebuild|build|gen|"
    r"dispatch|__init__)")


def strip_comments(lines, ext):
    """Blank out comment text, keeping one output line per input line so evidence keeps its number.

    Commented-out code is not evidence. This was not a hypothetical: `csrc/kernels/cache_kernels.cu`
    is 15 `__global__` functions whose only CK references are four commented-out `ck_tile::` lines,
    and it was reported as a CK kernel on the strength of them.

    Deliberately naive -- a `#` or `//` inside a string literal is stripped too. That direction is
    safe: it can only remove evidence and push the verdict toward "undecided", never invent one.
    """
    out = []
    if ext == ".py":
        for line in lines:
            out.append(line.split("#", 1)[0])
        return out
    in_block = False
    for line in lines:
        buf, i = [], 0
        while i < len(line):
            if in_block:
                end = line.find("*/", i)
                if end == -1:
                    i = len(line)
                else:
                    in_block = False
                    i = end + 2
                continue
            start = line.find("/*", i)
            slash = line.find("//", i)
            if slash != -1 and (start == -1 or slash < start):
                buf.append(line[i:slash])
                i = len(line)
            elif start != -1:
                buf.append(line[i:start])
                in_block = True
                i = start + 2
            else:
                buf.append(line[i:])
                i = len(line)
        out.append("".join(buf))
    return out


def iter_files(paths):
    for p in paths:
        if os.path.isfile(p):
            yield p
            continue
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in sorted(filenames):
                if os.path.splitext(fn)[1].lower() in SOURCE_EXT:
                    yield os.path.join(dirpath, fn)


def scan_file(path, entry):
    """(scores, decisive, evidence) for one file. `entry` present in the file lifts every weight."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            raw = f.read().splitlines()
    except OSError as e:
        return {}, set(), [{"file": path, "error": str(e)}]

    ext = os.path.splitext(path)[1].lower()
    scores, decisive, evidence = {}, set(), []

    if ext in BY_EXT:
        lang = BY_EXT[ext]
        scores[lang] = scores.get(lang, 0) + EXT_WEIGHT
        decisive.add(lang)
        evidence.append({"file": path, "line": 0, "language": lang,
                         "match": f"file extension {ext}", "weight": EXT_WEIGHT})

    lines = strip_comments(raw, ext)
    text = "\n".join(lines)
    # How much this file's verdict counts toward a directory's:
    #   defines the kernel under optimization -> 2x, it is the file the question is about
    #   named like a harness / build driver   -> 0.5x, it drives a kernel rather than being one
    factor = 1.0
    if entry and re.search(rf"\b{re.escape(entry)}\b", text):
        factor = 2.0
    elif HARNESS_HINT.search(os.path.basename(path).lower()):
        factor = 0.5

    for lang, pattern, weight, declares in MARKERS:
        rx = re.compile(pattern, re.MULTILINE)
        for i, line in enumerate(lines, 1):
            m = rx.search(line)
            if not m:
                continue
            scores[lang] = scores.get(lang, 0) + weight
            if declares:
                decisive.add(lang)
            evidence.append({"file": path, "line": i, "language": lang,
                             "match": m.group(0).strip()[:80], "weight": weight})
            break        # one hit per marker per file: repetition is not extra proof
    return scores, decisive, evidence, factor


def decide(scores, decisive):
    """(language, reason). Returns (None, why) rather than guessing when the evidence is thin."""
    if not scores:
        return None, "no authoring-language marker found in any scanned file"

    suppressed = {parent for child, parent in PARENT.items() if child in decisive}
    live = {k: v for k, v in scores.items() if k not in suppressed}
    if not live:
        return None, "every candidate was suppressed as a host language of another candidate"

    ranked = sorted(live.items(), key=lambda kv: (-kv[1], kv[0]))
    top, top_score = ranked[0]
    if top not in decisive:
        return None, (f"strongest signal is {top!r} at {top_score:g}, but only from namespace or "
                      f"helper references — no include, import or kernel decorator. That is a "
                      f"mention, not an implementation")
    if len(ranked) > 1:
        runner, runner_score = ranked[1]
        if runner in decisive and runner_score >= top_score * 0.75:
            return None, (f"ambiguous: {top!r} ({top_score:g}) and {runner!r} ({runner_score:g}) both "
                          f"declare themselves and are too close to call; a wrong language label is "
                          f"worse than none")
    note = ""
    if suppressed:
        note = f"; ignored {', '.join(sorted(suppressed))} as the host language of a more specific match"
    return top, f"{top!r} scored {top_score:g}{note}"


def tied_languages(scores, decisive):
    """The declaring languages too close to call in one file: what `decide` refused to pick among.

    Empty when one language clearly leads. A caller that knows which language it asked for may use
    this to confirm that request - never to pick a language the file does not declare.
    """
    suppressed = {parent for child, parent in PARENT.items() if child in decisive}
    live = {k: v for k, v in scores.items() if k in decisive and k not in suppressed}
    if len(live) < 2:
        return []
    top = max(live.values())
    tied = sorted(k for k, v in live.items() if v >= top * 0.75)
    return tied if len(tied) > 1 else []


def detect(paths, entry=None):
    """Decide each file, then let the files vote. Aggregating raw scores across files is wrong.

    The host-language suppression in `decide` is a statement about ONE file: "this file imports
    triton because Gluon is a Triton dialect". Summing scores over a tree and suppressing afterwards
    applies that statement to files it was never about. Measured on aiter: 309 Triton kernels
    containing two Gluon files reported the whole tree as Gluon, and `csrc/kernels` reported as CK
    because two of its 33 HIP files include a CK header.
    """
    votes, evidence, scanned, undecided = {}, [], 0, 0
    tied = set()
    for path in iter_files(paths):
        s, d, e, factor = scan_file(path, entry)
        scanned += 1
        evidence.extend(e)
        lang, _why = decide(s, d)
        if lang is None:
            undecided += 1
            tied.update(tied_languages(s, d))
            continue
        votes[lang] = votes.get(lang, 0) + factor

    ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    if not ranked:
        language, reason = None, (f"none of the {scanned} scanned file(s) declares an authoring "
                                  f"language")
    elif len(ranked) == 1:
        language, reason = ranked[0][0], f"{ranked[0][0]!r} in {ranked[0][1]:g} of the scanned files"
    else:
        (top, top_v), (runner, runner_v) = ranked[0], ranked[1]
        if top_v >= runner_v * 1.5:
            language = top
            reason = (f"{top!r} in {top_v:g} weighted file(s) against {runner!r} at {runner_v:g}"
                      + (f" (+{undecided} undecided)" if undecided else ""))
        else:
            language = None
            reason = (f"ambiguous: {top!r} ({top_v:g}) and {runner!r} ({runner_v:g}) are too close "
                      f"to call across the scanned files; name a single file or pass --entry")
            tied.update({top, runner})

    evidence.sort(key=lambda d: -d.get("weight", 0))
    return {
        "language": language,
        "reason": reason,
        "files_scanned": scanned,
        "files_undecided": undecided,
        "votes": {k: round(v, 2) for k, v in ranked},
        # Only when undecided: the declaring languages the verdict was refused between. A vendored
        # dispatcher that carries a FlyDSL path and a Triton fallback in one file lands here.
        "ambiguous_between": sorted(tied) if language is None else [],
        # Enough to check the verdict by hand, not the whole scan.
        "evidence": [e for e in evidence if e.get("language") == language][:5] or evidence[:5],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="source file(s) or directory(ies) to scan")
    ap.add_argument("--entry", default=None,
                    help="kernel symbol under optimization; files defining it outweigh the rest")
    ap.add_argument("--json", action="store_true", help="machine-readable output only")
    a = ap.parse_args()

    missing = [p for p in a.paths if not os.path.exists(p)]
    if missing:
        # Refused rather than reported as "no language found": a typo'd path is indistinguishable
        # from a kernel with no markers, and only one of them is the caller's fault.
        ap.error("path(s) do not exist: " + ", ".join(missing))

    res = detect(a.paths, a.entry)
    if a.json:
        print(json.dumps(res, ensure_ascii=False))
    else:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res["language"] else 1


if __name__ == "__main__":
    sys.exit(main())
