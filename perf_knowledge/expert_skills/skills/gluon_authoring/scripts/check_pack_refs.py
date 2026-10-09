#!/usr/bin/env python3
"""check_pack_refs.py - every in-pack pointer must resolve: the file AND the heading it names.

WHY. The pack is read by following pointers: `skill.md` routes to `references/method/<stage>.md`,
the runtime stage cards address a heading by its exact text, the lever cards cite the section whose
API a lever calls, and the scripts print "read X ## Y" when they refuse. A pointer whose target was
renamed or folded elsewhere fails silently -- the reader lands nowhere and improvises. This script
resolves all of them, so a restructure cannot leave one behind.

WHAT IS CHECKED (under the pack dir G, plus the GEAK repo root for cross-tree paths):
  * markdown links `[text](rel/path.md#anchor)` in every `.md` (anchor = GitHub slug or heading text)
  * backticked `path.md ## Heading` / `path.md#Heading` citations, and backticked/bare pack paths
    (`references/...`, `scripts/...`, `runtime/...`, `agents/...`, `scheduler/...`, `$KT/...`,
    `$SKILL/...`, `kernel_workflow/...`, `perf_knowledge/...`, `e2e_workflow/...`) in every `.md`
  * every string in the runtime JSON (`runtime/**/*.json`), `scripts/pack_facts.json`
    (incl. standing_references), `references/hardware/lever-cards.json` (authoring_ref / api_ref /
    source / note / onboarding) and the shared hardware data the pack reads
    (`perf_knowledge/hardware/data/{thresholds,workload_models}.json`)
  * `file.md ## Heading` pointers printed by the pack's scripts and the shared kernel tools
    (`G/scripts/*.py|*.sh`, `kernel_workflow/scripts/kernel_tools/*`)

SKIPPED: fenced code blocks (an example is not a pointer) and the body of a level-2 `## Sources`
section (provenance -- it names the files a page was merged from, which by design no longer exist).

RESOLUTION (mirrors check_term_index.py, plus the hardware dir): a relative path is tried against the
citing file's directory, the pack root G, `G/references`, `G/references/hardware`, then the repo root.
`$KT/` = kernel_workflow/scripts/kernel_tools, `$SKILL/` / `$SKILL_ROOT/` / `$PACK/` = G.
Heading match is normalized (leading `#`, backticks, whitespace, case) and a citation may be a PREFIX
of the real heading (the house "cite the opening words" shorthand). A bare basename that resolves
nowhere on that path names the one pack file carrying it, if exactly one does.

ALLOWLIST: `scripts/check_pack_refs.allow` -- one fnmatch pattern per line (matched against the
reference text, e.g. `SKILL.md` or `tile-programming-triton/*`), `#` comments. Only for refs that
are intentionally external (another pack, a run artifact name); never for a real broken pointer.

Usage:
  check_pack_refs.py [--pack G] [--allow FILE] [--quiet]
  check_pack_refs.py --selftest
Exit 0 = every pointer resolves; 1 = dangling refs listed; 2 = usage error.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PACK = HERE.parent

_FENCE = re.compile(r"^\s*(```|~~~)")
_HEAD = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_MDLINK = re.compile(r"(?<!!)\[[^\]\n]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_TICK = re.compile(r"`([^`\n]+)`")
# `path.md ## Heading` or `path.md#Heading` inside one backtick span
_CITE_HASH = re.compile(r"^\s*(?P<path>[\w$./{}-]+\.md)\s*#{1,6}\s*(?P<head>.+?)\s*$")
_CITE_ANCHOR = re.compile(r"^\s*(?P<path>[\w$./{}-]+\.md)#(?P<head>.+?)\s*$")
# a path token: optional $VAR/ or ../ prefix, then a/b/c.ext
_PATH = re.compile(
    r"(?<![\w./$*<>{}-])(?P<p>(?:\$\{?[A-Z_]+\}?/|(?:\.\./)+|\./)?[\w.-]+(?:/[\w.-]+)*"
    r"\.(?:md|py|sh|json|yaml|yml|js|tsv|gz))(?![\w/-])")
_PACK_DIRS = ("references/", "scripts/", "runtime/", "agents/", "scheduler/")
_REPO_DIRS = ("kernel_workflow/", "perf_knowledge/", "e2e_workflow/")
# pointers printed by scripts: X.md ## Heading (heading ends at a quote, backtick or line end)
_SCRIPT_CITE = re.compile(r"(?P<path>[\w./-]+\.md)\s?(?:##+)\s*(?P<head>[^\"'`\n\\]+)")


def _repo_root(pack: Path) -> Path:
    for cand in [pack, *pack.parents]:
        if (cand / "kernel_workflow").is_dir() and (cand / "perf_knowledge").is_dir():
            return cand
    return pack.parents[4] if len(pack.parents) > 4 else pack


def norm(s: str) -> str:
    s = re.sub(r"^#+\s*", "", s.strip())
    s = s.replace("`", "").replace("**", "")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s.rstrip("…").rstrip(".").rstrip() if s.endswith(("…", "...")) else s


def head_candidates(head: str) -> list[str]:
    """A citation embedded in prose/JSON/code has no closing delimiter: try the whole tail, then the
    tail cut at the first sentence / clause break (a heading never contains `. ` or ` -- `)."""
    out = [head]
    for sep in (". ", " -- ", "; ", ")", " (", " and ../", " — ", ", "):
        if sep in head:
            out.append(head.split(sep, 1)[0])
    return [h.strip() for h in out if h.strip()]


def slug(s: str) -> str:
    s = s.strip().lower().replace("`", "")
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def visible_lines(text: str) -> list[str]:
    out, fence = [], False
    for line in text.splitlines():
        if _FENCE.match(line):
            fence = not fence
            out.append("")
            continue
        out.append("" if fence else line)
    return out


class Checker:
    def __init__(self, pack: Path, allow: list[str]):
        self.pack = pack.resolve()
        self.repo = _repo_root(self.pack)
        self.kt = self.repo / "kernel_workflow" / "scripts" / "kernel_tools"
        self.allow = allow
        self.problems: list[str] = []
        self.checked = 0
        self._heads: dict[Path, list[str]] = {}

    # ---------------------------------------------------------------- resolution
    def headings(self, path: Path) -> list[str]:
        if path not in self._heads:
            hs = []
            for line in visible_lines(path.read_text(encoding="utf-8", errors="replace")):
                m = _HEAD.match(line)
                if m:
                    hs.append(norm(m.group(2)))
            self._heads[path] = hs
        return self._heads[path]

    def resolve(self, ref: str, citing: Path) -> Path | None:
        ref = ref.strip()
        m = re.match(r"^\$\{?([A-Z_]+)\}?/(.*)$", ref)
        if m:
            var, rest = m.group(1), m.group(2)
            base = {"KT": self.kt, "SKILL": self.pack, "SKILL_ROOT": self.pack, "PACK": self.pack,
                    "G": self.pack, "GEAK": self.repo, "GEAK_ROOT": self.repo, "REPO": self.repo,
                    "HD": self.repo / "perf_knowledge" / "hardware" / "data"}.get(var)
            if base is None:
                return Path("/__unknown_var__")  # sentinel: not checkable
            cand = base / rest
            return cand if cand.exists() else None
        bases = [citing.parent, self.pack, self.pack / "references", self.pack / "references" / "hardware",
                 self.repo]
        for b in bases:
            cand = (b / ref)
            if cand.exists():
                return cand
        if "/" not in ref:
            # house shorthand: a bare basename names the one pack file that carries it
            hits = self._basenames().get(ref, [])
            if len(hits) == 1:
                return hits[0]
        return None

    def _basenames(self) -> dict[str, list[Path]]:
        if not hasattr(self, "_bn"):
            self._bn: dict[str, list[Path]] = {}
            for f in self.pack.rglob("*"):
                if f.is_file() and "__pycache__" not in f.parts:
                    self._bn.setdefault(f.name, []).append(f)
        return self._bn

    def head_ok(self, target: Path, head: str) -> bool:
        return any(self._head_ok1(target, h) for h in head_candidates(head))

    def _head_ok1(self, target: Path, head: str) -> bool:
        if target.suffix != ".md" or not target.is_file():
            return True
        n = norm(head)
        if not n:
            return True
        hs = self.headings(target)
        if n in hs or any(h.startswith(n) for h in hs):
            return True
        s = slug(head)
        return any(slug(h) == s or slug(h).startswith(s) for h in hs)

    def allowed(self, where: str, ref: str) -> bool:
        cite = where.split(":", 1)[0]
        bare = re.split(r"\s*#", ref, 1)[0].strip()
        for pat in self.allow:
            fpat, sep, rpat = pat.partition("::")
            if not sep:
                fpat, rpat = "*", pat
            if not fnmatch.fnmatchcase(cite, fpat):
                continue
            if fnmatch.fnmatchcase(ref, rpat) or fnmatch.fnmatchcase(bare, rpat):
                return True
        return False

    def report(self, where: str, ref: str, why: str) -> None:
        if self.allowed(where, ref):
            return
        self.problems.append(f"{where}: {why}: {ref}")

    def check_ref(self, where: str, citing: Path, path: str, head: str | None, *, bare_ok=False) -> None:
        self.checked += 1
        path = path.strip().rstrip(".,;:")
        target = self.resolve(path, citing)
        shown = path + (f" ## {head}" if head else "")
        if target is not None and str(target) == "/__unknown_var__":
            return
        if target is None:
            if bare_ok:
                return
            self.report(where, shown, "file does not resolve")
            return
        if head and not self.head_ok(target, head):
            self.report(where, shown, "heading missing")

    # ---------------------------------------------------------------- scanners
    def _is_pointer_path(self, p: str) -> bool:
        if p.startswith(("$", "../", "./")):
            return True
        if p.startswith(_PACK_DIRS + _REPO_DIRS):
            return True
        if "/" not in p and p.endswith(".md"):
            return True  # bare .md basename: the house citation shorthand
        if p.startswith(("gluon/", "hardware/", "tile-programming/", "workloads/", "method/", "pitfalls/")):
            return True  # references-relative (lever-cards / thresholds style)
        return False

    def scan_md(self, f: Path) -> None:
        rel = f.relative_to(self.pack) if self.pack in f.parents else f
        provenance = False
        pending = None
        for i, line in enumerate(visible_lines(f.read_text(encoding="utf-8", errors="replace")), 1):
            where = f"{rel}:{i}"
            hm = _HEAD.match(line)
            if hm:
                # a `## Sources` section is provenance: it names the files a page was merged FROM,
                # which by design no longer exist. It is a record, not a pointer to follow.
                provenance = len(hm.group(1)) == 2 and norm(hm.group(2)).startswith("sources")
            if provenance:
                continue
            if pending:
                line, i, where = pending[0] + " " + re.sub(r"^\s*(?:>\s*)*", "", line), pending[1], pending[2]
                pending = None
            if line.count("`") % 2 == 1 and not line.rstrip().endswith("|"):
                pending = (line, i, where)  # a backtick span wraps onto the next line: join them
                continue
            for m in _MDLINK.finditer(line):
                url = m.group(1)
                if re.match(r"^[a-z]+:", url) or url.startswith("<"):
                    continue
                path, _, anchor = url.partition("#")
                if not path:
                    self.checked += 1
                    if anchor and not self.head_ok(f, anchor.replace("-", " ")) and not self.head_ok(f, anchor):
                        self.report(where, url, "heading missing")
                    continue
                self.check_ref(where, f, path, anchor or None)
            spans = _TICK.findall(_MDLINK.sub(" ", line))
            for span in spans:
                m = _CITE_HASH.match(span) or _CITE_ANCHOR.match(span)
                if m and not re.search(r"[<>*{}]", m.group("path")):
                    if self._is_pointer_path(m.group("path")):
                        self.check_ref(where, f, m.group("path"), m.group("head"))
                    continue
                for pm in _PATH.finditer(span):
                    p = pm.group("p")
                    if re.search(r"[<>*{}]", p) or not self._is_pointer_path(p):
                        continue
                    self.check_ref(where, f, p, None)
            # prose (outside backticks): only unambiguous pack/repo paths
            prose = _TICK.sub(" ", _MDLINK.sub(" ", line))
            for pm in _PATH.finditer(prose):
                p = pm.group("p")
                if re.search(r"[<>*{}]", p):
                    continue
                if p.startswith(_PACK_DIRS + _REPO_DIRS) or p.startswith(("$KT/", "$SKILL/")):
                    self.check_ref(where, f, p, None)

    def _json_strings(self, obj, trail=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from self._json_strings(v, f"{trail}.{k}" if trail else str(k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                yield from self._json_strings(v, f"{trail}[{i}]")
        elif isinstance(obj, str):
            yield trail, obj

    def scan_json(self, f: Path, *, refs_base: Path | None = None) -> None:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.problems.append(f"{f}: unreadable JSON: {exc}")
            return
        rel = f.relative_to(self.repo) if self.repo in f.parents else f
        citing = (refs_base / "_.json") if refs_base else f
        for trail, s in self._json_strings(data):
            where = f"{rel}:{trail}"
            done = set()
            # whole-string `path.md#Heading` (runtime cards) / `path.md ## Heading`
            m = _CITE_ANCHOR.match(s) or _CITE_HASH.match(s)
            if m and self._is_pointer_path(m.group("path")):
                self.check_ref(where, citing, m.group("path"), m.group("head"))
                continue
            for cm in re.finditer(r"(?P<path>[\w./-]+\.md)\s*##\s*(?P<head>[^;`()\"\n]+?)(?=\s*(?:[;`()\"\n]|$| \+ | / | or | and ))", s):
                if self._is_pointer_path(cm.group("path")):
                    self.check_ref(where, citing, cm.group("path"), cm.group("head").strip(" .,"))
                    done.add(cm.start("path"))
            for pm in _PATH.finditer(s):
                if pm.start("p") in done:
                    continue
                p = pm.group("p")
                if re.search(r"[<>*{}]", p) or not self._is_pointer_path(p):
                    continue
                self.check_ref(where, citing, p, None)

    def scan_script(self, f: Path) -> None:
        rel = f.relative_to(self.repo) if self.repo in f.parents else f
        text = f.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            for m in _SCRIPT_CITE.finditer(line):
                path = m.group("path")
                if not self._is_pointer_path(path) or re.search(r"[<>*{}]", path):
                    continue
                head = m.group("head").strip().rstrip(".,;:)")
                if len(head) < 3:
                    continue
                self.check_ref(f"{rel}:{i}", self.pack / "scripts" / "_.py", path, head)
            # bare pack paths a script prints or documents ("see references/method/profile.md")
            for pm in _PATH.finditer(line):
                p = pm.group("p")
                if not p.endswith(".md") or re.search(r"[<>*{}]", p):
                    continue
                if p.startswith(("references/", "../references/", "agents/")) or "gluon_authoring/" in p:
                    self.check_ref(f"{rel}:{i}", self.pack / "scripts" / "_.py", p, None)

    # ---------------------------------------------------------------- driver
    def run(self, with_shared=True) -> list[str]:
        for f in sorted(self.pack.rglob("*.md")):
            if "__pycache__" in f.parts or ".tile-runtime" in f.parts:
                continue
            self.scan_md(f)
        for f in sorted((self.pack / "runtime").rglob("*.json")) if (self.pack / "runtime").is_dir() else []:
            self.scan_json(f)
        for rel in ("scripts/pack_facts.json", "references/hardware/lever-cards.json"):
            if (self.pack / rel).is_file():
                self.scan_json(self.pack / rel, refs_base=self.pack / "references")
        if with_shared:
            hd = self.repo / "perf_knowledge" / "hardware" / "data"
            for name in ("thresholds.json", "workload_models.json"):
                if (hd / name).is_file():
                    self.scan_json(hd / name, refs_base=self.pack / "references")
        script_dirs = [self.pack / "scripts"] + ([self.kt] if with_shared and self.kt.is_dir() else [])
        for d in script_dirs:
            for f in sorted(list(d.glob("*.py")) + list(d.glob("*.sh"))):
                if f.name == Path(__file__).name:
                    continue
                self.scan_script(f)
        return self.problems


def load_allow(path: Path | None) -> list[str]:
    if path is None or not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip() if not line.lstrip().startswith("#") else ""
        if line:
            out.append(line)
    return out


# --------------------------------------------------------------------------- selftest

def _selftest() -> int:
    failures: list[str] = []

    def expect(label, problems, want_fail):
        ok = bool(problems) == want_fail
        print(f"  [{'ok' if ok else 'FAIL'}] {label}: {'problems=' + str(len(problems)) if problems else 'clean'}")
        for p in problems[:3]:
            print("        " + p)
        if not ok:
            failures.append(label)

    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        (repo / "kernel_workflow" / "scripts" / "kernel_tools").mkdir(parents=True)
        (repo / "kernel_workflow" / "scripts" / "kernel_tools" / "tool.py").write_text(
            "print('read method/a.md ## Real Heading')\n")
        (repo / "perf_knowledge" / "x").mkdir(parents=True)
        (repo / "perf_knowledge" / "x" / "doc.md").write_text("# Doc\n")
        pack = repo / "perf_knowledge" / "expert_skills" / "skills" / "p"
        (pack / "references" / "method").mkdir(parents=True)
        (pack / "references" / "hardware").mkdir(parents=True)
        (pack / "runtime" / "stages").mkdir(parents=True)
        (pack / "scripts").mkdir(parents=True)
        A = pack / "references" / "method" / "a.md"
        A.write_text("# A\n\n## Real Heading — with a tail\n\nbody\n\n### Sub `x`\n")
        (pack / "references" / "hardware" / "hw.md").write_text("# HW\n\n## Gating laws\n")
        entry = pack / "skill.md"
        good = ("# Entry\n\nSee [a](references/method/a.md#real-heading--with-a-tail) and\n"
                "`references/method/a.md ## Real Heading` and `method/a.md#Sub x` and `hw.md ## Gating`.\n"
                "Run `$KT/tool.py` and `perf_knowledge/x/doc.md`; artifact `decision_log.md` is allowed.\n"
                "```\n`references/nope.md ## fenced example`\n```\n")
        entry.write_text(good)
        (pack / "runtime" / "stages" / "s.json").write_text(json.dumps(
            {"stage_spec": {"reference": "references/method/a.md#Real Heading — with a tail"},
             "standing_references": {"hw": "references/hardware/hw.md"}}))
        (pack / "references" / "hardware" / "lever-cards.json").write_text(json.dumps(
            {"levers": {"x": {"authoring_ref": "method/a.md ## Real Heading", "source": "hardware/hw.md"}}}))
        allow = ["decision_log.md"]

        def run():
            return Checker(pack, allow).run()

        expect("clean pack", run(), False)
        entry.write_text(good + "`references/method/gone.md`\n")
        expect("mutation: cited file missing", run(), True)
        entry.write_text(good + "`references/method/a.md ## Renamed Heading`\n")
        expect("mutation: backticked heading missing", run(), True)
        entry.write_text(good + "[x](references/method/a.md#no-such-anchor)\n")
        expect("mutation: md-link anchor missing", run(), True)
        entry.write_text(good)
        (pack / "runtime" / "stages" / "s.json").write_text(json.dumps(
            {"stage_spec": {"reference": "references/method/a.md#Stop conditions"}}))
        expect("mutation: runtime card heading missing", run(), True)
        (pack / "runtime" / "stages" / "s.json").write_text(json.dumps(
            {"standing_references": {"budget": "references/phases/budget.md"}}))
        expect("mutation: standing reference missing", run(), True)
        (pack / "runtime" / "stages" / "s.json").write_text("{}")
        (pack / "references" / "hardware" / "lever-cards.json").write_text(json.dumps(
            {"levers": {"x": {"api_ref": "gluon/memory-reference.md ## Async"}}}))
        expect("mutation: lever-card api_ref missing", run(), True)
        (pack / "references" / "hardware" / "lever-cards.json").write_text("{}")
        (repo / "kernel_workflow" / "scripts" / "kernel_tools" / "tool.py").write_text(
            "print('read compiler-contract.md ## Scenario B')\n")
        expect("mutation: script-printed pointer missing", run(), True)
        (repo / "kernel_workflow" / "scripts" / "kernel_tools" / "tool.py").write_text("print('ok')\n")
        entry.write_text(good.replace("decision_log.md", "unknown_artifact.md"))
        expect("mutation: non-allowlisted bare name", run(), True)
        entry.write_text(good + "\n## Sources\n\nMerged from `references/phases/budget.md` (deleted by design).\n")
        expect("a ## Sources section is provenance, not a pointer", run(), False)
        entry.write_text(good + "\n## Sources\n\nx\n\n## Next\n\n`references/phases/budget.md`\n")
        expect("mutation: pointer after the Sources section is checked again", run(), True)
        entry.write_text(good + "Wrapped: `references/method/a.md ## Real\nHeading — with a tail`.\n")
        expect("a citation wrapped across two lines resolves", run(), False)
        entry.write_text(good + "Wrapped: `references/method/a.md ## Not\nThere`.\n")
        expect("mutation: a wrapped citation with a wrong heading fails", run(), True)
        entry.write_text(good)
        expect("restored pack is clean again", run(), False)

    print("\nselftest:", "PASS" if not failures else f"FAIL ({len(failures)})")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", default=str(DEFAULT_PACK), help="pack dir (default: this script's pack)")
    ap.add_argument("--allow", default=None, help="allowlist file (default: <pack>/scripts/check_pack_refs.allow)")
    ap.add_argument("--no-shared", action="store_true",
                    help="skip the shared hardware data and kernel_workflow/scripts/kernel_tools scans")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    pack = Path(args.pack).resolve()
    if not pack.is_dir():
        print(f"check_pack_refs: no such pack dir: {pack}", file=sys.stderr)
        return 2
    allow_path = Path(args.allow) if args.allow else pack / "scripts" / "check_pack_refs.allow"
    chk = Checker(pack, load_allow(allow_path))
    problems = chk.run(with_shared=not args.no_shared)
    if problems:
        print(f"check_pack_refs: {len(problems)} dangling reference(s) ({chk.checked} checked)")
        for p in problems:
            print("  " + p)
        return 1
    if not args.quiet:
        print(f"check_pack_refs: all {chk.checked} references resolve in {pack.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
