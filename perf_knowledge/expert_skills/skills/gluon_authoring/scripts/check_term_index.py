#!/usr/bin/env python3
"""check_term_index.py - validate every pointer in `references/hardware/term-index.md`.

WHY THIS EXISTS, AND WHAT IT IS NOT. `term-index.md` is a file made entirely of pointers, added to
fix the problem that readers could not find content they already had. An index of pointers that
nothing validates is a fresh instance of that same problem: the rows rot silently, and a reader
sent to a heading that no longer exists is worse off than one who never found the index.

Nothing in the build validates it today. Two checks were considered and neither covers this file:

  * `compose.py` `_filter_lever_cards` hard-errors on cross-DSL pointers that do not resolve, but
    only inside the `authoring_ref` / `api_ref` / `onboarding` fields of `lever-cards.json`. It is a
    JSON-field check; a markdown table is outside it.
  * `doc_links.py` checks that a cited FILE resolves inside the pack. BY DEFAULT it splits a
    citation on the heading separator and drops the heading half, so a citation naming a real file
    and a heading that does not exist passes it. It does have a heading check (`_ANCHORED` /
    `_heading_matches`, prefix-matching so the house "cite the opening words" shorthand passes),
    but it is opt-in behind `DOC_LINKS_ANCHORS=1` and OFF in the gate, because the tree it landed
    on has pre-existing anchor drift. So the heading half is unvalidated in practice, not in
    principle -- and two gaps survive even with the flag on: the check is skipped entirely when the
    cited FILE does not resolve in that pack (a bare basename absent from one pack is invisible),
    and it cannot see this index's `term-index-conditional:` declarations, so it would report the
    conditional rows below as failures.

The heading half is exactly what an index is for, so this script checks BOTH halves. It is a
standalone checker, not a gate hook: wiring it into the repo-level gate means editing the gate,
which is outside this pack.

RESOLUTION follows `doc_links.py`: a bare basename is tried against the citing file's directory,
the pack root, and the pack's `references/` directory, in that order.

CONDITIONAL ROWS. Some targets legitimately do not exist in every pack -- `benchmark-hygiene.md` is
a per-pack file whose copies differ by an order of magnitude in length, and the section the fitting
rows point at ships only in the Triton-family packs. Those rows are declared machine-readably in
`term-index.md` itself (`term-index-conditional:` lines) rather than hardcoded here, so the prose
caveat and the check share one source. A conditional row that is NOT declared is a failure; a
declared row is required to resolve in the packs it names and allowed to be absent elsewhere.

Usage:
  check_term_index.py --pack <composed-skill-dir> [--pack <dir> ...]
  check_term_index.py --selftest
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

_FENCE = re.compile(r"^```")
_CITE = re.compile(r"`([\w./-]+\.md)(?:\s+##\s+([^`]+))?`")
_COND = re.compile(r"term-index-conditional:\s*(?P<file>[\w./-]+\.md)\s*##\s*(?P<head>.+?)\s*\|\s*packs:\s*(?P<packs>.+)")
_INDEX_REL = "references/hardware/term-index.md"


def _visible(text: str) -> list[str]:
    """Lines outside fenced code blocks (a citation in a fence is an example, not a pointer)."""
    out, fence = [], False
    for line in text.splitlines():
        if _FENCE.match(line.strip()):
            fence = not fence
            out.append("")
            continue
        out.append("" if fence else line)
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("`", "")).strip().lower()


def _headings(text: str) -> list[str]:
    return [_norm(h) for h in re.findall(r"^#{1,6}\s+(.*)$", text, re.M)]


def _declared_conditionals(text: str) -> dict[tuple[str, str], set[str]]:
    out: dict[tuple[str, str], set[str]] = {}
    for m in _COND.finditer(text):
        key = (m.group("file"), _norm(m.group("head")))
        out[key] = {p.strip() for p in m.group("packs").split(",") if p.strip()}
    return out


def _resolve(index_path: Path, pack: Path, name: str) -> Path | None:
    for cand in (index_path.parent / name, pack / name, pack / "references" / name):
        if cand.is_file():
            return cand
    return None


def check_pack(pack: Path) -> list[str]:
    """Return a list of problems; empty means the index resolves fully in this pack."""
    pack = Path(pack)
    index = pack / _INDEX_REL
    if not index.is_file():
        return [f"{pack.name}: ships no {_INDEX_REL}"]
    text = index.read_text()
    conds = _declared_conditionals(text)
    # the pack's short name, as used in the `packs:` declaration
    short = pack.name.replace("tile-programming-", "")
    problems: list[str] = []
    for lineno, line in enumerate(_visible(text), 1):
        for m in _CITE.finditer(line):
            name, head = m.group(1), (m.group(2) or "").strip()
            target = _resolve(index, pack, name)
            key = (name, _norm(head)) if head else None
            declared = conds.get(key) if key else None
            if target is None:
                if declared is not None and short not in declared:
                    continue  # declared absent here
                problems.append(f"{pack.name}:{lineno}: file does not resolve: {name}")
                continue
            if not head:
                continue
            heads = _headings(target.read_text())
            n = _norm(head)
            if n in heads or any(h.startswith(n) for h in heads):
                continue
            if declared is not None:
                if short in declared:
                    problems.append(
                        f"{pack.name}:{lineno}: declared present in this pack but heading missing: "
                        f"{name} ## {head}")
                continue  # legitimately absent here
            problems.append(f"{pack.name}:{lineno}: heading missing: {name} ## {head}")
    return problems


def stale_declarations(packs: list[Path]) -> list[str]:
    """Conditional declarations whose reason to exist has gone away.

    A `term-index-conditional:` row says "this target is legitimately absent from some packs".
    When the gap is later filled -- someone adds the missing section to the remaining packs --
    the declaration keeps passing, because an unused exemption never fails. It then quietly
    becomes two wrong things: a blanket exemption nobody re-earned, and prose in `term-index.md`
    telling the reader the section ships in fewer packs than it does. Neither is visible from the
    passing side, which is the same way an exemption degrades into "always pass"; this is that
    failure in its other direction.

    Reported only when 2+ packs are scanned. With one pack the comparison has no control and
    cannot conclude anything, so it says so rather than passing silently.
    """
    packs = [Path(p) for p in packs]
    if len(packs) < 2:
        return [f"stale-declaration check SKIPPED: needs 2+ packs, got {len(packs)}"]
    declared: dict[tuple[str, str], set[str]] = {}
    for pack in packs:
        index = pack / _INDEX_REL
        if index.is_file():
            for key, names in _declared_conditionals(index.read_text()).items():
                declared.setdefault(key, set()).update(names)
    problems: list[str] = []
    for (name, nhead), names in sorted(declared.items()):
        resolved_in = []
        for pack in packs:
            index = pack / _INDEX_REL
            target = _resolve(index, pack, name) if index.is_file() else None
            if target is None:
                continue
            heads = _headings(target.read_text())
            if nhead in heads or any(h.startswith(nhead) for h in heads):
                resolved_in.append(pack.name)
        if len(resolved_in) == len(packs):
            problems.append(
                f"stale declaration: `{name} ## {nhead}` now resolves in ALL "
                f"{len(packs)} scanned packs, but is still declared conditional for "
                f"{sorted(names)}. Delete the declaration and the caveat prose it pairs with.")
    return problems


# --------------------------------------------------------------------------- selftest

_IDX = """# Term index

<!-- term-index-conditional: sometimes.md ## Only Here | packs: alpha
-->

| term | at |
| --- | --- |
| a | `target.md ## Real Heading` |
| b | `sometimes.md ## Only Here` |

```text
this is an example, not a pointer: `nonexistent.md ## Nope`
```
"""


def _fixture(root: Path, name: str, *, with_section: bool) -> Path:
    pack = root / f"tile-programming-{name}"
    (pack / "references" / "hardware").mkdir(parents=True, exist_ok=True)
    (pack / _INDEX_REL).write_text(_IDX)
    (pack / "references" / "hardware" / "target.md").write_text("# T\n\n## Real Heading\n\nbody\n")
    body = "# S\n\n## Only Here\n\nbody\n" if with_section else "# S\n\n## Something Else\n\nbody\n"
    (pack / "references" / "sometimes.md").write_text(body)
    return pack


def _selftest() -> int:
    failures = []

    def expect(label: str, problems: list[str], want_fail: bool):
        got_fail = bool(problems)
        ok = got_fail == want_fail
        print(f"  [{'ok' if ok else 'FAIL'}] {label}: "
              f"{'problems=' + str(len(problems)) if got_fail else 'clean'}")
        if not ok:
            failures.append(label)
        for p in problems[:3]:
            print(f"        {p}")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # baseline: the declaring pack has the section, a non-declared pack does not
        alpha = _fixture(root, "alpha", with_section=True)
        beta = _fixture(root, "beta", with_section=False)
        expect("clean pack (declared, section present)", check_pack(alpha), False)
        expect("clean pack (undeclared, section absent -> exempt)", check_pack(beta), False)

        # MUTATION 1: break a heading that is NOT conditional -> must fail
        (alpha / "references" / "hardware" / "target.md").write_text("# T\n\n## Renamed\n\nbody\n")
        expect("mutation: heading renamed", check_pack(alpha), True)
        (alpha / "references" / "hardware" / "target.md").write_text("# T\n\n## Real Heading\n\nbody\n")

        # MUTATION 2: remove a cited file entirely -> must fail
        (alpha / "references" / "hardware" / "target.md").unlink()
        expect("mutation: cited file deleted", check_pack(alpha), True)
        (alpha / "references" / "hardware" / "target.md").write_text("# T\n\n## Real Heading\n\nbody\n")

        # MUTATION 3: pack DECLARED to have the section loses it -> must fail.
        # This is the one that proves the exemption is not a blanket pass.
        (alpha / "references" / "sometimes.md").write_text("# S\n\n## Gone\n\nbody\n")
        expect("mutation: declared-present section removed", check_pack(alpha), True)
        (alpha / "references" / "sometimes.md").write_text("# S\n\n## Only Here\n\nbody\n")

        # MUTATION 4: drop the declaration -> the conditional row must stop being exempt.
        # Proves the declaration is load-bearing rather than decorative.
        (beta / _INDEX_REL).write_text(_IDX.replace("term-index-conditional:", "was-conditional:"))
        expect("mutation: declaration removed (exemption must vanish)", check_pack(beta), True)
        (beta / _INDEX_REL).write_text(_IDX)

        # MUTATION 5: a citation inside a code fence must stay ignored even though
        # `nonexistent.md` does not exist -- otherwise examples become failures.
        expect("fenced example stays ignored", check_pack(beta), False)

        # MUTATION 6: the gap the declaration exists for gets filled in every pack -> the
        # declaration must be reported as stale. Without this, a filled gap is invisible.
        expect("stale check quiet while the gap is real", stale_declarations([alpha, beta]), False)
        (beta / "references" / "sometimes.md").write_text("# S\n\n## Only Here\n\nbody\n")
        expect("mutation: gap filled everywhere -> declaration is stale",
               stale_declarations([alpha, beta]), True)
        (beta / "references" / "sometimes.md").unlink()

        # MUTATION 7: one pack is no control. Must refuse to conclude, not pass.
        expect("single pack refuses to conclude", stale_declarations([alpha]), True)

    print("\nselftest:", "PASS" if not failures else f"FAIL ({len(failures)})")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", action="append", default=[],
                    help="composed skill directory to check (repeatable)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if not args.pack:
        ap.error("--pack is required (or --selftest)")

    problems: list[str] = []
    for p in args.pack:
        problems.extend(check_pack(Path(p)))
    if len(args.pack) >= 2:
        problems.extend(stale_declarations([Path(p) for p in args.pack]))
    if problems:
        print(f"term-index: {len(problems)} unresolved pointer(s)")
        for p in problems:
            print("  " + p)
        return 1
    print(f"term-index: all pointers resolve in {len(args.pack)} pack(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
