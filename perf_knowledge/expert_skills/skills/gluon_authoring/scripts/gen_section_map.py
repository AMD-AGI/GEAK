#!/usr/bin/env python3
"""gen_section_map.py - (re)generate `reference.md ## E. Section map` from the pack's ## / ### headings.

The section map lists every `##` / `###` heading of every page under `references/`, one line per page,
so a key term is greppable from the entry index. Run it after adding, renaming or removing a heading;
`--check` reports a stale map without writing (exit 1).

Usage:
  gen_section_map.py [--pack <gluon_authoring dir>]          rewrite reference.md in place
  gen_section_map.py [--pack <dir>] --check                  exit 1 if the map is out of date
  gen_section_map.py --selftest                              offline self-check on a temp pack
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

MARK = "## E. Section map"
INTRO = ("Every `##` / `###` section of every page (generated from the headings; regenerate with "
         "`scripts/gen_section_map.py` after editing headings). Find the key term, open the page at "
         "that heading.")
ORDER = ["references/tile-programming", "references/gluon", "references/gluon/pipeline",
         "references/gluon/inline-asm", "references/workloads", "references/pitfalls",
         "references/hardware", "references/method"]
_HEAD = re.compile(r"^(#{2,3}) (.+?)\s*$")


def headings(page: Path) -> list[str]:
    """`##` / `###` headings outside fenced code; `###` prefixed with '› ', '|' made table-safe."""
    out, fence = [], False
    for ln in page.read_text().splitlines():
        if ln.startswith("```"):
            fence = not fence
        if fence:
            continue
        m = _HEAD.match(ln)
        if m:
            h = m.group(2).replace("|", "/")
            out.append(("" if m.group(1) == "##" else "› ") + h)
    return out


def render(pack: Path) -> str:
    """Full new text of reference.md (everything before MARK kept verbatim)."""
    ref = (pack / "reference.md").read_text()
    head = ref.split(MARK)[0].rstrip() + "\n\n"
    out = [MARK, "", INTRO, ""]
    for d in ORDER:
        for f in sorted((pack / d).glob("*.md")):
            hs = headings(f)
            if hs:
                out.append(f"- `{f.relative_to(pack).as_posix()}`: " + " · ".join(hs))
    return head + "\n".join(out) + "\n"


def selftest() -> int:
    with tempfile.TemporaryDirectory() as t:
        p = Path(t)
        (p / "references/workloads").mkdir(parents=True)
        (p / "reference.md").write_text("# idx\n\nrow\n\n## E. Section map\n\nold\n")
        page = "workloads/" + "x" + ".md"  # split so check_pack_refs does not read it as a pack link
        (p / "references" / page).write_text(
            "# X\n## A | b\n```\n## not a heading\n```\n### C\n#### D\n")
        new = render(p)
        assert new.startswith("# idx\n\nrow\n\n## E. Section map\n"), new
        assert f"- `references/{page}`: A / b · › C\n" in new, new
        assert "not a heading" not in new and "D" not in new.split(MARK)[1].split("x.md")[1]
        (p / "reference.md").write_text(new)
        assert render(p) == new, "not idempotent"
    print("SELFTEST PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", type=Path, default=Path(__file__).resolve().parent.parent,
                    help="gluon_authoring pack directory (default: this script's pack)")
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 if the map is stale")
    ap.add_argument("--selftest", action="store_true", help="run the offline self-check and exit")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    ref = a.pack / "reference.md"
    if not ref.is_file():
        ap.error(f"no reference.md under {a.pack}")
    new = render(a.pack)
    if a.check:
        if ref.read_text() != new:
            print(f"STALE: {ref} section map differs from the headings; run without --check")
            return 1
        print("section map up to date")
        return 0
    ref.write_text(new)
    print(f"wrote {ref} ({new.split(MARK)[1].count(chr(10) + '- ')} pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
