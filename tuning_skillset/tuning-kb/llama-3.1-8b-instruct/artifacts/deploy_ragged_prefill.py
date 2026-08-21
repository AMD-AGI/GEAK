#!/usr/bin/env python3
"""apply|revert the ragged no-prefix prefill path in SGLang's aiter backend.

Reversible file swap, never a hand edit: `ab_campaign.sh` flips arm order between rounds and
so calls apply/revert repeatedly, and a half-applied source tree is a silently wrong
measurement. Both sides are byte-compared against copies taken before the first edit
(analysis/base/aiter_backend.py.orig is pristine, .ragged carries the patch), and every
__pycache__ entry for the file is removed so the next server start cannot import a stale .pyc.

  python3 analysis/deploy_ragged_prefill.py apply|revert|status
"""
from __future__ import annotations

import filecmp
import hashlib
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
LIVE = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/layers/attention/aiter_backend.py"
)
ORIG = HERE / "base" / "aiter_backend.py.orig"
CAND = HERE / "base" / "aiter_backend.py.ragged"


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def _drop_pyc() -> None:
    for pyc in LIVE.parent.glob("__pycache__/aiter_backend.*.pyc"):
        pyc.unlink()


def main() -> int:
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    for p in (ORIG, CAND):
        if not p.exists():
            print(f"missing recorded base {p}", file=sys.stderr)
            return 2

    if action == "status":
        pass
    elif action in ("apply", "revert"):
        shutil.copyfile(CAND if action == "apply" else ORIG, LIVE)
        _drop_pyc()
    else:
        print(__doc__, file=sys.stderr)
        return 2

    state = (
        "cand(ragged)"
        if filecmp.cmp(LIVE, CAND, shallow=False)
        else "base(paged)" if filecmp.cmp(LIVE, ORIG, shallow=False) else "UNKNOWN"
    )
    print(f"{action}: live={_sha(LIVE)} -> {state}")
    return 0 if state != "UNKNOWN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
