#!/usr/bin/env python3
"""Validate a generated MegaMoE Step-2 analysis artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze import SCHEMA_VERSION, validate_analysis_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.analysis.read_text(encoding="utf-8"))
    validate_analysis_output(payload)
    print(f"GEAK_MOE_ANALYSIS_VALID={SCHEMA_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
