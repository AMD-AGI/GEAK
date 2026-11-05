#!/usr/bin/env bash
set -euo pipefail
: "${OUT:?OUT}"; : "${PARAMS:?PARAMS}"
python3 - <<'PY'
import json, os, shutil
from pathlib import Path
params = json.load(open(os.environ["PARAMS"]))
fixture = Path(os.path.expandvars(os.path.expanduser(params["fixture_dir"])))
if not fixture.is_absolute():
    fixture = Path(os.environ["GEAK_ROOT"]) / fixture
for name in ("rows.json", "folded.json"):
    shutil.copyfile(fixture / name, Path(os.environ["OUT"]) / name)
PY
