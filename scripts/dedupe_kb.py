"""Dedupe knowledge_base.json by (kernel_name, best_strategy) — keep highest-speedup entry."""

import json
from pathlib import Path

KB = Path(__file__).resolve().parents[1] / "src" / "minisweagent" / "memory" / "cross_session" / "knowledge_base.json"

data = json.loads(KB.read_text())
exps = data.get("experiences", [])
print(f"Before: {len(exps)} experiences")

best_by_key: dict[tuple[str, str], dict] = {}
for exp in exps:
    key = (exp.get("kernel_name", ""), exp.get("best_strategy", ""))
    sp = float(exp.get("best_speedup", 0))
    if key not in best_by_key or sp > float(best_by_key[key].get("best_speedup", 0)):
        best_by_key[key] = exp

deduped = list(best_by_key.values())
print(f"After:  {len(deduped)} experiences (removed {len(exps) - len(deduped)} duplicates)")

data["experiences"] = deduped
data["experience_count"] = len(deduped)
KB.write_text(json.dumps(data, indent=2))
print(f"Wrote {KB}")

# Per-kernel count
from collections import Counter
cnts = Counter(e.get("kernel_name", "") for e in deduped)
print("\nPer-kernel counts:")
for k, c in sorted(cnts.items(), key=lambda x: -x[1]):
    print(f"  {c:>3}  {k}")
