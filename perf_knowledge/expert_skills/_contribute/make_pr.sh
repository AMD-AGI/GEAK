#!/usr/bin/env bash
# make_pr.sh — validate an expert skill and open a PR for it.
#
# Usage:   bash _contribute/make_pr.sh <skill_id> [--allow-draft]
#
# Steps:
#   1. static validation (schema / operator alignment / required sections)
#   2. require validation_status == validated  (unless --allow-draft)
#   3. branch + commit the skill file (+ regenerated index.yaml)
#   4. push and open a PR (gh if available, else print the compare URL)
set -euo pipefail

ID="${1:-}"; shift || true
ALLOW_DRAFT=0
for a in "$@"; do [ "$a" = "--allow-draft" ] && ALLOW_DRAFT=1; done
[ -n "$ID" ] || { echo "usage: make_pr.sh <skill_id> [--allow-draft]"; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"                 # expert_skills/
SKILL="$ROOT/skills/$ID/skill.md"
[ -f "$SKILL" ] || { echo "no such skill: $SKILL"; exit 2; }

echo "==> static validation"
python3 "$HERE/validate_skill.py" "$ID" --static

echo "==> reindex"
python3 "$HERE/scaffold.py" --reindex

STATUS="$(python3 - "$SKILL" <<'PY'
import sys, re, yaml
t=open(sys.argv[1]).read(); m=re.match(r"^---\n(.*?)\n---\n",t,re.S)
print((yaml.safe_load(m.group(1)).get("validation") or {}).get("status","draft"))
PY
)"
echo "    validation_status = $STATUS"
if [ "$STATUS" != "validated" ] && [ "$ALLOW_DRAFT" -eq 0 ]; then
  echo "REFUSING: status is '$STATUS', not 'validated'. Run validate_skill.py --emit-plan, do the"
  echo "on-box measurement, then validate_skill.py $ID --record ...  (or pass --allow-draft for a WIP PR)."
  exit 1
fi

cd "$ROOT"
BRANCH="expert-skill/$ID"
git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
git add "skills/$ID" index.yaml
git commit -m "expert_skills: add $ID ($STATUS)" || { echo "nothing to commit"; }
git push -u origin "$BRANCH" 2>&1 || { echo "push failed (check remote/auth)"; exit 1; }

if command -v gh >/dev/null 2>&1; then
  gh pr create --fill --title "expert_skills: add $ID" \
    --body "Adds expert skill \`$ID\` (status: $STATUS). Validated via validate_skill.py; see the skill's validation.artifact." || true
else
  REMOTE="$(git remote get-url origin 2>/dev/null || echo '')"
  echo "gh not found. Push done on branch '$BRANCH'."
  echo "Open a PR manually for: $REMOTE  (branch $BRANCH)"
fi
