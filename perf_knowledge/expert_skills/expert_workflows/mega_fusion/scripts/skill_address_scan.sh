#!/usr/bin/env bash
# skill_address_scan.sh -- fail if an injected expert-skill card contains an address that RESOLVES
# to a reachable copy of the answer.
#
# WHY THIS EXISTS, SEPARATELY FROM reference_leak_sweep.sh
# --------------------------------------------------------
# The sweep looks for reference *content* copied into the run tree. This looks for the shorter path:
# a knowledge card that simply tells the engineer where the finished implementation lives. Once,
# with capability_eval on and use_expert_skills=true, a now-retired
# MegaMoE predecessor card opened its Procedure with
#
#     Reference implementation: `AITER` branch `geak/megamoe-v2-candidate`, HEAD `<9-hex sha>`.
#
# and cited five more commits in Sources. That card is read off disk by the engineer agent, so the
# orchestrator cannot redact it at injection time, and the run was asking that same engineer to
# DERIVE the megakernel. Prose in the injected block already said "use the mechanism, never the
# addresses"; prose had already proved insufficient twice.
#
# THE RULE IS RESOLVABILITY, NOT CITATION
# ---------------------------------------
# A commit hash in a Sources section is normal, honest provenance and most of them point at repos
# this machine cannot see (another skill cites `d7e8df1` in an internal perf-knowledge repo -- that
# is a citation, not a door). Failing on every hex token would push authors to strip real provenance
# and would cry wolf until it got switched off. So the test is the property that actually matters:
#
#     does this address resolve to a commit in a repository the run can reach?
#
# A citation that resolves nowhere passes. A citation that `git cat-file -t` turns into a commit from
# inside the run tree is an address, and it fails -- one `git show <sha>:<file>` away from the answer.
#
# Repos searched: every .git discoverable under the scan root, plus $AITER_JIT_DIR's repo (GEAK_TASK
# points every engineer at that one by name), plus anything given with --repo.
#
# THE SAME RULE, APPLIED TO FILESYSTEM PATHS
# ------------------------------------------
# A git sha is not the only kind of address. `/outside/hidden-reference/reference.patch`
# in a card is the same door with a shorter walk, and the git pass cannot see it. So the second pass
# asks the identical question of every absolute path a card names:
#
#     does this path exist, and can this uid read it?
#
# Symmetric to the git rule, and for the same reason: a path that names another machine's layout is a
# citation and passes; one that opens here is an address and fails. Three things are deliberately NOT
# findings:
#   * relative paths (`aiter/ops/flydsl/...`) -- they resolve against the engineer's own workspace,
#     which is the tree the task is asking them to edit. Naming the file to modify is the job.
#   * system prefixes (/usr, /opt/rocm, ...) -- present on every box, carry nothing.
#   * a path that EXISTS but this uid cannot read -- reported as CONTAINED, not as a failure.
#     READ THE CAVEAT: root ignores permission bits, so under uid 0 that branch is unreachable and
#     `chmod 000` on a quarantine tree contains NOTHING. This was going to be the plan for the
#     reference trees, and the scanner is where the assumption gets tested rather than assumed:
#     running as root, the only real containment is to move the tree off the machine or delete it.
#     The scan says so out loud when it finds an open path while uid 0.
# Add `--allow-prefix DIR` for a tree the cards may legitimately point into; the workflow's own
# directory is allowed by default, since a card citing `knowledge/foo.md` by absolute path is citing
# the workflow, not the answer.
#
# SCOPE: a scanner for INJECTED KNOWLEDGE, not a tree audit. Pointed at a whole project it also
# reports that project's own commits -- correct by its rule, useless as a signal, because the question
# there is not "does this hash resolve" but "is this a copy of the answer", which reference_leak_sweep.sh
# answers by content. Use each for what it sees.
#
# Exit: 0 clean, 1 resolvable address found, 2 bad invocation.

set -uo pipefail

SKILLS_DIR=""
SCAN_ROOT=""
EXTRA_REPOS=()
ALLOW_PREFIXES=()

usage() {
  cat >&2 <<'EOF'
usage: skill_address_scan.sh --skills-dir DIR [--scan-root DIR] [--repo GITDIR]...
                             [--allow-prefix DIR]...

  --skills-dir    expert_skills directory whose cards get injected into agents
  --scan-root     tree to auto-discover git repositories in (default: --skills-dir's ancestor)
  --repo          additional repository to test addresses against (repeatable)
  --allow-prefix  filesystem tree the cards may legitimately name (repeatable). The workflow's own
                  directory is always allowed.
EOF
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skills-dir)   SKILLS_DIR="${2:-}"; shift 2 ;;
    --scan-root)    SCAN_ROOT="${2:-}";  shift 2 ;;
    --repo)         EXTRA_REPOS+=("${2:-}"); shift 2 ;;
    --allow-prefix) ALLOW_PREFIXES+=("${2:-}"); shift 2 ;;
    -h|--help)      usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

[[ -n "$SKILLS_DIR" && -d "$SKILLS_DIR" ]] || { echo "skill_address_scan: --skills-dir must be an existing directory" >&2; exit 2; }
command -v git >/dev/null 2>&1 || { echo "skill_address_scan: git not found; cannot test resolvability" >&2; exit 2; }

if [[ -z "$SCAN_ROOT" ]]; then SCAN_ROOT="$(cd "$SKILLS_DIR/../.." && pwd)"; fi

# ---- assemble the set of repositories an engineer could reach -----------------------------------
repos=()
add_repo() {
  local r="$1"
  [[ -n "$r" ]] || return 0
  r="$(git -C "$r" rev-parse --show-toplevel 2>/dev/null)" || return 0
  local seen
  for seen in "${repos[@]+"${repos[@]}"}"; do [[ "$seen" == "$r" ]] && return 0; done
  repos+=("$r")
}
while IFS= read -r g; do add_repo "$(dirname "$g")"; done < <(find "$SCAN_ROOT" -maxdepth 4 -name .git -print 2>/dev/null)
# AITER_JIT_DIR is named in GEAK_TASK.md, so its repo is reachable by every engineer by construction.
[[ -n "${AITER_JIT_DIR:-}" ]] && add_repo "$AITER_JIT_DIR"
for r in "${EXTRA_REPOS[@]+"${EXTRA_REPOS[@]}"}"; do add_repo "$r"; done

if [[ ${#repos[@]} -eq 0 ]]; then
  # Not an exit: the filesystem pass below needs no repository, and skipping it here is how the
  # git-only version of this script would have missed a bare path leak on a machine with no checkout.
  echo "skill_address_scan: no reachable git repository found under $SCAN_ROOT -- nothing a COMMIT"
  echo "  address could resolve against. The git pass is a PASS by absence, not by checking; re-run"
  echo "  with --repo if the engineer can reach a repository this scan cannot see. The filesystem"
  echo "  pass still runs."
fi

# ---- collect candidate addresses from the cards --------------------------------------------------
# Hex runs of 7..40 chars (git's own abbreviation floor), and slash-bearing branch-like tokens.
# INCLUDES defaults to the card formats. Widen it to audit a whole tree, not just the skills dir:
#   INCLUDES='*.md *.txt *.json *.yaml' skill_address_scan.sh --skills-dir <tree>
: "${INCLUDES:=*.md *.yaml *.yml}"
# `for pat in $INCLUDES` would GLOB: the shell expands *.md against the cwd and the includes silently
# become whatever markdown files happen to sit there, so the scan looks clean by scanning nothing.
# read -a splits on IFS without globbing.
IFS=' ' read -r -a inc_pats <<<"$INCLUDES"
inc_args=(); for pat in "${inc_pats[@]}"; do inc_args+=(--include="$pat"); done
hits=0
# Two passes on purpose. Resolving a token costs a git process, and the naive
# line x token x repo loop spent over two minutes on one tree before finishing. Collect
# "token<TAB>file:line" first, resolve each DISTINCT token once, then report every site that named a
# token which resolved. Same verdict, one git call per unique address.
occ=$(mktemp); trap 'rm -f "$occ" "$toks"' EXIT
# All-digit tokens are dropped. Widened to a tree of profile JSON this scanner drowned in 7-digit
# CYCLE COUNTS: git resolves `1103344` to a unique abbreviated commit in a 555 MB repo, and every ATT
# wavestate dump is full of such numbers. A real abbreviated SHA is all-digits about 1.5% of the time
# ((10/16)^9); that is the miss this buys, and it is worth it, because the alternative is a report
# nobody reads. Stated out loud rather than filtered silently.
grep -rnoE '\b[0-9a-f]{7,40}\b' "${inc_args[@]}" "$SKILLS_DIR" 2>/dev/null |
  awk -F: 'NF>=3 {tok=$NF; if (tok ~ /^[0-9]+$/) next; site=""; for(i=1;i<NF;i++) site=site (i>1?":":"") $i; print tok "\t" site}' >>"$occ"
# Branch-like tokens: skip filesystem paths and filenames, which dominate these documents.
grep -rnoE '\b[A-Za-z0-9._-]+/[A-Za-z0-9._/-]+\b' "${inc_args[@]}" "$SKILLS_DIR" 2>/dev/null |
  awk -F: 'NF>=3 {tok=$NF; if (tok ~ /^\//) next; if (tok ~ /\.(py|md|json|log|yaml|yml|sh|txt|h|cpp|so)$/) next;
                  site=""; for(i=1;i<NF;i++) site=site (i>1?":":"") $i; print tok "\t" site}' >>"$occ"

toks=$(mktemp); cut -f1 "$occ" | sort -u >"$toks"
while IFS= read -r tok; do
  kind=""; where=""
  for repo in "${repos[@]+"${repos[@]}"}"; do
    if [[ "$(git -C "$repo" cat-file -t "$tok" 2>/dev/null)" == "commit" ]]; then kind="commit"; where="$repo"; break; fi
    if git -C "$repo" rev-parse --verify --quiet "refs/heads/$tok" >/dev/null 2>&1 ||
       git -C "$repo" rev-parse --verify --quiet "refs/remotes/$tok" >/dev/null 2>&1; then kind="ref"; where="$repo"; break; fi
  done
  [[ -n "$kind" ]] || continue
  echo "RESOLVABLE ADDRESS  $tok  ($kind in $where)"
  grep -P "^\Q$tok\E\t" "$occ" | cut -f2 | sort -u | sed 's/^/    at  /'
  hits=$((hits + 1))
done <"$toks"

# ---- pass 2: filesystem paths ---------------------------------------------------------------------
# Same rule, different namespace. Absolute paths only -- see the header for why relative ones are the
# engineer's own workspace and are not findings.
WF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALLOW_PREFIXES+=("$WF_DIR")
# Present on every box; naming one carries no information about this run.
SYS_PREFIXES=(/usr /bin /sbin /lib /lib64 /etc /proc /sys /dev /run /boot /opt/rocm /opt/conda /opt/venv)

paths=$(mktemp); trap 'rm -f "$occ" "$toks" "$paths"' EXIT
extract_paths() {  # stdin: grep -rno output; stdout: token<TAB>site
  awk -F: 'NF>=3 {tok=$NF; site=""; for(i=1;i<NF;i++) site=site (i>1?":":"") $i; print tok "\t" site}'
}
PATH_RE='/[A-Za-z0-9._+-]+(/[A-Za-z0-9._+-]+)+'
grep -rnoE "$PATH_RE" "${inc_args[@]}" "$SKILLS_DIR" 2>/dev/null | sed -E 's/[.,;:)`"'"'"']+$//' |
  extract_paths >"$paths"

# Fail-open guard. Both of this subsystem's past bugs reported clean by scanning nothing (a glob that
# ate the includes; a grep flag combination that matched nothing). A regex that silently stops
# matching would do it a third time, so prove the extractor fires on a string known to contain a path
# before believing any clean result it produces.
probe=$(echo "x:1:/opt/rocm/lib/libhsa.so" | extract_paths | cut -f1)
if [[ "$probe" != "/opt/rocm/lib/libhsa.so" ]]; then
  echo "skill_address_scan: the filesystem extractor did not fire on its own probe (got '$probe')." >&2
  echo "  Refusing to report clean: a scan that matches nothing looks exactly like a scan that found" >&2
  echo "  nothing, and this checker has shipped that bug twice already." >&2
  exit 2
fi

path_hits=0
contained=0
while IFS= read -r p; do
  [[ -n "$p" ]] || continue
  skip=0
  for a in "${ALLOW_PREFIXES[@]+"${ALLOW_PREFIXES[@]}"}"; do [[ -n "$a" && "$p" == "$a"/* || "$p" == "$a" ]] && { skip=1; break; }; done
  [[ $skip -eq 1 ]] && continue
  for s in "${SYS_PREFIXES[@]}"; do [[ "$p" == "$s"/* ]] && { skip=1; break; }; done
  [[ $skip -eq 1 ]] && continue
  [[ -e "$p" ]] || continue          # names another machine's layout: a citation, not a door
  if [[ -d "$p" ]]; then readable=$([[ -r "$p" && -x "$p" ]] && echo yes || echo no)
  else                   readable=$([[ -r "$p" ]] && echo yes || echo no); fi
  if [[ "$readable" == "no" ]]; then
    echo "CONTAINED PATH     $p  (exists, unreadable by uid $(id -u) -- enforcement, not trust)"
    contained=$((contained + 1))
    continue
  fi
  echo "RESOLVABLE PATH     $p  ($([[ -d "$p" ]] && echo directory || echo file), readable here)"
  grep -F -- "$p"$'\t' "$paths" | cut -f2 | sort -u | sed 's/^/    at  /'
  path_hits=$((path_hits + 1))
done < <(cut -f1 "$paths" | sort -u)

if [[ $hits -gt 0 || $path_hits -gt 0 ]]; then
  echo
  [[ $hits -gt 0 ]] &&
    echo "FAIL: $hits git address(es) in $SKILLS_DIR resolve inside a repository this run can read."
  if [[ $path_hits -gt 0 ]]; then
    echo "FAIL: $path_hits filesystem path(s) in $SKILLS_DIR open on this machine."
    [[ "$(id -u)" == "0" ]] &&
      echo "  (uid 0: permission bits are not containment here. chmod on the target changes nothing;" &&
      echo "   move it off this machine or delete it.)"
  fi
  echo "Redact the address, keep the mechanism. A skill is reproducible because it describes what to"
  echo "build, not because it says where the built copy is parked."
  exit 1
fi

echo "clean: no address in $SKILLS_DIR resolves -- ${#repos[@]} reachable repo(s), and no absolute"
echo "  path outside the allowed prefixes opens here."
echo "  repos tested: ${repos[*]-(none)}"
echo "  allowed prefixes: ${ALLOW_PREFIXES[*]}"
[[ $contained -gt 0 ]] &&
  echo "  $contained path(s) exist but are unreadable to this uid; they are contained by permissions."
echo "  NOTE this cannot see a repository the engineer reaches by cloning or fetching during the run,"
echo "  nor a path created after the scan, and it says nothing about how much of the ANSWER the card"
echo "  gives away in prose -- a card can be address-free and still be a full specification. That"
echo "  judgement stays human."
exit 0
