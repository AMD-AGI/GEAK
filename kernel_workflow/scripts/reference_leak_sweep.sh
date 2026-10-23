#!/usr/bin/env bash
# reference_leak_sweep.sh -- find every reachable copy of the reference inside a capability_eval run tree.
#
# The path check in kernel_mega.js only proves known_reference_paths sit outside the run tree. The
# reference can still be reachable as content: a control patch, an older wave's output or state, a
# git bundle, another branch, or a readable repo outside the tree (e.g. AITER_JIT_DIR). So this sweep
# looks for content, not paths. It is a shell script because workflow scripts have no filesystem
# access; run it before launching and after any wave that writes into the tree.
#
# Usage:
#   reference_leak_sweep.sh --tree <run tree root> [--allow <path>]... [--repo <path>]...
#                           [--markers <file>] [--derive <ref> <base>] [--given <path>]...
#
# --repo adds a repository outside <tree> whose refs are also scanned.
#
# Exit 0 = clean. Exit 1 = leaks found (listed on stdout). Exit 2 = bad invocation.
#
# A leak is a file inside <tree> containing a marker: an identifier the reference introduces and the
# frozen baseline never uses. --derive regenerates the list (identifiers added by `git diff base..ref`
# minus those in the baseline); regenerate it whenever the reference moves.

set -uo pipefail

# MARKER_FILE is not cleared here, so an exported MARKER_FILE is honoured.
TREE=""; ALLOW=(); EXTRA_REPOS=()
DERIVE_REF=""; DERIVE_BASE=""; GIVEN=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tree)    TREE="$2"; shift 2 ;;
    --allow)   ALLOW+=("$2"); shift 2 ;;
    --repo)    EXTRA_REPOS+=("$2/.git"); shift 2 ;;
    --markers) MARKER_FILE="$2"; shift 2 ;;
    --derive)  DERIVE_REF="$2"; DERIVE_BASE="$3"; shift 3 ;;
    --given)   GIVEN+=("$2"); shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The marker list lives outside SKILL_DIR: it names the reference's own symbols, so a copy engineers
# can read would hand them the design. A trusted launcher supplies MARKER_FILE out of band; do not
# write its path into this script or launch_args.json. Fails closed: no list, exit 2.
: "${MARKER_FILE:?set MARKER_FILE to the out-of-band reference marker list}"
# Refuse unconditionally if a copy sits under scripts/, even when MARKER_FILE also exists.
if [[ -f "$HERE/reference_leak_markers.txt" ]]; then
  echo "reference_leak_sweep: refusing to use $HERE/reference_leak_markers.txt -- the marker list is" >&2
  echo "verify-only and must not sit inside SKILL_DIR where engineers can read it. Move it to" >&2
  echo "$MARKER_FILE (or set MARKER_FILE) and delete the copy under scripts/." >&2
  exit 2
fi

# --- derive mode: regenerate the marker list -------------------------------------------------------
if [[ -n "$DERIVE_REF" ]]; then
  [[ -d "$DERIVE_REF" && -d "$DERIVE_BASE" ]] || { echo "--derive needs two existing trees" >&2; exit 2; }
  base_rev="$(cd "$DERIVE_BASE" && git rev-parse --short HEAD 2>/dev/null)"
  ref_rev="$(cd "$DERIVE_REF" && git rev-parse --short HEAD 2>/dev/null)"
  [[ -n "$base_rev" && -n "$ref_rev" ]] || { echo "both trees must be git checkouts" >&2; exit 2; }
  added=$(mktemp); baseline=$(mktemp)
  (cd "$DERIVE_REF" && git diff "$base_rev".."$ref_rev" -- . ) \
    | grep '^+' | grep -oE '\b[A-Za-z_][A-Za-z0-9_]{7,}\b' | sort -u > "$added"
  grep -rhoE '\b[A-Za-z_][A-Za-z0-9_]{7,}\b' "$DERIVE_BASE" 2>/dev/null | sort -u > "$baseline"

  # Subtract identifiers from every ref tip of the base repo, not only its checkout: names the team
  # already uses on other branches are not reference-derived, and an over-inclusive list makes the
  # output unreadable. Deduped by SHA and bounded; coverage is reported on stderr.
  refs_scanned=0; refs_total=0
  declare -A dseen=()
  while read -r sha _; do
    [[ -z "$sha" || -n "${dseen[$sha]:-}" ]] && continue
    dseen[$sha]=1; refs_total=$((refs_total+1))
  done < <(git -C "$DERIVE_BASE" for-each-ref --format='%(objectname) %(refname)' 2>/dev/null)
  # NOT `for ... done | sort -u >> "$baseline"`. A loop on the left of a pipe runs in a SUBSHELL, so
  # every refs_scanned increment is discarded when it exits and the counter reads 0 no matter how many
  # trees were harvested. That misfires the "NOT harvested / the list may over-report" warning below on
  # a run that in fact harvested everything -- and that warning is the operator's only signal that the
  # subtraction was incomplete, so a version of it that cries wolf is worse than none. Redirect per
  # iteration instead and keep the loop in this shell.
  for sha in "${!dseen[@]}"; do
    [[ $refs_scanned -ge ${REF_SCAN_MAX_TREES:-2000} ]] && break
    refs_scanned=$((refs_scanned+1))
    git -C "$DERIVE_BASE" grep -I -h -oE '\b[A-Za-z_][A-Za-z0-9_]{7,}\b' "$sha" 2>/dev/null >> "$baseline"
  done
  sort -u -o "$baseline" "$baseline"
  echo "--derive: subtracted the base working tree plus $refs_scanned of $refs_total unique ref trees." >&2
  [[ $refs_scanned -lt $refs_total ]] && echo "--derive: $((refs_total - refs_scanned)) ref tree(s) NOT harvested (REF_SCAN_MAX_TREES); the list may over-report." >&2
  [[ $refs_total -eq 0 ]] && echo "--derive: the base repo has no refs, so only its working tree was subtracted. If you stripped its refs for containment, derive the list BEFORE stripping." >&2

  # Subtract what the run is legitimately given (--given <path>, repeatable): task text, knowledge and
  # roles name flags and path markers that the reference also adds, and those must not count as leaks.
  # E.g. --given <task dir> --given <knowledge dir> --given <roles dir>.
  if [[ ${#GIVEN[@]} -gt 0 ]]; then
    given_n=0
    for g in "${GIVEN[@]}"; do
      [[ -e "$g" ]] || { echo "--given: no such path, ignored: $g" >&2; continue; }
      given_n=$((given_n+1))
      grep -rhoE '\b[A-Za-z_][A-Za-z0-9_]{7,}\b' "$g" 2>/dev/null >> "$baseline"
    done
    sort -u -o "$baseline" "$baseline"
    echo "--derive: also subtracted identifiers appearing in $given_n path(s) the run is given." >&2
  else
    echo "--derive: no --given paths. Tokens the task text and knowledge cards name on purpose (the opt-in flags, path=SCATTERED, ...) will be proposed as markers and must be removed by hand before installing." >&2
  fi

  comm -23 "$added" "$baseline"
  rm -f "$added" "$baseline"
  exit 0
fi

[[ -n "$TREE" && -d "$TREE" ]] || { echo "--tree <existing dir> required" >&2; exit 2; }
[[ -f "$MARKER_FILE" ]] || { echo "marker file not found: $MARKER_FILE" >&2; exit 2; }

TREE="$(cd "$TREE" && pwd)"
mapfile -t MARKERS < <(grep -vE '^\s*(#|$)' "$MARKER_FILE")
[[ ${#MARKERS[@]} -gt 0 ]] || { echo "marker file is empty: $MARKER_FILE" >&2; exit 2; }

# Markers are identifiers, so match them as whole identifiers, not substrings: a marker appearing as
# the tail or head of a longer identifier IS a different identifier.
#
# One alternation rather than `-f patternfile`: GNU grep's -P accepts only a single pattern and
# otherwise matches nothing silently. The self-test below guards against that.
pattern="(?<![A-Za-z0-9_])(?:$(printf '%s|' "${MARKERS[@]}" | sed 's/|$//'))(?![A-Za-z0-9_])"

# Self-test: prove the engine matches a synthetic marker and not the same marker inside a longer
# identifier before trusting a clean result.
probe="${MARKERS[0]}"
if ! printf ' %s \n' "$probe" | grep -q -P "$pattern" 2>/dev/null; then
  echo "reference_leak_sweep: grep -P cannot evaluate the marker pattern on this system; the sweep" >&2
  echo "  would report clean by matching nothing. Refusing to run." >&2
  exit 2
fi
if printf 'zz%szz\n' "$probe" | grep -q -P "$pattern" 2>/dev/null; then
  echo "reference_leak_sweep: identifier boundaries are not being enforced. Refusing to run." >&2
  exit 2
fi

# Scope: files a leak can be copied out of (source, patches, bundles); not measurement JSON, reports
# or __pycache__, which may legitimately name counters. Override with --ext.
: "${EXTS:=py diff patch bundle h hpp cpp cc cu hip s asm sh}"
find_args=(); first=1
for e in $EXTS; do
  if [[ $first -eq 1 ]]; then find_args+=(-name "*.$e"); first=0; else find_args+=(-o -name "*.$e"); fi
done

# Exclude this scanner and its marker list, which quote markers by construction.
hits=$(find "$TREE" -type f \( "${find_args[@]}" \) -not -path '*/__pycache__/*' \
         -not -name 'reference_leak_sweep.sh' -not -name 'reference_leak_markers.txt' -print0 2>/dev/null \
       | xargs -0 -r grep -l -P "$pattern" 2>/dev/null | sort -u)

# --allow trees the run is supposed to read (frozen baseline, isolated control workspace).
leaks=()
while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  skip=0
  for a in "${ALLOW[@]:-}"; do
    [[ -z "$a" ]] && continue
    a="$(cd "$a" 2>/dev/null && pwd || echo "$a")"
    [[ "$f" == "$a" || "$f" == "$a/"* ]] && { skip=1; break; }
  done
  [[ $skip -eq 0 ]] && leaks+=("$f")
done <<< "$hits"

# --- REFS ---
#
# A file sweep misses branches that are not checked out, so grep the tree at every ref of every
# repository under <tree> (deduped by commit SHA). NOT covered: unreferenced history, dangling objects,
# and repos reachable outside <tree> unless passed with --repo; verify's byte-identity check covers
# the rest.
gitdirs=()
while IFS= read -r g; do [[ -n "$g" ]] && gitdirs+=("$g"); done < <(
  find "$TREE" -name .git -maxdepth 6 -print 2>/dev/null | sort -u)
for r in "${EXTRA_REPOS[@]:-}"; do [[ -n "$r" ]] && gitdirs+=("$r"); done

ref_leaks=(); ref_skipped=0
# Exclude this scanner and its marker list at ref tips too.
pathspec=(); for e in $EXTS; do pathspec+=("*.$e"); done
pathspec+=(':!*reference_leak_sweep.sh' ':!*reference_leak_markers.txt')

for g in "${gitdirs[@]:-}"; do
  [[ -z "$g" ]] && continue
  repo="$(dirname "$g")"; [[ -d "$g" ]] || repo="$(dirname "$g")"
  repo="$(cd "$repo" 2>/dev/null && pwd)" || continue
  skip=0
  for a in "${ALLOW[@]:-}"; do
    [[ -z "$a" ]] && continue
    a="$(cd "$a" 2>/dev/null && pwd || echo "$a")"
    [[ "$repo" == "$a" || "$repo" == "$a/"* ]] && { skip=1; break; }
  done
  [[ $skip -eq 1 ]] && continue

  # Dedupe by SHA over every ref plus HEAD (a detached HEAD is not listed by for-each-ref).
  declare -A seen_sha=()
  while read -r sha rname; do
    [[ -z "$sha" ]] && continue
    [[ -n "${seen_sha[$sha]:-}" ]] && continue
    seen_sha[$sha]="$rname"
  done < <( { git -C "$repo" for-each-ref --format='%(objectname) %(refname)' 2>/dev/null
              printf '%s HEAD\n' "$(git -C "$repo" rev-parse HEAD 2>/dev/null)"; } )

  n=0
  for sha in "${!seen_sha[@]}"; do
    n=$((n+1))
    if [[ $n -gt ${REF_SCAN_MAX_TREES:-200} ]]; then ref_skipped=$((ref_skipped+1)); break; fi
    while IFS= read -r line; do
      [[ -n "$line" ]] && ref_leaks+=("$repo  ${seen_sha[$sha]} (${sha:0:8})  ${line#*:}")
    done < <(git -C "$repo" grep -I -l -P "$pattern" "$sha" -- "${pathspec[@]}" 2>/dev/null)
  done
  unset seen_sha
done

if [[ ${#ref_leaks[@]} -gt 0 ]]; then
  echo "REFERENCE LEAK IN GIT REFS: ${#ref_leaks[@]} blob(s) carry reference-only markers at a ref tip."
  echo "These are invisible to a working-tree grep and reachable with one \`git checkout\`. Remove the"
  echo "ref (or move the whole .git outside the run tree) before launching."
  printf '  %s\n' "${ref_leaks[@]}" | sort -u | head -60
  [[ ${#ref_leaks[@]} -gt 60 ]] && echo "  ... $(( ${#ref_leaks[@]} - 60 )) more"
fi
[[ $ref_skipped -gt 0 ]] && echo "NOTE $ref_skipped repo(s) had more unique ref trees than REF_SCAN_MAX_TREES=${REF_SCAN_MAX_TREES:-200}; the remainder were NOT scanned."

if [[ ${#leaks[@]} -eq 0 && ${#ref_leaks[@]} -eq 0 ]]; then
  echo "LEAK SWEEP clean: ${#MARKERS[@]} reference markers, 0 hits inside $TREE outside the allow-list."
  echo "Covered: working-tree files, and the tree at every ref tip of ${#gitdirs[@]} repository(ies)."
  echo "NOT covered: unreferenced history, dangling objects, and repos outside <tree> not passed with"
  echo "--repo. Only verify's byte-identity check covers those."
  exit 0
elif [[ ${#leaks[@]} -eq 0 ]]; then
  exit 1
fi

echo "REFERENCE LEAK: ${#leaks[@]} file(s) inside $TREE carry reference-only markers."
echo "Each of these is a copy of the answer that an engineer reaches with one grep. Move them outside"
echo "the tree (a stale path in an old report is harmless once it resolves to nothing), or --allow them"
echo "if the run is genuinely supposed to read them."
# NAME THE MARKER, NOT JUST THE FILE: print which markers matched, so a hit can be judged without
# opening the file (the containment gate forbids opening them).
for f in "${leaks[@]}"; do
  echo "  $f"
  grep -o -P "$pattern" "$f" 2>/dev/null | sort | uniq -c | sort -rn | head -12 |
    while read -r n m; do echo "      ${n}x  $m"; done
done
exit 1
