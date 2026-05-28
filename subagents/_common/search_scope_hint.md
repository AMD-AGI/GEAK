<!-- Source of truth for the bash filesystem-search-scope rule shared by all
     GEAK subagents.  When edited, also propagate the same block to every
     subagents/*/SYSTEM_PROMPT.md (delimited by the BEGIN/END markers
     below) and to:
       - src/minisweagent/tools/tools.json (bash schema description)
       - src/minisweagent/run/preprocess_v3/subagent.py (_factory_bash)
       - src/minisweagent/tools/bash_command.py (_format_scope_block) -->
<!-- BEGIN GEAK_SEARCH_SCOPE_HINT -->
Filesystem search scope (enforced by the bash tool):

- `find`, `grep -r`, `rg`, `tree`, `du -a`, `ls -R` MUST target one of:
    * `$GEAK_REPO_ROOT` — original user-provided repo (read-only reference)
    * `$GEAK_WORK_DIR`  — current worktree (may differ from `$GEAK_REPO_ROOT`)
    * the current working directory (when it is inside a repo/worktree)
    * `/tmp`, `/var/tmp` — scratch space
    * `/opt` (e.g. `/opt/rocm`), `/usr`, `/etc`, `/var/lib` — system dirs
- Scans rooted at `/`, `/wekafs` (top-level), `/home`, `/root`, `/proc`,
  `/sys`, `/mnt`, `/media`, or any ancestor of the repo are
  auto-rejected.
- For `/opt`, `/usr`, and other system dirs prefer `-maxdepth 3` to keep
  latency low.
- Prefer `rg <pattern> $GEAK_REPO_ROOT` over `find <root> -name <file>`
  for content searches.
- A wall-clock timeout (~10 min, override with `$GEAK_BASH_TIMEOUT_SEC`)
  applies to every bash command; on timeout the entire process group is
  killed. Design searches accordingly.
<!-- END GEAK_SEARCH_SCOPE_HINT -->
