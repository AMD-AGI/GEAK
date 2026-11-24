/**
 * Paths that unittest/harness rewrite in avo_repo and must not fail
 * checkoutAvoRepo's clean-worktree guard.
 *
 * Prefixes: directory trees (Kimi-style overlay rebuilt from kernel_src).
 * Files: exact repo-relative paths (GLM 0917 `_runs.jsonl` append-only log).
 */
export const HARNESS_GENERATED_PREFIXES = ['_cand_overlay', '_priv_overlay'];
export const HARNESS_GENERATED_FILES = ['_runs.jsonl'];

export function isHarnessGeneratedPath(relPath) {
  const p = String(relPath || '').replace(/^\.\//, '');
  if (!p) return false;
  if (HARNESS_GENERATED_FILES.includes(p)) return true;
  const base = p.split('/').pop();
  if (HARNESS_GENERATED_FILES.includes(base)) return true;
  return HARNESS_GENERATED_PREFIXES.some((prefix) =>
    p === prefix || p.startsWith(`${prefix}/`) || p.startsWith(`${prefix}_`));
}
