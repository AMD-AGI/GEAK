// Select the task-authoritative commit score while retaining the historical
// verified_geomean default for every existing campaign.

export function normalizeCommitScoreField(value = '') {
  const field = String(value || 'verified_geomean').trim();
  if (!['verified_geomean', 'verified_weighted'].includes(field)) {
    throw new Error(`unsupported commit_score_field: ${field}`);
  }
  return field;
}

export function verifiedCommitScore(ver, field = 'verified_geomean') {
  const selected = Number(ver && ver[field]);
  if (Number.isFinite(selected) && selected > 0) return selected;
  const fallback = Number(ver && ver.verified_geomean);
  return Number.isFinite(fallback) && fallback > 0 ? fallback : 0;
}

export function withCommitScore(ver, field = 'verified_geomean') {
  if (!ver) return ver;
  const score = verifiedCommitScore(ver, field);
  if (!(score > 0) || field === 'verified_geomean') return ver;
  return {
    ...ver,
    raw_verified_geomean: ver.verified_geomean,
    verified_geomean: score,
    commit_score_field: field,
  };
}
