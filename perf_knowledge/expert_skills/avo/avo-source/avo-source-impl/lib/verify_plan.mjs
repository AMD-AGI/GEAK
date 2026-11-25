const finitePositive = (value) => Number.isFinite(Number(value)) && Number(value) > 0;

/**
 * Collapse workers that produced the same candidate bytes from the same base ref.
 * The caller supplies candidateIdentity because hashing the artifact is runtime I/O.
 */
export function dedupeVerifyItems(items) {
  const unique = [];
  const duplicates = [];
  const canonicalByKey = new Map();

  for (const item of items || []) {
    const identity = String(item?.candidateIdentity || '').trim();
    const baseRef = String(item?.startRef || '').trim();
    if (!identity) {
      unique.push(item);
      continue;
    }
    const key = `${baseRef}\0${identity}`;
    const canonical = canonicalByKey.get(key);
    if (!canonical) {
      canonicalByKey.set(key, item);
      unique.push(item);
      continue;
    }
    duplicates.push({ ...item, duplicateOf: canonical.i, dedupeKey: key });
  }

  return { unique, duplicates };
}

/**
 * Pick one clean preliminary winner for full verification.
 * Stable worker-index ordering makes ties deterministic.
 */
export function selectScreeningFinalist(results) {
  return [...(results || [])]
    .filter((item) => item?.ver?.status === 'ok'
      && item.ver.correctness === 'pass'
      && finitePositive(item.ver.verified_geomean))
    .sort((left, right) => {
      const delta = Number(right.ver.verified_geomean)
        - Number(left.ver.verified_geomean);
      return delta || Number(left.i) - Number(right.i);
    })[0] || null;
}
