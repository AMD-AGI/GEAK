const VALIDATE_SCHEMA = obj({
  kernel_name: { type: 'string' },
  director_verified_speedup_geomean: { type: 'number' },
  director_verified_speedup_arithmetic: { type: 'number' },
  director_verified_speedup_weighted: { type: 'number' }, // PRIMARY when workload_aligned
  tech_lead_reported_speedup_geomean: { type: 'number' },
  validation_status: { type: 'string' }, correctness: { type: 'string' },
  // clean | host_bound | unprimed | unknown | not_applicable. Declared so the relaying agent has no
  // reason to drop it -- the gate calls it REQUIRED so a speedup never travels downstream unlabelled.
  timing_basis: { type: 'string' },
  per_case: perCase, applied_to_original: { type: 'string' },
  arbitration_note: { type: 'string' }, final_patch: { type: 'string' },
}, ['director_verified_speedup_geomean', 'validation_status']);

// Warm-start resolver output = experience_store.py `resolve` JSON, verbatim.
const WARMSTART_RESOLVE_SCHEMA = obj({
  read_reason: { type: 'string' }, slug: { type: 'string' },
  // What this run's name derived to vs the page actually served, so a surprising match is visible.
  requested_slug: { type: 'string' }, match_tier: { type: 'string' },
  // store mode only: the key the candidates came from. Declared rather than left to
  // additionalProperties so the agent relaying this JSON has no reason to drop it.
  canonical_id: { type: 'string' },
  // Which of the keyed read's planes answered ("remote" | "local"). Both are addressed by canonical
  // id, so this is the ONLY field telling them apart — and the attest step picks its flags from it.
  read_plane: { type: 'string' },
  // Same kernel, another language: the wrong target_language was passed, not an empty store.
  other_language_pages: { type: 'array', items: { type: 'string' } },
  filtered: obj({
    total: { type: 'number' }, retired: { type: 'number' }, below_min_speedup: { type: 'number' },
    same_direction_collapsed: { type: 'number' }, demoted_by_hint: { type: 'number' },
    // The caveat on every number above: the service pages `--scan` rows, so a saturated read
    // describes a prefix of the page and reads identically to a thin page.
    scanned: { type: 'number' }, scan_saturated: { type: 'boolean' },
  }),
  candidates: {
    type: 'array',
    items: obj({
      rank: { type: 'number' }, slug: { type: 'string' }, speedup: { type: 'number' },
      exp_dir: { type: 'string' }, arch: { type: 'string' },
      // Declared for the same reason canonical_id is: without it the attest step below has no
      // address to write this run's verdict back to.
      session_id: { type: 'string' },
      patch_path: { type: 'string' }, prose_path: { type: 'string' },
      strategy: { type: 'string' }, status: { type: 'string' },
      // direction = the IDEA (one rank each). comparable=false: recorded on a different bench key
      // than rank 1's, so the order between them is a prior only.
      direction: { type: 'string' }, bench_key: { type: 'string' }, comparable: { type: 'boolean' },
    }, ['rank', 'patch_path']),
  },
}, ['read_reason']);
// Warm-start writer output = experience_store.py `write` JSON, verbatim. written=false with reason
// "duplicate_impl" is a success: the store already held this code and counted a reproduction.
const WARMSTART_WRITE_SCHEMA = obj({
  written: { type: 'boolean' }, reason: { type: 'string' }, slug: { type: 'string' },
  dir: { type: 'string' }, speedup: { type: 'number' },
  reproduced: { type: 'string' }, reproductions: { type: 'number' },
  // store mode only: where the same result landed on the KB plane. `replaced` says which of the
  // two outcomes it was — a new candidate under the key, or this patch's own one measured again.
  remote: obj({
    written: { type: 'boolean' }, reason: { type: 'string' }, canonical_id: { type: 'string' },
    session_id: { type: 'string' }, champion: { type: 'boolean' }, replaced: { type: 'boolean' },
  }),
}, ['written']);

