
// PRIMARY metric of a worker (kernel_lane.js) result: its final_speedup is already the weighted number
// when the run was workload-aligned, else the geomean. Declared weighting must remain available.
const primSpeedup = (r) => {
  if (!r) return 0;
  if (r.workload_aligned === true) {
    return Number.isFinite(r.final_weighted) && r.final_weighted > 0 ? r.final_weighted : null;
  }
  const g = r.final_speedup != null ? r.final_speedup
          : (r.final_weighted != null ? r.final_weighted
             : (r.final_geomean != null ? r.final_geomean : 0));
  return Number.isFinite(g) ? g : 0;
};

// Gate fields like `smoke` are free strings in the schemas, so an agent legitimately answers
// "PASS - 15/15 parity draws, geomean 0.994" and a `=== 'pass'` test throws the whole run away.
// Match the leading word instead: "PASS - ..." / "passed" gate open, "FAIL"/"did not pass" stay shut.
const says = (v, w) => String(v == null ? '' : v).trim().toLowerCase().startsWith(w);

