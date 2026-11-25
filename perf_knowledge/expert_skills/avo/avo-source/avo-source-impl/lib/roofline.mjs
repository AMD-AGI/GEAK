// Pure, testable roofline model for AVO (no I/O, no runtime globals).
//
// This turns AVO's objective from a PURELY RELATIVE one ("faster than the running
// best") into a HARDWARE-ANCHORED one ("what % of the roofline are we at, and which
// ceiling binds us"). Approaching the roofline — not just beating the last version —
// is what lets the agent surpass an expert and stop at the real limit.
//
// Inputs are per-case telemetry {latency_ms, flops, bytes, dtype}; everything is
// derived from the CDNA3/MI300X peak table (kernel_knowledge/hardware/cdna3_mi300/
// peak_tables.md). All fields degrade gracefully to null when flops/bytes are
// missing, so a kernel whose COMMANDMENT does not yet emit FLOPs/bytes keeps working
// on pure speedup (the roofline signal is simply inert).
//
// Unit-tested by tests/roofline.test.mjs (run: node tests/roofline.test.mjs).

// --- Hardware peak table ---------------------------------------------------
// Peak compute in FLOP/s by dtype, and HBM bandwidth in B/s. Source: peak_tables.md
// (MI300X gfx942 @ 2.1 GHz, 304 CU). MI325X shares compute; only hbm_bw differs.
export const PEAKS = {
  gfx942: {
    label: 'MI300X',
    compute: {
      fp64: 81.7e12, fp64_matrix: 163.4e12,
      fp32: 163.4e12, tf32: 653.7e12,
      fp16: 1307.4e12, bf16: 1307.4e12,
      fp8: 2614.9e12, int8: 2614.9e12,
    },
    hbm_bw: 5.325e12,            // B/s
  },
  gfx942_mi325: {
    label: 'MI325X',
    compute: {
      fp64: 81.7e12, fp64_matrix: 163.4e12,
      fp32: 163.4e12, tf32: 653.7e12,
      fp16: 1307.4e12, bf16: 1307.4e12,
      fp8: 2614.9e12, int8: 2614.9e12,
    },
    hbm_bw: 6.0e12,
  },
  // CDNA4 / MI355X. Source: perf_knowledge/hardware/cdna4_mi350/peak_tables.md
  gfx950: {
    label: 'MI355X',
    compute: {
      fp64: 78.6e12, fp32: 157.3e12,
      fp16: 2.5e15, bf16: 2.5e15,
      fp8: 5.0e15, int8: 5.0e15,
      fp6: 10.0e15, fp4: 10.0e15,
      mxfp8: 5.0e15, mxfp6: 10.0e15, mxfp4: 10.0e15,
    },
    hbm_bw: 8.0e12,
  },
};
PEAKS.gfx950_mi355 = PEAKS.gfx950;

const TF = 1e12;   // FLOP/s per TFLOP/s
const GB = 1e9;    // B/s per GB/s

function hw(name) { return PEAKS[name] || PEAKS.gfx942; }
function peakCompute(dtype, name) {
  const c = hw(name).compute;
  return c[String(dtype || 'fp16').toLowerCase()] || c.fp16;
}

/** Roofline ridge point (arithmetic intensity, FLOP/byte) for a dtype. */
export function ridgePoint(dtype = 'fp16', hwName = 'gfx942') {
  return peakCompute(dtype, hwName) / hw(hwName).hbm_bw;
}

/**
 * Analyze a single case's telemetry against the roofline.
 *
 * @param {object} c     {latency_ms, flops, bytes, dtype?}
 * @param {object} opts  {hw?, basementThreshold?, nearRooflineFrac?}
 * @returns {object} analysis (fields null when telemetry is insufficient)
 */
export function analyzeCase(c, opts = {}) {
  const hwName = opts.hw || 'gfx942';
  const basementThreshold = opts.basementThreshold != null ? opts.basementThreshold : 0.35;
  const nearFrac = opts.nearRooflineFrac != null ? opts.nearRooflineFrac : 0.80;
  // Max believable fraction_of_roofline. Achieved cannot exceed the hardware ceiling,
  // so anything above 1 + this tolerance means the flops/bytes are untrustworthy.
  const fracTol = opts.fractionTolerance != null ? opts.fractionTolerance : 0.05;
  const dtype = (c && c.dtype) || opts.dtype || 'fp16';

  const latency_s = c && Number(c.latency_ms) > 0 ? Number(c.latency_ms) / 1000 : null;
  const flops = c && Number.isFinite(Number(c.flops)) && Number(c.flops) > 0 ? Number(c.flops) : null;
  // A2: prefer rocprof-MEASURED HBM bytes over the analytic estimate — analytic bytes
  // can mis-classify a cache-friendly kernel (counts traffic that never hits HBM).
  const measured = c && Number.isFinite(Number(c.measured_bytes)) && Number(c.measured_bytes) > 0
    ? Number(c.measured_bytes) : null;
  const analytic = c && Number.isFinite(Number(c.bytes)) && Number(c.bytes) > 0 ? Number(c.bytes) : null;
  const bytes = measured != null ? measured : analytic;
  const bytes_source = measured != null ? 'measured' : (analytic != null ? 'analytic' : null);
  const counters = (c && c.counters && typeof c.counters === 'object') ? c.counters : null;

  const pc = peakCompute(dtype, hwName);
  const bw = hw(hwName).hbm_bw;
  const ridge_ai = pc / bw;

  const base = {
    name: c && c.name, dtype,
    ai: null, achieved_tflops: null, achieved_gbps: null,
    peak_compute_tflops: pc / TF, peak_bw_gbps: bw / GB, ridge_ai,
    roofline_tflops: null, fraction_of_roofline: null,
    mem_frac: null, cmp_frac: null,
    regime: 'unknown', near_roofline: false,
    potential_speedup: null, headroom: null,
    telemetry_invalid: false,
    bytes_source, counters,
  };
  if (latency_s == null || flops == null || bytes == null) return base;

  const ai = flops / bytes;
  const achieved = flops / latency_s;                 // FLOP/s
  const achieved_gbps = bytes / latency_s / GB;
  const roofline = Math.min(pc, ai * bw);             // attainable FLOP/s at this AI
  const fraction = roofline > 0 ? achieved / roofline : null;
  const mem_frac = achieved_gbps / (bw / GB);
  const cmp_frac = achieved / pc;

  // Telemetry sanity gate: fraction_of_roofline must be <= 1 (achieved cannot beat
  // the hardware ceiling). A value meaningfully above 1 means the flops/bytes
  // derivation is wrong (e.g. theoretical-FLOP overcount) or latency was mismeasured.
  // Treat such a case as INVALID telemetry — degrade exactly like missing telemetry
  // (null fraction/regime) so it neither fires the roofline early-stop nor distorts
  // steering, and flag it so the caller can warn. A within-tolerance >1 is kept as
  // measurement noise.
  if (fraction == null || !Number.isFinite(fraction) || fraction > 1 + fracTol) {
    return {
      ...base,
      ai, achieved_tflops: achieved / TF, achieved_gbps,
      roofline_tflops: roofline / TF, mem_frac, cmp_frac,
      telemetry_invalid: fraction != null && Number.isFinite(fraction) && fraction > 1 + fracTol,
    };
  }

  const regime = classifyRegime({ ai, ridge_ai, mem_frac, cmp_frac, basementThreshold });
  const potential = achieved > 0 ? roofline / achieved : null;

  return {
    ...base,
    ai, achieved_tflops: achieved / TF, achieved_gbps,
    roofline_tflops: roofline / TF, fraction_of_roofline: fraction,
    mem_frac, cmp_frac, regime,
    near_roofline: fraction >= nearFrac,
    potential_speedup: potential,
    headroom: Math.max(0, 1 - fraction),
    telemetry_invalid: false,
  };
}

/**
 * Which ceiling binds this point. 'basement' = far below BOTH ceilings (so the
 * limiter is latency / occupancy / launch overhead, not BW or FLOPs).
 */
export function classifyRegime({ ai, ridge_ai, mem_frac, cmp_frac, basementThreshold = 0.35 }) {
  if (![mem_frac, cmp_frac].every(Number.isFinite)) return 'unknown';
  if (Math.max(mem_frac, cmp_frac) < basementThreshold) return 'basement';
  return ai >= ridge_ai ? 'compute-bound' : 'memory-bound';
}

const _geomean = (xs) => {
  const v = (xs || []).map(Number).filter((s) => Number.isFinite(s) && s > 0);
  if (!v.length) return null;
  return Math.exp(v.reduce((a, s) => a + Math.log(s), 0) / v.length);
};

/**
 * Aggregate per-case analyses into a steering summary for the loop/prompts.
 *
 * dominant_regime / worst_case point at the BIGGEST remaining opportunity (the case
 * with the most potential speedup), so the operator attacks the binding constraint
 * where it matters most — not an already-saturated shape.
 *
 * @param {Array} perCase  raw telemetry [{name, latency_ms, flops, bytes, dtype}]
 * @param {object} opts    forwarded to analyzeCase (hw, thresholds)
 */
export function rooflineSummary(perCase, opts = {}) {
  const analyses = (perCase || []).map((c) => analyzeCase(c, opts));
  const valid = analyses.filter((a) => a.fraction_of_roofline != null);
  // Cases whose telemetry put them impossibly above the ceiling (excluded above).
  const n_invalid = analyses.filter((a) => a && a.telemetry_invalid).length;
  if (!valid.length) {
    return { ok: false, geomean_fraction: null, dominant_regime: 'unknown',
      worst_case: null, n_valid: 0, n_invalid, n_total: analyses.length, analyses };
  }
  const geomean_fraction = _geomean(valid.map((a) => a.fraction_of_roofline));
  // Worst case = most potential speedup left (largest roofline/achieved).
  const worst = valid.reduce((w, a) =>
    (!w || (a.potential_speedup || 0) > (w.potential_speedup || 0)) ? a : w, null);
  return {
    ok: true,
    geomean_fraction,
    dominant_regime: worst ? worst.regime : 'unknown',
    worst_case: worst ? { name: worst.name, regime: worst.regime,
      fraction_of_roofline: worst.fraction_of_roofline, potential_speedup: worst.potential_speedup } : null,
    n_valid: valid.length, n_invalid, n_total: analyses.length, analyses,
  };
}

/**
 * Map a regime to a concrete, ranked lever list + the matching specialist track
 * (algorithm | memory | compute | host_runtime — same taxonomy as team_workflow).
 * This is the deterministic core of gap-driven steering; the prompt turns it into
 * one focused change.
 */
export function recommendLevers(regime) {
  switch (regime) {
    case 'memory-bound':
      return { specialty: 'memory', levers: [
        'fuse adjacent ops to cut HBM round-trips',
        'vectorize + coalesce global loads/stores (128B transactions)',
        'stage reused data in LDS; maximize reuse before eviction',
        'XCD-aware tiling for L2 (4 MiB/XCD) + Infinity Cache (256 MiB) locality',
        'reduce bytes moved: lower precision where safe, or recompute vs reload',
        'async global->LDS copy to overlap load with compute',
      ] };
    case 'compute-bound':
      return { specialty: 'compute', levers: [
        'raise MFMA/matrix-core utilization; pick the right MFMA shape for register footprint',
        'cut VALU / packing / conversion overhead around the MMA loop',
        'balance register pressure vs occupancy (do not spill)',
        'software-pipeline to hide MFMA latency (issue next while current accumulates)',
        'remove LDS bank conflicts feeding the matrix cores',
      ] };
    case 'basement':
      return { specialty: 'host_runtime', levers: [
        'raise wave occupancy (tune grid/block, LDS & register budget)',
        'collapse kernel dispatches / fuse launches; cut per-launch overhead',
        'double-buffer to hide global-load latency',
        'persistent / CUDA-graph style launch to remove host gaps',
        'rebalance work across all CUs / XCDs of this device (avoid idle CUs, tail effects)',
      ] };
    default:
      return { specialty: 'algorithm', levers: [
        'capture FLOPs/bytes + rocprof counters so the roofline regime is known',
        'establish strong tiling / work decomposition structure first',
      ] };
  }
}

// Specialist track per regime + the orthogonal cover order (dominant ceiling first,
// then the next likely ceiling, ...). Used by operator=team to put the RIGHT
// specialist on the binding constraint and spread the rest over distinct ceilings.
const _SPECIALIST_ORDER = {
  'memory-bound':  ['memory', 'compute', 'host_runtime', 'algorithm'],
  'compute-bound': ['compute', 'memory', 'host_runtime', 'algorithm'],
  'basement':      ['host_runtime', 'memory', 'compute', 'algorithm'],
  'unknown':       ['algorithm', 'memory', 'compute', 'host_runtime'],
};

/**
 * Assign K specialist tracks for a Team round given the current binding regime:
 * the dominant-regime specialist first (attack the ceiling we're actually hitting),
 * then orthogonal coverage of the other ceilings (which one binds next shifts after
 * each commit). Cycles if k > 4. Graceful: an unknown/absent regime → algorithm-first
 * (today's generic order).
 *
 * @param {string} regime  'memory-bound'|'compute-bound'|'basement'|'unknown'
 * @param {number} k       number of workers / directions
 * @returns {string[]}     length k, e.g. ['memory','compute','host_runtime']
 */
export function assignSpecialists(regime, k) {
  const order = _SPECIALIST_ORDER[regime] || _SPECIALIST_ORDER.unknown;
  const n = Math.max(1, k | 0);
  const out = [];
  for (let i = 0; i < n; i++) out.push(order[i % order.length]);
  return out;
}

export const _internals = { _geomean, peakCompute, hw, TF, GB, _SPECIALIST_ORDER };
