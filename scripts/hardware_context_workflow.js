// Shared pure workflow contract. Embedded because GEAK workflow JS has no FS/import API.
// Edit here, then run scripts/sync_hardware_context_workflow.py.
function parseHardwareContext(value) {
  if (!value) return null;
  const c = typeof value === 'string' ? JSON.parse(value) : value;
  const o = c.observed || {};
  if (c.schema_version !== 1 || !/^gfx[0-9a-f]+$/.test(o.gfx || '') ||
      !Number.isInteger(o.physical_cu_count) || o.physical_cu_count <= 0 ||
      !['matched', 'conflict', 'unknown'].includes(c.status) ||
      !/^[a-f0-9]{64}$/.test(c.catalog_sha256 || '') ||
      c.may_gate !== false || c.analysis_role !== 'planning') {
    throw new Error('invalid hardware_context: use scripts/hardware_context.py');
  }
  if (c.status === 'matched') {
    const s = c.specification || {};
    if (!c.sku || c.sku === 'unknown' || c.sku !== c.candidate_sku ||
        s.arch !== o.gfx || s.cus !== o.physical_cu_count || s.product !== o.marketing_name) {
      throw new Error('hardware_context SKU/resource mismatch');
    }
  } else if (c.sku !== 'unknown' || c.specification !== null) {
    throw new Error('unresolved hardware_context must not provide product peaks');
  }
  return c;
}
let HARDWARE_CONTEXT = parseHardwareContext((A.state || {}).hardware_context);
let HARDWARE_CONTEXT_PATH = String(A.hardware_context_path || (A.state || {}).hardware_context_path || '');
function adoptHardwareContext(value, gfx, name, cu, path, declaredSku) {
  const next = parseHardwareContext(value) || HARDWARE_CONTEXT;
  if (next) {
    const o = next.observed;
    if (o.gfx !== gfx || o.physical_cu_count !== cu || (name && o.marketing_name !== name)) {
      throw new Error('hardware_context disagrees with setup GPU identity');
    }
    if (HARDWARE_CONTEXT) {
      const previous = HARDWARE_CONTEXT.observed;
      for (const k of ['gfx', 'marketing_name', 'physical_cu_count', 'compute_partition', 'memory_partition', 'memory_bytes']) {
        if (previous[k] != null && previous[k] !== o[k]) throw new Error('hardware_context changed across workflow handoff: ' + k);
      }
      if (HARDWARE_CONTEXT.catalog_sha256 !== next.catalog_sha256 || HARDWARE_CONTEXT.sku !== next.sku) {
        throw new Error('hardware_context catalog or SKU changed across workflow handoff');
      }
    }
  }
  if (A.expected_sku && A.expected_sku !== 'unknown' && (!next || next.sku !== A.expected_sku)) {
    throw new Error('expected_sku disagrees with hardware_context');
  }
  if (declaredSku && (!next || declaredSku !== next.sku)) {
    throw new Error('setup device_sku disagrees with hardware_context');
  }
  HARDWARE_CONTEXT = next;
  HARDWARE_CONTEXT_PATH = String(path || HARDWARE_CONTEXT_PATH);
}
function hardwareInputs() {
  return { DEVICE_SKU: HARDWARE_CONTEXT ? HARDWARE_CONTEXT.sku : 'unknown',
    HARDWARE_CONTEXT, HARDWARE_CONTEXT_PATH };
}
const initialHardwareContext = parseHardwareContext(A.hardware_context) || HARDWARE_CONTEXT;
if (initialHardwareContext) {
  const o = initialHardwareContext.observed;
  adoptHardwareContext(initialHardwareContext, A.expected_gfx || o.gfx,
    A.expected_device_name || o.marketing_name, Number(A.expected_physical_cu_count || o.physical_cu_count));
}
function hardwarePrelude() {
  return '\n## Shared GEAK hardware context\n' + JSON.stringify(hardwareInputs()) + '\n' +
    'Read scripts/hardware_context.md under the GEAK repository root. This applies to every language and optimization role, independently of expert-skill switches. ' +
    'Setup/Freeze must run scripts/hardware_context.py in the workload environment, write EVAL_DIR/hardware_context.json, and return hardware_context plus hardware_context_path. ' +
    'Before numerical planning, validate that file with scripts/hardware_context.py --check FILE. ' +
    'Use observed visible CUs for grid sizing; use verified resource scope for memory/cache planning. Missing facts stay unknown; never promote an ISA default to a product. ' +
    'Export GEAK_HARDWARE_CONTEXT to that file for local and native roofline tools. TraceLens retains native models/calibration; local specification values are planning-only.\n\n';
}
