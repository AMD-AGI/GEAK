// Shared pure workflow contract. Embedded because GEAK workflow JS has no FS/import API.
// Edit here, then run scripts/sync_hardware_context_workflow.py.
// Explicit false overrides a saved opt-in; an old context alone never enables it.
const hardwareOption = A.use_hardware_context != null ? A.use_hardware_context : (A.state || {}).use_hardware_context;
const USE_HARDWARE_CONTEXT = ['true', '1', 'yes'].includes(String(hardwareOption || '').trim().toLowerCase());
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
let HARDWARE_CONTEXT = USE_HARDWARE_CONTEXT ? parseHardwareContext((A.state || {}).hardware_context) : null;
let HARDWARE_CONTEXT_PATH = USE_HARDWARE_CONTEXT ? String(A.hardware_context_path || (A.state || {}).hardware_context_path || '') : '';
function adoptHardwareContext(value, gfx, name, cu, path, declaredSku) {
  if (!USE_HARDWARE_CONTEXT) return;
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
  return { USE_HARDWARE_CONTEXT: true, DEVICE_SKU: HARDWARE_CONTEXT ? HARDWARE_CONTEXT.sku : 'unknown',
    HARDWARE_CONTEXT, HARDWARE_CONTEXT_PATH };
}
function hardwareArgs() {
  return USE_HARDWARE_CONTEXT ? { use_hardware_context: true,
    hardware_context: HARDWARE_CONTEXT, hardware_context_path: HARDWARE_CONTEXT_PATH,
    expected_sku: HARDWARE_CONTEXT ? HARDWARE_CONTEXT.sku : 'unknown' } : { use_hardware_context: false };
}
const initialHardwareContext = USE_HARDWARE_CONTEXT ? parseHardwareContext(A.hardware_context) || HARDWARE_CONTEXT : null;
if (initialHardwareContext) {
  const o = initialHardwareContext.observed;
  adoptHardwareContext(initialHardwareContext, A.expected_gfx || o.gfx,
    A.expected_device_name || o.marketing_name, Number(A.expected_physical_cu_count || o.physical_cu_count));
}
function hardwarePrelude(prompt) {
  if (!USE_HARDWARE_CONTEXT) return '';
  const role = /(?:^|\n)You are the ([\w-]+)\. PHASE=([\w-]+)\./.exec(String(prompt || ''));
  const worker = /^You are Engineer /.test(String(prompt || ''));
  if (!worker && (!role || /^(report|finalize|validate)$/.test(role[2]) ||
      !/^(director|oracle_freezer|tech_lead|profiler|profile_engineer|engineer|deep_engineer|author_engineer|tuning_specialist|system_architect|op_benchmarker)$/.test(role[1]))) return '';
  const setup = role && /^(director|oracle_freezer)$/.test(role[1]) && /^(setup|freeze)$/.test(role[2]);
  return '\n## Shared GEAK hardware context\n' + JSON.stringify(hardwareInputs()) + '\n' +
    'Hardware context was explicitly enabled. Read scripts/hardware_context.md under the GEAK repository root. ' +
    (setup ? 'Run scripts/hardware_context.py in the workload environment after creating EVAL_DIR, write EVAL_DIR/hardware_context.json, and return the complete hardware_context plus hardware_context_path and device_sku. Check any supplied context against this fresh observation. ' : '') +
    'Before numerical planning, validate that file with scripts/hardware_context.py --check FILE. ' +
    'Use observed visible CUs for grid sizing; use verified resource scope for memory/cache planning. Missing facts stay unknown; never promote an ISA default to a product. ' +
    'Export GEAK_HARDWARE_CONTEXT to that file for local and native roofline tools. TraceLens retains native models/calibration; local specification values are planning-only.\n\n';
}
