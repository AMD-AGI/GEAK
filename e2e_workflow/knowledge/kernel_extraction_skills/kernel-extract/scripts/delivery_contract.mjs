// Copyright (c) 2026 Advanced Micro Devices, Inc. Apache-2.0.
// Standalone adaptation, 2026-10-12: extracted pure schemas/evidence gates.
// No workflow runner or external repository is loaded.
const obj = (props, required) => ({ type: 'object', properties: props, required: required || [], additionalProperties: true });
const arrObj = { type: 'array', items: { type: 'object', additionalProperties: true } };
const arrStr = { type: 'array', items: { type: 'string' } };

const SMOKE_STATUS = { type: 'string', enum: ['pass', 'fail'] };
// Large machine verdicts travel as immutable task-local files, not LLM summaries.
const SELECTION_VERDICT_REF_SCHEMA = obj({
  path: { type: 'string' }, sha256: { type: 'string' },
}, ['path', 'sha256']);
const EXTRACTION_SELECTION_FIELDS = {
  selection_validation_ref: SELECTION_VERDICT_REF_SCHEMA,
  source_delivery_ref: SELECTION_VERDICT_REF_SCHEMA,
  joint_ut: { type: 'object', additionalProperties: true },
  phase_selection_validations: { type: 'array', items: obj({
    regime: { type: 'string' }, target_callable: { type: 'string' }, seam_candidates: arrObj,
    selection_validation: { type: 'object', additionalProperties: true },
    selection_validation_ref: SELECTION_VERDICT_REF_SCHEMA,
  }, ['regime', 'target_callable']) },
};

const EXTRACT_OP_SCHEMA = obj({
  ...EXTRACTION_SELECTION_FIELDS,
  short_name: { type: 'string' }, op_kind: { type: 'string' }, editable: { type: 'boolean' },
  task_dir: { type: 'string' }, shapes: { type: 'object', additionalProperties: true },
  workload_path: { type: 'string' }, // per-(shape,dtype) weighted workload model for this kernel (optional)
  dtype: { type: 'string' }, synthesized: { type: 'boolean' }, regimes_captured: arrStr,
  candidate_backends: arrStr, reference_io_sha256: { type: 'string' },
  target_callable: { type: 'string' }, // module:attr rebind seam for an authored kernel ('' if none)
  baseline_callable: { type: 'string' }, // module:attr of the FROZEN real online kernel (the speedup denominator)
  baseline_frozen: { type: 'boolean' }, // true only when baseline_callable resolves outside the task dir
  device_kernel: { type: 'string' },
  seam_candidates: arrObj,
  selection_validation: { type: 'object', additionalProperties: true },
  smoke: SMOKE_STATUS, smoke_detail: { type: 'string' }, notes: { type: 'string' },
}, ['op_kind', 'task_dir', 'smoke']);

// A replay bootstrap is an intermediate artifact, never a completed UT or a
// performance result. The full extractor below still runs every normal gate.
const JOINT_BOOTSTRAP_SCHEMA = obj({
  ready: { type: 'boolean' }, task_dir: { type: 'string' },
  report_path: { type: 'string' }, phase_results: arrObj, notes: { type: 'string' },
}, ['ready', 'task_dir', 'report_path', 'phase_results']);


const EXTRACT_SCHEMA = obj({
  ...EXTRACTION_SELECTION_FIELDS,
  short_name: { type: 'string' }, editable: { type: 'boolean' }, task_dir: { type: 'string' },
  source_path_in_sglang: { type: 'string' }, target_callable: { type: 'string' },
  num_cases: { type: 'number' }, regimes_captured: arrStr, candidate_backends: arrStr,
  build: { type: 'boolean' }, unittest_smoke: SMOKE_STATUS, smoke_detail: { type: 'string' },
  // the baseline leg is an ENVIRONMENT (baseline_overlay/ = a frozen CURRENT_OVERLAY snapshot), and
  // candidate_bind is the ONE entry layered on top of it to make the candidate leg.
  candidate_bind: { type: 'object', additionalProperties: true },
  baseline_overlay: { type: 'string' },
  baseline_frozen: { type: 'boolean' }, // true only when baseline_overlay/ was seeded AND candidate_bind is declared
  device_kernel: { type: 'string' },
  seam_candidates: arrObj,
  selection_validation: { type: 'object', additionalProperties: true },
  reference_io_sha256: { type: 'string' }, notes: { type: 'string' },
}, ['editable', 'task_dir', 'unittest_smoke']);


const CALLABLE_SPEC_RX = /^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$/;
const validCallableSpec = (s) => CALLABLE_SPEC_RX.test(String(s || '').trim());
// parse_profile.short_name truncates display names at 60 chars; a declared name that hit the limit
// is our own doing, so a prefix match at it is accepted rather than read as a mismatch.
const SHORT_NAME_LIMIT = 60;
// Balanced groups are removed innermost-first until the symbol stops changing. One greedy pass
// spans from the FIRST opener to the LAST closer, so `k<a>(t<b>)` loses the '(' that marks the end
// of the name, and a nested `k<pair<a,b>>` leaves the leftover delimiter behind.
const stripBalanced = (symbol, opener, closer) => {
  const esc = (c) => c.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const rx = new RegExp(`${esc(opener)}[^${esc(opener)}${esc(closer)}]*${esc(closer)}`, 'g');
  let text = symbol;
  for (let previous = null; previous !== text;) {
    previous = text;
    text = text.replace(rx, ' ');
  }
  return text;
};
// Must stay identical to canonical_kernel_name in kernel_selection.py -- the JS gate and the Python
// verdict compare the same two symbols, so any drift lets a kernel pass one side and fail the other.
// Parentheses are stripped as balanced groups rather than by cutting at the first '(': ROCm spells
// the unnamed namespace `(anonymous namespace)`, which puts a parenthesis BEFORE the kernel, so
// cutting there collapsed every such symbol to the return type `void`, and two unrelated kernels
// that both collapsed to `void` then certified each other. Brackets go the same way, which subsumes
// the `[clone .1]` rule: this pipeline appends its own annotations to a kernel name
// (`_fwd_grouped_kernel_stage1 [sliding_attention]`, `main_kernel[prefill]`), and rocprof spells
// memory ops `Memcpy DtoD (Device -> Device)`. The return type is then dropped by name and the FIRST
// identifier taken, as parse_profile.short_name does; taking the last whitespace token also drops
// `void`, but on any of those annotated spellings it answers with the annotation.
function canonicalDeviceKernel(s) {
  const stripped = stripBalanced(stripBalanced(
    stripBalanced(String(s || ''), '[', ']'), '<', '>'), '(', ')');
  // Whatever opener is left never closes, so the symbol was cut off inside it -- profile artifacts
  // store kernel names elided mid-template. The name is what precedes that opener; reading on
  // instead lifts a fragment out of the template arguments and states it with the same confidence
  // as a real name, which then matches the wrong kernel rather than refusing.
  // Removing a group also leaves a gap where it stood, and an unnamed namespace sits mid-
  // qualification: `at::native::(anonymous namespace)::CatArrayBatchedCopy` becomes
  // `at::native:: ::CatArray...`, where the identifier probe stops at the space.
  const cut = stripped.split(/[<([]/)[0].trim().replace(/\s*::\s*/g, '::');
  const match = /^[\w:]+/.exec(cut.replace(/^void\s+/, ''));
  return match ? match[0].split('::').pop().toLowerCase().replace(/[^a-z0-9_]+/g, '') : '';
}
// A name that hit the display limit ends mid-token, so no word boundary can follow it. Tested both
// ways because either side may be the shortened one: heads carry a short_name while traces carry
// the full symbol, and which is which is not fixed.
const truncatedPrefix = (needle, haystack) =>
  needle.length >= SHORT_NAME_LIMIT && haystack.startsWith(needle);
// [arguments, closed] for the symbol's first `<...>`. Reading only what follows the `<` keeps this
// independent of the return type and namespace, which one side routinely spells and the other does
// not. `closed` is false when the symbol was cut off inside the template -- profile artifacts elide
// long names, and only the visible part of an elided argument list can be held against anything.
// Separators become `_` rather than vanishing, so `<128, 4, ...>` cannot read as a prefix of
// `<128, 48, ...>`.
function templateArguments(value) {
  const text = String(value || ''), start = text.indexOf('<');
  const fold = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, '_');
  if (start < 0) return ['', true];
  let depth = 0;
  for (let i = start; i < text.length; i++) {
    if (text[i] === '<') depth++;
    else if (text[i] === '>' && --depth === 0) return [fold(text.slice(start + 1, i)), true];
  }
  return [fold(text.slice(start + 1)), false];
}
function kernelIdentitiesMatch(a, b) {
  const x = canonicalDeviceKernel(a), y = canonicalDeviceKernel(b);
  if (!x || !y) return false;
  if (x !== y && !truncatedPrefix(x, y) && !truncatedPrefix(y, x)) return false;
  // The base token deliberately drops template arguments so a bare declared name can match its
  // decorated spelling. When BOTH sides carry them the information is present on both, and ignoring
  // it certifies the wrong kernel: one capture here held 20 distinct kernels named
  // at::native::vectorized_elementwise_kernel, separated only by their functor.
  const [xa, xc] = templateArguments(a), [ya, yc] = templateArguments(b);
  if (!xa || !ya) return true;
  if (xc && yc) return xa === ya;
  // An elided list still has to agree as far as both sides actually spell it out.
  return xa.startsWith(ya) || ya.startsWith(xa);
}
function requiredDeviceKernel(h) {
  if (!h || h.entity_kind !== 'gpu_kernel') return '';
  // entity_evidence.matched_profiled_kernel is what parse_profile.py --annotate stamped on the row,
  // so it is the profiler's own name for this entity and outranks the display short_name.
  return String(h.device_kernel ||
    (h.entity_evidence && h.entity_evidence.matched_profiled_kernel) ||
    h.short_name || h.name || '').trim();
}
const candidateSpec = (c) =>
  String((c && (c.target_callable || c.callable || c.spec)) || '').trim();
// A declared depth is a STRUCTURAL claim about the call chain. It orders candidates before any
// runtime evidence exists, so a value that is absent or not a finite number is unusable: `Number()`
// turns it into NaN, every comparison against it is silently false, and a genuinely deeper candidate
// stops being able to outrank anything. Refuse the candidate instead of ranking it as depth 0.
const candidateDepth = (c) => {
  const depth = Number(c && c.depth);
  return Number.isFinite(depth) ? depth : null;
};
function selectionCandidatesForHead(h, ext) {
  const required = requiredDeviceKernel(h);
  const fused = !!(h && (h.is_fused_kernel === true || h.op_kind === 'moe'));
  const bySpec = new Map();
  for (const candidate of [
    ...((h && h.seam_candidates) || []),
    ...((ext && ext.seam_candidates) || []),
  ]) {
    const spec = candidateSpec(candidate);
    // Architect classifications win for an existing spec; Extractor may append missing inner launchers.
    if (spec && !bySpec.has(spec)) bySpec.set(spec, candidate);
  }
  return Array.from(bySpec.values()).filter((candidate) => {
    if (!candidate || typeof candidate !== 'object' || !validCallableSpec(candidateSpec(candidate)))
      return false;
    const role = String(candidate.role || '').toLowerCase();
    if (role === 'kernel_entry') return false; // native/JIT object identity is unsafe to monkeypatch
    if (candidateDepth(candidate) === null) return false;
    const kernels = Array.isArray(candidate.device_kernels) ? candidate.device_kernels : [];
    if (required && !kernels.some((kernel) => kernelIdentitiesMatch(required, kernel))) return false;
    return (fused
      ? ['outer_wrapper', 'dispatcher', 'op_seam']
      : ['outer_wrapper', 'dispatcher', 'op_seam', 'inner_launcher']).includes(role);
  });
}
function prepareHeadSelection(h) {
  if (!h || typeof h !== 'object') return h;
  const out = { ...h };
  const required = requiredDeviceKernel(out);
  if (required && !out.device_kernel) out.device_kernel = required;
  // live_call_seam and malformed target_callable values are prose, never machine targets.
  if (!validCallableSpec(out.target_callable)) out.target_callable = '';
  const fused = out.is_fused_kernel === true || out.op_kind === 'moe';
  const jointPhase = out.joint_ut && (out.joint_ut.phases || []).find(p =>
    (p.required_device_kernels || []).some(k => kernelIdentitiesMatch(k, required)));
  const deployable = selectionCandidatesForHead(jointPhase ? {
    ...out, seam_candidates: jointPhase.seam_candidates || out.seam_candidates,
  } : out).filter((candidate) => {
    if (jointPhase && !(jointPhase.required_device_kernels || []).every(k =>
        (candidate.device_kernels || []).some(v => kernelIdentitiesMatch(k, v)))) return false;
    const role = String(candidate.role || '').toLowerCase();
    return fused ? role === 'op_seam' : ['inner_launcher', 'op_seam'].includes(role);
  });
  deployable.sort((a, b) => candidateDepth(b) - candidateDepth(a));
  out.target_callable = deployable.length ? candidateSpec(deployable[0]) : '';
  out.selection_status = out.target_callable
    ? 'candidate_selected_needs_runtime_verification'
    : 'extractor_must_discover_live_launcher';
  return out;
}
function kernelSelectionVerified(h, ext, jointScopeMembers = null) {
  if (ext && ext.selection_evidence_error) return { ok: false, why: ext.selection_evidence_error };
  if (h && h.dispatch_ut) {
    if (h.joint_ut) return { ok: false, why: 'choose joint_ut or dispatch_ut, not both' };
    const phases = h.dispatch_ut.phases, rows = ext && ext.phase_selection_validations;
    if (!Array.isArray(phases) || !phases.length || !Array.isArray(rows) || rows.length !== phases.length)
      return { ok: false, why: 'dispatch UT requires one selection verdict per declared branch' };
    const seen = new Set();
    for (const spec of phases) {
      if (!spec.regime || !spec.device_kernel || !Array.isArray(spec.seam_candidates) ||
          !spec.seam_candidates.length || seen.has(spec.regime))
        return { ok: false, why: 'dispatch branches must have distinct nonempty regime labels' };
      seen.add(spec.regime);
      const matches = rows.filter(r => r.regime === spec.regime);
      if (matches.length !== 1 || matches[0].target_callable !== ext.target_callable)
        return { ok: false, why: `${spec.regime}: dispatch branch must verify the same whole-op callable` };
      const check = kernelSelectionVerified({ ...h, dispatch_ut: undefined,
        is_fused_kernel: true, device_kernel: spec.device_kernel, seam_candidates: spec.seam_candidates }, matches[0]);
      if (!check.ok) return { ok: false, why: `${spec.regime}: ${check.why}` };
    }
    if ([requiredDeviceKernel(h), ...(h.selected_device_kernels || [])].some(k =>
        !phases.some(p => kernelIdentitiesMatch(k, p.device_kernel))))
      return { ok: false, why: 'dispatch branches omit a selected device kernel' };
    return { ok: true, why: 'verified whole-op dispatch branches independently' };
  }
  if (ext && ext.selection_evidence_error)
    return { ok: false, why: ext.selection_evidence_error };
  if (h && h.joint_ut) {
    const phases = h.joint_ut.phases;
    const results = ext && ext.phase_selection_validations;
    if (!Array.isArray(phases) || !phases.length || !Array.isArray(results))
      return { ok: false, why: 'joint UT requires declared phases and per-phase selection verdicts' };
    if (!phases.some(p => (p.required_device_kernels || []).some(k =>
        kernelIdentitiesMatch(k, requiredDeviceKernel(h)))))
      return { ok: false, why: 'joint UT does not contain its profiled anchor kernel' };
    const regimes = new Set();
    for (const spec of phases) {
      const regime = String(spec.regime || '').trim();
      if (!regime || regimes.has(regime))
        return { ok: false, why: 'joint UT phases must have distinct nonempty regime labels' };
      regimes.add(regime);
      const requiredMembers = spec.required_device_kernels;
      if (!Array.isArray(requiredMembers) || requiredMembers.length < 2 ||
          !requiredMembers.every(k => typeof k === 'string' && k.trim()) ||
          requiredMembers.some((k, i) => requiredMembers.slice(0, i).some(v => kernelIdentitiesMatch(k, v))) ||
          !requiredMembers.some(k => kernelIdentitiesMatch(k, spec.device_kernel)))
        return { ok: false, why: `${regime}: joint phase must declare its anchor and companion kernels` };
      const matches = results.filter(r => r.regime === regime);
      if (matches.length !== 1)
        return { ok: false, why: `${regime}: expected one joint selection verdict` };
      const result = matches[0], verdict = result.selection_validation;
      if (!verdict || verdict.selection_scope !== 'joint_chain')
        return { ok: false, why: `${regime}: a leaf verdict cannot certify a joint chain` };
      const reported = verdict.required_device_kernels || [];
      if (reported.length !== requiredMembers.length ||
          requiredMembers.some(k => !reported.some(v => kernelIdentitiesMatch(k, v))))
        return { ok: false, why: `${regime}: joint verdict has the wrong member kernel set` };
      if (requiredMembers.some(k => !(verdict.chain_members || []).some(m =>
          kernelIdentitiesMatch(k, m.device_kernel) && Number(m.matched_kernel_calls) > 0)))
        return { ok: false, why: `${regime}: missing positive member-kernel execution evidence` };
      if (!Array.isArray(verdict.joint_call_coverage) || !verdict.joint_call_coverage.length ||
          verdict.joint_call_coverage.some(c => !Array.isArray(c.missing_members) || c.missing_members.length))
        return { ok: false, why: `${regime}: incomplete joint invocation coverage` };
      const check = kernelSelectionVerified({ ...h, joint_ut: undefined,
        device_kernel: spec.device_kernel, seam_candidates: spec.seam_candidates || result.seam_candidates },
        result, requiredMembers);
      if (!check.ok) return { ok: false, why: `${regime}: ${check.why}` };
    }
    if (results.length !== phases.length)
      return { ok: false, why: 'joint UT returned undeclared or duplicate phase results' };
    if (!results.some(r => r.target_callable === ext.target_callable))
      return { ok: false, why: 'joint UT primary callable is not one of its verified phase entries' };
    return { ok: true, why: `verified complete joint chains for ${Array.from(regimes).join(', ')}` };
  }
  const required = requiredDeviceKernel(h);
  if (!required) return { ok: true, why: 'not a profiled GPU-head extraction' };
  const target = String((ext && ext.target_callable) || '').trim();
  if (!validCallableSpec(target))
    return { ok: false, why: `target_callable '${target}' is not an exact module:attr spec` };
  const verdict = ext && ext.selection_validation;
  if (!verdict || verdict.contract !== 'kernel_selection')
    return { ok: false, why: 'kernel_selection.py verdict is missing' };
  if (verdict.selection_scope === 'joint_chain' && !jointScopeMembers)
    return { ok: false, why: 'joint-chain selection requires a declared joint UT boundary' };
  if (verdict.ok !== true)
    return { ok: false,
      why: `kernel_selection.py failed: ${(verdict.failed || []).join(', ') || 'unknown'}` };
  if (String(verdict.target_callable || '').trim() !== target)
    return { ok: false,
      why: `selection verdict certifies '${verdict.target_callable || ''}', not '${target}'` };
  if (!kernelIdentitiesMatch(required, verdict.device_kernel || ext.device_kernel))
    return { ok: false,
      why: `selection verdict certifies '${verdict.device_kernel || ext.device_kernel || ''}', ` +
        `not profiled kernel '${required}'` };
  if (Number(verdict.total_calls_observed || 0) <= 0 ||
      Number(verdict.target_marker_calls || 0) <= 0 ||
      Number(verdict.matched_kernel_calls || 0) <= 0)
    return { ok: false, why: 'selection verdict lacks positive call/marker/kernel evidence' };
  const candidates = selectionCandidatesForHead(h, ext);
  if (!candidates.length)
    return { ok: false, why: 'no relevant structured seam_candidates were returned' };
  const tested = new Set((verdict.candidate_targets_tested || []).map((value) => String(value || '').trim()));
  const omitted = candidates.map(candidateSpec).filter((spec) => !tested.has(spec));
  if (omitted.length)
    return { ok: false, why: `selection probe omitted candidate(s): ${omitted.join(', ')}` };
  const selected = candidates.find((candidate) => candidateSpec(candidate) === target);
  const selectedRole = String((selected && selected.role) || '').toLowerCase();
  const fused = h.is_fused_kernel === true || h.op_kind === 'moe';
  if (!selected || (fused ? selectedRole !== 'op_seam'
    : !['inner_launcher', 'op_seam'].includes(selectedRole)))
    return { ok: false,
      why: fused
        ? `fused head must select the whole-operation op_seam, got '${selectedRole || 'unknown'}'`
        : `selected target role '${selectedRole || 'unknown'}' is not a deployable inner/op seam` };
  // "Deepest" is settled by the profiler's OBSERVED host-span nesting (kernel_selection.py computes
  // deeper_live_candidates from it), not by the declared `depth`. The declared integer is a claim by
  // the same agent whose choice is under review, and an appended candidate carrying a large depth
  // would otherwise outrank a genuinely deeper one. The structural claim is still checked, but only
  // as an ADDITIONAL way to fail: it can never buy a pass that the observed nesting did not grant.
  const observedDeeper = (verdict.deeper_live_candidates || [])
    .map((value) => String(value || '').trim()).filter(Boolean);
  if (observedDeeper.length)
    return { ok: false,
      why: `deeper live candidate(s) observed across calls/ranks: ${observedDeeper.join(', ')}` };
  const live = new Set((verdict.live_candidate_targets || []).map((value) => String(value || '').trim()));
  const selectedDepth = candidateDepth(selected);
  const deeper = candidates.filter((candidate) =>
    candidateSpec(candidate) !== target && live.has(candidateSpec(candidate)) &&
    candidateDepth(candidate) > selectedDepth);
  if (deeper.length)
    return { ok: false,
      why: `declared-deeper live candidate(s): ${deeper.map(candidateSpec).join(', ')}` };
  if (verdict.deepest_verified !== true)
    return { ok: false, why: 'selected callable is not machine-verified as deepest' };
  return { ok: true, why: `${target} launches profiled kernel ${required}` };
}

// Opt-in transport only: hydrate exact Python verdicts before applying the same
// selection gate. A file or hash error cannot fall back to an inline claim of PASS.
// This function is also used by kernel_extraction_delivery.mjs for preflight.
async function hydrateExtractionVerdicts(ext) {
  if (!ext || typeof ext !== 'object') return ext;
  const phases = Array.isArray(ext.phase_selection_validations) ? ext.phase_selection_validations : [];
  if (!ext.selection_validation_ref && !phases.some(p => p && p.selection_validation_ref)) return ext;
  const out = { ...ext };
  try {
    const fs = await import('node:fs/promises');
    const path = await import('node:path');
    const { createHash } = await import('node:crypto');
    const task = await fs.realpath(ext.task_dir);
    const readVerdict = async (row) => {
      if (!row || !row.selection_validation_ref) return row;
      const ref = row.selection_validation_ref;
      if (typeof ref.path !== 'string' || !ref.path.trim() ||
          typeof ref.sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(ref.sha256))
        throw new Error('selection_validation_ref requires path and lowercase SHA-256');
      const file = await fs.realpath(path.resolve(task, ref.path));
      const relative = path.relative(task, file);
      if (!relative || relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative))
        throw new Error('selection verdict must be a file inside task_dir');
      const bytes = await fs.readFile(file);
      if (createHash('sha256').update(bytes).digest('hex') !== ref.sha256)
        throw new Error(`selection verdict SHA-256 mismatch: ${ref.path}`);
      const verdict = JSON.parse(bytes.toString('utf8'));
      if (!verdict || verdict.contract !== 'kernel_selection')
        throw new Error(`not a kernel_selection verdict: ${ref.path}`);
      return { ...row, selection_validation: verdict };
    };
    Object.assign(out, await readVerdict(out));
    if (Array.isArray(ext.phase_selection_validations)) {
      out.phase_selection_validations = [];
      for (const row of phases) out.phase_selection_validations.push(await readVerdict(row));
    }
  } catch (error) {
    out.selection_evidence_error = `selection evidence file invalid: ${error.message || error}`;
  }
  return out;
}


const hasFrozenBaseline = (ext) =>
  !!(ext && (ext.baseline_frozen === true ||
             (ext.candidate_bind && typeof ext.candidate_bind === 'object') ||
             (typeof ext.baseline_callable === 'string' && ext.baseline_callable.trim() !== '')));


export { EXTRACT_SCHEMA, EXTRACT_OP_SCHEMA, kernelSelectionVerified,
  hydrateExtractionVerdicts, hasFrozenBaseline };
