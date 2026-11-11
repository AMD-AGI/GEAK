// Agent calls, extraction, integration and workflow deadlines.
function agentBounded(rawPrompt, opts, onHung, timeoutFor = agentTimeoutFor) {
  const prompt = withProcessSafety(rawPrompt);
  const timeoutMs = timeoutFor();
  if (typeof setTimeout !== 'function' || !(timeoutMs > 0)) return agent(prompt, opts);
  let to;
  const run = agent(prompt, opts);
  const guard = new Promise((resolve) => {
    to = setTimeout(() => {
      log(`  [hung-agent guard] ${(opts && opts.label) || 'agent'} exceeded ${Math.round(timeoutMs / 60000)}min with no return — treating as a failed attempt.`);
      if (onHung) onHung(run);
      resolve(null);
    }, timeoutMs);
  });
  return Promise.race([
    run.then((r) => { clearTimeout(to); return r; }, (e) => { clearTimeout(to); throw e; }),
    guard,
  ]);
}

// Attempts the hung-agent guard abandoned inside the reserve, by label. Each is still running; an
// integrator's eventual return is its A/B verdict (adoptAbandonedVerdict).
const ABANDONED_RUNS = new Map();

async function safeAgent(prompt, opts, tries = 3, timeoutFor = agentTimeoutFor) {
  let lastErr = 'unknown';
  for (let i = 0; i < tries; i++) {
    try {
      const ev = tlAgent(prompt, opts, i + 1);   // record AT DISPATCH; ok=false until it resolves
      let hung = null;
      const r = await agentBounded(prompt, opts, (run) => { hung = run; }, timeoutFor);
      if (r) {
        if (r.llm_timeline) LLM_TL.nested.push(r.llm_timeline);
        if (ev) ev.ok = true;
        return r;
      }
      const inReserve = !!hung && atReserveFloor();
      if (hung && (!AGENT_TIMEOUT_RETRY || inReserve)) {
        if (inReserve) ABANDONED_RUNS.set((opts && opts.label) || '', hung);
        log(`agent[${(opts && opts.label) || '?'}] DEGRADED to null without a retry: the timed-out attempt ` +
          `may still be running (${inReserve ? 'the final reserve has begun' : 'agent_timeout_retry=false'}).`);
        return null;
      }
      lastErr = 'null/empty result';
    } catch (e) { lastErr = String(e); }
    log(`agent[${(opts && opts.label) || '?'}] attempt ${i + 1}/${tries} failed: ${String(lastErr).slice(0, 160)}`);
  }
  log(`agent[${(opts && opts.label) || '?'}] DEGRADED to null after ${tries} tries (${String(lastErr).slice(0, 120)})`);
  return null;
}

// --- FlyDSL provisioning gate ----
// Once strategize routes a flydsl backend onto any candidate, the orchestrator (not the architect
// LLM) runs the ensure_flydsl skill before any consumer can select it (roles/system_architect.md
// §0a). Reuse-or-build is decided only by ensure_flydsl.sh's own version gate.
let flydslProvisioned = false;   // guard so the two call sites (strategize / re-strategize) provision at most once
const _wantsFlydsl = (c) => ((c && c.candidate_backends) || []).some((b) => String(b).toLowerCase() === 'flydsl');
const flydslRouted = (...queues) => queues.some((q) => (q || []).some(_wantsFlydsl));
function stripFlydslFromQueues(...queues) {
  let n = 0;
  for (const q of queues) for (const c of (q || [])) {
    if (_wantsFlydsl(c)) { c.candidate_backends = (c.candidate_backends || []).filter((b) => String(b).toLowerCase() !== 'flydsl'); n++; }
  }
  return n;
}
async function ensureFlydslGate() {
  if (flydslProvisioned) return;                      // already provisioned this run
  if (!flydslRouted(headQueue, kernelQueue)) return;  // strategize did not route flydsl -> nothing to do
  log('[flydsl-gate] strategize routed a flydsl backend -> running ensure_flydsl (detect -> reuse-or-build) BEFORE any consumer.');
  const g = await safeAgent(
    `You are the flydsl provisioner. Run the ensure_flydsl expert skill EXACTLY as ` +
    `${WORKFLOW_DIR}/../perf_knowledge/expert_skills/skills/ensure_flydsl/skill.md specifies: launch ` +
    `${WORKFLOW_DIR}/../perf_knowledge/expert_skills/skills/ensure_flydsl/ensure_flydsl.sh DETACHED and poll ` +
    `(success = flydsl_env.sh written; failure = .flydsl_build.failed). The script's own version gate ` +
    `(import flydsl AND kernels.moe_gemm_2stage AND __version__ >= FLYDSL_MIN_VERSION, default 0.2.2) is the ` +
    `SOLE authority: reuse an ambient flydsl if satisfied, else clone+build the pin (~15-18min). Do NOT use ` +
    `is_flydsl_available() to decide anything. Return {ok,built,env_file,version,reason}; ok=true iff, after ` +
    `sourcing flydsl_env.sh, "import flydsl, kernels.moe_gemm_2stage" works.`,
    { phase: 'Strategize', label: 'architect:ensure_flydsl', schema: FLYDSL_GATE_SCHEMA });
  if (g && g.ok) {
    flydslProvisioned = true;
    log(`[flydsl-gate] FlyDSL ready (${g.built ? 'built' : 'reused'} ${g.version || '?'}); env=${g.env_file || '(default /opt/flydsl/FlyDSL/flydsl_env.sh)'}.`);
  } else {
    const n = stripFlydslFromQueues(headQueue, kernelQueue);
    log(`[flydsl-gate] ensure_flydsl FAILED (${(g && g.reason) || 'no result'}) -> stripped 'flydsl' from ${n} candidate backend list(s); those heads fall back to their other backends (no silent flydsl selection).`);
  }
}

// A profile identity is a device symbol; the replacement target is a live Python callable. Keep the
// two machine fields separate and require runtime evidence before bake-off or authoring.
const CALLABLE_SPEC_RX = /^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$/;
const validCallableSpec = (s) => CALLABLE_SPEC_RX.test(String(s || '').trim());
// parse_profile.short_name truncates display names at 60 chars; a declared name that hit the limit
// is our own doing, so a prefix match at it is accepted rather than read as a mismatch.
const SHORT_NAME_LIMIT = 60;
// Removes balanced groups innermost-first until the symbol stops changing (a single greedy pass
// would span `k<a>(t<b>)` from the first opener to the last closer).
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
// Must stay identical to canonical_kernel_name in kernel_selection.py (the JS gate and the Python
// verdict compare the same symbols). Brackets, templates and parentheses are stripped as balanced
// groups, not cut at the first '(' (ROCm's `(anonymous namespace)` precedes the kernel name, and
// annotations like `[clone .1]`, `[prefill]` or `(Device -> Device)` follow it); then the return type
// is dropped and the FIRST identifier taken, as parse_profile.short_name does.
function canonicalDeviceKernel(s) {
  const stripped = stripBalanced(stripBalanced(
    stripBalanced(String(s || ''), '[', ']'), '<', '>'), '(', ')');
  // A leftover opener never closes (the name was elided mid-template), so the name is what precedes
  // it. Gaps left by removed groups are closed around `::` (`at::native:: ::CatArray...`).
  const cut = stripped.split(/[<([]/)[0].trim().replace(/\s*::\s*/g, '::');
  const match = /^[\w:]+/.exec(cut.replace(/^void\s+/, ''));
  return match ? match[0].split('::').pop().toLowerCase().replace(/[^a-z0-9_]+/g, '') : '';
}
// A name at the display limit ends mid-token (no word boundary follows). Called both ways: either
// side may be the shortened one (heads carry short_name, traces the full symbol).
const truncatedPrefix = (needle, haystack) =>
  needle.length >= SHORT_NAME_LIMIT && haystack.startsWith(needle);
// [arguments, closed] for the symbol's first `<...>`, independent of return type and namespace.
// `closed` is false when the name was elided inside the template. Separators fold to `_`, so
// `<128, 4, ...>` is not a prefix of `<128, 48, ...>`.
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
  // The base token drops template arguments so a bare name matches its decorated spelling; when
  // BOTH sides carry them they must agree (many kernels differ only by their template functor).
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
// A declared depth orders candidates before any runtime evidence; a missing or non-finite one
// returns null so the candidate is refused (NaN would silently break every comparison).
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
  // A completed UT already verified its callable; do not replace it with an
  // untested nomination when the optimization queue is admitted.
  if (typeof KERNEL_EXTRACT_ON !== 'undefined' && KERNEL_EXTRACT_ON && h.prepared_ut) return { ...h };
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
  // "Deepest" is settled by the profiler's OBSERVED nesting (kernel_selection.py's
  // deeper_live_candidates), not the agent-declared `depth`, which can only add a failure.
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

// The profiled row, not the Architect's restated entity_kind, is the authority: a head adopts
// gpu_kernel only if every row its device_kernel matches is a named gpu_kernel row (exact names
// first; a canonical match folds `aten::mul` with `mul`). Rows are identified by their most
// specific name, since one short_name is shared across instantiations.
function adoptProfiledEntityKind(head, rows) {
  if (!head || head.entity_kind === 'gpu_kernel' || !head.device_kernel) return false;
  const want = String(head.device_kernel).trim();
  const identity = (r) => [r.device_kernel, r.name, r.short_name]
    .map((v) => String(v || '').trim()).find(Boolean) || '';
  const profiled = (rows || []).filter((r) => r && identity(r));
  let matched = profiled.filter((r) => identity(r) === want);
  if (!matched.length) matched = profiled.filter((r) => kernelIdentitiesMatch(identity(r), want));
  if (!matched.length || matched.some((r) => r.entity_kind !== 'gpu_kernel' || !r.device_kernel)) return false;
  head.entity_kind = 'gpu_kernel';
  return true;
}

// A selection verdict may arrive as `selection_validation_ref` (path + SHA-256). This script cannot
// read or hash files (see RUNTIME SANDBOX CONTRACT), so the extractor's preflight
// (scripts/kernel_extraction_delivery.mjs::hydrateExtractionVerdicts) resolves it. Backstop: a ref
// still unresolved here becomes an explicit evidence error, never an inline PASS.
function guardUnhydratedVerdicts(ext) {
  if (!ext || typeof ext !== 'object') return ext;
  // A hydration that ran and failed already gave the more specific diagnosis, so it wins.
  if (ext.selection_evidence_error) return ext;
  const phases = Array.isArray(ext.phase_selection_validations) ? ext.phase_selection_validations : [];
  const unresolved = (row) => !!(row && row.selection_validation_ref && !row.selection_validation);
  if (!unresolved(ext) && !phases.some(unresolved)) return ext;
  return { ...ext, selection_evidence_error:
    'selection_validation_ref arrived unresolved: the workflow sandbox cannot read or hash files. ' +
    'Run scripts/kernel_extraction_delivery.mjs as the extraction preflight so the verdict is ' +
    'hydrated into selection_validation before it is returned.' };
}

const PRE_FLAGGED_HEADS = [];
function admitHeads(queue, stage) {
  const admitted = [];
  const profiledRows = (profile && profile.top_kernels) || [];
  const unnamed = profiledRows.filter((r) => r && r.entity_kind === 'gpu_kernel' && !r.device_kernel);
  if (unnamed.length)
    log(`  ⚠️ profile contract: ${unnamed.length} gpu_kernel row(s) carry no device_kernel ` +
      `(${unnamed.map((r) => r.short_name || r.name || '?').join(', ')}); heads cannot adopt ` +
      `entity_kind from them (${stage}).`);
  for (const head of (queue || []).filter(Boolean)) {
    const label = head.short_name || head.name || '(unnamed)';
    const restated = head.entity_kind;
    if (adoptProfiledEntityKind(head, profiledRows))
      log(`  [admit] ${label}: entity_kind=${restated || 'missing'} -> gpu_kernel from the profiled row (${stage}).`);
    if (head.entity_kind !== 'gpu_kernel') {
      log(`  ⚠️ FLAG ${label}: entity_kind=${head.entity_kind || 'missing'}; ` +
        `the head track requires a profiler-confirmed gpu_kernel (${stage}).`);
      PRE_FLAGGED_HEADS.push({ short_name: label,
        pct_gpu_time: head.pct_gpu_time, stage, gate: 'wrong_head_granularity',
        reason: `entity_kind=${head.entity_kind || 'missing'}; gpu_kernel required` });
      continue;
    }
    // A gpu_kernel head with no device identity would pass kernelSelectionVerified vacuously
    // ("not a profiled GPU-head extraction"), so refuse it at admission.
    if (!requiredDeviceKernel(head)) {
      log(`  ⚠️ FLAG ${label}: entity_kind=gpu_kernel but no device_kernel/short_name/name; ` +
        `there is no profiled symbol to verify a seam against (${stage}).`);
      PRE_FLAGGED_HEADS.push({ short_name: label,
        pct_gpu_time: head.pct_gpu_time, stage, gate: 'missing_kernel_identity',
        reason: 'gpu_kernel head carries no device_kernel/short_name/name' });
      continue;
    }
    admitted.push(prepareHeadSelection(head));
  }
  return admitted;
}

// A FROZEN baseline is resolvable when the extractor seeded baseline_overlay/ + declared
// meta.candidate_bind (kernel track), or set an importable meta.baseline_callable (op track).
// That is the language-independent speedup denominator.
const hasFrozenBaseline = (ext) =>
  !!(ext && (ext.baseline_frozen === true ||
             (ext.candidate_bind && typeof ext.candidate_bind === 'object') ||
             (typeof ext.baseline_callable === 'string' && ext.baseline_callable.trim() !== '')));

// Why an extraction failed, for logs / ledger / flagged heads. The status field is an enum
// ("pass"/"fail"), so the prose lives in smoke_detail; notes may be unrelated, so read both.
const extractFailWhy = (e) => !e ? 'none'
  : [e.smoke_detail, e.notes].filter(Boolean).join(' | ') || e.smoke || e.unittest_smoke || 'none';

// Re-asserts in pure JS what the shipped envelope must still prove. agentx_kernel_preparation.mjs
// gated the on-disk RETURN.json while packing, but its output reaches us through an agent (see
// RUNTIME SANDBOX CONTRACT), so a truncated or hand-built reply must not pass. Checked at intake
// (readAgentXPreparation) and at dispatch (reusePreparedUT); on-disk hashes stay preparation's job.
function preparedUtDefect(kernel, result, phase) {
  if (!result || typeof result !== 'object')
    return 'prepared UT carries no validated result; re-pack the delivery with --emit-prepared';
  if (typeof result.task_dir !== 'string' || !result.task_dir.trim())
    return 'prepared UT result names no task_dir';
  if (result.editable === false) return 'prepared UT task is not editable';
  // extract reports unittest_smoke, extract_op reports smoke -- same split checkPreparedUT uses.
  const status = phase === 'extract' ? result.unittest_smoke : result.smoke;
  if (status !== 'pass') return `prepared UT did not pass (${phase}: ${status ?? 'no status'})`;
  if (result.baseline_frozen !== true)
    return 'prepared UT has no frozen baseline, so there is no speedup denominator';
  if (!kernel || typeof kernel.pct_gpu_time !== 'number' ||
      !Number.isFinite(kernel.pct_gpu_time) || kernel.pct_gpu_time < 0 || kernel.pct_gpu_time > 100)
    return 'prepared candidate has no attributed GPU-time share in [0, 100]';
  // `attention` and `attn` are one kind (agentx_kernel_preparation.mjs::opKind); a declaration that
  // disagrees with its own result is a defect, not a routing choice.
  const canonical = (v) => v === 'attention' ? 'attn' : v;
  const requested = canonical(kernel.op_kind), returned = canonical(result.op_kind);
  if (requested && returned && requested !== returned)
    return `op_kind differs between declaration and result: ${requested} / ${returned}`;
  // A weighted UT reaches its lanes' weighted primary only through the packed workload binding.
  if (kernel.prepared_ut && kernel.prepared_ut.weighted === true &&
      !(typeof result.workload_path === 'string' && result.workload_path.startsWith('/')))
    return 'weighted prepared UT carries no absolute workload_path binding';
  return '';
}

// Hand an optimization lane the UT preparation validated, after re-asserting the invariants above;
// no prepared_ut.result fails closed. Known limit: a task dir mutated after packing is not detected.
function reusePreparedUT(kernel) {
  const fail = (why) => ({ smoke: 'fail', unittest_smoke: 'fail',
    smoke_detail: `prepared UT is no longer valid: ${why}` });
  const ref = kernel && kernel.prepared_ut;
  if (!ref) return fail('missing prepared UT; complete the extraction stage before optimization');
  const defect = preparedUtDefect(kernel, ref.result, ref.phase);
  if (defect) return fail(defect);
  const canonical = (v) => v === 'attention' ? 'attn' : v;
  return { ...ref.result, op_kind: canonical(ref.result.op_kind) || canonical(kernel.op_kind),
    smoke: 'pass', unittest_smoke: 'pass' };
}

// AgentX reuses its prepared UT. Otherwise run kernel_extractor and, while smoke passes but the
// baseline is not frozen or the kernel selection is unverified, re-invoke it with a corrective (up to
// BASELINE_EXTRACT_RETRIES); if still incomplete, return an extract failure so the head is flagged
// or skipped, never timed against its own scaffold. Failures stay local to the caller; GPU work
// uses the skill's per-device locks.
async function extractWithBaseline(role, phase, intro, inputs, opts) {
  const smokeOk = (e) => !!(e && e.task_dir && (e.smoke === 'pass' || e.unittest_smoke === 'pass'));
  const head = (inputs && inputs.KERNEL) || {};
  if (typeof KERNEL_EXTRACT_ON !== 'undefined' && KERNEL_EXTRACT_ON) return reusePreparedUT(head);
  const captureIntro = `${intro} CAPTURE STORAGE BOUNDS (issue #429): every capture EXTRA_ENV MUST include ` +
    `\`${CAPTURE_STORAGE_ENV}\`. Use kernel_selection.py with --task-dir "$TASK" so the selected oracle is ` +
    `promoted and all capture.pid-* dirs are reclaimed. Unittests for large MoE oracles MUST use ` +
    `h.iter_eager_cases_from_oracle / h.check_correct_multi_lazy.`;
  let ext = await safeAgent(roleAgent(role, phase, captureIntro, {
    ...(inputs || {}),
    CAPTURE_STORAGE_ENV,
    CAPTURE_BYTE_BUDGET,
    CAPTURE_CASE_BYTE_LIMIT,
    CAPTURE_PERSIST_POLICY,
  }), opts);
  ext = guardUnhydratedVerdicts(ext);
  let tries = 0;
  const attemptedTargets = [];
  const complete = (e) => hasFrozenBaseline(e) && kernelSelectionVerified(head, e).ok;
  while (smokeOk(ext) && !complete(ext) && tries < BASELINE_EXTRACT_RETRIES) {
    tries++;
    const selection = kernelSelectionVerified(head, ext);
    const needBaseline = !hasFrozenBaseline(ext);
    const priorTarget = String((ext && ext.target_callable) || '').trim();
    if (priorTarget && !attemptedTargets.includes(priorTarget)) attemptedTargets.push(priorTarget);
    const selectionCorrective = selection.ok ? '' :
      ` PRIOR ATTEMPT DID NOT SELECT THE PROFILED GPU KERNEL: ${selection.why}. ` +
      'Treat KERNEL.live_call_seam as prose only. Merge KERNEL.seam_candidates with any missing inner ' +
      'launcher found from source/runtime inspection; preserve existing candidate classifications. ' +
      'BEFORE re-running capture, reclaim prior process-local artifacts: ' +
      '`python3 "$SKILL_DIR/scripts/capture_shapes.py" --cleanup-task-dir "$TASK" --no-promote` ' +
      '(issue #429 — never accumulate capture.pid-* oracles across retries). ' +
      'Install safe markers for every relevant candidate, never native/JIT kernel_entry objects. Run ' +
      'kernel_selection.py over every process-local capture and all root-call traces with --task-dir "$TASK". ' +
      'Return its JSON verbatim as selection_validation. Select the deepest live inner_launcher/op_seam across all ' +
      'calls/ranks; a fused head must select the whole-op op_seam. Rejecting the previous outer wrapper ' +
      'is not success. Do not return any ATTEMPTED_TARGET_CALLABLES value again.';
    const baselineCorrective = needBaseline ?
      ' PRIOR ATTEMPT DID NOT FREEZE A BASELINE. You MUST seed baseline_overlay/ from ' +
      'CURRENT_OVERLAY (the live serving stack = the speedup denominator), declare meta.candidate_bind ' +
      '(the ONE overlay entry built from kernel_src/), prove both legs differ via h.assert_legs_differ, ' +
      'then return baseline_frozen:true. An extraction with no frozen baseline is INVALID and will be discarded.' : '';
    log(`  ${(opts && opts.label) || role}: extraction contract incomplete ` +
      `(${selection.ok ? 'kernel selected' : selection.why}; ` +
      `${needBaseline ? 'baseline missing (baseline_overlay/ + meta.candidate_bind)' : 'baseline frozen'}). ` +
      `RE-EXTRACTING (retry ${tries}/${BASELINE_EXTRACT_RETRIES}).`);
    // Best-effort capture reclaim before the agent retries.
    if (ext && ext.task_dir) {
      try {
        const { execFileSync } = require('child_process');
        execFileSync('python3', [
          `${WORKFLOW_DIR}/scripts/capture_shapes.py`,
          '--cleanup-task-dir', String(ext.task_dir),
          '--no-promote',
        ], { stdio: 'pipe', timeout: 120000 });
        log(`  ${(opts && opts.label) || role}: reclaimed capture artifacts under ${ext.task_dir}`);
      } catch (cleanupErr) {
        log(`  ${(opts && opts.label) || role}: capture reclaim skipped (${cleanupErr && cleanupErr.message ? cleanupErr.message : cleanupErr})`);
      }
    }
    ext = await safeAgent(
      roleAgent(role, phase, captureIntro + selectionCorrective + baselineCorrective, {
        ...(inputs || {}),
        CAPTURE_STORAGE_ENV,
        CAPTURE_BYTE_BUDGET,
        CAPTURE_CASE_BYTE_LIMIT,
        CAPTURE_PERSIST_POLICY,
        PRIOR_TARGET_CALLABLE: priorTarget,
        PRIOR_SELECTION_VALIDATION: (ext && ext.selection_validation) || {},
        ATTEMPTED_TARGET_CALLABLES: attemptedTargets.slice(),
      }),
      opts);
    ext = guardUnhydratedVerdicts(ext);
  }
  const finalSelection = kernelSelectionVerified(head, ext);
  if (smokeOk(ext) && !finalSelection.ok) {
    log(`  ${(opts && opts.label) || role}: kernel selection still unverified after ` +
      `${BASELINE_EXTRACT_RETRIES} re-extractions — ABORTING (${finalSelection.why}).`);
    return { ...ext, smoke: 'fail', unittest_smoke: 'fail', selection_failed: true,
      smoke_detail: `workflow forced fail: kernel selection failed: ${finalSelection.why}`,
      notes: `kernel selection failed: ${finalSelection.why} — ${ext.notes || ''}` };
  }
  if (smokeOk(ext) && !hasFrozenBaseline(ext)) {
    log(`  ${(opts && opts.label) || role}: STILL no frozen baseline after ${BASELINE_EXTRACT_RETRIES} ` +
      `re-extractions — ABORTING this extraction (refusing a fake speedup vs the candidate's own scaffold).`);
    return { ...ext, smoke: 'fail', unittest_smoke: 'fail',
      smoke_detail: `workflow forced fail: no frozen baseline after ${BASELINE_EXTRACT_RETRIES} re-extractions`,
      notes: `no frozen baseline after ${BASELINE_EXTRACT_RETRIES} re-extractions ` +
        `(baseline_overlay/ + meta.candidate_bind required as the speedup denominator) — ${ext.notes || ''}` };
  }
  // source_delivery_required is set only by the AgentX preparation reader, whose mode returned at
  // reusePreparedUT (verifySourceDelivery ran at packing). Reaching here means source evidence is
  // required on a path that cannot hash kernel_src (no filesystem), so fail closed.
  if (smokeOk(ext) && head.source_delivery_required === true) {
    return { ...ext, smoke: 'fail', unittest_smoke: 'fail', source_delivery_failed: true,
      smoke_detail: 'source delivery incomplete: this head requires source evidence, but it was ' +
        'extracted outside the AgentX preparation stage, which is what verifies it' };
  }
  return ext;
}

// abDone == the integrator measured BOTH legs (ref + cand) and emitted a real
// verdict. gate:'incomplete' or ab_complete:false means a leg is still missing.
const abDone = (integ) => !!(integ && integ.gate !== 'incomplete' && integ.ab_complete !== false);

// Publish an e2e delta with its two medians from the same A/B (ref_med before the win, cand_med
// after), so consumers can restate wins against one fixed baseline; bare percentages don't compose.
const e2eFrom = (integ) => ({
  e2e_delta_pct: integ.e2e_delta_pct,
  base_tput: integ.ref_med,
  new_tput: integ.cand_med,
});
// The KERNEL_RESULT a given integrate A/B gated. Call sites read both patch spellings
// (head=code_patch, milestone=final_patch; tryCorrectiveReauthor sets both).
const krOf = (inputs) => (inputs && inputs.KERNEL_RESULT) || {};

// Bank an accepted win with the fields a KB reader needs to replay it elsewhere. e2e_store.py's
// _accepted_kernels keeps every field but looks up `name`, `language`, `isolated_speedup` and
// `patch`, so those aliases sit beside short_name/backend/isolated (read by the report, resume state
// and run_e2e.py). `e.patch` wins: a corrective re-author supplies the fixed patch.
const bankAccepted = (list, e, kr) => {
  const k = kr || {};
  list.push({
    ...e,
    name: e.short_name || '',
    language: e.backend || '',
    isolated_speedup: Number(e.isolated) || 0,
    winner_kind: e.kind || k.winner_kind || '',
    patch: e.patch || k.code_patch || k.final_patch || '',
    target_callable: k.target_callable || '',
    source_path_in_sglang: k.source_path_in_sglang || '',
    apply_env: e.apply_env || k.apply_env || '',
    apply_flags: e.apply_flags || k.apply_flags || '',
    pct_gpu_time: e.pct_gpu_time != null ? e.pct_gpu_time : (k.pct_gpu_time || 0),
  });
};

// An integrator abandoned at the reserve boundary is still measuring; with no FINISH retry allowed
// inside the reserve, its late return is this A/B's only verdict. Wait for it through at most the
// first LATE_VERDICT_RESERVE_FRAC of the reserve, never into AGENTX_FINAL_PHASE_MS.
const LATE_VERDICT_RESERVE_FRAC = 0.25;
async function adoptAbandonedVerdict(integ, label) {
  const run = ABANDONED_RUNS.get(label);
  ABANDONED_RUNS.delete(label);
  if (integ || !run || !atReserveFloor() || typeof setTimeout !== 'function') return integ;
  const waitMs = remainingMs() - Math.max(Math.ceil(FINAL_RESERVE_MS * (1 - LATE_VERDICT_RESERVE_FRAC)),
    AGENTX_FINAL_PHASE_MS);
  if (!(waitMs > 0)) return integ;
  log(`  ${label}: abandoned at the reserve boundary — waiting up to ${Math.round(waitMs / 60000)}min ` +
    'for its own verdict instead of starting a FINISH retry.');
  let to;
  const late = await Promise.race([
    run.then((r) => r || null, () => null),
    new Promise((resolve) => { to = setTimeout(() => resolve(null), waitMs); }),
  ]);
  clearTimeout(to);
  if (!late) {
    log(`  ${label}: no late verdict within ${Math.round(waitMs / 60000)}min; the A/B stays incomplete.`);
    return integ;
  }
  if (late.llm_timeline) LLM_TL.nested.push(late.llm_timeline);
  log(`  ${label}: ADOPTED the late verdict (${late.gate || '?'}${late.e2e_delta_pct != null ? `, ${late.e2e_delta_pct}%` : ''}).`);
  return late;
}

const SPEEDUP_REVIEW_RESULTS = new Map();
async function reviewSpeedup(integ, inputs, label, phaseName) {
  const kr = inputs.KERNEL_RESULT || {};
  const pct = kr.pct_gpu_time || 0, isolated = kr.verified_isolated_speedup || 0;
  if (!abDone(integ) || !['accepted', 'stack'].includes(integ.gate) || integ.output_parity === 'fail'
      || !speedupNeedsReview(pct, isolated, integ) || confirmedSpeedup(integ)) return integ;
  if (integ.speedup_review && ['pending', 'inconclusive', 'rejected'].includes(integ.speedup_review.status)) return integ;
  const original = integ.measurement_evidence || {};
  const estimate = speedupEstimate(pct, isolated, integ);
  // Cache by measurement AND candidate/reference inputs. Returning the same cached A/B cannot
  // restart a review, including when Finalize revisits a pending integration.
  const key = JSON.stringify([original, kr, inputs.CURRENT_OVERLAY, inputs.CURRENT_FLAGS,
    inputs.CURRENT_ENV, inputs.WORKLOAD, integ.ref_med, integ.cand_med]);
  if (SPEEDUP_REVIEW_RESULTS.has(key)) return SPEEDUP_REVIEW_RESULTS.get(key);
  const pending = { ...integ, speedup_review: { status: 'inconclusive', original_measurement: original, estimate },
    reason: 'speedup_review_inconclusive: independent A/B confirmation unavailable' };
  SPEEDUP_REVIEW_RESULTS.set(key, pending);
  log(`  ${label}: speedup review (+${integ.e2e_delta_pct}% vs estimate +${estimate.estimate_pct.toFixed(2)}%, ` +
    `${estimate.pct_gpu_time}% GPU, ${estimate.source}); one independent A/B, no kernel rewrite.`);
  if (!['measurement_id', 'candidate_digest', 'reference_digest', 'workload_digest', 'ref_summary', 'cand_summary']
      .every((field) => typeof original[field] === 'string' && original[field].length > 0)) {
    pending.reason = 'speedup_review_inconclusive: original A/B provenance missing; cannot bind a confirmation';
    return pending;
  }
  if (searchClosed(`${label}: speedup review`)) return pending;
  const reviewInputs = { ...inputs, FORCE_FRESH_AB: true, RESUME_AB: false, REUSE_REF: false,
    REPLICAS: Math.max(2, Number(inputs.REPLICAS) || 1),
    CAND_OVERLAY_DIR: integ.accepted_overlay || '',
    SPEEDUP_REVIEW: { original_measurement: original, estimate },
  };
  delete reviewInputs.SHARED_REF_MED;
  // A confirmation abandoned at the boundary confirms nothing.
  if (abWindowShort(abLegs(reviewInputs), `${label}: speedup review`, reviewInputs.REPLICAS)) {
    pending.reason = 'speedup_review_inconclusive: deferred, the confirmation A/B cannot finish before the final reserve';
    return pending;
  }
  const fresh = await safeAgent(roleAgent('e2e_integrator', 'integrate',
    'Confirm this surprising speedup with ONE independent A/B. Follow FORCE_FRESH_AB in your role: ' +
    'use separate ref/cand artifact directories, preserve the exact candidate and current reference, ' +
    'measure at least two replicas per leg and repeat the configured correctness and engagement gates. ' +
    'Do not reuse old benchmark legs or rewrite the kernel. Amdahl is an estimate, never a correctness verdict. ' +
    'Return the fresh measurements and their provenance; if the gain reproduces, accept even above the estimate.',
    reviewInputs), { phase: phaseName, label: `${label} speedup review`, schema: INTEGRATE_SCHEMA }, 1);
  let result = pending;
  if (fresh && fresh.ab_complete === true && fresh.gate !== 'incomplete') {
    const checked = { ...fresh, speedup_review: { status: 'confirmed', original_measurement: original, estimate } };
    if (confirmationEvidenceMatches(checked)) {
      if (fresh.gate === 'accepted' && fresh.output_parity === 'pass'
          && ['accuracy', 'byte_exact'].includes(fresh.parity_kind)) result = checked;
      else if (fresh.gate === 'rejected' || fresh.output_parity === 'fail') {
        result = { ...checked, gate: 'rejected', speedup_review: { ...checked.speedup_review, status: 'rejected' } };
      }
    }
  }
  SPEEDUP_REVIEW_RESULTS.set(key, result);
  return result;
}
// Run ONE integrate A/B: an incomplete result is re-invoked in resume mode for the missing leg (up
// to AB_FINISH_RETRIES) within the remaining budget. Acceptance requires both legs; a leg still
// missing at the deadline stays incomplete. Every integrate site uses this.
async function runIntegrateBothLegs(intro, inputs, label, phaseName) {
  // Incomplete, so callers keep it PENDING, not rejected.
  if (abWindowShort(abLegs(inputs), label)) {
    return { gate: 'incomplete', ab_complete: false,
      reason: 'deferred: the trace-replay A/B cannot finish before the final reserve' };
  }
  // Caller inputs win for the candidate (the accuracy gate and the tuning carve-out only ADD keys);
  // correctness always uses the retained original seed.
  const withTuning = {
    MEASUREMENT_PURPOSE: 'search', REPLICAS: SEARCH_REPLICAS,
    PROFILE_TOPN_JSON: (profile && profile.profile_topN_json) || '',
    ...ACCURACY_INPUTS, ...tuningIntegrateInputs(), ...inputs,
    BASELINE_FLAGS, BASELINE_ENV, BASELINE_OVERLAY,
  };
  let integ = await adoptAbandonedVerdict(await safeAgent(
    roleAgent('e2e_integrator', 'integrate', intro, withTuning),
    { phase: phaseName, label, schema: INTEGRATE_SCHEMA }), label);
  let tries = 0;
  while (!abDone(integ) && tries < AB_FINISH_RETRIES) {
    if (searchClosed(`${label}: FINISH retry`)) break;
    if (abWindowShort(1, `${label}: FINISH retry`)) break;
    tries++;
    log(`  ${label}: A/B incomplete (${integ ? (integ.reason || integ.gate) : 'null/timeout'}); ` +
      `re-invoking integrator to FINISH the missing leg (retry ${tries}/${AB_FINISH_RETRIES}).`);
    const finishLabel = `${label} (finish ${tries})`;
    integ = await adoptAbandonedVerdict(await safeAgent(
      roleAgent('e2e_integrator', 'integrate',
        'FINISH this e2e A/B. A reference leg may already exist on disk under $CB/ref — reuse it and ' +
        'run ONLY the missing candidate leg (or re-run both if no usable ref exists), then return ' +
        'accepted/stack/rejected with ab_complete:true only after both legs satisfy the declared ' +
        'measurement protocol. Keep the required cases, samples, replicas and lifecycle. If time ' +
        'or execution fails before completion, return gate:"incomplete" with the saved evidence.',
        { ...withTuning, RESUME_AB: true }),
      { phase: phaseName, label: finishLabel, schema: INTEGRATE_SCHEMA }), finishLabel);
  }
  integ = await reviewSpeedup(integ, withTuning, label, phaseName);
  if (integ && integ.short_name) {
    // Preserve every candidate's review when several backends share one head short_name.
    const measurementId = (integ.measurement_evidence && integ.measurement_evidence.measurement_id) || withTuning.CAND_TAG || '';
    const candidateId = [integ.short_name, measurementId].filter(Boolean).join('_').replace(/[^A-Za-z0-9_.-]+/g, '_');
    const accepted = integEligible(integ);
    // integration_result keeps any speedup review as evidence; only a rejected one blocks.
    await persistE2EValidationCheckpoint(`overlay/candidate_${candidateId}/e2e_validation.json`, {
      phase: 'Overlay', validation_level: 'integrator', gate: integ.gate || 'incomplete',
      committed: accepted, validation_status: accepted ? 'accepted_intermediate' : 'incomplete',
      integration_result: integ,
      baseline_throughput_tok_s: integ.ref_med || 0,
      final_throughput_tok_s: integ.cand_med || integ.e2e_throughput_tok_s || 0,
      throughput_speedup: (integ.ref_med && integ.cand_med) ? integ.cand_med / integ.ref_med : 0,
      baseline_config: { flags: withTuning.EXTRA_SERVER_ARGS || '', env: withTuning.EXTRA_ENV || '' },
      accepted_config: { flags: withTuning.EXTRA_SERVER_ARGS || '', env: withTuning.EXTRA_ENV || '' },
      accepted_kernels: accepted ? [{ short_name: integ.short_name,
        kernel_slot: integ.target_callable || integ.short_name }] : [],
      accepted_heads: [], final_patch: [], final_overlay: { path: integ.accepted_overlay || '' },
      final_launch_script: { path: `${EVAL_DIR}/bench_e2e.sh` },
      bench_script: { path: `${EVAL_DIR}/bench_e2e.sh` },
      measurement: { measurement_mode: MEASUREMENT_MODE, metric_basis: 'aggregate_output_tok_s',
        workload: WORKLOAD, source_artifact: `${EVAL_DIR}/overlay`,
        acceptance: { gain_exceeds_noise: accepted, correctness_passed: integ.output_parity !== 'fail' } },
      stack: { kernel_slots: accepted ? [{ kernel_slot: integ.target_callable || integ.short_name,
        selected: true, candidate_id: candidateId }] : [] },
      replay: { asset_paths: [`${EVAL_DIR}/bench_e2e.sh`, integ.accepted_overlay].filter(Boolean) },
      integrity: { checkpoint_assets: [] },
    });
  }
  return integ;
}

// The lane's accepted primary metric; geomean only for an explicitly unweighted legacy result.
function isolatedPrimary(result, workloadPath) {
  if (!result) return null;
  return result.final_speedup === undefined && !workloadPath && result.workload_aligned === false
    ? result.final_geomean : result.final_speedup;
}

// Lightweight SURGICAL FIX (Tier 1 of the corrective): one kernel_surgeon agent makes the smallest
// fix that keeps the isolated win, self-verifies on the immutable unittest, and returns a
// kernel_workflow-shaped result, or null to escalate to the heavy re-author.
const SURGEON_SCHEMA = obj({
  fixed: { type: 'boolean' }, final_patch: { type: 'string' }, final_geomean: { type: 'number' },
  eval_dir: { type: 'string' }, root_cause: { type: 'string' }, note: { type: 'string' },
}, ['fixed']);
async function trySurgicalFix(spec, reason, fixClass, attempt) {
  if (!SURGICAL_FIX || spec.workload_path || spec.workload_aligned !== false) return null;
  const intro = `A verified-isolated ${spec.language || 'triton'} kernel winner for op "${spec.short_name}" ` +
    `(${(spec.isolated || 0).toFixed(2)}x isolated) ENGAGED live but was REJECTED at the e2e gate. ` +
    `Reject class = ${fixClass}. Reject reason: "${reason}". Make the SMALLEST possible fix that clears it ` +
    `while KEEPING the algorithm + the isolated win; do NOT re-optimize or re-explore. Self-verify on the ` +
    `IMMUTABLE unittest, then emit a minimal final_patch.`;
  const s = await safeAgent(
    roleAgent('kernel_surgeon', 'surgical_fix', intro, {
      TASK_DIR: spec.task_dir || '', KERNEL_EVAL_DIR: spec.kernel_eval_dir || '',
      CURRENT_PATCH: (spec.base_inputs && spec.base_inputs.KERNEL_RESULT &&
        (spec.base_inputs.KERNEL_RESULT.code_patch || spec.base_inputs.KERNEL_RESULT.final_patch)) || '',
      REJECT_REASON: reason, FIX_CLASS: fixClass, ISOLATED: spec.isolated || 0,
      LANGUAGE: spec.language || 'triton', GPU_ID: spec.gpu_id, KERNEL_WF_DIR,
    }),
    { phase: spec.phase_name || 'HeadKernel', label: `surgeon ${spec.short_name} (${attempt})`, schema: SURGEON_SCHEMA });
  if (s && s.fixed && s.final_patch && (s.final_geomean || 0) > 1.0) {
    return { authored: true, final_patch: s.final_patch, final_geomean: s.final_geomean, workload_aligned: false,
      eval_dir: s.eval_dir || spec.kernel_eval_dir || '', reason: s.root_cause || 'surgical fix' };
  }
  log(`  ${spec.short_name}: surgical fix did not produce a verified patch (${s ? (s.note || s.root_cause || 'not fixed') : 'null'}) — escalating to re-author.`);
  return null;
}

// CORRECTIVE RE-AUTHOR, used by every head-integration site. A candidate that passed the isolated
// oracle and engaged live but was rejected at the e2e gate for a fixable class (rejectClass) gets up
// to HEAD_CORRECTIVE_MAX fix-and-retries of the EXISTING kernel, each re-gated by a both-legs A/B.
// Not charged to HEAD_BUDGET; skipped once the deadline fired. spec.base_inputs is the integrate
// inputs template; spec.cur = {overlay,flags,env,tput}. Returns { banked, integ, isolated, ... }.
async function tryCorrectiveReauthor(spec) {
  let reason = spec.reason || '';
  // Which fix-and-retry class is this reject? '' = terminal (not auto-correctable).
  let fixClass = spec.fix_class || rejectClass(reason);
  const eligible = HEAD_CORRECTIVE_MAX > 0 && !((FAST_MODE && FAST_DEADLINE_HIT) || TIME_DEADLINE_HIT)
    && (spec.kernel_eval_dir || spec.task_dir) && (spec.isolated || 0) > 1.0 && fixClass !== '';
  if (!eligible) return { banked: false };
  const curTput = (spec.cur && spec.cur.tput) || 0;
  // Class-specific, generic instructions: `integration` = wrong posture (JIT/capture/host-sync);
  // `correctness` = wrong live output (parity/accuracy failure). Both keep the algorithm and isolated win.
  const INTEGRATION_FIX_TASK =
    `Fix ONLY the integration posture per knowledge/learned/method-cudagraph-safe-integration: ` +
    `precompile/register EVERY (shape-bucket × config) the LIVE workload hits — PREFILL buckets AND decode buckets, ` +
    `every per-bucket tile/config the kernel selects — at WARMUP before capture (an *_overlay_precompile(weights, ` +
    `scales, buckets) hook the integrator calls once, pre-capture) so ALL TP workers load a prebuilt code object ` +
    `instead of lazily compiling in the multiproc warmup (the cause of NO_BINARY_FOR_GPU / capture hang). If the cause ` +
    `is a host-sync (.item()/.cpu()/.sum().item()) on the hot path, remove it and cache weight prep by data_ptr(). ` +
    `Keep the steady-state hot path host-sync-free. ` +
    `If the reject is no_rebind_seam / no_engagement (the overlay bound but never ran on the live path): FIND the ` +
    `method the live server ACTUALLY dispatches (grep the cand server.log for which entry engaged — e.g. the ` +
    `modular 'TritonExperts.apply'/'invoke_fused_moe_triton_kernel' path, NOT a dead legacy '*_impl'), then rebind ` +
    `the overlay at THAT seam and MATCH its call signature so the kernel is invoked. Prove it re-engages ` +
    `(engagement_check > 0) before returning.`;
  const CORRECTNESS_FIX_TASK =
    `The kernel is WRONG on the LIVE path (reject: "${reason}") even though it passed the isolated oracle — the ` +
    `classic single-snapshot OVER-FIT. Live serving calls this op every step with CHANGING contents (routing / ` +
    `token→expert assignment / gather-scatter indices / masks) while REUSING the same buffers (stable data_ptr, ` +
    `new contents); the fixed-tensor unittest hides this. Fix the CORRECTNESS without giving up the speedup: ` +
    `(1) NEVER key a cache on tensor.data_ptr()/id() for any tensor whose CONTENTS vary per call (routing/gather/ ` +
    `scatter/mask/index) — recompute it each call (cheap, and CUDA-graph-capture-safe) or key on a value hash; ` +
    `only cache things that depend purely on SHAPE/DTYPE (e.g. tile/occupancy config). (2) Do not carry per-call ` +
    `state across calls or assume the previous call's indices/mask still apply. (3) If you use atomics/scatter-add ` +
    `into a reused output, zero it each call inside the captured region. VALIDATE your fix adversarially: call the ` +
    `kernel on TWO different routing snapshots back-to-back REUSING the same input buffers, and on repeated calls ` +
    `with fresh contents, asserting bit/tol correctness on EACH — a data_ptr cache or stale reuse must fail this. ` +
    `A gain above the conditional Amdahl projection alone does not establish wrong output; preserve actual correctness evidence.`;
  for (let cAttempt = 1; cAttempt <= HEAD_CORRECTIVE_MAX; cAttempt++) {
    log(`  ${spec.short_name}: ${fixClass.toUpperCase()} reject (${reason}) — corrective ${cAttempt}/${HEAD_CORRECTIVE_MAX} (fix-and-retry the iso winner; NOT a new head, NOT charged to head_budget).`);
    // TIER 1: lightweight surgical patch (one agent, minutes).
    let fix = await trySurgicalFix(spec, reason, fixClass, cAttempt);
    const viaSurgical = !!(fix && fix.final_patch && (fix.final_geomean || 0) > 1.0);
    // TIER 2: the heavyweight kernel_workflow re-author, only if the surgical patch failed.
    if (!viaSurgical) {
      try {
        fix = await fastBoundedWorkflow({ scriptPath: KERNEL_WF_SCRIPT }, {
          kernel_path: spec.kernel_eval_dir || spec.task_dir, workflow_dir: KERNEL_WF_DIR,
          mode: 'optimize', target_language: spec.language || 'triton',
          op_spec: { op_kind: spec.op_kind, shapes: spec.shapes || {}, dtype: spec.dtype || 'bf16', regime: spec.regime || '', cuda_graph_safe: true, ...(spec.workload_path ? { workload_path: spec.workload_path } : {}) },
          perf_knowledge_dir: KERNEL_KNOWLEDGE_DIR,
          use_expert_skills: USE_EXPERT_SKILLS ? 'true' : 'false', expert_skills_dir: EXPERT_SKILLS_DIR,
          // Keep the isolated win: KB history is readable but never auto-applied here.
          ...KB_ARGS, warm_start: WARM_START_OFF ? KB_ARGS.warm_start : 'reference',
          budget: KERNEL_BUDGET, gpu_ids: spec.gpu_id, exp_root: `${EVAL_DIR}/kernels/_exp`,
          task: `CORRECTIVE FIX — do NOT re-discover the algorithm; KEEP the ${(spec.isolated || 0).toFixed(2)}x isolated win. ` +
            `This kernel PASSED the isolated oracle and ENGAGED on all live workers but was REJECTED at the e2e serving gate ` +
            `for: "${reason}". ` + (fixClass === 'correctness' ? CORRECTNESS_FIX_TASK : INTEGRATION_FIX_TASK) +
            ` Emit a fixed final_patch. ` + GRAPH_REQ + (TASK || ''),
          apply_to_original: 'false',
        }, `${spec.short_name}:corrective`);
      } catch (e) { fix = { authored: false, validation_status: 'error', reason: String(e) }; }
    }
    const fixedIsolated = isolatedPrimary(fix, spec.workload_path);
    if (!fix || fix.authored === false || !Number.isFinite(fixedIsolated) || !(fixedIsolated > 1.0)
        || (!viaSurgical && fix.validation_status !== 'accepted')
        || typeof fix.final_patch !== 'string' || !fix.final_patch.trim()) {
      log(`  ${spec.short_name}: corrective produced no usable kernel (${fix ? fix.reason || fix.validation_status : 'null'}).`);
      return { banked: false };
    }
    log(`  ${spec.short_name}: fix via ${viaSurgical ? 'SURGICAL patch (light)' : 'kernel_workflow re-author (heavy)'}; isolated primary ${fixedIsolated.toFixed(2)}. Re-gating e2e.`);
    // Re-gate the fixed kernel in the caller's KERNEL_RESULT shape, swapping in the corrected patch,
    // eval_dir and speedup. Both patch fields get the new patch; a stale one would re-apply the bug.
    const base = spec.base_inputs || {};
    const fixInputs = { ...base,
      KERNEL_RESULT: { ...(base.KERNEL_RESULT || {}),
        code_patch: fix.final_patch, final_patch: fix.final_patch,
        authored_kernel_eval_dir: fix.eval_dir || spec.kernel_eval_dir || (base.KERNEL_RESULT && base.KERNEL_RESULT.authored_kernel_eval_dir) || '',
        verified_isolated_speedup: fixedIsolated, isolated_geomean: fix.final_geomean ?? null,
        pct_gpu_time: spec.pct_gpu_time || (base.KERNEL_RESULT && base.KERNEL_RESULT.pct_gpu_time) || 0,
        corrective_fix_of: reason } };
    if (spec.cur) {
      fixInputs.CURRENT_OVERLAY = spec.cur.overlay; fixInputs.CURRENT_FLAGS = spec.cur.flags;
      fixInputs.CURRENT_ENV = spec.cur.env; fixInputs.CURRENT_THROUGHPUT = spec.cur.tput;
    }
    const integ2 = await runIntegrateBothLegs(
      'Apply the CORRECTIVELY-FIXED kernel winner; gate on e2e throughput.', fixInputs,
      `integrate ${spec.short_name} corrective`, spec.phase_name || 'HeadKernel');
    const ab2 = !!(integ2 && integ2.gate !== 'incomplete' && integ2.ab_complete !== false);
    const pctForDiagnostic = spec.pct_gpu_time || (spec.base_inputs && spec.base_inputs.KERNEL_RESULT && spec.base_inputs.KERNEL_RESULT.pct_gpu_time) || 0;
    if (ab2 && integAccepted(integ2, pctForDiagnostic, fixedIsolated) && integ2.e2e_throughput_tok_s > curTput) {
      // Carry the corrected patch out to bankAccepted: the call site's KERNEL_RESULT still holds the
      // rejected one.
      return { banked: true, integ: integ2, isolated: fixedIsolated, isolated_geomean: fix.final_geomean ?? null, patch: fix.final_patch,
        kernel_eval_dir: fix.eval_dir || spec.kernel_eval_dir || '' };
    }
    reason = gateRejectReason(integ2, pctForDiagnostic, fixedIsolated) || reason;
    // Structured stop, not a re-classification of the prose reason: once parity no longer fails, a
    // throughput or review issue cannot justify another correctness rewrite.
    if (ab2 && integ2.output_parity !== 'fail') {
      log(`  ${spec.short_name}: corrective has no bankable e2e win (${reason}) — stopping kernel rewrites.`);
      break;
    }
    log(`  ${spec.short_name}: corrective still rejected (${reason}).`);
    fixClass = rejectClass(reason);
    if (fixClass === '') break;   // new failure not auto-correctable -> stop retrying
    // Progressive: the NEXT attempt builds on this attempt's (partially) fixed kernel, not the original.
    spec.kernel_eval_dir = fix.eval_dir || spec.kernel_eval_dir;
  }
  return { banked: false };
}

// --- FAST-MODE wall-clock control (no-op unless FAST_MODE) ----
// Date.now() is unavailable here (it would break resume), so budgets use setTimeout: a one-shot flag
// stops the head loop STARTING new ops, and fastBoundedWorkflow() caps each nested kernel workflow.
let FAST_DEADLINE_HIT = false;
if (FAST_MODE && typeof setTimeout === 'function' && FAST_HEAD_DEADLINE_MS > 0) {
  setTimeout(() => {
    FAST_DEADLINE_HIT = true;
    log(`[fast-mode] head-dispatch deadline (${Math.round(FAST_HEAD_DEADLINE_MS / 60000)}min) reached — no NEW head ops will start; finishing the in-flight head then proceeding to Finalize/Validate within the ${Math.round(FAST_BUDGET_MS / 60000)}min budget.`);
  }, FAST_HEAD_DEADLINE_MS);
}
// A nested kernel workflow raced against the tighter of the fast-mode cap and the trace-replay
// reserve bound. On expiry it resolves null, which callers treat as "no kernel".
async function fastBoundedWorkflow(ref, wfArgs, label) {
  if (laneWindowShort(`nested kernel workflow ${label || ''}`)) return null;
  wfArgs = { ...wfArgs, ...(await laneDeadlineArgs(label, 'HeadKernel')) };
  // noteKernelKB returns `r` unchanged, so the caller's null-guards are untouched.
  const p = workflow(ref, laneArgs(wfArgs)).then((r) => noteKernelKB(r, label));
  const fastCap = FAST_MODE && FAST_HEAD_WF_MS > 0 ? FAST_HEAD_WF_MS : null;
  const resCap = laneReserveCapMs();
  const capMs = [fastCap, resCap].reduce((m, c) => (c != null && (m == null || c < m) ? c : m), null);
  if (typeof setTimeout !== 'function' || !(capMs > 0)) return p;
  return raceLaneCap(p, capMs, () => {
    if (capMs === resCap) return logLaneAtReserve(label);
    log(`  [fast-mode] nested kernel workflow ${label || ''} exceeded ${Math.round(capMs / 60000)}min — abandoning (null) to stay on budget.`);
  });
}
// Milestone lanes: the trace-replay reserve bound alone.
async function reserveBoundedWorkflow(ref, wfArgs, label) {
  if (laneWindowShort(`nested kernel workflow ${label || ''}`)) return null;
  wfArgs = { ...wfArgs, ...(await laneDeadlineArgs(label, 'Milestone')) };
  const p = workflow(ref, laneArgs(wfArgs)).then((r) => noteKernelKB(r, label));
  const capMs = laneReserveCapMs();
  if (typeof setTimeout !== 'function' || !(capMs > 0)) return p;
  return raceLaneCap(p, capMs, () => logLaneAtReserve(label));
}
function raceLaneCap(p, capMs, onCap) {
  let to;
  const guard = new Promise((resolve) => {
    to = setTimeout(() => { onCap(); resolve(null); }, capMs);
  });
  return Promise.race([p.then((r) => { clearTimeout(to); return r; }, (e) => { clearTimeout(to); throw e; }), guard]);
}
function logLaneAtReserve(label) {
  log(`  [time-budget] nested kernel workflow ${label || ''} still running at the final reserve — abandoning (null).`);
}

// --- DEEP-MODE wall-clock control (no-op unless DEEP_MODE) ----
// Same mechanism: a one-shot flag stops NEW co-opt waves once the HeadKernel budget is spent, and
// deepBoundedWorkflow() caps each nested kernel_workflow burst so one backend can't stall a wave.
let DEEP_DEADLINE_HIT = false;
if (DEEP_MODE && typeof setTimeout === 'function' && DEEP_HEAD_BUDGET_MS > 0) {
  setTimeout(() => {
    DEEP_DEADLINE_HIT = true;
    log(`[deep-mode] HeadKernel budget (${Math.round(DEEP_HEAD_BUDGET_MS / 3600000)}h) reached — no NEW co-opt waves will start; finishing the in-flight wave then proceeding to Finalize/Validate.`);
  }, DEEP_HEAD_BUDGET_MS);
}
// --- DEFAULT-MODE wall-clock control (no-op unless time_budget_s passed) ----
// One flag for every mode: stop STARTING new head/milestone work at TIME_HEAD_DEADLINE_MS, leaving
// the tail for in-flight work and Finalize/Report/Validate.
let TIME_DEADLINE_HIT = false;
if (TIME_HEAD_DEADLINE_MS != null && typeof setTimeout === 'function' && TIME_HEAD_DEADLINE_MS > 0) {
  setTimeout(() => {
    TIME_DEADLINE_HIT = true;
    log(`[time-budget] dispatch deadline (${Math.round(TIME_HEAD_DEADLINE_MS / 60000)}min of ${Math.round(TIME_BUDGET_MS / 60000)}min budget) reached — no NEW head/milestone work will start; finishing in-flight then proceeding to Finalize/Report/Validate before the hard kill.`);
  }, TIME_HEAD_DEADLINE_MS);
}
// --- RESERVE BOUNDARY ----
// The one flag saying "the reserve has begun" (it also bounds the Finalize-gate drain): a single
// absolute setTimeout, so unlike remainingMs() it cannot drift. Inert without time_budget_s.
let TIME_FINAL_DEADLINE_HIT = false;
if (TIME_BUDGET_EFFECTIVE_MS != null && typeof setTimeout === 'function' && TIME_BUDGET_EFFECTIVE_MS > 0) {
  setTimeout(() => {
    TIME_FINAL_DEADLINE_HIT = true;
    log(`[time-budget] RESERVE BOUNDARY reached (${Math.round(TIME_BUDGET_EFFECTIVE_MS / 60000)}min of the ${Math.round(TIME_BUDGET_MS / 60000)}min budget spent) — the remaining ~${Math.round(FINAL_RESERVE_MS / 60000)}min belongs to Finalize/Report/Validate. No NEW A/B or measured work starts from here.`);
  }, TIME_BUDGET_EFFECTIVE_MS);
}
// The same rule for everything ahead of the Finalize-gate: the search phases, their re-profiles, the
// KB replays and FINISH retries. Each checks before it dispatches; see atReserveFloor() for why.
function searchClosed(what) {
  if (FINAL_PHASE_STARTED || (!TIME_FINAL_DEADLINE_HIT && !atReserveFloor())) return false;
  log(`[time-budget] ${what} SKIPPED — the final reserve has begun.`);
  return true;
}
// The tuning loop is hours of GPU work under ONE agent, which the reserve can abandon but not stop,
// so it starts only with this much left before the reserve. tuning_min_window_s=0 disables.
const _tuningMinWindowS = parseInt(A.tuning_min_window_s, 10);
const TUNING_MIN_WINDOW_MS = (Number.isFinite(_tuningMinWindowS) && _tuningMinWindowS >= 0
  ? _tuningMinWindowS : 21600) * 1000;   // 6h
function tuningWindowShort() {
  if (TIME_BUDGET_MS == null || !(TUNING_MIN_WINDOW_MS > 0)) return false;
  const windowMs = remainingMs() - FINAL_RESERVE_MS;
  if (windowMs >= TUNING_MIN_WINDOW_MS) return false;
  log(`[time-budget] TuningSkillset SKIPPED — ~${Math.max(0, Math.round(windowMs / 60000))}min left before ` +
    `the final reserve, under the ${Math.round(TUNING_MIN_WINDOW_MS / 60000)}min the tuning loop needs to ` +
    'finish its A/B before Finalize and Validate need the GPUs.');
  return true;
}
function deepBoundedWorkflow(ref, wfArgs, label) {
  const p = workflow(ref, laneArgs(wfArgs)).then((r) => noteKernelKB(r, label));
  if (!DEEP_MODE || typeof setTimeout !== 'function' || !(DEEP_HEAD_WF_MS > 0)) return p;
  let to;
  const guard = new Promise((resolve) => {
    to = setTimeout(() => {
      log(`  [deep-mode] nested kernel_workflow burst ${label || ''} exceeded ${Math.round(DEEP_HEAD_WF_MS / 60000)}min — abandoning (null) so the wave barrier proceeds; STATE_DIR keeps its progress for the next wave.`);
      resolve(null);
    }, DEEP_HEAD_WF_MS);
  });
  return Promise.race([p.then((r) => { clearTimeout(to); return r; }, (e) => { clearTimeout(to); throw e; }), guard]);
}
// --- TRACE-REPLAY SEARCH vs THE RESERVE ----
// Trace-replay lanes and A/Bs start only if they fit before the boundary (inert off AgentX/time_budget_s).
const AGENTX_LANE_ROUND_S = 3600;    // a lane round in flight at its stop, plus the lane's report
const AGENTX_EXTRACT_S = 4200;       // kernel extraction
const agentxGuardsOn = () => IS_AGENTX && TIME_BUDGET_MS != null && !FINAL_PHASE_STARTED;
const searchWindowMs = () => remainingMs() - FINAL_RESERVE_MS;
// A reused or resumed reference leg is not run again.
const abLegs = (inputs) => (inputs && (inputs.REUSE_REF || inputs.RESUME_AB) ? 1 : 2);
// A leg timed over several passes costs that many legs (an upper bound under warm_server).
const agentxAbMs = (legs, passes = 1) =>
  1000 * (legs * passes * agentxLegS(_declaredWindowS(AGENTX.geak_loop_duration_s, 900)) + AGENTX_AB_OVERHEAD_S);
const agentxLaneTailMs = () => agentxAbMs(2) + AGENTX_LANE_ROUND_S * 1000;
const agentxLaneStartMs = () => agentxLaneTailMs() + AGENTX_LANE_ROUND_S * 1000;
// need() reads AGENTX, so it runs only once the guard holds.
function windowShort(need, what, verdict, cost) {
  if (!agentxGuardsOn()) return false;
  const windowMs = searchWindowMs(), needMs = need();
  if (windowMs >= needMs) return false;
  log(`[time-budget] ${what} ${verdict} — ~${Math.max(0, Math.round(windowMs / 60000))}min left before the ` +
    `final reserve, under the ~${Math.round(needMs / 60000)}min ${cost}.`);
  return true;
}
// An A/B that cannot finish is deferred (incomplete, pending), not abandoned mid-leg at the boundary.
function abWindowShort(legs, what, passes = 1) {
  return windowShort(() => agentxAbMs(legs, passes), what, 'DEFERRED',
    `a ${legs}-leg trace-replay A/B${passes > 1 ? ` with ${passes} timed passes a leg` : ''} takes`);
}
// A lane needs one round before its stop plus the A/B for what it found.
function laneWindowShort(what) {
  return windowShort(agentxLaneStartMs, what, 'NOT STARTED', 'a kernel lane and its A/B need');
}
// Every search candidate needs kernel extraction and a lane; without both the run ends early.
function searchWindowShort(what) {
  return windowShort(() => AGENTX_EXTRACT_S * 1000 + agentxLaneStartMs(), what, 'SKIPPED',
    'kernel extraction plus one kernel lane and its A/B need. Going straight to Finalize and Validate');
}
// kernel_lane.js starts no round after deadline_epoch, so the stop is a lane tail ahead of the boundary;
// no_stop_s past the window leaves the TechLead free to end it early. No clock (Date.now() breaks
// resume): ONE `date +%s` read, anchored to ELAPSED_MS; unreadable, only the backstop bounds the lane.
const CLOCK_SCHEMA = obj({ epoch: { type: 'number' } }, ['epoch']);
let EPOCH_AT_T0 = null;
async function epochNow(phaseName) {
  if (EPOCH_AT_T0 == null) {
    EPOCH_AT_T0 = safeAgent('Run EXACTLY this command and nothing else:\n```bash\ndate +%s\n```\n' +
      'Return {"epoch": <the integer it printed>}. Do NOT modify any file and do NOT run anything else.',
      { phase: phaseName, label: 'clock', effort: 'low', schema: CLOCK_SCHEMA }, 1, () => 5 * 60000)
      .then((r) => (r && Number.isFinite(r.epoch) ? Math.floor(r.epoch) - Math.floor(ELAPSED_MS / 1000) : null));
  }
  const t0 = await EPOCH_AT_T0;
  if (t0 == null) EPOCH_AT_T0 = null;
  return t0 == null ? null : t0 + Math.floor(ELAPSED_MS / 1000);
}
async function laneDeadlineArgs(label, phaseName) {
  if (!agentxGuardsOn()) return {};
  const stopInS = Math.floor((searchWindowMs() - agentxLaneTailMs()) / 1000);
  if (!(stopInS > 0)) return {};
  const now = await epochNow(phaseName);
  if (now == null) {
    log(`  [time-budget] ${label}: wall clock unreadable, no round deadline.`);
    return {};
  }
  log(`  [time-budget] ${label}: no new round after ~${Math.round(stopInS / 60)}min ` +
    `(${Math.round(agentxLaneTailMs() / 60000)}min before the final reserve).`);
  return { deadline_epoch: now + stopInS, no_stop_s: stopInS + 86400 };
}
// Backstop: never wait for a lane past the boundary.
const laneReserveCapMs = () => (agentxGuardsOn() ? Math.max(AGENT_TIMEOUT_FLOOR_MS, searchWindowMs()) : null);

// GPU semaphore (FAST MODE only for head jobs; deep mode's co-opt pools also use one): exclusive
// leases of physical card ids, so concurrent isolated measurements never share a GPU. Deadlock-free:
// a waiter holds no cards and acquires its full count atomically.
