// Agent calls, extraction, integration and workflow deadlines.
function agentBounded(rawPrompt, opts, onHung) {
  const prompt = withProcessSafety(rawPrompt);
  const timeoutMs = agentTimeoutFor();
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

// Attempts the hung-agent guard gave up on inside the reserve, by label. Each is still running, and
// for an integrator its eventual return is the verdict of the A/B it was measuring
// (adoptAbandonedVerdict).
const ABANDONED_RUNS = new Map();

async function safeAgent(prompt, opts, tries = 3) {
  let lastErr = 'unknown';
  for (let i = 0; i < tries; i++) {
    try {
      const ev = tlAgent(prompt, opts, i + 1);   // record AT DISPATCH; ok=false until it resolves
      let hung = null;
      const r = await agentBounded(prompt, opts, (run) => { hung = run; });
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

// --- FlyDSL provisioning gate -------------------------------------------------
// Design intent (roles/system_architect.md §0a): the MOMENT strategize routes a flydsl backend onto
// any head/kernel candidate, FlyDSL must be provisioned via the ensure_flydsl expert skill BEFORE any
// consumer can select it. We enforce it in the ORCHESTRATOR — not by trusting the architect LLM to have
// run it — so "decide to use flydsl" and "ensure_flydsl" are one inseparable step. The detect->reuse-or-build
// decision lives ENTIRELY inside ensure_flydsl.sh (its own version gate: import flydsl + kernels.moe_gemm_2stage
// AND __version__ >= FLYDSL_MIN_VERSION, default 0.2.2). is_flydsl_available() is NOT used to decide anything.
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
  const deployable = selectionCandidatesForHead(out).filter((candidate) => {
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
function kernelSelectionVerified(h, ext) {
  const required = requiredDeviceKernel(h);
  if (!required) return { ok: true, why: 'not a profiled GPU-head extraction' };
  const target = String((ext && ext.target_callable) || '').trim();
  if (!validCallableSpec(target))
    return { ok: false, why: `target_callable '${target}' is not an exact module:attr spec` };
  const verdict = ext && ext.selection_validation;
  if (!verdict || verdict.contract !== 'kernel_selection')
    return { ok: false, why: 'kernel_selection.py verdict is missing' };
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

// The Architect restates each head and has emitted the row's classification (library_gemm,
// triton_kernel) as entity_kind. The profiled row is the authority: a head whose device_kernel
// matches a profiled gpu_kernel row takes that row's entity_kind.
// Every row kind is evidence, not only gpu_kernel rows: canonicalization drops the namespace, so
// `aten::mul` (dispatcher_op) and `mul` (gpu_kernel) fold together, as parse_profile.py guards.
// Exact names decide first; a canonical match that reaches rows of different kinds is ambiguous.
// A row is identified by its most specific name only: a display short_name is shared across
// distinct kernels (every vectorized_elementwise_kernel instantiation), so it must not stand in for
// a concrete symbol the row already carries.
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
    // A gpu_kernel head with NO device identity at all makes requiredDeviceKernel return '', and
    // kernelSelectionVerified then passes it as "not a profiled GPU-head extraction". That vacuous
    // pass is the whole contract switched off for this head, so refuse it at admission instead.
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

// Run a kernel_extractor agent and GUARANTEE it froze a real baseline. safeAgent already retries
// transient failures; this wraps it to ALSO re-extract when the extraction succeeds (smoke passed,
// task dir present) but produced NO frozen baseline — re-invoking with a corrective instruction up
// to BASELINE_EXTRACT_RETRIES times. If a baseline still can't be frozen, we force the caller's
// existing extract-failure path (smoke/unittest_smoke -> 'fail' + a reason) so the head is flagged
// (if dominant) or skipped, NEVER timed against its own scaffold. `role`/`phase`/`intro`/`inputs`
// are the roleAgent args; `opts` is the safeAgent opts (phase/label/schema). Used by every extract
// site (deep, opt-A, milestone/head extract_op, and the non-op milestone extract).
async function extractWithBaseline(role, phase, intro, inputs, opts) {
  const smokeOk = (e) => !!(e && e.task_dir && (e.smoke === 'pass' || e.unittest_smoke === 'pass'));
  const head = (inputs && inputs.KERNEL) || {};
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
    // Best-effort reclaim before the agent retries (issue #429 storage amplification).
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
  return ext;
}

// abDone == the integrator measured BOTH legs (ref + cand) and emitted a real
// verdict. gate:'incomplete' or ab_complete:false means a leg is still missing.
const abDone = (integ) => !!(integ && integ.gate !== 'incomplete' && integ.ab_complete !== false);

// An e2e delta is a RATIO. These are the two numbers it is a ratio of, taken
// off the same A/B that produced it: ref_med is where the workload stood
// before this win, cand_med is where it stood after. Published together, a
// consumer can restate the win in points of one fixed baseline. Published
// alone, the only thing a consumer can do is add percentages that were each
// measured against a different denominator, and those do not compose.
const e2eFrom = (integ) => ({
  e2e_delta_pct: integ.e2e_delta_pct,
  base_tput: integ.ref_med,
  new_tput: integ.cand_med,
});
// The KERNEL_RESULT that a given integrate A/B actually gated. Both patch spellings are read at the
// call sites because the two tracks fill different ones (head=code_patch, milestone=final_patch) and
// tryCorrectiveReauthor deliberately sets BOTH to the corrected patch.
const krOf = (inputs) => (inputs && inputs.KERNEL_RESULT) || {};

// Bank an accepted win, carrying the fields a KB reader needs to REPLAY it on another box.
// e2e_store.py:_accepted_kernels starts from dict(item) so anything here survives verbatim into the
// record — but it LOOKS UP `name`, `language`, `isolated_speedup` and `patch`, while this workflow
// only ever pushed `short_name`, `backend` and `isolated`. Every accepted kernel was therefore
// matched by nothing and silently dropped on the way into the store, which is why both e2e records
// already in the remote KB were written with accepted_kernels:[] and no patch at all. The old
// spellings stay (the report, the resume state and run_e2e.py all read them); the aliases are added
// alongside. `e.patch` wins over the KERNEL_RESULT's because a corrective re-author supplies the
// FIXED patch and the gated one is the broken kernel.
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

// An integrator abandoned at the reserve boundary is still measuring, and is usually minutes from its
// verdict. No FINISH retry may start inside the reserve, so that return is the only verdict this A/B
// can still get: wait for it, but only through the first LATE_VERDICT_RESERVE_FRAC of the reserve, so
// Finalize/Report/Validate keep the rest.
const LATE_VERDICT_RESERVE_FRAC = 0.25;
async function adoptAbandonedVerdict(integ, label) {
  const run = ABANDONED_RUNS.get(label);
  ABANDONED_RUNS.delete(label);
  if (integ || !run || !atReserveFloor() || typeof setTimeout !== 'function') return integ;
  const waitMs = remainingMs() - Math.ceil(FINAL_RESERVE_MS * (1 - LATE_VERDICT_RESERVE_FRAC));
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

// Retry an incomplete A/B within the remaining budget. Acceptance requires both
// legs; a missing leg at the deadline remains incomplete.
async function runIntegrateBothLegs(intro, inputs, label, phaseName) {
  // Caller inputs win for the candidate; correctness always uses the retained original seed.
  const withTuning = {
    MEASUREMENT_PURPOSE: 'search', REPLICAS: SEARCH_REPLICAS,
    ...ACCURACY_INPUTS, ...tuningIntegrateInputs(), ...inputs,
    BASELINE_FLAGS, BASELINE_ENV, BASELINE_OVERLAY,
  };
  let integ = await adoptAbandonedVerdict(await safeAgent(
    roleAgent('e2e_integrator', 'integrate', intro, withTuning),
    { phase: phaseName, label, schema: INTEGRATE_SCHEMA }), label);
  let tries = 0;
  while (!abDone(integ) && tries < AB_FINISH_RETRIES) {
    if (searchClosed(`${label}: FINISH retry`)) break;
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
  if (integ && integ.short_name) {
    const candidateId = String(integ.short_name).replace(/[^A-Za-z0-9_.-]+/g, '_');
    const accepted = integ.gate === 'accepted' || integ.gate === 'stack';
    await persistE2EValidationCheckpoint(`overlay/candidate_${candidateId}/e2e_validation.json`, {
      phase: 'Overlay', validation_level: 'integrator', gate: integ.gate || 'incomplete',
      committed: accepted, validation_status: accepted ? 'accepted_intermediate' : 'incomplete',
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

// Lightweight SURGICAL FIX (Tier 1 of the corrective). One focused kernel_surgeon agent reads the reject
// diagnosis + the failing kernel + the live call seam (from the task's meta.json), makes the SMALLEST
// edit that fixes the defect WITHOUT giving up the isolated win, self-verifies on the IMMUTABLE unittest
// (correctness PASS + geomean preserved), and emits a final_patch. Returns a fix object shaped like the
// kernel_workflow result ({final_patch, final_geomean, eval_dir, authored}) so the corrective re-gate path
// is identical, or null if it could not produce a verified patch (=> escalate to the heavy re-author).
const SURGEON_SCHEMA = obj({
  fixed: { type: 'boolean' }, final_patch: { type: 'string' }, final_geomean: { type: 'number' },
  eval_dir: { type: 'string' }, root_cause: { type: 'string' }, note: { type: 'string' },
}, ['fixed']);
async function trySurgicalFix(spec, reason, fixClass, attempt) {
  if (!SURGICAL_FIX) return null;
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
    return { authored: true, final_patch: s.final_patch, final_geomean: s.final_geomean,
      eval_dir: s.eval_dir || spec.kernel_eval_dir || '', reason: s.root_cause || 'surgical fix' };
  }
  log(`  ${spec.short_name}: surgical fix did not produce a verified patch (${s ? (s.note || s.root_cause || 'not fixed') : 'null'}) — escalating to re-author.`);
  return null;
}

// Reusable CORRECTIVE RE-AUTHOR (general; used by every head-integration site, any mode). A head
// candidate that PASSED the isolated oracle and ENGAGED live but was REJECTED at the e2e gate for a
// FIXABLE integration reason (JIT/DSL kernel lazily compiling in TP>1 warmup -> NO_BINARY_FOR_GPU /
// cuda_graph_capture_unsafe / capture hang / host-sync) earns up to HEAD_CORRECTIVE_MAX cheap fix-and-
// retries: re-OPTIMIZE the EXISTING kernel (keep the algorithm + isolated win; fix only the integration
// posture), then re-run the both-legs e2e A/B. This is NOT a new head discovery, so it does NOT consume
// HEAD_BUDGET; it is bounded per head and skipped once the kernel-phase wall-clock deadline has fired.
// spec: { short_name, op_kind, shapes, dtype, regime, gpu_id, kernel_eval_dir, task_dir, language,
//         isolated, base_inputs (the integrate inputs template, carries KERNEL_RESULT), reason,
//         cur:{overlay,flags,env,tput} }.  Returns { banked, integ, isolated } (banked=false if
// ineligible or still rejected). See knowledge/learned/method-cudagraph-safe-integration.
async function tryCorrectiveReauthor(spec) {
  let reason = spec.reason || '';
  // Which fix-and-retry class is this reject? '' = terminal (not auto-correctable).
  let fixClass = spec.fix_class || rejectClass(reason);
  const eligible = HEAD_CORRECTIVE_MAX > 0 && !((FAST_MODE && FAST_DEADLINE_HIT) || TIME_DEADLINE_HIT)
    && (spec.kernel_eval_dir || spec.task_dir) && (spec.isolated || 0) > 1.0 && fixClass !== '';
  if (!eligible) return { banked: false };
  const curTput = (spec.cur && spec.cur.tput) || 0;
  // The corrective instruction is CLASS-SPECIFIC. `integration` = the posture is wrong (JIT/capture/
  // host-sync); `correctness` = the output is wrong on the live path (parity/accuracy fail or an
  // implausible speedup), i.e. the kernel over-fit the single captured snapshot. Both KEEP the
  // algorithm + isolated win and fix only the defect. GENERIC — the text describes the failure CLASS,
  // never this specific kernel/model.
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
    `An implausible speedup (far above the op's Amdahl ceiling) means it is doing less/degenerate work — treat as wrong.`;
  for (let cAttempt = 1; cAttempt <= HEAD_CORRECTIVE_MAX; cAttempt++) {
    log(`  ${spec.short_name}: ${fixClass.toUpperCase()} reject (${reason}) — corrective ${cAttempt}/${HEAD_CORRECTIVE_MAX} (fix-and-retry the iso winner; NOT a new head, NOT charged to head_budget).`);
    // TIER 1 — lightweight SURGICAL patch (one agent, minutes). Most rejects are a tiny seam bug.
    let fix = await trySurgicalFix(spec, reason, fixClass, cAttempt);
    const viaSurgical = !!(fix && fix.final_patch && (fix.final_geomean || 0) > 1.0);
    // TIER 2 — escalate to the HEAVYWEIGHT kernel_workflow re-author ONLY if the surgical patch failed.
    if (!viaSurgical) {
      try {
        fix = await fastBoundedWorkflow({ scriptPath: KERNEL_WF_SCRIPT }, {
          kernel_path: spec.kernel_eval_dir || spec.task_dir, workflow_dir: KERNEL_WF_DIR,
          mode: 'optimize', target_language: spec.language || 'triton',
          op_spec: { op_kind: spec.op_kind, shapes: spec.shapes || {}, dtype: spec.dtype || 'bf16', regime: spec.regime || '', cuda_graph_safe: true, ...(spec.workload_path ? { workload_path: spec.workload_path } : {}) },
          perf_knowledge_dir: KERNEL_KNOWLEDGE_DIR,
          use_expert_skills: USE_EXPERT_SKILLS ? 'true' : 'false', expert_skills_dir: EXPERT_SKILLS_DIR,
          // A corrective must KEEP the isolated win; adopting a stored patch over it would be exactly the
          // re-discovery the task forbids. History is still readable, just never auto-applied.
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
    if (!fix || fix.authored === false || !(fix.final_geomean > 1.0) || !fix.final_patch) {
      log(`  ${spec.short_name}: corrective produced no usable kernel (${fix ? fix.reason || fix.validation_status : 'null'}).`);
      return { banked: false };
    }
    log(`  ${spec.short_name}: fix via ${viaSurgical ? 'SURGICAL patch (light)' : 'kernel_workflow re-author (heavy)'}; iso geomean ${(fix.final_geomean || 0).toFixed(2)}. Re-gating e2e.`);
    // Re-gate the FIXED kernel. Preserve the caller's KERNEL_RESULT SHAPE (head=authored/code_patch,
    // milestone=editable/final_patch) and only swap in the corrected patch + eval_dir + iso speedup, so the
    // helper is track-agnostic. Set BOTH patch fields to the new patch — the integrator reads whichever
    // matches its apply mode; leaving the other stale would re-apply the broken kernel.
    const base = spec.base_inputs || {};
    const fixInputs = { ...base,
      KERNEL_RESULT: { ...(base.KERNEL_RESULT || {}),
        code_patch: fix.final_patch, final_patch: fix.final_patch,
        authored_kernel_eval_dir: fix.eval_dir || spec.kernel_eval_dir || (base.KERNEL_RESULT && base.KERNEL_RESULT.authored_kernel_eval_dir) || '',
        verified_isolated_speedup: fix.final_geomean,
        corrective_fix_of: reason } };
    if (spec.cur) {
      fixInputs.CURRENT_OVERLAY = spec.cur.overlay; fixInputs.CURRENT_FLAGS = spec.cur.flags;
      fixInputs.CURRENT_ENV = spec.cur.env; fixInputs.CURRENT_THROUGHPUT = spec.cur.tput;
    }
    const integ2 = await runIntegrateBothLegs(
      'Apply the CORRECTIVELY-FIXED kernel winner; gate on e2e throughput.', fixInputs,
      `integrate ${spec.short_name} corrective`, spec.phase_name || 'HeadKernel');
    const ab2 = !!(integ2 && integ2.gate !== 'incomplete' && integ2.ab_complete !== false);
    const pctForGuard = spec.pct_gpu_time || (spec.base_inputs && spec.base_inputs.KERNEL_RESULT && spec.base_inputs.KERNEL_RESULT.pct_gpu_time) || 0;
    // Implausible-speedup guard: a "win" whose e2e delta blows past the op's Amdahl ceiling is corruption
    // masquerading as a win (does less/degenerate work) — never bank it; treat as a correctness reject and
    // let the next corrective attempt fix it. GENERIC (uses only pct_gpu_time + isolated).
    const implausible2 = ab2 && (integ2.gate === 'accepted' || integ2.gate === 'stack')
      && isImplausibleSpeedup(pctForGuard, fix.final_geomean, integ2);
    if (ab2 && (integ2.gate === 'accepted' || integ2.gate === 'stack') && integ2.e2e_throughput_tok_s > curTput && !implausible2) {
      // Carry the CORRECTED patch out to the caller's bankAccepted. The gated KERNEL_RESULT at the
      // call site still holds the broken kernel, and recording that into the KB would publish a
      // patch that was rejected on this very box as if it were the win.
      return { banked: true, integ: integ2, isolated: fix.final_geomean, patch: fix.final_patch,
        kernel_eval_dir: fix.eval_dir || spec.kernel_eval_dir || '' };
    }
    reason = implausible2
      ? `implausible_speedup (+${(integ2.e2e_delta_pct || 0).toFixed(1)}% >> Amdahl ceiling +${amdahlCeilingPct(pctForGuard, fix.final_geomean).toFixed(1)}% — corruption)`
      : ((integ2 && (integ2.reason || integ2.gate)) || reason);
    // STRUCTURED stop (do NOT re-classify from the prose reason — it may MENTION "corruption" while saying
    // it was RESOLVED, which would loop wastefully). If this attempt made the kernel CORRECT (parity no
    // longer failing) and it was not an implausible speedup, the ONLY remaining problem is throughput/
    // do-no-harm — more re-authoring cannot create Amdahl headroom that isn't there, so STOP now.
    if (ab2 && integ2.output_parity !== 'fail' && !implausible2) {
      log(`  ${spec.short_name}: corrective produced a CORRECT kernel with no e2e win (do-no-harm: ${reason}) — stopping; throughput headroom for this op is exhausted.`);
      break;
    }
    log(`  ${spec.short_name}: corrective still rejected (${reason}).`);
    fixClass = implausible2 ? 'correctness' : rejectClass(reason);
    if (fixClass === '') break;   // new failure not auto-correctable -> stop retrying
    // Progressive: the NEXT attempt builds on this attempt's (partially) fixed kernel, not the original.
    spec.kernel_eval_dir = fix.eval_dir || spec.kernel_eval_dir;
  }
  return { banked: false };
}

// --- FAST-MODE wall-clock control (no-op unless FAST_MODE) -------------------------------------------
// Date.now()/new Date() are unavailable in workflow scripts (they would break resume), so the budget is
// enforced with setTimeout: (1) a one-shot deadline flag that stops the head loop from STARTING new ops,
// and (2) fastBoundedWorkflow() which races each nested kernel workflow against a per-head cap so the
// in-flight op can't overrun. Both are inert when FAST_MODE is off → default path is byte-identical.
let FAST_DEADLINE_HIT = false;
if (FAST_MODE && typeof setTimeout === 'function' && FAST_HEAD_DEADLINE_MS > 0) {
  setTimeout(() => {
    FAST_DEADLINE_HIT = true;
    log(`[fast-mode] head-dispatch deadline (${Math.round(FAST_HEAD_DEADLINE_MS / 60000)}min) reached — no NEW head ops will start; finishing the in-flight head then proceeding to Finalize/Validate within the ${Math.round(FAST_BUDGET_MS / 60000)}min budget.`);
  }, FAST_HEAD_DEADLINE_MS);
}
// Run a nested kernel workflow with a fast-mode time cap. When FAST_MODE is off it returns the raw
// workflow() promise (identical to a direct call); on cap-expiry it resolves null so the caller's
// existing null-guards treat it as "no kernel" and continue.
function fastBoundedWorkflow(ref, wfArgs, label) {
  // noteKernelKB returns `r` unchanged, so the caller's null-guards are untouched.
  const p = workflow(ref, laneArgs(wfArgs)).then((r) => noteKernelKB(r, label));
  const capMs = FAST_MODE && FAST_HEAD_WF_MS > 0 ? FAST_HEAD_WF_MS : null;
  if (typeof setTimeout !== 'function' || !(capMs > 0)) return p;
  let to;
  const guard = new Promise((resolve) => {
    to = setTimeout(() => {
      log(`  [fast-mode] nested kernel workflow ${label || ''} exceeded ${Math.round(capMs / 60000)}min — abandoning (null) to stay on budget.`);
      resolve(null);
    }, capMs);
  });
  return Promise.race([p.then((r) => { clearTimeout(to); return r; }, (e) => { clearTimeout(to); throw e; }), guard]);
}

// --- DEEP-MODE wall-clock control (no-op unless DEEP_MODE) -------------------------------------------
// Same mechanism as fast mode: a one-shot deadline flag stops the deep head scheduler from starting NEW
// co-opt waves once the 20h HeadKernel budget is spent (the in-flight wave + Finalize/Validate still run),
// and deepBoundedWorkflow() caps each nested kernel_workflow BURST so a slow backend can't stall the
// per-wave barrier. Both inert when DEEP_MODE is off → default/fast paths byte-identical.
let DEEP_DEADLINE_HIT = false;
if (DEEP_MODE && typeof setTimeout === 'function' && DEEP_HEAD_BUDGET_MS > 0) {
  setTimeout(() => {
    DEEP_DEADLINE_HIT = true;
    log(`[deep-mode] HeadKernel budget (${Math.round(DEEP_HEAD_BUDGET_MS / 3600000)}h) reached — no NEW co-opt waves will start; finishing the in-flight wave then proceeding to Finalize/Validate.`);
  }, DEEP_HEAD_BUDGET_MS);
}
// --- DEFAULT-MODE wall-clock control (no-op unless time_budget_s passed) -----------------------------
// fast/deep have their own deadline flags (capped above); the DEFAULT pipeline had NO time awareness at
// all — its Milestone editable-kernel loop and head bake-off track could run past the orchestrator's kill.
// Mirror the fast/deep mechanism with ONE general flag: stop STARTING new head/milestone work after 60% of
// the effective budget (same 60/40 carve as fast — leaves the tail for the in-flight task + Finalize/
// Report/Validate). Active in ALL modes when time_budget_s is set (harmless for fast/deep, which stop even
// earlier on their own flags); fully inert (never registered) when time_budget_s is absent => byte-identical.
let TIME_DEADLINE_HIT = false;
if (TIME_HEAD_DEADLINE_MS != null && typeof setTimeout === 'function' && TIME_HEAD_DEADLINE_MS > 0) {
  setTimeout(() => {
    TIME_DEADLINE_HIT = true;
    log(`[time-budget] dispatch deadline (${Math.round(TIME_HEAD_DEADLINE_MS / 60000)}min of ${Math.round(TIME_BUDGET_MS / 60000)}min budget) reached — no NEW head/milestone work will start; finishing in-flight then proceeding to Finalize/Report/Validate before the hard kill.`);
  }, TIME_HEAD_DEADLINE_MS);
}
// --- RESERVE BOUNDARY ---------------------------------------------------------------------------
// TIME_DEADLINE_HIT above only stops STARTING new head/milestone work; it does not bound the
// Finalize-gate that sits between it and phase('Finalize'), whose unbounded drain loop ate the whole
// reserve in the 20260823 sessions. This is the one flag saying "the reserve has begun": a single
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

// GPU semaphore (FAST MODE only — the default path never constructs one). Hands out EXCLUSIVE leases of
// physical card ids from a pool so two concurrent isolated jobs never share a GPU -> their op-bench /
// kernel-layer speed measurements never contend. Deadlock-free: a waiter holds 0 cards while queued and
// acquires its full count atomically (no hold-and-wait). Uses only Promises/arrays (no Date.now/Math.random,
// which the Workflow runtime forbids).
