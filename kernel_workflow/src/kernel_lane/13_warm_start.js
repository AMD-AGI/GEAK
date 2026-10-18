// ===========================================================================
// PHASE: WarmStart. The recorded speedup only RANKS; adoption is decided by a fresh
// measurement through the verify gate here. gfx comes from the baseline profile's
// on-box `device` string (no extra probe).
// ===========================================================================
// One benched candidate -> one of kb/attest.py's four outcomes. `inapplicable` is a verdict on the
// PAIRING, excluded from the retire arithmetic — both "this workspace lacks those files" and "it
// won but could not be committed here" belong there. A patch that ran and gave a WRONG ANSWER is
// `failed`: it applied, it ran, and it did not deliver what the record promised.
const kbVerdict = (ver, sp) => {
  if (!ver) return 'not_reproduced';
  if (says(ver.correctness, 'fail')) return 'failed';
  if (says(ver.status, 'apply_failed')) {
    return /outside_editable_set/i.test(String(ver.notes || '')) ? 'inapplicable' : 'not_reproduced';
  }
  if (!says(ver.status, 'verified')) return 'not_reproduced';
  if (!(sp > 1.0)) return 'failed';              // applied, ran, correct, just did not go faster
  return says(ver.correctness, 'pass') ? 'inapplicable' : 'not_reproduced';   // fast but never adopted
};
let warm_start = { adopted: false, read_reason: WARM_START_ON ? 'read' : 'disabled', candidates: [] };
let skipLoop = false;
if (WARM_START_ON && !setup.resumed && KB_ROOT_OK) {
  phase('WarmStart');
  if (!GFX) {
    warm_start.read_reason = 'missing_arch';
    log('[kb] warm-start skipped: no gfx detected from the baseline profile device string.');
  } else {
    // The two planes take a different root flag and a different name→page rule (the store addresses
    // by canonical id, so there is nothing to match fuzzily), and print the same JSON.
    const localResolveCmd = KB_MODE === 'store'
      ? `resolve-remote --plane local --store ${JSON.stringify(KB_STORE_DIR)}${KB_VERSION_FLAG}`
      : `resolve --root ${JSON.stringify(KB_ARTIFACTS_DIR)} --match ${WARM_START_MATCH}`;
    // Bring the local store level with the tree before reading it, scoped to this kernel and arch
    // so it costs a stat walk over one page. Writes file both schemes, so this is usually a no-op;
    // what it catches is the backlog and anything a curation pass imported by hand — without it,
    // reading by key looks like a cold start on the kernels with the longest history. Idempotent
    // and non-destructive (cmd_sync_local): the session id is a digest of the patch, and the write
    // carries the store's ledger, retraction and reproduction count forward.
    //
    // `|| true`: a sync that cannot run must degrade to a thinner page, never a failed warm start.
    // Its summary goes to a file rather than /dev/null because its failures are otherwise
    // indistinguishable from an empty KB — `skipped.unreadable` above all, which is what a tree
    // written by a root container looks like to everyone else.
    const syncCmd = KB_MODE === 'store' && KB_ARTIFACTS_DIR
      ? `python3 ${JSON.stringify(EXPERIENCE_STORE)} sync-local --root ${JSON.stringify(KB_ARTIFACTS_DIR)} \\
  --store ${JSON.stringify(KB_STORE_DIR)} --kernel-name ${JSON.stringify(KERNEL_NAME)} --gfx ${GFX} \\
  > ${JSON.stringify(EVAL_DIR + '/kb_sync.json')} 2>&1 || true\n`
      : '';
    const commonArgs =
      `--kernel-name ${JSON.stringify(KERNEL_NAME)} --language ${JSON.stringify(TARGET_LANGUAGE)} \\\n` +
      `  --gfx ${GFX} --top-n 3 --min-speedup ${WARM_START_MIN_SPEEDUP} \\\n` +
      `  --refs-dir ${JSON.stringify(EVAL_DIR + '/kb_references')}`;
    // Deliberately NO `--precision` on the read, though the write below states it. The filter earns
    // its keep on the tuned lane, where a table's destination filename encodes its dtype and a
    // mismatched one is simply unusable. A PATCH is source: a Triton kernel proven at bf16 is
    // usually the same code that helps at fp8, so narrowing here would turn a thin page into an
    // empty one and cost a cold start — the exact hours this read exists to avoid. Recording the
    // dtype without filtering on it keeps that call reversible once the pages are thick enough to
    // show whether cross-dtype patches actually carry.
    // Remote first, local curated tree as the fallback. The service is the shared plane and should
    // win when it has an answer, but it is still filling up, while `kb_artifacts/` holds a hand-
    // curated history (retired entries, one entry per direction) that a thin remote page must not
    // shadow. An empty remote answer is indistinguishable from a 404 on this scheme, so "no
    // candidates" — not "no error" — is what triggers the second read. Both reads are seconds and
    // no GPU; the thing they protect against is a cold start that costs hours.
    //
    // The e2e lane pushed this branch into cmd_resolve via `read_planes`; this one stays in bash
    // because its fallback is a DIFFERENT SUBCOMMAND — off the store scheme `localResolveCmd` reads
    // the curated tree, not a second plane of the same store, and no plane list expresses that.
    const resolveScript = KB_REMOTE === 'off'
      ? `${syncCmd}python3 ${JSON.stringify(EXPERIENCE_STORE)} ${localResolveCmd} \\\n  ${commonArgs}`
      : `${syncCmd}${KB_ENV_PRELUDE}
REMOTE_OUT=''
if [ -n "$KB_STORE_TOKEN" ]; then
  REMOTE_OUT=$(python3 ${JSON.stringify(EXPERIENCE_STORE)} resolve-remote --plane remote${KB_VERSION_FLAG} \\
  ${commonArgs} 2>/dev/null || true)
  if printf '%s' "$REMOTE_OUT" | python3 -c 'import json,sys; sys.exit(0 if (json.load(sys.stdin).get("candidates") or []) else 1)' 2>/dev/null; then
    printf '%s\\n' "$REMOTE_OUT"; exit 0
  fi
fi
python3 ${JSON.stringify(EXPERIENCE_STORE)} ${localResolveCmd} \\
  ${commonArgs}`;
    const resolved = await agentT(
      `You are the warm-start resolver. Run EXACTLY this ` +
      `${KB_REMOTE === 'off' && !syncCmd ? 'command' : 'script'} ` +
      `and return its single-line JSON stdout verbatim as StructuredOutput — do not add, drop, reorder, ` +
      `or reinterpret any field. ` +
      // Saying the sync line is not the answer keeps the agent from relaying its summary, or from
      // treating its `|| true` as an error worth retrying.
      (syncCmd ? `Its first line brings the on-disk knowledge base up to date and prints nothing ` +
        `(its summary is redirected to a file); the JSON you return is the LAST command's. ` : '') +
      (KB_REMOTE === 'off' ? '' :
        `It tries the shared KB Store service first and falls back to the on-disk knowledge base by ` +
        `itself; run it as one script, do not split it into separate commands. `) +
      `An empty candidate list is a VALID answer meaning "nothing recorded for this kernel yet". Do not ` +
      `retry with different flags and do not invent candidates:
\`\`\`bash
${resolveScript}
\`\`\``,
      { phase: 'WarmStart', label: 'warm_start:resolve', schema: WARMSTART_RESOLVE_SCHEMA }) || {};
    warm_start.read_reason = resolved.read_reason || 'read';
    warm_start.slug = resolved.slug || '';
    warm_start.match_tier = resolved.match_tier || '';
    warm_start.filtered = resolved.filtered || null;
    const cands = Array.isArray(resolved.candidates) ? resolved.candidates : [];
    warm_start.candidates = cands.map(c => ({
      rank: c.rank, slug: c.slug, speedup: c.speedup, direction: c.direction || '', status: 'read',
      session_id: c.session_id || '', exp_dir: c.exp_dir || '',
    }));
    const f = resolved.filtered || {};
    // Which plane answered — read off the answer, not inferred from the mode. Both keyed reads
    // emit `canonical_id`, so only `read_plane` (kb/plane.py:read_planes) separates service from
    // disk. Not cosmetic: it picks the flags the attest step files this run's verdict with. No
    // canonical id means neither keyed read answered — on `store` mode that is an empty page
    // (`none`), since the tree is only consulted under `--kb_mode local`.
    warm_start.plane = resolved.read_plane === 'remote' ? 'remote'
      : resolved.canonical_id ? 'store'
        : KB_MODE === 'store' ? 'none' : 'local';
    if (resolved.canonical_id) log(`[kb] plane=${warm_start.plane} key=${resolved.canonical_id}`);
    else if (KB_REMOTE !== 'off') log(`[kb] plane=${warm_start.plane} (service had no candidates, ` +
      `or no credentials)`);
    log(`[kb] experience read: slug=${resolved.slug || '?'} (${resolved.match_tier || 'exact'} match of ` +
      `${resolved.requested_slug || KERNEL_NAME}) reason=${warm_start.read_reason} ` +
      `candidates=${cands.length}${f.total ? ` of ${f.total} recorded [${f.retired || 0} retired, ` +
      `${f.below_min_speedup || 0} below ${WARM_START_MIN_SPEEDUP}x, ` +
      `${f.same_direction_collapsed || 0} same-direction]` : ''}` +
      // Every count on this line is over the rows FETCHED; a saturated scan means the page held
      // more.
      `${f.scan_saturated ? ` (SCAN SATURATED at ${f.scanned} — the page holds more)` : ''}`);
    const otherLangs = Array.isArray(resolved.other_language_pages) ? resolved.other_language_pages : [];
    if (!cands.length && otherLangs.length) {   // wrong target_language, not an empty store
      log(`[kb] NOTE: no ${TARGET_LANGUAGE} page, but the store holds ${otherLangs.join(', ')} — ` +
        `check the target_language this lane was invoked with.`);
    }
    // reference-only: prose is already mirrored to EVAL_DIR/kb_references by the resolver; do not apply.
    if (!WARM_START_REF_ONLY) {
      const editableCsv = ((analysis && analysis.modifiable_files) || []).join(',');
      for (const c of cands) {                              // already rank-ordered (fastest first)
        const rec = warm_start.candidates.find(x => x.rank === c.rank);
        const remapOut = `${EVAL_DIR}/warm_start/cand_${c.rank}/remapped.diff`;
        let ver = await agentT(
          roleAgent('verify_engineer', 'verify',
            'Validate a HISTORICAL warm-start patch — the SAME gate as any round candidate. The patch was ' +
            'won in the workspace that produced it, whose paths need not exist here, so FIRST run EXACTLY:\n' +
            '```bash\n' +
            `mkdir -p ${JSON.stringify(`${EVAL_DIR}/warm_start/cand_${c.rank}`)}\n` +
            `python3 ${JSON.stringify(EXPERIENCE_STORE)} remap --patch ${JSON.stringify(c.patch_path)} \\\n` +
            `  --out ${JSON.stringify(remapOut)} --editable ${JSON.stringify(editableCsv)} \\\n` +
            `  --workspace ${JSON.stringify(CANONICAL)}\n` +
            '```\n' +
            'It prints one line of JSON. remapped=true: apply the REMAPPED patch (its paths were rewritten ' +
            'onto this workspace). reason "no_change_needed": apply the original PATCH. Any other reason ' +
            '(notably "unmapped_paths" — the patch touches files this workspace does not have): return ' +
            'status:"apply_failed", notes "patch_outside_editable_set", and DO NOT apply or benchmark. ' +
            'Apply with `git apply "$P" || git apply --3way "$P"`. On ANY failure restore the working copy ' +
            'fully (git checkout -- . and delete untracked files) before returning so the next candidate ' +
            'starts from a clean tree.', {
            CANONICAL, PATCH: c.patch_path, REMAP_OUT: remapOut,
            VERIFY_DIR: `${EVAL_DIR}/warm_start/cand_${c.rank}`,
            EDITABLE_SET: (analysis && analysis.modifiable_files) || [],
            GPU_ID: GPU_LIST[0], SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
            ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
            ...(REQUIRE_GRAPH_CAPTURE ? { REQUIRE_GRAPH_CAPTURE: '1' } : {}),
          }),
          { phase: 'WarmStart', label: `warm_start:verify c${c.rank}`, schema: VERIFY_SCHEMA });
        let checkedWarmPatch = c.patch_path;
        if (EXTERNAL_OBJECTIVE) {
          // Match the adoption command's exact -f choice; the model does not
          // choose a different patch for checking than the one later applied.
          const exists = await runProcess({cmd: '/usr/bin/test', args: ['-f', remapOut],
            timeoutMs: 5000, graceMs: 1000, settleMs: 1000});
          if (exists.status !== 'completed' || ![0, 1].includes(exists.code)
              || exists.strays_killed || !exists.group_cleared || exists.stdio_held) {
            ver = null;
          } else if (exists.code === 0) checkedWarmPatch = remapOut;
        }
        ver = await checkedExternalMeasurement(ver, CANONICAL, checkedWarmPatch);
        const sp = primSpeedup(ver);
        if (ver && says(ver.status, 'verified') && says(ver.correctness, 'pass') && sp > 1.0) {
          const adopt = await agentT(
            `You are the TechLead adopting a validated warm-start patch into the canonical workspace.
\`\`\`bash
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
cd ${CANONICAL}
git checkout -- .
# the remapped patch when verify produced one (its paths fit THIS workspace), else the stored one
P=${JSON.stringify(remapOut)}; [ -f "$P" ] || P=${JSON.stringify(c.patch_path)}
git apply "$P" || git apply --3way "$P"
git -c user.email=team@workflow -c user.name=team add -A
git -c user.email=team@workflow -c user.name=team commit -q -m "warm-start adopt: ${c.slug} (${sp.toFixed(2)}x)"
git --no-pager diff "$(git rev-list --max-parents=0 HEAD)..HEAD" > ${EVAL_DIR}/current_best.diff
\`\`\`
If BOTH applies fail, apply manually to match intent, then add -A + commit and RE-RUN the COMMANDMENT
correctness check; only report committed=true if it still passes. Return JSON {committed, current_best_diff, note}.`,
            { phase: 'WarmStart', label: `warm_start:adopt c${c.rank}`, schema: COMMIT_SCHEMA });
          if (adopt && adopt.committed) {
            // The optimize loop now builds ON this patch: cumulative starts at the adopted Nx, so this
            // run's own rounds only earn the delta above it (split out as incremental_speedup below).
            cumulative = sp;
            bestPerCase = (ver.per_case && ver.per_case.length) ? ver.per_case : bestPerCase;
            finalWinner = { source: `warm_start:${c.slug}`, geomean: sp,
              arithmetic: ver.verified_arithmetic || sp, per_case: bestPerCase, patch: c.patch_path,
              ...externalFields(ver) };
            if (bestSeen < sp) bestSeen = sp;
            warm_start.adopted = true;
            warm_start.adopted_speedup = sp;
            warm_start.slug = c.slug;
            // Carried into the write-back: the idea this run started from, and the entry it
            // descends from — lineage a later curation pass cannot recover from the diff alone.
            warm_start.direction = c.direction || '';
            warm_start.exp_dir = c.exp_dir || '';
            if (rec) { rec.status = 'adopted'; rec.outcome = 'validated'; rec.verified_speedup = sp; }
            log(`[kb] warm-start ADOPTED ${c.slug} @ ${sp.toFixed(2)}x — optimizing from the patched state.`);
            profileSummary = await agentT(
              roleAgent('profile_engineer', 'reprofile',
                'Re-profile the adopted warm-start state and classify the new bottleneck.', {
                WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_LIST[0], ROUND: 0,
                COMMANDMENT, PREVIOUS_METRICS: profileSummary,
              }),
              { phase: 'WarmStart', label: 'warm_start:reprofile', schema: PROFILE_SCHEMA }) || profileSummary;
            if (history) history.bottleneck_now = profileSummary ? profileSummary.bottleneck : history.bottleneck_now;
            if (WARM_START_RETURN_AFTER) { skipLoop = true; warm_start.returned_after_read_kb = true; }
            break;
          }
        }
        if (rec && rec.status !== 'adopted') {
          rec.status = (ver && ver.status) ? `rejected:${ver.status}` : 'rejected:apply_failed';
          rec.outcome = kbVerdict(ver, sp);
          rec.verified_speedup = sp || 0;
        }
        log(`[kb] warm-start candidate c${c.rank} ${rec ? rec.status : 'rejected'} (${sp ? sp.toFixed(2) + 'x' : 'no measure'}).`);
      }
      // Record what THIS box saw, so the next one reads a ledger and not just a speedup. Only
      // candidates carrying an `outcome` reached the GPU — the loop breaks after adopting, and
      // counting an unbenched record would enter an attempt that never happened.
      const benched = warm_start.candidates.filter(x => x.outcome && (x.session_id || x.exp_dir));
      if (benched.length) {
        const planeFlags = warm_start.plane === 'remote' ? '--plane remote'
          : warm_start.plane === 'store' ? `--plane local --store ${JSON.stringify(KB_STORE_DIR)}` : '';
        const cmds = benched.map(x =>
          `python3 ${JSON.stringify(EXPERIENCE_STORE)} attest ${planeFlags} ` +
          (x.exp_dir ? `--exp-dir ${JSON.stringify(x.exp_dir)} ` : '') +
          (planeFlags && x.session_id ? `--session-id ${JSON.stringify(x.session_id)} ` : '') +
          `--kernel-name ${JSON.stringify(KERNEL_NAME)} --language ${JSON.stringify(TARGET_LANGUAGE)} ` +
          `--gfx ${GFX}${KB_VERSION_FLAG} --outcome ${x.outcome} ` +
          `--measured-speedup ${Number(x.verified_speedup) || 0} ` +
          `--note ${JSON.stringify(`kernel_lane warm-start c${x.rank}: ${x.status}`.slice(0, 200))} ` +
          `--measured-by ${JSON.stringify('kernel_lane:' + GFX)} --apply || true`);
        await agentT(
          `You are the kernel knowledge-base attestor. Run EXACTLY these commands in order and ` +
          `return {"ran": <how many you ran>, "note": "<anything that failed>"}. Each records what ` +
          `this box saw when it benched a stored patch. Do NOT edit them, do NOT add or drop any, ` +
          `and do NOT retry a failure — a repeat would double-count the attempt.\n` +
          '```bash\n' + KB_ENV_PRELUDE + cmds.join('\n') + '\n```',
          { phase: 'WarmStart', label: 'kb:attest',
            schema: obj({ ran: { type: 'number' }, note: { type: 'string' } }, []) });
        log(`[kb] attested ${benched.length} benched candidate(s): ` +
          benched.map(x => `c${x.rank}=${x.outcome}`).join(' '));
      }
    }
  }
}
