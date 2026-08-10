// Distil already-finished kernel runs into learned-knowledge proposals. NO re-optimization.
//
// `kernel_workflow.js learn=true` distils at the end of a run. This driver does the same thing for
// runs that finished BEFORE the feature existed, or when you want to (re)distil a whole sweep in one
// pass: it runs the Curator alone against each completed eval dir and writes proposals into the
// KB's _inbox/. It never touches a GPU and never edits a workspace.
//
// args = {
//   exp_base:      "/abs/.../exp/<sweep>",       // holds one dir per kernel
//   workflow_dir:  "/abs/.../kernel_workflow",
//   learned_kb_dir:"<workflow_dir>/knowledge/learned",   // optional
//   split_path:    "<exp_base>/ab_split.json",           // optional but strongly advised
//   only:          ["kernel", ...],                      // optional explicit subset
//   box_quiet:     "true",
// }
//
// The split matters more than it looks. Cards are distilled from finished runs and read by later
// ones, and sweeps re-run the SAME kernels — so any kernel that contributes a card is spent as an
// evaluation subject forever after. Passing `split_path` makes the driver refuse the held-out
// kernels; without it, one careless pass over the whole sweep destroys the only instrument that can
// tell you whether the KB works.
export const meta = {
  name: 'kernel-distil',
  description: 'Run the Curator over already-completed kernel eval dirs and file learned-knowledge proposals. No GPU, no re-optimization.',
  whenToUse: 'After a sweep finishes, to turn its validated runs into distilled cards. Pass args.exp_base and args.workflow_dir.',
  phases: [
    { title: 'Survey', detail: 'find completed eval dirs and read each one\'s validation gate' },
    { title: 'Distil', detail: 'one Curator per eligible kernel -> _inbox/<run_id>.json' },
  ],
};

const A = args || {};
const EXP_BASE = String(A.exp_base || '').replace(/\/+$/, '');
if (!EXP_BASE) throw new Error('args.exp_base is required (the sweep dir holding one subdir per kernel)');
const WF = String(A.workflow_dir || '').replace(/\/+$/, '');
if (!WF) throw new Error('args.workflow_dir is required');
const LEARNED_KB_DIR = String(A.learned_kb_dir || (WF + '/knowledge/learned')).replace(/\/+$/, '');
const SPLIT_PATH = String(A.split_path || (EXP_BASE + '/ab_split.json')).trim();
const ONLY = Array.isArray(A.only) ? A.only : [];
const BOX_QUIET = String(A.box_quiet != null ? A.box_quiet : 'true') === 'true';

const SURVEY_SCHEMA = {
  type: 'object',
  properties: {
    split_found: { type: 'boolean' },
    kernels: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          eval_dir: { type: 'string' },
          validation_status: { type: 'string' },
          correctness: { type: 'string' },
          verified_speedup: { type: ['number', 'null'] },
          held_out: { type: 'boolean' },
          eligible: { type: 'boolean' },
          skip_reason: { type: 'string' },
          kk_operator: { type: ['string', 'null'] },
          kk_language: { type: ['string', 'null'] },
          device: { type: 'string' },
          regime: { type: 'string' },
        },
        required: ['name', 'eval_dir', 'eligible'],
      },
    },
  },
  required: ['kernels'],
  additionalProperties: true,
};

const CURATE_SCHEMA = {
  type: 'object',
  properties: {
    proposed: { type: 'boolean' }, reason: { type: 'string' }, cards: { type: 'number' },
    rejected: { type: 'array', items: { type: 'object', properties: {
      title: { type: 'string' }, reasons: { type: 'array', items: { type: 'string' } } } } },
    proposal_path: { type: 'string' },
  },
  required: ['proposed'],
  additionalProperties: true,
};

phase('Survey');
// One agent does the filesystem work the script cannot: list the eval dirs, read each validation
// verdict, and apply the split. It reports eligibility per kernel and WHY a kernel was skipped —
// a silent skip and a genuine "nothing to learn" look identical otherwise.
const survey = await agent(
  `Survey the completed kernel runs under ${EXP_BASE} and report which may be distilled.

1. List its immediate subdirectories. Ignore anything ending in \`.aborted_*\`, and ignore dirs with
   no \`director_validation.json\`.
2. Read ${SPLIT_PATH} if it exists. Any kernel listed under \`held_out\` is NOT eligible —
   set held_out=true, eligible=false, skip_reason="held-out split". If the file does not exist, set
   split_found=false and treat nothing as held out.
3. For each remaining kernel read \`director_validation.json\`:
   - \`validation_status\` must be \`accepted\` (case-insensitive) and \`correctness\` must be \`pass\`
     (case-insensitive — BOTH cases occur in these files). Otherwise eligible=false with the reason.
   - verified_speedup: prefer \`director_verified_speedup_geomean\`; if that key is absent, fall back
     to \`speedup_vs_remeasured_baseline.geomean\` (some runs write it only there — a reader that
     knows one key reports "unvalidated" for a run that was in fact validated more carefully than
     most). null if neither exists, and then eligible=false.
4. Also read \`analysis.json\` for kk_operator / kk_language / regime, and \`baseline_metrics.json\`
   (or \`profiling_summary.md\`) for the device string. Empty string if absent — do not invent them.
${ONLY.length ? `5. Consider ONLY these kernels: ${JSON.stringify(ONLY)}.` : ''}

Do NOT modify anything. Return the JSON.`,
  { phase: 'Survey', label: 'distil:survey', schema: SURVEY_SCHEMA, effort: 'low' });

if (!survey || !survey.kernels) {
  log('Survey failed — nothing distilled.');
  return { surveyed: 0, distilled: 0, error: 'survey returned null' };
}
if (survey.split_found === false) {
  // Loud, because the consequence is invisible and permanent: distil everything and there is no
  // held-out set left to measure the KB with, ever.
  log(`WARNING: no split file at ${SPLIT_PATH}. Every kernel distilled here is spent as an ` +
      `evaluation subject — the A/B loses its control group. Continuing because it was not passed.`);
}

const eligible = survey.kernels.filter(k => k.eligible);
const skipped = survey.kernels.filter(k => !k.eligible);
log(`Survey: ${survey.kernels.length} completed run(s), ${eligible.length} eligible, ` +
    `${skipped.length} skipped (${skipped.map(k => `${k.name}: ${k.skip_reason || '?'}`).join('; ') || 'none'}).`);

phase('Distil');
const results = await pipeline(
  eligible,
  (k) => agent(
    `You are the curator. PHASE=distill.
First Read ${WF}/roles/curator.md and follow its instructions for PHASE=distill.
Also Read ${LEARNED_KB_DIR}/README.md — it is the contract, and ${WF}/scripts/kb.py enforces it.
Do all filesystem/shell work yourself (Bash/Read/Write).

This kernel's run is already finished; you are distilling it after the fact. Everything you need is
on disk in the eval dir: \`insight_log.md\` (the insight blackboard + hypothesis ledger),
\`tech_lead_report.md\`, \`director_validation.json\`, \`analysis.json\`.

## Inputs
- EVAL_DIR: ${k.eval_dir}
- SKILL_DIR: ${WF}
- LEARNED_KB_DIR: ${LEARNED_KB_DIR}
- RUN_ID: ${k.name}-${EXP_BASE.split('/').filter(Boolean).slice(-1)[0]}
- KERNEL_NAME: ${k.name}
- KK_OPERATOR: ${k.kk_operator || ''}
- KK_LANGUAGE: ${k.kk_language || ''}
- DEVICE: ${k.device || ''}
- REGIME: ${k.regime || ''}
- VALIDATION: read ${k.eval_dir}/director_validation.json (verified speedup ${k.verified_speedup})
- CITATIONS: [] (this run predates the read path, so it cited no cards)
- BOX_QUIET: ${BOX_QUIET}
- HELD_OUT: false

Return ONLY the structured JSON the role file specifies.`,
    { phase: 'Distil', label: `distil:${k.name}`, schema: CURATE_SCHEMA }),
);

const clean = results.filter(Boolean);
const proposed = clean.filter(r => r.proposed);
const cards = proposed.reduce((s, r) => s + (r.cards || 0), 0);
const rejected = clean.flatMap(r => r.rejected || []);
log(`Distil: ${proposed.length}/${eligible.length} eligible runs produced a proposal, ${cards} card(s) total. ` +
    `${rejected.length} card(s) rejected by the linter. ` +
    `${eligible.length - clean.length} curator(s) returned null.`);
log(`Next: python3 ${WF}/scripts/kb.py --kb-dir ${LEARNED_KB_DIR} drain --validated-runs ${eligible.length}` +
    ` (dry run first — it is the only writer of the KB, and you want to read what it would do).`);

return {
  surveyed: survey.kernels.length,
  eligible: eligible.length,
  skipped: skipped.map(k => ({ name: k.name, reason: k.skip_reason })),
  proposals: proposed.length,
  cards,
  // A curator that dies on an API fault returns null and its run's knowledge vanishes with no error.
  // Reported so a KB built from a third of the sweep does not look like one built from all of it.
  curator_nulls: eligible.length - clean.length,
  rejected_cards: rejected,
  nothing_to_say: clean.filter(r => !r.proposed).map(r => r.reason),
};
