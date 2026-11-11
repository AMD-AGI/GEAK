// LLM token+time accounting (PURELY ADDITIVE; args.llm_stats="false" makes it a no-op).
// DELIBERATELY NO TIMESTAMPS: Date.now()/new Date() are unavailable in workflow scripts. Every
// duration in the report comes from the transcripts (scripts/llm_ledger.py); this records only the
// role/phase/attempt identity of each agent call so the ledger attributes tokens to the right agent.
// Each workflow declares its own LLM_TL (it names the workflow).
const LLM_STATS = String(A.llm_stats != null ? A.llm_stats : 'true').trim().toLowerCase() !== 'false';
const TL_ROLE_RE = /You are the ([A-Za-z0-9_.\-]+)\.\s*PHASE=([A-Za-z0-9_.\-]+)\./;
// Called AT DISPATCH (before the await), so LLM_TL.events is in dispatch order, not completion
// order. Two same-key agents that finish out of order no longer swap identities under the parser's
// positional join. Returns the event so the caller flips `ok` once the attempt resolves (null when
// off). Even an attempt that hangs or throws is recorded, because the record predates the await.
function tlAgent(prompt, o, attempt) {
  if (!LLM_STATS) return null;
  // Identity comes from the PROMPT, not opts.label: labels are free-form display strings. The
  // prompt's opening line always carries `You are the <role>. PHASE=<sub_phase>.`, the same line the
  // transcript records, so the ledger folds this call under its own agent instead of its predecessor.
  const m = TL_ROLE_RE.exec(String(prompt || ''));
  const e = {
    seq: LLM_TL.events.length,
    phase: (o && o.phase) || '',
    label: (o && o.label) || 'agent',
    role: m ? m[1] : '',
    sub_phase: m ? m[2] : '',
    attempt: attempt,
    ok: false,
  };
  LLM_TL.events.push(e);
  return e;
}
