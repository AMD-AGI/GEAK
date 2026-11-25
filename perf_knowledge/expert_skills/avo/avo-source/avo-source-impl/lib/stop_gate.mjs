// Deterministic gate for honoring a TechLead stop request.
//
// TechLead's stop is advisory: it is a judgement about convergence, made by an agent that
// cannot measure the run's wall-clock budget. With early_stop off the run is contracted to
// spend the FULL budget, so the orchestrator — not the agent — decides when spending ends.
//
// Requiring N consecutive confirmations was not sufficient on its own: a TechLead that
// believes the kernel has converged simply re-issues the stop every round until the streak
// is met, which ended 12h runs at round 7 with ~10h of budget unspent. The budget check
// below is what actually holds the contract.
//
// Kept free of workflow runtime state so it is unit-testable (see tests/stop_gate.test.mjs).

// Wall-clock cost of one more round: the round itself plus the supervisor/plan overhead
// that precedes it.
export function roundCostS({ estStepS, overheadS }) {
  return Number(estStepS || 0) + Number(overheadS || 0);
}

// How many further rounds still fit in the remaining budget. Returns null when no round has
// completed yet, because there is then no basis for an estimate — callers MUST treat null as
// affordable so the gate never fires on missing data.
export function affordableRounds(budget) {
  if (!Number(budget?.estStepS)) return null;
  const cost = roundCostS(budget);
  if (cost <= 0) return null;
  return Math.floor(Math.max(0, Number(budget.remainingS) || 0) / cost);
}

export function canAffordAnotherRound(budget) {
  const n = affordableRounds(budget);
  return n === null || n >= 1;
}

// Returns { honored, reason }. `reason` is what the caller logs, so the decision is legible
// in the run log rather than being inferred from the absence of further rounds.
export function shouldHonorTeamStop({
  stop, earlyStop, streak, confirmations, remainingS, estStepS, overheadS,
}) {
  if (!stop) return { honored: false, reason: 'no_stop_requested' };
  // early_stop is the operator explicitly opting into ending on convergence.
  if (earlyStop) return { honored: true, reason: 'early_stop_enabled' };
  if (Number(streak) < Number(confirmations)) {
    return { honored: false, reason: 'awaiting_confirmations' };
  }
  if (canAffordAnotherRound({ remainingS, estStepS, overheadS })) {
    return { honored: false, reason: 'budget_remains' };
  }
  return { honored: true, reason: 'budget_spent' };
}
