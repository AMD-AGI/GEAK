// Hung-agent + API-fault guard around agent(): retries API errors, resolves null on timeout.
async function agentT(p, o) {
  // Parallel members share installed dependencies, not mutable compiler caches.
  if (A.population_lineage) o = { ...o, env: { ...(o && o.env), GEAK_ISOLATE_WORKSPACE_CACHES: '1' } };
  const label = (o && o.label) ? o.label : 'agent';
  for (let attempt = 1; attempt <= AGENT_RETRIES; attempt++) {
    const ev = tlAgent(p, o, attempt);   // record AT DISPATCH; ok=false until it resolves
    try {
      // The standalone runtime's bounded mode terminates and settles the actual
      // subprocess. Never race it with a Promise that abandons a live role.
      if (agent.processDeadlineActive || typeof setTimeout !== 'function' || !(AGENT_TIMEOUT_MS > 0)) {
        const opts = agent.processDeadlineActive ? { ...o, processTimeoutMs: AGENT_TIMEOUT_MS } : o;
        const r0 = await agent(p, opts); if (ev && r0) ev.ok = true; return r0;
      }
      let to;
      const guard = new Promise((resolve) => {
        to = setTimeout(() => {
          log(`  [hung-agent guard] ${label} exceeded ${Math.round(AGENT_TIMEOUT_MS / 60000)}min with no return — resolving null so the round proceeds.`);
          resolve(null);
        }, AGENT_TIMEOUT_MS);
      });
      // A timeout resolves null (returned as-is, no retry). An API/agent error rejects -> caught below.
      const r = await Promise.race([
        agent(p, o).then((rr) => { clearTimeout(to); return rr; }, (e) => { clearTimeout(to); throw e; }),
        guard,
      ]);
      if (ev && r) ev.ok = true;
      return r;
    } catch (e) {
      if (agent.processDeadlineActive && e
          && (e.code === 'GEAK_PROCESS_DEADLINE' || e.code === 'GEAK_AGENT_TIMEOUT')) {
        log(`  [deadline] ${label} stopped (${e.code}); no retry after its process cutoff.`);
        return null;
      }
      const msg = String(e && e.message ? e.message : e).slice(0, 200);
      if (attempt < AGENT_RETRIES) {
        log(`  [api-fault guard] ${label} attempt ${attempt}/${AGENT_RETRIES} hit an API/agent error (${msg}) — retrying so a transient outage doesn't kill the run.`);
        continue;
      }
      log(`  [api-fault guard] ${label} still failing after ${AGENT_RETRIES} attempts (${msg}) — resolving null so the workflow degrades gracefully instead of exiting.`);
      return null;
    }
  }
  return null;
}
