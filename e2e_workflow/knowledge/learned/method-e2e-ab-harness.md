---
key: e2e A/B measurement · any gfx · sglang/vllm
type: method
confidence: ★★★
effect: stops false wins — a positive median inside the noise band is a NULL, not a win; and stops
  UNMEASURABLE rounds — a candidate whose Amdahl ceiling is below the box's session spread cannot be
  resolved in either direction, so the A/B buys noise
confirms: 8
last_seen: 2026-08-24
---
# Honest e2e A/B: tight interleave + non-overlap gate (not just a positive median)
- lever: run a tight INTERLEAVED A/B (REF, CAND, REF, CAND, …) on a SINGLE GPU with a PINNED port, then
  gate on BOTH `delta_med > noise_band` AND non-overlapping distributions (`cand_min > ref_max`). The
  ~0.5% noise band is real: clean ref/cand medians overlap routinely, so a sub-band delta with
  overlapping [min,max] is a NULL.
- apply: ≥5–7 repeats/leg, back-to-back, same GPU. Combine an accepted-config stack and gate the SUM vs
  the TRUE baseline (small real wins only count when stacked).
- verify: sglang derives `grpc_port = port + 10000` and rejects >65535 → an OS ephemeral port >55535
  crashes launch; ALWAYS pin PORT to a low value. Budget for grpc-port-flake retries.
- caution (also verify the DENOMINATOR, not just the candidate): a stored baseline goes stale. A run
  whose finalize bench divided by a ~22-h-old baseline reported **+0.86%**; re-measuring the identical
  no-overlay config in the same session gave a 13%-lower reference and the same final stack scored
  **+15.72%** (Director-validated). Re-measure the reference leg in the SAME session as the candidate,
  and treat a leg whose spread exceeds ~5% as invalid — discard and re-run it rather than dividing by it
  (a 15.7%-spread reference leg would have inflated the same delta to +24.8%). Two independent
  same-session A/Bs run ~14 h apart agreeing (+16.77% vs +15.72%) while the box's ABSOLUTE level moved
  13% is the strongest available evidence that a win is real — ratios survive box drift, levels do not.
- caution (also verify the win is not a WARMING artifact — cheap, and it is what makes an accept
  defensible): if throughput trends MONOTONICALLY UPWARD within each leg and the candidate happened to run
  second, "cand > ref" is confounded by warming no matter how big the gap. Fix: add interleaved
  DRIFT-CONTROL legs (ref, cand, ref_b, cand_b, ref_c) and pool per side, then check the single decisive
  fact — **the LAST leg chronologically is a REF and it is among the LOWEST ref values**. Worked case:
  ref 5479->5807 and cand 5860->6352 both warming; pooled REF med 5579 / CAND med 6250 = +12.01% with
  non-overlapping ranges (cand_min 5859.8 > ref_max 5807.0) and the final ref_c the lowest ref of all
  ⇒ the win is not drift. Change NOTHING but the overlay between legs; inherit every workload knob
  (num_prompts / warmups / range-ratio) from the orchestrator rather than overriding one.
- caution (also verify the BASELINE is self-deterministic before claiming byte parity): probe the greedy
  parity set (temp=0/seed=0/ignore_eos) TWICE per leg on the warm server, between timed rounds, on BOTH
  legs — four hashes, not two. Only then does `hash(cand) == hash(ref)` mean anything: it separates "the
  candidate is byte-identical" from "this server is nondeterministic and both legs are noise". Runs on
  other stacks have hit a no-overlay baseline that diverged from ITSELF on 8/12 prompts, which silently
  makes byte parity unavailable and forces the weaker accuracy gate; the four-hash protocol detects that
  in the same pass instead of after the fact.
- caution (also verify an OVER-CEILING accept arithmetically): an accepted delta larger than its own naive
  Amdahl ceiling (pct_gpu x iso speedup) is either a second-order win or output corruption. Distinguish
  them by reconciling the per-step budget — us saved per launch x layers vs the measured TPOT drop. One
  accept at +12.01% against a +1.90% ceiling reconciled to within ~11% (modelled vs measured per-step), and
  was byte-exact; the implausible-speedup guard is properly scoped to SOFT/accuracy accepts, since a
  device-time profile legitimately under-counts a kernel whose patch also collapses adjacent graph nodes
  and de-serializes its neighbours.
- caution (also verify your driver is not DELETING the leg it is about to read, and keep a second source
  of truth for every number): a driver that `rm -rf`'d a leg's output subdir before the bench wrote its
  per-run result json left one whole leg with no `bench_summary.json`. It was recoverable ONLY because the
  same metric is echoed to the console — grep the leg log for `Output token throughput (tok/s)` and
  cross-check on the SURVIVING leg that the console values equal that leg's `all_throughput` before
  trusting the recovered ones. Rule: never clean a leg directory inside the timed phase, and always tee
  each leg's console to a file so a lost artifact costs a grep, not a re-run.
- caution (also verify the reference against ITSELF when the server does CONTINUOUS BATCHING): the
  four-hash protocol fired again — a ref-vs-ref probe on one warm vLLM server was byte-identical on only
  2/12 greedy prompts (mean seqmatch 0.72). Under continuous batching the batch composition, not the
  overlay, decides the tokens, so `byte_exact` is simply UNAVAILABLE as a gate and a ref-vs-cand 2/12 is
  the SAME number, not a regression. Fall back to accuracy (inspect the diverging pairs for coherence and
  factual correctness) and say so in the verdict; do not report a parity FAILURE you cannot distinguish
  from the baseline's own noise.
- caution (RESOLVABILITY — also verify the gate CAN see the candidate, BEFORE spending the launches):
  compare the candidate's Amdahl ceiling (`pct_gpu_time` x isolated speedup) against the box's MEASURED
  within-session throughput spread, not against the nominal 0.5% noise band. One round spent four A/Bs on
  candidates whose ceilings were +1.1–1.8% while the box (under foreign tenants) was swinging 6–11%
  between repeats of the SAME leg; the results came back −20.5%, +5.9%, −12.4%, −6.1% and only the two
  disjoint ones carried information. Rule: if `ceiling < spread`, the gate cannot resolve the candidate in
  either direction — either buy quiet (hold the GPU locks and run both legs back-to-back inside minutes,
  which is what made the two disjoint results trustworthy) or do not fund the round.
- caution (a `stack` verdict is a HYPOTHESIS, not a small win — also verify it at the next reprofile):
  when the median favours the candidate but ref/cand ranges OVERLAP, the harness is right to gate `stack`
  instead of `accepted` — but a `stack` must not then be carried as banked gain. A +5.90% `stack` on 2
  overlapping repeats was falsified by the very next reprofile (0.995x, i.e. flat), and the
  in-trace per-launch decomposition said why: the kernel the patch retiled improved only −1.24% per decode
  launch (inside its own repeat band) while an untouched neighbour went +9.2%. Re-verify every `stack`
  against in-trace per-launch us before it enters a cumulative claim, and never let one become the
  reference leg of the next round without re-measuring it.
- caution (also verify the box is quiet): a self-replicating helper-process storm can depress both legs
  ~13%. Both legs inside the same regime keeps the RATIO sound, but never quote the absolute level.
- caution (also verify REUSE_SERVER points at the port your server is really on): the bench harness
  enforces a port WINDOW (`PORT_BASE`/`PORT_SPAN`, `PORT_ENFORCE_RANGE=1`) and SILENTLY re-allocates any
  `PORT` outside it. A hand-written validation driver that launched its own healthy server on an
  out-of-window port then called the harness with `REUSE_SERVER=1` got a freshly allocated in-window port
  with nothing behind it -> "No healthy server", rc=2, on BOTH legs, after both servers and both
  correctness probes had already succeeded. Derive the driver's port as `PORT_BASE + offset`, and assert
  the port the harness actually probes equals the one you launched on, BEFORE the timed phase.
- caution (also verify the run has budget left for the validation A/B itself): a candidate can be real at
  the integrate gate and the RUN still close `flagged` (mechanism + parity reproduced, zero independent
  timed samples). When that happens, do not promote the cross-session ratio to a headline and do not call
  it a rejection — say "not validated", leave the reversible overlay bundle in place, and record the fix
  needed to close it out. Same box: a 21-h-old baseline sat 7.4% above a same-day no-overlay leg at an
  identical config, i.e. drift alone was worth more than half the claimed run-level delta.
- source: exp/e2e_*Qwen3.5-27B*/ 2026-06-07 / 06-09; exp/e2e_*Qwen3-14B-FP8-vllm*/e2e_cycle0 2026-08-22
  (stale-denominator + discarded high-spread reference leg, Director validate phase);
  exp/e2e_*Qwen3.5-27B-FP8*/ 2026-08-22 (port-window REUSE_SERVER trap; flagged run, 7.4% drift in 21 h);
  exp/e2e_*gpt-oss-120b*/ 2026-08-23 (drift-control legs + four-hash parity + the over-ceiling
  reconciliation, on a byte-exact +12.01% MoE-prologue accept, TP2 gfx950 vLLM);
  exp/e2e_*Qwen3.5-122B-A10B-FP8*/ 2026-08-23 (deleted-leg artifact recovered from the console log;
  continuous-batching reference self-divergence 2/12 forcing the accuracy gate; TP2 gfx950 vLLM);
  exp/e2e_*gpt-oss-120b*/ 2026-08-24 (the resolvability screen + the falsified `stack`: four MoE-GEMM
  A/Bs at +1.1–1.8% ceilings against a 6–11% box spread; TP2 gfx950 vLLM)
