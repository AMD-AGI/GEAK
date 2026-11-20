# Why v2 looks the way it does

Source: the 1114–1116 GEAK × EAT run on four Muse-Spark kernels on MI355X: splitk GEMM, fused norm,
compact MXFP4 MoE, and FlyDSL GQA verify attention. It ran 69 supervisor iterations, 35 of them substantive (splitk idled for 34), about $4.0K and
~41 h. All four tasks started from the same seed team (2 engineers + integrate_verify, built from the
v1 skills). The EAT supervisor then redesigned the team every iteration. Each rule in v2 is something
that recurred in at least two tasks, or a v1 rule the supervisor blamed for a failure.
Sources: `eat_muse_1113/tasks/*/{SYNTHESIS,LEARNINGS,EVOLUTION_HISTORY}.md`, the supervisor transcripts
and `solutions/results.tsv`.

Results (official score vs the same frozen baseline; the previous GEAK result measured in the same eval):

| kernel | previous GEAK | this run |
|---|---|---|
| splitk | 1.157 | 1.191 |
| norm | 1.166 | 1.291 |
| compact | 1.086 | 1.172 |
| gqa | 1.058 | 1.187 (scored path; the deployed path was fixed in iteration 7) |

| v2 rule | evidence |
|---|---|
| Round 1 = time budget + deployment map before optimizing (lead, investigator `budget`/`deployment`) | norm iter 2: decode_profiler's floor breakdown was the largest gain. gqa iter 5: per-wave stamps found 1.8 + 1.3 µs; "most productive role per dollar" |
| Challenge every declared ceiling with an independent probe (`ceiling`) | false roofs in all four tasks: splitk "13.7 µs fixed / 6 TB/s", compact "6.0 TB/s", gqa "5.8 TB/s at the HBM ceiling", norm "prefill roof" measured with the team's own kernel. Each refutation opened the largest step that followed (HIP rewrites at 7.1–7.6 TB/s) |
| `MODE=fresh-arch` only at a design bottleneck (plateau + one design family + headroom), probe-first, judged over two rounds | gqa iter 3: an architect barred from editing c301 shipped only +1-2% (c908) but its x4 design carried iter 4 to 1.1475 and the final c1801. norm iter 4: prefill rewrite fnr_r, prefill 1.34-1.40 → 1.47-1.55. compact iter 4: 12 persistent-kernel probes all slower; one decisive probe (c306) closed the direction, but the round cost $323 for 0 score. splitk iter 6 (gap < 2%): clean-room design lost by 3.5-5% for $36, used as convergence evidence |
| Cut work by ownership (regime / size band / cost component), not by technique | norm iter 3 (decode/prefill: prefill gave ~3/4 of the gain), gqa iter 4 (large_n / small_n: biggest jump), compact (M bands), splitk (end / wide_m) |
| Deliverables (bound, ≥ 2 designs, a mechanism per tie) instead of stop rules | gqa iter 3: "engineers quit early … the geak-engineer skill's stop rules ('stop once 3 timings within 1%', 'change approach after 8 steps') gave them permission to stop". Engineers used 56–75 of 360–420 min. With deliverables they worked 5–6 h |
| `prior-art` mode | gqa iter 3: aiter's asm PA ran at 0.92–0.96× of c301, an existence proof. norm iter 2: kernarg preload came from outside knowledge |
| `instrument` mode, steering tools validated against officials, no in-process A/B decisions | compact: in-process A/B carried state between kernels and overstated c385 ~2×; GPU0/GPU1 bias 0.8–1.8%. splitk: a probe-induced "poisoned graph" regime, host-socket offset (pinning cut spread 6×). gqa: in-process A/B "can be wrong in direction" |
| Cross-over pairs (`xover.sh`) and per-GPU A/A | compact iter 5: the same file scored ±0.013 across days, so decisions moved to GPU-swapped pairs. norm: per-orientation medians |
| Pre-registered ship rule; non-inferiority for speed-neutral fixes; fix a rule that blocks a real gain | splitk iters 3, 8, 9: three supervisor ship rules blocked audited gains (c012, c015) before revision. norm iter 5: a per-orientation rule blocked c480. c016/c017/c580/c600 shipped on non-inferiority |
| Ship first, continuously | compact iter 1, norm iter 5, splitk iter 9: verified work was lost to time-outs or node reclaims while unshipped |
| Adversarial audit list in geak-ship | gqa: m0 hardware hazard, a race in c1501, and process-global counters corrupting output under concurrent calls (c2001). compact: crash on parallel graph branches. splitk: out-of-LDS crash on legal shapes, then 25/31 deployment threats failing (5 silently wrong), then dtype / device / int32 bugs |
| Deployment-path check (`deployment`) | gqa iter 6: the server uses `defer_reduce=True`, a path the UT never times, so the X4 gains did not reach serving. norm: the UT's flushed regime ranked store policies opposite to deployment (c450 vs c520) |
| LEARNINGS (live, with evidence) + SYNTHESIS (rewritten) | norm iter 2: "mid-run sharing let the two roles stack their changes". gqa iter 5: `xchat.md` hand-offs |
| The lead must end the run on independent evidence | splitk: converged at iter 14, then ran 34 no-op iterations (~$4 each) because EAT has no exit |

Not yet shown: whether v2 as fixed skills matches the evolving team at equal budget. That is the next
experiment (H4 vs H1 / H3 from c000, same budget, ≥ 2 seeds). Until then, treat v2 as a hypothesis.
