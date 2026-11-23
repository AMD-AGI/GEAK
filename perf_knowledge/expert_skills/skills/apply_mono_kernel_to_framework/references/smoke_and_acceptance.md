# Smoke and acceptance: what proves a delivered mono kernel

Status: these checks were **designed, and no checker was ever built or run** (see the status note in
[skill.md](../skill.md)). The design exists because the first version let the integrator run its own smoke
and report its own verdict, and that fails quietly: a counter read wrongly, or two runs that both had the
switch off, still report a pass. The rule that came out of it: **every verdict is computed from files the
servers left behind (token dumps, counter lines, logs), by a script or by hand, never asserted.**

Deliver **one patch per model**. A patch per layer kind does not work: kinds often share one framework
layer class, so the patches edit the same file, define the same names and cannot be applied together.

## 1. Checks that need no GPU
Run these before any server starts. Each is a few seconds of CPU.

| Check | Pass condition |
|---|---|
| Applies | The patch applies cleanly to a clean clone at the target commit (`git apply --check`) |
| Kernel unchanged | Every kernel file in the patch has the same hash as the file that passed the layer oracle, and the runtime support library appears exactly once |
| Call consistency | The set of argument names the patch passes to the kernel's builder and launcher equals the set the layer oracle's adapter passes |
| Selection table | `{layer index: kind}` equals the layer indices each kind's oracle covers, and holds only kinds that pass |
| Framework reads | Every framework-internal name the glue reads, kept as one constant list in the patch, exists in the source at the commit |
| Counter line | The patch contains the code that formats the counter line of section 3 |

Why "kernel unchanged" and "call consistency" exist: the glue is rewritten for the framework interface, while
the adapter that passed the layer oracle was typically a global class rebind, which delivery forbids. Without
these two checks the call path you ship is one no oracle ever exercised.

## 2. The three-server check
One full TP group, installed through the workflow's overlay. Start three servers, one after the other, with
the same serving configuration and the same fixed prompt set: at least 32 prompts, several context lengths, a
few hundred generated tokens each, temperature 0.

| Server | Switch | Purpose |
|---|---|---|
| Unpatched | none | The true baseline |
| Patched | off | Shows the patch is inert |
| Patched | on | The candidate |

| Check | Pass condition | Required |
|---|---|---|
| Off is stock | Switch-off tokens are identical to the unpatched server's, prompt by prompt, and `launches` is 0 on every rank | yes |
| Greedy match | Switch-on tokens against switch-off tokens, prompt by prompt; a divergence passes only as a tie (section 4) | yes |
| Effect | Section 3 | yes |
| Accuracy | Section 5 | optional for the smoke (reported, not in its pass rule); required by the e2e gate once any prompt diverged |
| Speed | Section 6 | recorded only, not a pass condition of the smoke |

The smoke passes only when the three required checks hold. If no serving smoke was run, report `skipped`,
never `pass`; if one check fails, report which. Where the native path is itself not reproducible from run to
run (see the SGLang reference), measure unpatched against unpatched first and judge against that baseline.

## 3. Effect proof with a launch counter
Every rank prints one fixed line periodically. Use the counter line or metric the framework commit already
defines for specialized decode if it has one; otherwise a line of this shape:

`MK_COUNTERS rank=<r> kind=<kind> launches=<n> skips=<json of reason: count>`

What upstream had when this was checked (see the two framework references for commits and symbols): one vLLM
package counts mono and stock steps and logs launch coverage now and then without a host sync; SGLang's
binding logs, after capture, which graph keys run the kernels, and has no launch counter. Neither counts skips
by reason, so that part is yours to add.

- `launches` is read from the kernel's control region at log intervals. Reading it is a host sync, so never
  read it per step.
- `skips` is counted where the routing decision is made, one counter per reason.

The effect proof holds when all of these are true:
1. With the switch on, `launches` grows during decode on **every rank** and for **every delivered kind**.
2. The growth matches eligible decode steps (those not skipped) × replaced instances of that kind × the
   kernel's launches per layer. Take that last factor from the kernel's own contract: a kernel designed for
   one launch per layer has 1, but a deployed kernel may legitimately use two (the public plugin documents one
   or two launches per layer, and specs that cover several layers in one launch).
3. Skips appear only under the expected reasons: prefill, a mixed batch, a verify step the kernel does not
   cover, a width outside the list, a piecewise-graph step. A skipped step of a kind and width the kernel
   covers fails the check. Rows, and so widths, are requests × tokens per request of the step.
4. With the switch off, `launches` stays 0 on every rank.

A server that is healthy and matches greedy output with `launches` at 0 has routed every step away. That is a
failed effect proof, and no speed number from that run may be reported.

## 4. Greedy match and logit ties
- Compare per prompt. At the first divergence record the step and the top-5 logits on both sides.
- A divergence whose top-2 logits are within the bound the kernel's oracle derived is a **tie**: report it as
  a tie, not as a failure. Any other divergence fails.
- Scope of the tie rule: only this check (switch on against switch off), only at the first divergence of a
  prompt, only within the derived bound. "Off is stock" has no tie rule.
- Why ties are legitimate: a fused layer and an in-kernel all-reduce change accumulation order, so results
  differ in the last place now and then. The native path itself forks at near-ties between batch sizes and
  between eager and graph mode. Compare the kernel's agreement with native against native's agreement with
  itself, and never widen the bound to turn a divergence into a tie.
- **A tie is a smoke classification, not an acceptance.** The e2e gate judges parity against the unpatched
  server ([e2e_integrator.md](../../../../../e2e_workflow/roles/e2e_integrator.md), gate 4) and treats a
  candidate that diverges on any prompt from a deterministic baseline as a real output change: it rejects a
  non-quantized change outright and accepts a quantized kernel only after a task-accuracy gate. So once any
  prompt diverged, tie or not, the paired accuracy run of section 5 is required before acceptance, and the
  accept is reported with `parity_kind` `accuracy`, never `byte_exact`.

## 5. Accuracy check: a correctness gate, not speed evidence
Optional for the smoke and not part of its pass rule; required by the e2e gate as soon as any prompt diverged
from the unpatched baseline (section 4). Run the full accuracy benchmark with the switch on and with it off,
paired on the same items, and report both scores and the paired difference. It answers one question only: did
near-tie forks change task quality? The e2e gate decides what difference it accepts.
Its wall-clock is not a speed result: it runs at its own concurrency, which need not be the one you serve at.

## 6. Speed, and reconciling it with the layer speedup
Record single-concurrency decode at width 1, time per output token and throughput, on versus off, then run
the timed A/B the e2e workflow prescribes. Compare the change with layer speedup × the share of step time the
replaced instances hold. When the two do not agree, the usual causes are host overhead in the glue (metadata
assembly, counter reads) or a graph capture that does not include the kernel.

## 7. Partial delivery and bisection
- Only kinds that pass their oracle, and that are faster than native, enter the patch. The report lists every
  kind with replaced or not, and why.
- If no kind qualifies, skip delivery and report that nothing was delivered.
- If the smoke fails, enable a subset of kinds, find the kind that causes the divergence and remove it from
  the patch. The verdict still comes from a full rerun of the smoke on the reduced patch.

## 8. Test the checker before trusting it
Build a native-equivalent patch: each selected instance is replaced by a subclass that calls the native
forward and prints the counter line. It must pass "off is stock" and "greedy match" and must **fail** the
effect proof, because `launches` stays 0. A checker that passes it is broken.

## 9. What delivery assumes the kernel has already shown
The layer oracle covers one instance, the listed widths, a few steps and a bounded number of replays. If the
kernel's owner cannot show the properties below, check them before replacing many instances.

| Property | What to look for |
|---|---|
| Several instances | A few instances in one process, each with its own layer's weights, called in turn over consecutive steps, also under graph replay: each matches its native layer, and swapping the call order changes nothing. If this fails, only one instance can be replaced, which is rarely worth delivering |
| Switching | Covered width, then an uncovered one, then covered again; decode, prefill, decode; next to a native layer of another kind. Outputs and KV match native, and `launches` advances only on eligible calls |
| Long sequence | Tens of consecutive decode steps with a growing KV cache across block boundaries: outputs within tolerance at every step, KV exact |
| Graph replay | One decode step with the layer captured in the framework's capture context and replayed 1000 times with fresh inputs: the last replay matches native, no error on the way |
| Fault injection | One rank held on the host past the deadline: every rank returns within deadline plus margin, the error decodes to a timeout with a readable site on every rank, no rank produces an output, and a clean launch afterwards succeeds. A hang is a failure |
| Bounded waits | Every loop that waits on memory written by another workgroup or GPU has a deadline. A kernel without one hangs forever on a rank disagreement and must not be delivered |
| Wrap of tags or epochs | A server runs far more steps than any unit test: step tags are sized for the longest serving session, or the wrap is handled and tested |
| Memory ledger | Per rank, width and instance count: weight copies, peer buffers, control region, scratch. Nothing allocated per call except outputs, and weight copies allocated before KV sizing |

## 10. Measured evidence (hand-driven integrations, not this procedure)
MiMo-V2.6-Flash in vLLM, TP4 on gfx942. Each figure is a ratio against that integration's own fastest native
baseline. The baselines differ, so compare within a row, never the HIP rows with the FlyDSL row.

| Integration | Result | Caveat |
|---|---|---|
| Two independent HIP kernels | Concurrency-1 P90 inter-token latency better by 3.05× and 3.73×; accuracy not significantly different | Independently reviewed |
| The same two, one-row steps only | About 1.4–1.5×; the largest single step in both was taking multi-row decode steps into the kernel | Read the skip counters before polishing width 1 |
| One FlyDSL kernel | Concurrency-1 P90 inter-token latency better by 2.25×, output throughput +27%; accuracy scores 0.3 points apart, not significant (paired p = 0.34) | Its baseline is the fastest native configuration **with multi-token prediction on**, which the HIP rows' baseline did not have, so this ratio must not be compared with theirs. Not independently reviewed. At concurrency 8 the accuracy run took about 1.4× as long with the kernel on, and latency at concurrency 2 to 16 was not measured |

## 11. What the report must contain
Image, framework commit, the patch, the exact server command of each leg, the commands for the smoke and for
the layer oracle, the counter lines of every rank, the list of kinds replaced or left out with reasons, and
every refusal reason and skip reason seen.
