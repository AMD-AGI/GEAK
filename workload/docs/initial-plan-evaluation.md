# Evaluating the initial serving plan

The starting route in Brain was tested with fresh, text-only planners. Historical
donors were DeepSeek V4 Pro and V4.1 Flash, Kimi K3.1 Pro, GLM 5.2 and MiniMax
M3.1 Flash. Target history for the final transfer workload was excluded.

## Why client concurrency is only a hint

| Evidence | What changes the initial plan |
| --- | --- |
| At C1, two single-pair GPU tests of persistent execution changed output/P90 by +31.00%/+55.03% on Flash and +0.69%/−3.56% on Pro. | Check the live target and draft dispatch, numerical contract and KV cost before selecting mono. Different model and draft implementations limit attribution. [Flash](../brain/experience/cases/deepseek-v41-flash/gpu-deepseek-v41-flash-4fab-mono-single-pair-20261114.md) · [Pro](../brain/experience/cases/deepseek-v4-pro/gpu-pro1028-mono-single-pair-20261113.md) |
| At Flash C64 the two emitting lanes had median widths 15/16 in one release and 13/13 in the next. The ordinary-path release improved output/P90 by 10.79%/18.89%. | Profile KV reads and expert dispatch at those widths. Grouping verify rows and changing index pages jointly affect the producer and consumer. Their separate gains are unknown. [Before](https://inferencex.semianalysis.com/api/v1/reproduce?id=444981) · [After](https://inferencex.semianalysis.com/api/v1/reproduce?id=446046) |
| At Kimi C32 the fraction of sampled decodes with queued work rose from 35.09% to 69.51%. Full-response P90 improved 40.54%, output improved 0.77%, and P90 TTFT rose from 4.70 to 12.20 seconds. | Compare one admission or prefill policy at fixed compute. The published change also changed draft depth. [Before](https://inferencex.semianalysis.com/api/v1/reproduce?id=447679) · [After](https://inferencex.semianalysis.com/api/v1/reproduce?id=448018) |

## Held-out planning replay

Five anonymous packets covered narrow compute, wide compute, queue and rank pressure,
a workload without a baseline, and a synthetic C64 workload admitting only one
request. Plans received identical current observations and six case cards from three
other origins. Both DeepSeek targets were absent from the history shelf. Each call
used GPT-6.1-Sol at xhigh with no tools or preceding conversation. A separate
anonymous review scored ten predefined actions.

| Policy | Explicit checks met | Wide-compute decision |
| --- | ---: | --- |
| Original geak-x | 3/10 | Profile phases and intermediate traffic. Parallel resource assignments and a fusion control were absent. |
| First branch draft | 8/10 | Qualify the writer/reader boundary. The answer still left assignments and the unfused baseline implicit. |
| Revised branch | 9/10 | Conditionally dispatch representative attention/index and GEMM UTs on separate devices, reserve both GPUs for TP2, and compare fusion with the best compatible unfused chain. |

The revised answer omitted an explicit throughput tradeoff in the eight-rank
packet. The checklist rewards explicit decisions, and the revision reused the
initial five packets. A further replay withheld Kimi history, supplied only its
C32 reference, and proposed isolating prefill cadence at fixed admission and draft
depth. Its serving effect was unknown when the plan was frozen. The subsequent local screen appears below. The eight
public send windows were clipped by native client timestamps and analyzed per
emitting lane. Those periodic samples cannot identify individual tail requests.

[Packets, planner answers, rubric, window counts and source hashes](initial-plan-replay.json)

## Final transfer protocol

Reserve Muse Spark for the final transfer run. Give a fresh planner the original CI
inputs, fresh target observations and allowlisted donor cases. Exclude prior target
PRs, plans, candidate patches, results and derived summaries from retrieval and
from worker handoffs. Instruction examples and suggested numerical thresholds also
need provenance review. This is an experiment-specific holdout. The reusable
starting plan and its decision criteria apply to any workload.

Before the live scheduling comparison, a tool-less planner received a source-hashed packet with 37 case cards from the five
donor origins and only the anonymized original launch fields. It selected Baseline
and Profile before choosing a path. An independent UT worker, given no validated
oracle or tolerance, requested the current validation owner's contract. These
experiments cover the text inputs actually supplied. The final GPU campaign must
bind its own implementation inputs from the same permitted sources.

## A live high-concurrency decision

The Kimi-held-out planner proposed changing the prefill interval from 16 to 8
with admission 32 and six verify tokens fixed. Its original-image reference
could not initialize the host cache on the experimental node. The eight ranks
requested approximately 1,526 GB against a shared GTT limit of 1,477 GB. A
common change to the recurrent-state host ratio gave both arms 1,227 GB of
host cache. Both ran on the same eight MI355X GPUs with TP8/EP1 and DCP8.

| Prefill interval | Total output (tok/s) | Full-response P90 (tok/s/user) | P90 TTFT (s) | Scored requests |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 836.70 | 34.75 | 7.51 | 1,265 |
| 8 | 850.98 | 32.42 | 3.60 | 1,287 |

Each AgentX screen sent for 1,200 seconds. Shortening the interval improved output
by 1.71% and reduced P90 TTFT by 52.07%, while full-response P90 fell by 6.72%.
All 1,251 shared requests produced the same output lengths. Returning the measurements
to a planner that still had no access to Kimi history led it to retain 16 and
prioritize a profile of prefill interruptions and the slow requests. This is one
ordered performance comparison using synthetic acceptance. No new model-quality
test was run. [Native evidence](../brain/experience/raw/agentx-kimi-c32-initial-plan-20261126.json).

The final transfer packet also includes this Kimi donor result. A fresh planner
received 38 permitted case cards and the anonymized original launch. The 44 input
files were verified against the actual request. No prior target optimization
records were supplied. The final target's E2E campaign is still a separate experiment.
