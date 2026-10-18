# Run a search that can keep learning

The goal is better validated programs as the search continues. A larger archive
or a longer clock does not establish that result. Keep the selected source at
fixed times, then replay those sources under the same conditions.

```text
Understand the full operation and its measured costs
                       |
Choose a question, a parent and useful donor ideas
                       |
Implement, measure, explain the result and save the source
                       |
Keep the best programs, different ideas and unfinished work
                       |
Choose what to continue, combine, repair or stop
                       |
Replay fixed checkpoints and validate the serving workload
```

## What each module does

| Module | Contract |
| --- | --- |
| Archive | Keep exact programs, ideas, measurements and next tests. A slow or failed implementation can remain useful research material. |
| Selection | Show recent complete and incomplete development evidence alongside the existing cell and selection advice. Retaining an idea does not automatically allocate work to it. |
| Combining ideas | Accept any useful number of donors. Explain each contribution, its prerequisites and conflicts. Parent links alone do not show that an idea survived or helped. |
| Engineer history | Keep a useful bounded view of parents and recent quantities, with references to every full record. |
| Hardware evidence | Supply the same reference files to both arms through their normal input paths. Check actual reads and experiments before claiming the evidence helped. |
| Checkpoints | Save exact selected sources and their history. Distinguish new code, selecting older code and remeasuring the same code. |

The archive uses the existing `ResearchStore`; there is no second program
database. See [archive.md](archive.md) for the folder view, five short engineer
questions and exact program/idea references.

## Configure the full allowance

Use the existing admitted-command configuration and a pinned release. Read back
the effective values before starting:

- Model and reasoning level for both planner and engineers.
- Search seconds, final reserve and the original clock used on resume.
- Planner decisions, loop steps, total evaluations and evaluations per engineer.
- Continuation, cells, protection, PUCT and program-lease settings.
- Engineer and measurement time envelopes, including the last time new work fits.
- Checkpoints and the source query used to read them.

Eight hours of search means 28,800 search seconds plus the declared final
reserve. A pilot launcher with fixed short-run limits needs a separate reviewed
configuration; changing its wall time alone is insufficient. Evaluation limits
also affect selection advice, so arbitrary large limits can change the policy.

The engineer parent view is capped at 24 KiB and the quantity catalog at 16 KiB.
Small records remain unchanged. Larger views preserve useful excerpts and exact
retrieval references. These are input-view bounds, not a bound on every tool
response or the full conversation.

## Measure whether later hours help

Use ORIGINAL at hour zero and declared checkpoints such as 0.5, 1, 2, 4, 6 and 8
hours. Preserve each arm's actual selection and stopping rule. Replay both arms'
sources together with ORIGINAL. Keep unavailable results and failed attempts
visible; a valid unchanged ORIGINAL can score 1.0×.

Compare hour-eight code with hour-four code in the same replay. A different
source selected late may have been discovered earlier. Record its first saved
occurrence before calling it a late discovery. Repeat independent searches to
measure variability; repeated timings of one program are not search repeats.

The source hooks and their interruption limits are documented in
[checkpoint_sources.md](checkpoint_sources.md). Source tests establish their
behavior. Installed use, eight-hour discovery and serving improvements each
require their own measured results.
