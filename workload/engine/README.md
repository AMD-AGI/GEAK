# engine — the plumbing

Everything under here exists so that the skill above it can be read without it. Nothing
in `engine/` knows an individual stage skill by name, and nothing a reader or an operator
normally edits lives here: `config/`, `stages/` and `runners/` are one level up, and
`run.py` forwards into `engine/core/stage.py` so that the entry point does not move when
the implementation behind it does.

Four directories, each named after the question it answers.

| | |
| --- | --- |
| `core/` | how one campaign walks from stage to stage — the driver (`stage.py`), the agent-mode stage (`agent_stage.py`) with everything it tells the agent in `core/prompt/`, the naming/identity gate (`target.py`), the report renderer (`report.py`) and the periodic monitor (`watch.py`) |
| `campaign/` | how eight GPUs are held, isolated and dispatched to — the scheduler, the per-GPU worker, container isolation, the durable credit file, the frozen-source bookkeeping, and the task entry point |
| `evidence/` | how measurement evidence is retained, placed and reused across attempts and resumes |
| `tools/` | single-purpose utilities: vendored-skill manifest verification, and the CPU-only fixture generator the self-check needs |

## Why this is code at all

`PRINCIPLES.md` R1 says that where prose will do, do not write code. These modules are
what survived that rule, and they buy exactly three things prose cannot:

1. **Physical isolation and parallelism** — one container per GPU with a single render
   node mounted, a lease per card, heartbeats, and eight of them at once. An env-only pin
   gets reset by the agent itself; only the container holds.
2. **Resumability** — evidence retained, revalidated and handed back, so an interrupted
   campaign continues from what is on disk rather than recomputing it.
3. **An after-the-fact verdict** — the target gate, the retired-knob gate, and
   `result.json` validation, each of which has to be able to refuse a run that already
   produced numbers.

Anything here that stops serving one of those three belongs in markdown instead.
`core/prompt/` is the worked example: what an unattended session is told — the contract,
the continuation, the repair attempt — is text, it is the text most worth rewriting as we
learn how these sessions fail, and the person best placed to rewrite it does not have to
be someone who edits Python. `agent_stage.py` keeps only the assembly.

## The one rule about direction

A slot may reach into `stages/_shared/`, because the pipeline implements the stage
contract. A slot that reached up into `engine/` would tie a replaceable skill to the
plumbing it is supposed to be independent of, and the dependency would point the wrong
way.
