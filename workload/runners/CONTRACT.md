# Runner contract

A runner isolates "start one unattended agent session" from any particular CLI. The
scheduler, the 48-hour accounting, the per-GPU containers and the fault-tolerance logic
know nothing about claude and nothing about codex — only about this contract.

Changing agent means `"runner": "codex"` in the config. No scheduling code changes.

## Interface

```bash
bash runners/<name>.sh <prompt_file> <workdir> <log_file>
```

| Argument | Meaning |
| --- | --- |
| `prompt_file` | a file whose contents are the complete instruction for the agent. The runner must pass it through verbatim — no truncation, no added phrasing of its own |
| `workdir` | the agent's working directory. The runner must `cd` into it |
| `log_file` | all agent stdout/stderr goes here. The runner must not swallow output |

**Exit codes**

| Code | Meaning |
| --- | --- |
| `0` | the session ended normally |
| `78` | an unmet precondition (binary absent, sidecar missing, no credentials). The caller treats this as `blocked` and must **not** retry it as a failure |
| anything else | the session failed. The caller may retry |

"Cannot start" and "started but did not succeed" are separated because the scheduler
retries, and retrying the former ten thousand times produces the same result.

## What a runner must do

1. **Self-check preconditions and return 78 when something is missing.** Is the binary
   present, is the sidecar it depends on present, are credentials readable? Starting
   silently and then failing every tool call closed is far worse than reporting the
   missing piece.
2. **Run non-interactively.** No waiting on a TTY, no confirmation prompts. The agent
   must be able to edit files and run commands on its own.
3. **Keep state directories on `/dev/shm`, not NFS.** Both claude's `CLAUDE_CONFIG_DIR`
   and codex's state are sqlite; killed hard on NFS they leave a stale `-shm` behind, and
   every subsequent start fails with `code 15 locking protocol`. Give each worker its own
   state directory so they cannot collide.
4. **Set no internal timeout.** The time budget is enforced from outside with
   `timeout(1)`. A runner that imposes its own limit fights the 48-hour budget.
5. **Pass GPU visibility through untouched.** The caller has already established physical
   isolation via container device mounts; the runner must not overwrite
   `ROCR_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES`. Measured: an env-only pin gets reset by
   the agent itself — only physical isolation holds.

## What the caller must do

- Time budget: `timeout <sec> bash $RUNNER ...`.
- Physical isolation: one container per GPU, mounting exactly one
  `/dev/dri/renderD<128+8*i>`.
- Log ownership: one `log_file` per worker, never shared.

## Shipped implementations

| File | Status |
| --- | --- |
| `claude.sh` | validated on real sessions |
| `codex.sh` | written to the contract, **not yet validated** (no codex binary in the development container); returns 78 when pieces are missing, as specified |
| `stub.sh` | calls no agent at all; exists to prove the scheduler is agent-independent |

## Credentials

Credentials never enter the repository. `claude.env.example` is a template: copy it
outside the tree, fill it in, and point `runner_env.env_file` at it from the scenario
config. On this cluster the agent reaches an enterprise gateway
(`ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN` + a subscription-key
`ANTHROPIC_CUSTOM_HEADERS`) rather than `~/.claude` credentials. When that file is absent
or incomplete, `claude.sh` returns 78 and the worker stops immediately instead of burning
the budget on retries that cannot succeed.
