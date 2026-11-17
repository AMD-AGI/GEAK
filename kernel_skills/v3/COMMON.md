# Rules every v3 role follows

Every role reads this file first. `SK` = `kernel_skills/`, `KW` = `$SK/../kernel_workflow`, `RUN` = the run dir.

## Files and communication
- Facts with evidence (command, log path, numbers) go to `RUN/LEARNINGS.md`, appended as you go.
- Messages for other roles go to `RUN/inbox.md` as `- (<from> → <to|all>, <UTC time>) <message>`. Read it at
  every step boundary. The lead does not relay messages between roles.
- If the Write tool refuses a file, write it with a Bash heredoc (`cat > file <<'EOF'`). Never return a
  report only as text: it must be on disk.
- Keep `RUN/<your-dir>/progress.md` current: step, UTC time, what is running (pid, log path), next command.
  After an interruption, a fresh agent must be able to continue from it alone.

## GPU
- GPU work only via `cd <ws> && bash $KW/scripts/gpu_lock.sh <GPU spec> <cmd>`. Use the spec in
  `RUN/cmd.env`, never `HIP_VISIBLE_DEVICES`. `cmd.env` also sets `GEAK_GPU_POOL_WAIT` and
  `GEAK_GPU_RUN_TIMEOUT`; export them.
- Any GPU job longer than about 2 minutes runs detached:
  `nohup setsid bash -c '<cmd>' > <log> 2>&1 &`. Write its pid and log path to `progress.md`, then poll
  the log. An agent or container restart must not lose the job.
- Hold the GPU lease only for the measurement itself: build and compile outside it. With one GPU shared by
  several roles, a role keeps one lease of at most about 10 minutes; geak-ship and `MODE=final` take priority.
- Exit 86 / `GEAK_SOURCE_INVALID` = invalid measurement, not a result.

## Measurement honesty
- Decisions use only instruments validated against the official eval (see geak-benchmark). In-process A/B of
  two kernels on shared buffers can be biased by 1–7% and even flip sign. Use fresh processes.
- Ties come with a mechanism. A "ceiling", "floor" or "fixed cost" is a claim until geak-roofline or an
  `investigator MODE=ceiling` confirms it with a probe that is independent of the team's design.
- Hold candidates to the oracle's contract (its tolerances), not to a stricter bar you set yourself, such as
  bit-exactness that the contract does not require. Measure the tolerance before designing around numerics.

## Candidates
- Workspaces: `bash $SK/bin/mkws.sh $RUN <dst> [parent patch]`.
- Save only editable paths: `git add -A && git diff --binary --cached base -- <EDITABLE> > patch.diff`
  (and the same with `parent` for `delta.diff`). Keep ablation and probe code out of the patch.
- Snapshot every step that gains ≥ ~3% as `cands/<id>/v<n>.diff`, so the lead can ship it early.
