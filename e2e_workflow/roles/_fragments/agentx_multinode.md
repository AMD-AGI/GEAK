# Kernel extraction (AgentX) — multi-node

Read this only when `SERVING_TOPOLOGY` is in your inputs.

- `CAPTURE_BY_ENGINE` names each engine's capture inputs; `CAPTURE_ENGINE_LOG` is that engine's
  server log.
- Each sampling point is one serving leg: `bench_e2e.sh` with `PROFILE=1` and the point's CONC/ISL,
  with the serving env every leg in this run uses. Never start a server yourself and never run
  serving_probe.py. Phase and shape checks of the collected traces run here, on the driver.
- `SERVING_TOPOLOGY=pd_disagg`: write `engine` (`prefill` | `decode`) into each task's
  `run/requested_kernel.json` — the engine whose trace (`mn_trace_roles.json`) the kernel was selected
  from. The packer refuses a task that names no engine.
- Copy the skill's GPU scripts (regen_reference.py, oracle_power.py, kernel_self_noise.py) into the
  task directory, as op_bench.py is, and run each through gpu_lock.sh, which places it on a node.
  gpu_lock_set.py does not run on a multinode run.
