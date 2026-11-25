# Multi-node from a Claude Code prompt — the main session

Read this only when the prompt serves the model on other machines (it names node IPs, prefill and
decode hosts, or a `multinode` block), or when the workflow refused `args.multinode`. There is no
`run_e2e.py` in front of you: you run its check yourself, before the Workflow call.

1. Claude must already run with `GEAK_MN_STATE=<shared_fs>/pins/<run>/mn_state.json` and
   `ROCR_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES=`. The kernel tools read the pin
   from this process's environment, so exporting it in a Bash call does not help: if it is unset,
   stop and give the user the start command the check prints.
2. Build the block from the prompt (fields: `docs/multi_node.md`):
   - `nodes` in rank order: each its IP, or `{host, role}` with `role` `prefill` | `decode` under PD;
   - `topology`: `pd_disagg` when roles are named, else `tp_span`;
   - `pd`: `{backend, ib_device}` from the KV transfer backend and the RDMA NICs the prompt names;
   - `ssh`: `{user, key, port, prefix}` from how the prompt says to reach the nodes;
   - a node's `launch_command` and a `router` only when the prompt gives them.

   Leave out `shared_fs`, `state` and `work_dir`: the check takes them from `GEAK_MN_STATE`.
3. Run the check with the block plus `"tp"` (each engine's TP):
   `GEAK_MN_SPEC='<block>' python3 <this workflow dir>/scripts/multi_node/mn.py check --prompt`
4. A refusal (exit 2) starts nothing. Report every `problems` line and stop. A busy card, a bound
   port or an unreachable node is the user's to fix: do not edit the block or `MEM_FRACTION` to pass.
5. On success, merge the JSON after `MN_PROMPT_ARGS=` into the Workflow args as it is: `tp`,
   `eval_dir`, `multinode`, `multinode_check`, `expected_gfx`, `expected_target`. Pass no other
   `eval_dir` or `exp_root`. Everything else (model, backend, workload, server flags and env) comes
   from the prompt as on one node.
