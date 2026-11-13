# Execution and evidence tools

[Brain](../brain/SKILL.md) makes campaign decisions. Engine tools provide
mechanical execution, isolation, recovery and evidence retention.
Modules may reuse these tools by path; their native validators still own acceptance.
The [sequential driver](core/README.md) is an optional executor with a narrower protocol.

| File | Used by / responsibility | Scope |
| --- | --- | --- |
| `core/stage.py` | `run.py`: config, stage execution, validation and resume | Sequential configured stages |
| `core/agent_stage.py`, `core/prompt/*.md` | Driver: runner invocation, continuation and repair prompts | Full driver deliveries |
| `core/target.py` | Driver: declared/observed stack and naming checks | One configured target |
| `core/report.py` | CLI and watch: render stage results; preserve driver/native rejection | Driver layout, including legacy applyback summary |
| `core/watch.py` | Driver: local process/resource monitoring and report files | Does not deliver chat updates |
| `campaign/scheduler.py` | Optimize: task queue, declared GPU widths and worker recovery | Kernel campaign; Docker isolation or local CPU smoke |
| `campaign/worker.sh` | Scheduler: attempts, credit, KEEP/revert and measurement hooks | Qualified kernel task protocol |
| `campaign/container_runtime.py` | Scheduler: configured runtime preparation and container arguments | Docker |
| `campaign/durable_write.py` | Scheduler/worker: atomic state writes and credit recovery | File-based state |
| `campaign/source_tree.py` | Worker/Verify: editable-source and frozen-harness snapshots | Declared kernel task layout |
| `campaign/run_task_entry.py` | Measurement/Verify/UT: native test entry and receipts | Python kernel tasks |
| `evidence/retain_measurement.py` | Worker/task entry: retain and recover measurement evidence | Kernel artifact layout |
| `evidence/retention_reuse.py` | Optional retention helper: reuse immutable archives | Recognized campaign layouts; copies otherwise |
| `evidence/generated_oracle_archive.py` | Task entry: retain generated oracle versions | Bundled UT harness adapter |
| `tools/make_fake_tasks.py` | CPU smoke checks: generate fake tasks | Test fixture |
| `tools/skill_manifest.py` | Required self-check: verify vendored trees | Manifest integrity |

Keep planning policy in Markdown. Hardware/configuration and protocol adapters are not
universal orchestration; reuse only the tools whose input contract matches the call.
