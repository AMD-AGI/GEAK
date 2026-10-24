# Native task measurement adapter

`direct_external_command.py` connects a supplied native task evaluator to the
ordinary GEAK and GEAK-EVO measurement/result command interface. It uses
`geak.native` for task validation, source capture, native execution, and raw
report reading. The caller provides the task environment, GPU access, and lock.

- [Configure native measurements](../../docs/evo/direct-external-command.md)
- [Share immutable task inputs](../../docs/evo/shared-task-inputs.md)
- [Run GEAK-EVO](../../docs/evo/stock-workflow.md)

The former pilot launchers, provider patches, and research receipts have been
removed from this directory. Historical deployments keep their pinned sources;
this adapter does not update them.
