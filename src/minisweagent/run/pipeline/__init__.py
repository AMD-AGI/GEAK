"""Pipeline sub-package: orchestration, preprocessing, and task management.

Key entry points:

- :func:`~.preprocessor.run_preprocessor` -- sequential pre-processing pipeline
- :func:`~.orchestrator.run_orchestrator` -- multi-round optimisation loop
- :func:`~.dispatch.run_task_batch` -- run tasks in parallel via agent pool
- :func:`~.helpers.run_discovery` -- kernel test discovery
- :func:`~.helpers.inject_resolved_kernel` -- resolve kernel URL
"""
