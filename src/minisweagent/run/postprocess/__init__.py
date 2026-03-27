"""Post-processing: benchmark parsing, result collection, patch selection.

Modules in this package handle everything that happens after agent runs
complete:

- ``benchmark_parsing`` -- parse benchmark output (latency, speedup,
  shape counts) and select the best patch by measured performance.
"""
