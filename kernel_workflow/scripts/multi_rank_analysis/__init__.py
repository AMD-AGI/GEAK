"""Generic multi-GPU-rank analysis primitives for GEAK Kernel Workflow.

This package is deliberately operator-agnostic: it must never import or reference a specific
kernel, model, or backend (no MegaMoE/AITER/MORI/FlyDSL names). It exists so that any distributed
UT/benchmark's per-rank timing and profiling output can be merged into a stable, versioned report
shape, and so that operator-specific analysis Skills (see
``kernel_workflow/knowledge/analysis_skills/``) have a shared, tested foundation to build on instead
of re-deriving all_gather/mean/max plumbing per operator.

See README.md in this directory for the full design rationale and the contract each module honors.
"""

from .aggregate import (
    all_ranks_true,
    classify_case_speedup,
    merge_rank_records,
    reduce_scalar,
    time_distributed,
)
from .bundle import (
    ANALYSIS_BUNDLE_SCHEMA_VERSION,
    bundle_from_rank_report,
    validate_analysis_bundle,
    validate_measurement_tracks,
)
from .experiments import (
    EXPERIMENT_SCHEMA_VERSION,
    compare_controlled_variants,
    validate_experiment_manifest,
)
from .evidence import (
    EVIDENCE_CATALOG_SCHEMA_VERSION,
    resolve_measurement_tracks,
    validate_evidence_catalog,
)
from .hardware import (
    HARDWARE_CONTEXT_SCHEMA_VERSION,
    validate_hardware_context,
)
from .instruction_analysis import (
    ATT_COLUMNS,
    load_instruction_category_map,
    parse_att_stats_csv,
    read_att_occupancy,
)
from .intervals import (
    analyze_category_overlap,
    critical_path,
    interval_overlap_us,
    interval_union_us,
    merge_intervals,
)
from .provenance import (
    COLLECTION_PROVENANCE_SCHEMA_VERSION,
    validate_collection_provenance,
)
from .schema import SCHEMA_VERSION, build_report
from .trace_categories import bucket_trace_events, load_category_map

__all__ = [
    "SCHEMA_VERSION",
    "EXPERIMENT_SCHEMA_VERSION",
    "EVIDENCE_CATALOG_SCHEMA_VERSION",
    "HARDWARE_CONTEXT_SCHEMA_VERSION",
    "ATT_COLUMNS",
    "ANALYSIS_BUNDLE_SCHEMA_VERSION",
    "COLLECTION_PROVENANCE_SCHEMA_VERSION",
    "analyze_category_overlap",
    "all_ranks_true",
    "bucket_trace_events",
    "bundle_from_rank_report",
    "build_report",
    "classify_case_speedup",
    "compare_controlled_variants",
    "critical_path",
    "interval_overlap_us",
    "interval_union_us",
    "load_category_map",
    "load_instruction_category_map",
    "merge_intervals",
    "merge_rank_records",
    "reduce_scalar",
    "resolve_measurement_tracks",
    "parse_att_stats_csv",
    "read_att_occupancy",
    "time_distributed",
    "validate_experiment_manifest",
    "validate_evidence_catalog",
    "validate_analysis_bundle",
    "validate_measurement_tracks",
    "validate_collection_provenance",
    "validate_hardware_context",
]
