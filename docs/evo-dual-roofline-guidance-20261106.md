# Concurrent kernel and workload roofline objectives

This component records a prospective guidance change: pursue supported kernel/region
attainment and independently modeled full-workload attainment concurrently. Kernel
attainment neither establishes nor must precede workload attainment. Preserve unknown
reference coverage and use actual workload evidence to select substantial opportunities.
The declared numerical, quality, latency, configuration and time contracts remain intact.

## Source separation

The parent component branch starts at `9f925de2902815d3125f0a51fd4d0dc2d7116f01`.
Its first commit imports only two preexisting, reviewed common14 knowledge guides.
The following component commit adds exactly 11 workload-guide lines and 6 hardware-guide
lines, plus this note; the imported prerequisite content is not a new 17-line contribution.

| Guide | Existing common14 SHA-256 | With dual-objective guidance SHA-256 |
| --- | --- | --- |
| e2e_workflow/knowledge/e2e_optimization.md | 5fff7c48fbc6f1b5717e30a7c215c660520da2ec1017f475e55c8b8d531f2900 | d46112dfb549a918d017b171b3e33a5aff0a6f7478a42b93ada2f323e4ebcb5c |
| kernel_workflow/knowledge/kernel_hardware_evidence.md | 4feb0aa83b7400ddca45ac79f5c5caedf3be45631451696954daacdb762b227d | f43ea058e152088aad94a75fe13a0296a4cb7e2d11aea6d2c65c315ba6baffc5 |

The exact four-leaf GEAK/GEAK_A3 mirrored review patch has SHA-256
`cedf42a23d19ed4e581852ec1726683cb5946bbe9bbd7f27cf7c18a50372336b`.
The corresponding binding has SHA-256
`3ef5d77b0647456c4604ab7819c33190178a694559fd4bdf755083cfa7142421`.
Codex root, actual Claude and actual Researcher3 concurred on that exact prospective
source change on November 6, 2026. This repository represents the single GEAK tree;
the separately bound runtime overlay contains identical GEAK and GEAK_A3 guide bytes.

## Evidence and dependency limits

This is guidance source tracking, not an installed or qualified common14 runtime.
At the chosen Git base, the kernel guide's linked `tracelens_evidence.md` is absent;
`profiling_guide.md` and `kernel_workflow/scripts/profile_kernel.sh` exist. No direct
references to either guide were found in the base role files; this check does not
establish how dynamically assembled contexts load them. These dependencies and
context assembly are not imported or qualified by this component.

No experiment, runtime import, test or GPU work is part of this change. No frozen
source or running experiment is altered. Source concurrence does not establish that
an agent consumed the guidance, that either 85% objective was reached, or that this
component improved discovery, kernel performance or serving performance. Prospective
experiments need their own complete source binding and declared evaluation contract.
