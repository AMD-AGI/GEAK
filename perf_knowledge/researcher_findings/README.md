# Researcher findings collection

This is the generated, persistent collection for Deep Research Agent findings. It lives inside the
GEAK knowledge base but is provenance-separate from:

- curated `perf_knowledge` reference cards, and
- measurement-derived `e2e_workflow/knowledge/learned` cards.

The unchanged online Researcher remains the sole author of knowledge content. After Stage 7,
`kernel_workflow/scripts/research_kb.py ingest` immediately transforms `deep_search.json` into
canonical cards. The transformer is deterministic and performs no model or network calls.

## Generated layout

```text
researcher_findings/
├── observations/<run-id>.json       # immutable normalized Stage-7 observation bundle
├── cards/<operator>/<card-id>.json  # machine-readable canonical card
├── cards/<operator>/<card-id>.md    # human-readable view of that card
├── snapshots/<snapshot-id>.json     # checksummed immutable card manifest
├── channels/latest.json             # atomically updated snapshot pointer
├── validation/events.jsonl           # append-only Director outcomes by card/snapshot
├── index.json                       # generated machine index
└── INDEX.md                         # generated human index
```

The generated paths appear only after the first successful online Researcher run.

## Merge policy

The unit of storage is a **scoped mechanism**, not a CI run.

1. Normalize operator, language/backend, GPU architecture, dtype, regime, bottleneck, and source
   kernel fingerprint.
2. Search only scope-compatible cards.
3. Match weekly wording variation with normalized mechanism tokens.
4. A match appends an observation/evidence and updates aggregate metadata; it does not create a new
   card or continually rewrite the first canonical mechanism.
5. A genuinely different mechanism or incompatible scope creates a new card.
6. If a later Researcher run rejects a previously preferred mechanism in the same scope, preserve
   both observations and mark the card `contested`; retrieval lowers its rank and surfaces a caution.
7. The entire merge runs under a filesystem lock and publishes one atomic snapshot.

Raw open measurements and rejected directions remain in the run observation for audit. Only the
Researcher's final ranked `directions[]` become planner-visible cards in this first version, matching
the existing online `deep_search_brief.md` contract and avoiding per-question KB growth.

## Online and offline modes

- `dra_mode=online`: run the existing web Researcher unchanged, pass its fresh brief directly to the
  TechLead, and immediately merge the findings here. The current run does not retrieve its duplicate.
- `dra_mode=offline`: invoke no Researcher and no web tools; retrieve matching cards from a snapshot
  and materialize `EVAL_DIR/deep_search_brief.offline.md` for the same TechLead handoff.
- `dra_mode=off`: neither path runs. The legacy `dra_enabled=true` argument remains an alias for
  `dra_mode=online`.

All cards are advisory. Correctness, on-box verification, and final Director validation remain the
only performance authority. Online and offline outcomes are appended to `validation/events.jsonl`;
this metadata never rewrites Researcher-authored card content and does not affect retrieval ranking in
the initial experiment.

For an online→offline fidelity check, run:

```bash
python3 kernel_workflow/scripts/research_kb.py compare \
  --kb-dir <researcher_findings> \
  --online-json <online-eval>/deep_search.json \
  --offline-retrieval <offline-eval>/research_kb_retrieval.json \
  --output <offline-eval>/online_offline_comparison.json
```

The comparison reports direction recall, mechanism similarity, specialty agreement, and the
direction-to-card mapping. This measures whether the offline planner starts with substantially the
same information as the online Researcher; Director-verified speedup remains a separate outcome.
