# Raw evidence

Retain acquisition collections unchanged. Add a new collection for later downloads or
measurements; write revised interpretations in `cases/` and `routes/`. Cite the collection
ID and original relative path, with the file hash where available, so reorganizing those
documents does not change their evidence.

```text
raw/
  <collection-id>.json       provenance and checksum scope
  <collection-id>.sha256     original evidence anchors
  collections/<collection-id>/
                            complete acquisition directory, excluded from Git
```

[AgentX MI355X collection](agentx-mi355x-20261113.json) preserves the complete original
research directory. `sources/` holds acquired evidence; `experiments/` retains original
run outputs. Historical `analysis/`, `data/` and `reports/` are companions, not all raw
measurements. Keep their evidence level and original content.

The shared archive outside the code checkout retains large source trees, Git mirrors,
image layers and run artifacts; `collections/` is a local view. Git versions this catalogue
and its checksum anchors, not the payload. A fresh clone needs
the archive supplied at the declared location or an archive root recorded in the task brief.
Native Git/OCI/acquisition inventories remain in each collection; the anchor checksum list
does not claim a full recursive content audit. Preserve old paths when relocating a collection.

[Muse Spark gap study](agentx-muse-gap-20261125.json) adds the November 25 public release,
in-window serving logs and native 20-minute MI355X comparisons.

[Day-zero source ports](agentx-muse-day0-ports-20261126.json) records original-image
backports, request-level outcomes and a source-matched historical numerical rejection.

[Kimi C32 scheduling experiment](agentx-kimi-c32-initial-plan-20261126.json) retains
the fixed-image interval comparison, startup rejections and host-pool observations.
