# Which file do I deploy?

**`deploy/` is the win. `evidence_only_001/` is not.** Deploying the wrong one is the single
mistake this entry exists to prevent, so the two are in separate directories and neither is at
the top level.

| directory | file | status |
| --- | --- | --- |
| `deploy/` | `mixtral8x7b_bf16_tuned_gemm.csv` (2 rows) | **the deployable win** — round 2, +0.474% (6918.43 → 6951.22 tok/s) |
| `deploy/` | `002_aiter_tuned_gemm_mixtral.patch` | the same two rows as a `git apply`-able diff |
| `evidence_only_001/` | `mixtral8x7b_tuned_fmoe.csv` (12 rows) | **not deployed.** Round 1's win, superseded. See `evidence_only_001/NOT_THE_DEPLOYABLE_WIN.md` |
| `evidence_only_001/` | `001_aiter_tuned_fmoe_mixtral.patch` | the same twelve rows as a diff |

Do not install both. Round 2 measured the stack and it does not add throughput — see the parent
`README.md`, section "Two rounds".

```bash
# the deployable win
cp deploy/mixtral8x7b_bf16_tuned_gemm.csv /sgl-workspace/aiter/aiter/configs/model_configs/
rm -rf /tmp/aiter_configs        # MANDATORY — the merge is cached
# then start the server; the CSV must be in place BEFORE launch
```

md5: `deploy/mixtral8x7b_bf16_tuned_gemm.csv` = `c85d3ec60d2d2dd0ad6e61fee6ac7c4b`,
`evidence_only_001/mixtral8x7b_tuned_fmoe.csv` = `3910458156ceae313ad9d8a41f7edd87`.
