# scripts — usage

Six tools for one job: **transcribing a tuned plain-Triton kernel into Gluon, and re-injecting the
pipeline afterwards**. Run from the GEAK repo root with
`SKILL=perf_knowledge/expert_skills/skills/gluon_authoring`.

## Transcription

| tool | what it does |
| --- | --- |
| `dump_ir.sh` | Runs a compile command with IR dumping on and collects `.ttir` / `.ttgir` / `.amdgcn` per variant. `--emit-gluon layouts\|anchor\|pipeline` additionally emits a Gluon skeleton from the dumped TTGIR. |
| `recover_gluon.py` | The driver: dump → recover layouts → emit an anchor → **`--verify`** the recompiled anchor's layout attributes against plain. Always finish with `--verify`; it is the only gate that catches a layout recovered wrong (wrong `order`/`kWidth`) behind a passing numeric oracle. |
| `ttgir_to_gluon.py` | The parser/emitter underneath. Pure text, no GPU and no `triton` import needed. `--selftest` checks it offline against three built-in TTGIR samples; `--pipeline` also emits a pipeline skeleton. Covers `#blocked`, `#amd_mfma`, `#swizzled_shared`, `#padded_shared`, `#linear`, `ttg.dot_op`, `ttg.slice`. Does **not** place `convert_layout` (manual, see `references/tile-programming/layout-recipes.md`) and cannot name `amd_rotating_shared` (language-surface gap). |
| `smoke_test_recover.sh` | Offline end-to-end check of the recovery toolchain. |
| `smoke_recover_gpu.py` | On-GPU version; needs `torch` + `triton`. |

## Pipeline re-injection

| tool | what it does |
| --- | --- |
| `probe_levers.py` | Per-build capability probe. `reinject_ttgir_pipeliner` answers whether plain's `add_schedule_loops` / `add_pipeline` are present in *this* `libtriton.so` before you edit `compiler.py`. That is the probe this skill uses; the other five it exposes belong to lever cards that are not part of this package. |

Both `--selftest` entry points run with no GPU and no ROCm:

```bash
python3 "$SKILL/scripts/ttgir_to_gluon.py" --selftest
python3 "$SKILL/scripts/probe_levers.py"   --selftest
```

## Two things the emitted text gets wrong here

The scripts are an unmodified upstream snapshot, so two strings they print do not match this skill:

- `recover_gluon.py --record` / `--verify` prints `perf_delta_vs_plain: <fill> # regression expected,
  NOT a reject`. That is upstream's transcribe-only step, where the pipeline layer came later. **Here the
  port is transcribe *plus* re-injection and closes at ≥95% of tuned plain** — due in round 1, or round 2
  when plain is auto-pipelined. Take the gate from `skill.md ## Procedure` step 3, not from this line.
- Several docstrings and `cmd` fields cite files this package does not carry (`compiler-contract.md`,
  `transcribe.md`, `experiment-records.md`, `lever-cards.json`, `opt_swp_test.py`, `bench.py`,
  `prof_driver.py`). Dead by design — do not go looking for them.
