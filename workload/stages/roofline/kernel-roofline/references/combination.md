# Mode: combination

Run both legs over one aligned case set and report them side by side. Record
`mode: combination` and keep outputs under `results/combination/`.

The purpose is **not** a better single number. It is to locate a disagreement: when
TL says a seam is at 40% of roof and rocprof says 93%, one of them is wrong about
the numerator (FLOPs/bytes) or the denominator (peaks), and only a joined table
shows which.

## 1. What this mode must not do

**TL's % and rocprof's % cannot be compared directly.** They are ratios against
different denominators. On gfx950, denominator ratio TL/rocprof:

| axis | ratio |
|---|---|
| HBM bandwidth | 0.856 |
| bf16 compute | 0.573 |
| fp8 compute | 0.628 |
| fp4 compute | 0.502 |

TraceLens' bandwidth figure is **already derated**. Stacking a `target_eff=0.90` on
top of it is a double derate and inflates `attainable_speedup` by about 17%.

Never divide one leg's percentage by the other's. Report both columns.

## 2. The legitimate graft

If you want one number that borrows the best of each: take the **numerator** from
TL's `PerfModel` (its `InferenceAttention` splits QK and V head_dim, which is finer
than a hand-rolled count) and the **denominator plus the verdict logic** from the
rocprof side. That is a third, explicitly labelled quantity — not either leg's
percentage.

## 3. Folding rules

**The folding arithmetic is engine-independent and lives in
[folding.md](folding.md).** Read it first; it binds this mode and the other two
alike, and `scripts/join_legs.py` / `scripts/compare_legs.py` enforce it. In one
line: fold in the time domain, sum numerator and denominator separately, divide
once — never average percentages, never average AIs.

This mode adds exactly two things to those rules.

- **The division of labour between the engines.** Within a case, the per-kernel
  weights `t_k` are **TraceLens'** device ms; the ratio `Eff_k` is **rocprof's**
  (achieved/attainable, from its counters). rocprof's own durations come from a
  serialized-dispatch profiling run and are not the deployment time distribution.
  **rocprof contributes the ratio; TL contributes where the time goes.**

- **Both columns must be folded over the same case set.** The TL leg costs no GPU and
  covers every shape; the rocprof leg costs a full profile per (shape, arm) and in
  practice covers the dominant case per seam. `join_legs.py` therefore re-folds the TL
  column over exactly the subset rocprof covered, so the two columns share a
  denominator — which means **the `TL%` in a join table is not the `TL%` of the
  standalone tl-only rollup.** Do not quote them interchangeably. See
  [folding.md](folding.md) §"Comparing two tables".

Identical folding makes the two columns structurally comparable. It does **not** make
their percentages the same kind of number — §1 still holds.

## 4. How to read the disagreement

Over a 40-case run: ranking and bound type came out essentially equivalent (36–38 of
40 cases agreed on axis). **Headroom did not.** So the disagreement between the two
engines is a dispute about **byte and FLOP counting**, not about percentages. Chase
it in the numerator first.

`compare_legs.py` is the tool for this: it prints TL AI / ridge / cmp% beside
rocprof AI / ridge / cmp% and the AI ratio. A row where the two agree on % but sit on
different axes is not agreement — it is two errors cancelling.

## 5. `x_TL%` is an identity, not a test

TL derives bytes and FLOPs from the shape, so its roofs are bit-identical across a
baseline and a champion leg. Its roofline-% therefore *necessarily* tracks the
measured speedup exactly. That ratio proves nothing.

**Only `x_rocprof%` is a real test**, because its numerator is measured per leg. For
a memory-bound row it decomposes cleanly:

```
x_rocprof%  =  x_bytes · x_meas
```

which says directly whether the win came from moving fewer bytes or from moving the
same bytes faster.

## 6. The only experiment that adjudicates

Run the **same kernel in two implementations** (baseline and champion) through
**both** legs. The measured speedup is an answer key neither engine can refute: an
engine whose predicted ratio contradicts it is the one that is wrong. Use the §5
decomposition to localize which term.

This two-implementation natural experiment is the only construction that settles
"which engine is wrong". A single implementation profiled by both legs can only show
*that* they disagree.

## 7. The arch JSON is not a constant

Four distinct MI355X arch JSONs exist on this cluster, with **18.6% dispersion in
the bf16 peak**:

| file | `mem_bw` | `matrix_bf16` | `source` |
|---|---|---|---|
| `MI355X.bw6857_bf1449.json` | 6857 | 1449 | benchmark-derived |
| `MI355X.bw6857_bf1452.json` | 6857 | 1452 | benchmark-derived |
| `MI355X.bw6877_bf1422.json` | 6877 | 1422 | benchmark-derived |
| `MI355X.bw8000_bf1686.json` | 8000 | 1686 | **absent** |

The most-copied one is the last: it has **no `source` field**, exactly equals the
datasheet pin, and carries only five `matrix_*` keys instead of eight. It is
handwritten. Do not infer provenance from the filename.

So every output row records **arch path + md5 + `source` + the peak key actually
resolved**. MD5s and the full comparison are in
[`../assets/README.md`](../assets/README.md).

One further trap inside the JSON: `matrix_fp4` hides two backends. The microbench
measures CK at 5146 and Triton at 2722 TFLOP/s but writes only the **max** into the
single `matrix_fp4` key. A Triton or FlyDSL fp4 kernel scored against that key is
underestimated by 1.89×. `assets/MI355X.measured.json` carries hand-added
`matrix_fp4_ck` / `matrix_fp4_triton` keys recovered from the same run's SUMMARY
block; use those when the backend is known.

## Running it

```bash
# 1. TL leg -> rows.json  (see tl-only.md)
# 2. rocprof leg, one case per command (scripts/rocprof_leg_template.sh,
#    which drives case_runner_template.py) -> <dir>/<kernel>_<case>_<arm>_roofline.txt

python3 scripts/join_legs.py    --rocprof-dir results/rocprof/leg \
                                --tl-rows results/tl-only/rows.json \
                                --out results/combination/per_case_join.json
python3 scripts/compare_legs.py --rocprof-dir results/rocprof/leg \
                                --tl-rows results/tl-only/rows.json
```

**With two arms, pass `--arm` and run the pair.** Section 6's experiment puts both
arms' files in one directory, so their names carry an arm tag and an untagged lookup
matches *nothing* — the tables come out empty rather than wrong, which is easy to
misread as "the join found no overlap". The TL rows must be the matching arm:

```bash
for arm in baseline candidate; do
  python3 scripts/join_legs.py    --rocprof-dir results/rocprof/leg --arm $arm \
                                  --tl-rows results/tl-only/rows_$arm.json \
                                  --out results/combination/per_case_$arm.json
  python3 scripts/compare_legs.py --rocprof-dir results/rocprof/leg --arm $arm \
                                  --tl-rows results/tl-only/rows_$arm.json
done
```

Both scripts fold through `scripts/fold.py`, the same module a single-engine leg
calls directly (`python3 scripts/fold.py --rows <rows.json>`) — which is what makes
the columns comparable rather than merely adjacent.

Both scripts end with a coverage line counting TL rows that had no rocprof file.
Read it. A leg truncated by one timeout or by the stdin trap in
[rocprof.md §6](rocprof.md) otherwise produces a table that looks complete,
because the missing rows are simply not in it.

`--tl-rows` is a JSON file with a `rows` list; each row needs `kernel`, `regime`,
`case`, `measured_us`, `roofline_us`, optional `case_weight`, and a `per_kernel` list
of `{"kernel": <device symbol>, "ms": <device ms>}`. `compare_legs.py` additionally
needs `flops_g`, `bytes_mb`, `bound`, `peak_tflops`, `peak_bw_gbps`, `peak_spec`.
Rows are matched to `<rocprof-dir>/<kernel>_<case>_roofline.txt`.

rocprof truncates long mangled kernel names in its per-kernel headers, so the join
falls back to 60-character prefix containment — an exact lookup silently misses
exactly the templated ATen and rocprim symbols that matter.
