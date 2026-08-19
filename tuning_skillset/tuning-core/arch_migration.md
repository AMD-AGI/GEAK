# Moving a tuning setup to a new architecture

Re-tuning per architecture is the easy half — you already know configs do not
transfer. The hard half is that a tuning *harness* also encodes hardware facts,
and when those are wrong they mostly do not error. They produce a plausible
number.

This is the checklist for what to re-derive, ordered by how quietly it fails.

## Fails silently, in order of how much it costs you

### 1. Prune rules

Any rule that rejects a candidate before launching it is a measurement of one
compiler on one part. The common case is an LDS/shared-memory bound: a tile
that will not fit is rejected without a compile.

Derive it, do not reason it out. Sweep tiles until they fail, parse the
compiler's own `Required: N, Hardware limit: N` numbers, and fit the relation.
A rule reasoned from first principles typically carries a factor the compiler
does not actually charge — for double-buffering, say — and a rule that is 2x
conservative rejects most of the legal space.

Why this is first: **an over-tight prune is invisible.** The sweep reports "no
candidate beat the noise floor," which reads as a finding about the hardware.
The wins simply never appear. There is no error and no warning.

Read the LDS budget from the device rather than hardcoding it. The *formula*
still needs re-derivation; only the limit is free.

Reading it is less free than it looks. `torch.cuda.get_device_properties(0)
.shared_memory_per_block` exists on torch 2.10 and **not** on torch 2.9.1, so
the obvious one-liner raises `AttributeError` on one of the two production
images. Fall back to `rocminfo`, which has reported it all along:

```python
def lds_bytes(dev=0):
    try:
        return torch.cuda.get_device_properties(dev).shared_memory_per_block
    except AttributeError:
        # GROUP segment in rocminfo's pool listing, in KB.
        out = subprocess.run(["rocminfo"], capture_output=True, text=True).stdout
        m = re.search(r"Segment:\s+GROUP.*?Size:\s+(\d+)\(", out, re.S)
        return int(m.group(1)) * 1024
```

Measured this way: **65536 bytes on gfx942, 163840 on gfx950** — 2.5x, and the
prune formula itself was unchanged (`lds_calibrate.py` re-derived it on gfx950
and the compiler's `Required` / `Hardware limit` ratio came back 1.00, same as
gfx942). So the formula ported and the constant did not, which is the worst
combination: nothing to notice. Carrying 65536 onto gfx950 rejects **28% of
the tiles that actually fit**, silently.

### 2. Roofline ceilings

Peak TFLOP/s and peak HBM GB/s are used as implausibility gates: a measured
result above peak means the measurement is wrong, not that the kernel is fast.
That only works if the ceiling is right for the part.

Two failure directions, and they are not symmetric:

- **Ceiling too low** → correct results get flagged impossible. Worse: if you
  set the ceiling from a *measured* `torch.matmul` figure, you have set it
  below what a tuned kernel achieves — so every successful tune trips the gate.
  Take ceilings from the vendor spec.
- **Ceiling too high** → some measurement bugs go uncaught. Tolerable.

So: spec peak for the gate, measured figure for the *achievable* baseline you
judge headroom against. Keep them as separate constants; they answer different
questions.

Mark un-measured constants as un-measured, in the source. A placeholder that
looks like a measurement will be quoted as one.

### 3. Dtype dialects and support

Same bit width does not mean same format. FP8 on CDNA3 is FNUZ
(`float8_e4m3fnuz`); on CDNA4 it is OCP (`float8_e4m3fn`). Different exponent
bias. Feed one to a kernel built for the other and you get plausible wrong
numbers — not an exception.

Resolve the dialect from the live device at runtime, never by naming a dtype
in source.

The support really does invert, measured both directions: gfx942 computes FNUZ
and refuses OCP, gfx950 computes OCP and refuses FNUZ. What does *not* invert
is the error text, and that asymmetry costs more than the symmetry saves:

| | gfx942 given OCP | gfx950 given FNUZ |
| --- | --- | --- |
| error | `HIPBLAS_STATUS_NOT_SUPPORTED` | `could not find valid hipblaslt solution` |

The second string is what hipBLASLt also emits for an unsupported *shape*. So
code that learned to recognise a dialect mismatch by matching the gfx942 text
reads the gfx950 failure as "this shape has no solution, try another one" and
tunes around a numerics bug instead of reporting it.

**Allocation is not support.** Check separately:

```python
a = torch.zeros((64, 64), dtype=dt, device=dev)   # may well succeed
torch._scaled_mm(a, a.t().contiguous().t(), ...)  # and this may not
```

Measured on CDNA3: OCP fp8 allocates without complaint and then fails at
`_scaled_mm` with `HIPBLAS_STATUS_NOT_SUPPORTED`. A support probe that only
allocates concludes the dtype works.

**And non-allocation is not non-support**, which is the trap in the other
direction and the more expensive one. On gfx950:

```python
torch.zeros(64, 64, dtype=torch.float4_e2m1fn_x2, device="cuda")
# RuntimeError: "fill_cuda" not implemented for 'Float4_e2m1fn_x2'
```

That is a missing torch *elementwise fill*, not a missing FP4 matrix core.
`torch.empty` succeeds, a uint8 buffer `.view()`s to it, and `_scaled_mm` with
e8m0 block scales runs at full rate. An allocation probe written the obvious
way reports FP4 unsupported on the only architecture that has an FP4 matrix
core — and FP4 is where roughly 80 of aiter's gfx950 entry points live. Probe
with `empty`, never with `zeros`.

The other direction exists too: a dtype the hardware genuinely lacks may be
*emulated* in a wider type. It runs, it is correct, and the throughput number
is measuring something other than what the label says. Record known-unsupported
pairs explicitly and refuse them with a reason.

### 4. Occupancy-derived heuristics

Anything reasoned from CU count, register file size, or waves-per-EU is
part-specific. A split-K factor balanced for 304 CUs is unbalanced on 256. Read
these from the device.

Read them, and then check what a *failed* read returns. A CU helper that falls
back to `304` when it cannot reach the GPU is worse than one that raises: on
gfx950 it builds the entire tuning space around an anchor that is 19% too
large, and nothing downstream can tell that from a real reading. Same for a
`_current_arch()` that defaults to `"gfx942"`. If the device cannot be read,
the answer is an exception, not a plausible number.

### 5. The dispatch floor

Below some duration a measurement is dispatch overhead, not kernel. Harnesses
encode that as a constant and use it to label a case overhead-bound, meaning
"there is no work here; tile tuning will return noise."

Measured with `tools/launch_floor.py`:

| | bare dispatch | minimal GEMM | floor used |
| --- | --- | --- | --- |
| gfx942 / ROCm 7.2 | 17.9 us | 41.7 us | 0.042 ms |
| gfx950 / ROCm 7.2 | 6.2 us | 17.0 us | 0.017 ms |

This is the one constant in the list where the **newer part having a better
number is what causes the bug**. Every other stale constant makes the new part
look artificially constrained in a direction you might notice. This one puts
the "too small to bother" threshold at 0.052 ms on a machine whose real floor
is 0.021, so kernels running at 0.031 and 0.046 ms — both of which tune fine,
one of them by 11% — get labelled overhead-bound and dropped, with a specific
and confident explanation attached. Nobody re-examines a case the tool has
already explained.

Re-measure per box, not merely per arch. It is a property of the driver and
runtime at least as much as of the silicon.

## Fails loudly — cheap to fix

- Tuned config files: wrong lookup key, so the library falls back. Loud only if
  you check engagement, which is why engagement verification is step 6 of the
  core loop and not optional.
- Kernels using arch-specific intrinsics or MFMA shapes: compile error.

## Order of operations

1. Probe the new device: dtype matrix (alloc *and* GEMM, separately, and with
   `empty` rather than `zeros`), LDS budget, CU count, wave size, dispatch
   floor, measured compute, and HBM swept across buffer sizes.
2. Fix ceilings, prune rules, and the dispatch floor from that probe.
3. Re-baseline every case before tuning anything. If baselines look wrong, the
   constants are still wrong — tuning on top of that just buries the error.
4. Delete old tuned artifacts. Do not port them.
5. Re-sweep, verify engagement, re-measure.

Step 3 is the one people skip. A baseline that is quietly wrong makes every
subsequent speedup ratio wrong in the same direction, and the ratios still look
reasonable.

## Worked example: gfx942 → gfx950

Every number below was measured on the two parts with the tools named, not
taken from a datasheet or inferred from one part. Where a spec figure appears
it is labelled as such and used only as an implausibility ceiling.

| | gfx942 (MI300X) | gfx950 (MI355X) | tool |
| --- | --- | --- | --- |
| CUs | 304 | 256 | `torch` properties |
| LDS / block | 65 536 B | 163 840 B | `lds_calibrate.py` |
| LDS prune ratio | 1.00 | 1.00 | `lds_calibrate.py` |
| bare dispatch | 17.9 us | 6.2 us | `launch_floor.py` |
| minimal GEMM | 41.7 us | 17.0 us | `launch_floor.py` |
| HBM, triad plateau | 4 200 GB/s | 6 000 GB/s | `hbm_probe.py` |
| HBM, datasheet | 5 300 GB/s | 8 000 GB/s | spec (ceiling only) |
| bf16 via torch | — | 1 441.6 TFLOP/s | `arch_calibrate.py` |
| bf16 dense peak | 1 307 | 2 516.6 TFLOP/s | spec (ceiling only) |
| fp8 via torch | — | 2 891 TFLOP/s | `arch_calibrate.py` |
| fp8 dialect that computes | FNUZ | OCP | `arch_calibrate.py` |
| fp4 matrix core | absent | present | `probe_fp4_ops.py` |
| Gluon dialect | `cdna3` | `cdna4` | `validate/claims.py` |

Three of these are worth more than the rest of the table.

**Measure bandwidth across sizes, not at one size.** gfx950 has 256 MB of
Infinity Cache. A copy benchmark at 128 MB reports 6 897 GB/s; at 16 GB it
reports 5 227. "512 MB, surely that's plenty" is still half a cache benchmark.
Sweeping to 16 GB is what makes the plateau visible. And one access pattern is
not enough either: the same tool times a `copy_` at ~4 570 GB/s and a triad at
~6 000. Taking the copy figure would have set gfx950's achievable bandwidth
*below* what its kernels routinely hit, so every memory-bound case would have
reported over 100% of achievable and the gate would have been meaningless.

**Torch reaching 57% of spec is not a broken measurement.** The bf16 gap
(1 441 measured against 2 516 spec) looks alarming until you note gfx942 sits
at the same ~51%: it is torch leaving the matrix core partly idle, on both
parts. fp32 and fp64 come in at 97% and 91% of spec because there is far less
scheduling headroom to lose there. Keep both columns. The spec figure is the
implausibility gate; the measured figure is what you judge headroom against.
Conflating them is how a harness ends up flagging every successful tune as
impossible.

**Some things that look like missing tuning are structural.** aiter ships one
`"any"` entry for the Gluon MXFP4 GEMM on gfx950, against M-binned entries in
forty-odd N/K-specialised files for the Triton path to the same math. That
reads as an untuned kernel. It is not: the Gluon kernel writes its register
layouts as literal basis vectors, which pins the tile to 256x256x256 on 8
warps, and any other value is `LLVM ERROR` + `abort()` rather than a slow
config. Two more knobs are accepted and never read. The real surface is four
knobs wide and one entry covers it. Before concluding a config file is thin,
find out how many degrees of freedom the kernel actually has — and probe that
in a subprocess, because the failure mode is a process abort that will take an
in-process sweep down with it.

One thing this table cannot capture: none of it is stable across *images* on
the same part. Two containers on this same MI355X, both reporting Triton
3.6.0, compile the same Gluon kernel 1.89x apart because one lands two
registers over the VGPR cap. Re-deriving constants fixes the harness; it does
not make results comparable across images. See `../env-setup/SKILL.md` §4.

## The pattern worth internalizing

Harness bugs are not randomly signed. A prune that is too tight, a ceiling that
is too high, a reference that is too permissive, a candidate list that collapsed
to one config — these all push the reported result **up**. Bugs that make a
tuner look worse get found immediately, because someone investigates a
disappointing number. Bugs that make it look better get shipped.

On new hardware, where you have no priors about what is normal, that asymmetry
is at its worst. Be more suspicious of a good result than a bad one until the
constants are re-derived.
