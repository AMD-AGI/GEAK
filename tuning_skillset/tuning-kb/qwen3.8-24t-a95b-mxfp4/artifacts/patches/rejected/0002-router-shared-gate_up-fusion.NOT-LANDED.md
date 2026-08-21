# 0002 — fuse the router `gate` and the shared-expert `gate_up` into one N=1024 GEMM

**Status: never landed. Priced first, and the price killed it.** There is no diff to review; this file
exists so the measurement is not lost.

Base: sglang `29481685462732237d80d86076d6563e1f658102` / aiter
`d9e5ef7ce08ee7045d583aed768cff41aa9210fe`.
Measured by: `/work/analysis/check_fuse.py` (re-runnable, needs one GPU, no server).

## What it would have been

`Qwen2MoeSparseMoeBlock.forward` issues two column-parallel bf16 GEMMs against the *same* post-attention
hidden states:

- `self.gate` — (M, 512, 8192), replicated, produces the router logits
- `self.shared_expert.gate_up_proj` — (M, 512, 8192) at TP=8

184 launches per forward, 9.5% of decode. Concatenating the two weight matrices into one (1024, 8192)
buffer at load time and re-pointing both `.weight.data` at row-slices of it is mathematically exact,
memory-neutral, and replaces two launches with one. At M=64 these GEMMs are launch-bound, so one
N=1024 GEMM should cost barely more than one N=512 GEMM — and it does.

## Why it does not work

Both consumers need contiguous input: `topk_softmax` on the router logits and `silu_and_mul` on the
gate_up half. The fused output's two column-slices are not contiguous, so each needs a `.contiguous()`.

```
     M  2x torch N512  2x tgemm N512  1x tgemm N1024  +2 contig  vs torch  vs tuned
    64          19.48          19.20           10.65      18.82    1.035x    1.020x   err=0.0e+00
  8192         153.95         130.23          117.11     128.78    1.195x    1.011x   err=3.1e-02
 16384         258.76         215.81          177.11     197.32    1.311x    1.094x   err=3.1e-02
```

At the decode shape the fused GEMM saves 8.8 µs and the two copies give 8.2 µs of it straight back.
Both copies move 64 KB; that time is pure launch overhead, not bandwidth, so it does not shrink with a
better copy kernel. A row-major GEMM cannot write two separately-contiguous outputs, and splitting the
output back into two buffers is the two GEMMs again.

## Numbers

**1.035× on 9.5% of decode = 0.3% end to end, against a measured noise floor of 1.03%.** Not benched
end to end and not gated — there was nothing to measure. The 1.195×/1.311× prefill columns are real but
irrelevant: patch 0001 established that at ISL 8192 / OSL 1024 a prefill-side gain of 3% moves
`output_throughput` by 0.08%, so 20–30% of 9.5% of prefill is far below anything measurable.

Recorded because the two-GEMMs-on-one-input pattern looks like free money in a profile, and it is not.
