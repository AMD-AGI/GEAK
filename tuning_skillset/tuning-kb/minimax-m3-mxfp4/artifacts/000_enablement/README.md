# 000_enablement — required to boot, not an optimization

This patch is part of the frozen baseline. **The configuration in `scripts/launch_server.sh` does not
start without it.** It is numbered `000` and kept separate from the numbered directories you will
create for your own work, which start at `001`.

```bash
./apply.sh            # idempotent
./apply.sh --check    # exit 0 if applied
./apply.sh --revert   # restore pristine files
```

`scripts/preflight.sh` and `scripts/launch_server.sh` both refuse to proceed without it.

## What fails without it

vLLM's v1 engine defaults to `cudagraph_mode=FULL_AND_PIECEWISE`, so after the PIECEWISE
mixed-prefill-decode graphs it captures a second set of **decode-only, FULL** graphs. To capture those
it hands the attention metadata builders a *fully padded dummy batch*: `query_start_loc_cpu` is all
zeros, `num_reqs == num_actual_tokens`, `max_query_len == 1`.

MiniMax-M3's two builders read the leading query length out of that array and assert it is positive:

```python
qsl_cpu = common_attn_metadata.query_start_loc_cpu
query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
decode_query_len = int(query_lens_cpu[0].item())
assert decode_query_len > 0
```

On a padded capture batch that leading entry is 0, so the assertion fires — at
`models/minimax_m3/common/indexer.py:317`, and then immediately again at the identical block in
`models/minimax_m3/common/sparse_attention.py:288`. The engine never reaches the serving loop. What
you see is:

```
(Worker_TP0 pid=...) ERROR [multiproc_executor.py:1007] AssertionError
(EngineCore pid=...) ERROR [core.py:1330] EngineCore failed to start.
```

which is bare enough that it is easy to misread as an OOM or a bad flag.

## What it changes

Both files get the same guard. When the leading decode query length is non-positive, the builder falls
back to the *uniform* decode query length it was configured with (`reorder_batch_threshold`, i.e. 1, or
1 + `num_speculative_tokens`) instead of asserting, and the
`num_decode_tokens == num_decodes * decode_query_len` invariant is relaxed for that padded case only.
The uniformity assertion is untouched, and on a real batch every code path is unchanged — the fallback
is reachable only from a batch that would previously have crashed.

`patch -p1` applied from the directory containing the `vllm` package. Diff:
`minimax_m3_uniform_decode_capture_guard.patch`.

## Why it is in the baseline rather than worked around

The alternative is `-cc.cudagraph_mode=PIECEWISE`, which skips the FULL decode capture and boots
unpatched. That was deliberately not done. Pinning PIECEWISE is a downgrade from the engine default,
it changes what the decode path actually executes, and folding it into the baseline would move the
floor that every later measurement is compared against. The reference run took the source fix and left
the cudagraph mode at its default, and `launch_server.sh` asserts `FULL_AND_PIECEWISE` on every boot
for exactly that reason.

Practical consequence for you: if a change of yours makes the server fall back to PIECEWISE, or breaks
FULL decode capture, the server will still come up and still serve correct output — at a different
throughput, for a reason that has nothing to do with your change. `launch_server.sh` checks that the
`Capturing CUDA graphs (decode, FULL)` progress bar reached 100% on the boot you are about to measure.

## Scope note

The same six-line block exists elsewhere in this vLLM tree (the MSA indexer and the FP8 sparse MLA
decode path). Those are not on this stack's code path, so they are left alone.
