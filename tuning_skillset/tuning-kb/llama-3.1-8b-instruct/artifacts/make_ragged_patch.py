#!/usr/bin/env python3
"""Generate analysis/base/aiter_backend.py.ragged from the pristine .orig.

The candidate is produced by a scripted, idempotent text insertion rather than by hand-editing the
live file, so the patch always has an exact base to diff against and can be regenerated after any
change to the recorded pristine copy. See FINDINGS.md for how .orig was obtained.
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ORIG = HERE / "base" / "aiter_backend.py.orig"
CAND = HERE / "base" / "aiter_backend.py.ragged"

ANCHOR = """                return o.view(-1, layer.tp_q_head_num * layer.head_dim)

            if self.kv_cache_is_vectorized_5d:
                return forward_extend_vectorized_5d(
"""

INSERT = '''                return o.view(-1, layer.tp_q_head_num * layer.head_dim)

            if (
                is_gfx95_supported()
                and forward_batch.forward_mode.is_extend()
                and window_size == (-1, -1)
                and sinks is None
                and self.logits_soft_cap == 0.0
                and layer.qk_head_dim == 128
                and layer.v_head_dim == 128
                and self.kv_cache_dtype != fp8_dtype
                and not self.kv_cache_is_vectorized_5d
                and q.dtype == torch.bfloat16
                and self.forward_metadata.kv_indices is not None
                and self.forward_metadata.max_kv_len is not None
            ):
                # bf16 head_dim-128 twin of the fp8/head_dim-256 branch above, generalised to
                # batches that carry a prefix.
                #
                # The paged kernel (mha_batch_prefill_func, ck_tile) resolves the page table
                # *inside* the softmax loop; the v3 ASM kernel behind flash_attn_varlen_func
                # (hsa/gfx950/fmha_v3_fwd/fwd_hd128_bf16_causal_group.co) has no paged variant at
                # page_size 1 -- it rejects any block table whose page size is not a multiple of
                # 128 -- so feeding it means materialising the KV span contiguously first.
                # Two index_selects move ~100 MB per layer at this shape and the ASM kernel wins
                # far more than that back: raced at the shape the benchmark actually produces
                # (3 seqs, q 2/8193/8189, kv 8193/8193/8189) the gather+ASM pair is 37.22% faster
                # than the paged call *including* the gather, max_rel_err 6.4e-04.
                #
                # Gating on the prefix being empty instead -- the obvious cheaper thing, and what
                # the fp8 branch above does -- is worthless here: the served prompts are 8192
                # tokens, the tokenizer prepends BOS, and 2*8193 exceeds the 16384-token
                # chunked-prefill budget, so from the second batch onward every prefill batch
                # contains one chunk continuation and a no-prefix gate never fires.
                #
                # causal=True with cu_seqlens_k != cu_seqlens_q is bottom-right aligned, which is
                # the correct mask for a chunk continuation.
                k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
                idx = self.forward_metadata.kv_indices
                kb = k_cache.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
                vb = v_cache.view(-1, layer.tp_v_head_num, layer.v_head_dim)
                kg, vg = self._ragged_kv_scratch(
                    idx.shape[0], layer, kb.dtype, kb.device
                )
                torch.index_select(kb, 0, idx, out=kg)
                torch.index_select(vb, 0, idx, out=vg)

                ragged_kwargs = {}
                attn_out = getattr(forward_batch, "_attn_output", None)
                if attn_out is not None:
                    ragged_kwargs["out"] = attn_out.view(
                        -1, layer.tp_q_head_num, layer.v_head_dim
                    )
                o = flash_attn_varlen_func(
                    q.contiguous().view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    kg,
                    vg,
                    self.qo_indptr[:bs0],
                    self.forward_metadata.kv_indptr[:bs0],
                    self.forward_metadata.max_q_len,
                    self.forward_metadata.max_kv_len,
                    softmax_scale=layer.scaling,
                    causal=True,
                    **ragged_kwargs,
                )
                return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

            if self.kv_cache_is_vectorized_5d:
                return forward_extend_vectorized_5d(
'''

# The gathered KV is re-materialised for all 32 layers of every prefill batch, so it gets a
# persistent scratch pair in the same style as _get_kv_indices_scratch rather than a fresh
# allocation per layer. Prefill is not HIP-graph captured, so growing it lazily is safe.
SCRATCH_ANCHOR = """        return self._kv_indices_scratch[:required_tokens]
"""

SCRATCH_INSERT = '''        return self._kv_indices_scratch[:required_tokens]

    def _ragged_kv_scratch(self, n_tokens, layer, dtype, device):
        """Contiguous K/V staging buffers for the gather+ASM prefill path."""
        need = n_tokens * layer.tp_k_head_num * layer.qk_head_dim
        cur = getattr(self, "_ragged_kv_buf", None)
        if (
            cur is None
            or cur[0].device != device
            or cur[0].dtype != dtype
            or cur[0].numel() < need
        ):
            self._ragged_kv_buf = (
                torch.empty(need, dtype=dtype, device=device),
                torch.empty(need, dtype=dtype, device=device),
            )
        kb, vb = self._ragged_kv_buf
        shape = (n_tokens, layer.tp_k_head_num, layer.qk_head_dim)
        return kb[:need].view(shape), vb[:need].view(shape)
'''


def main() -> int:
    src = ORIG.read_text()
    for anchor in (ANCHOR, SCRATCH_ANCHOR):
        if src.count(anchor) != 1:
            print(f"anchor not unique ({src.count(anchor)}x):\n{anchor}", file=sys.stderr)
            return 2
    out = src.replace(ANCHOR, INSERT).replace(SCRATCH_ANCHOR, SCRATCH_INSERT)
    CAND.write_text(out)
    added = len(out.splitlines()) - len(src.splitlines())
    print(f"wrote {CAND}  (+{added} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
