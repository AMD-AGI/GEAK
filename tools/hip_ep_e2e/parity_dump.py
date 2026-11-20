#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Teacher-forced logits of an OGA model, for parity.py. Runs in the package's
Python (OGA's wheel is cp314-only), started by parity.py through the sysroot
loader with CWD at <pkg>/lib.

For each text, cut to its target length T:
  - prefill path: one append of all T tokens; saves argmax at every position and
    the full logits at `--sample` evenly spaced positions;
  - decode path: a fresh generator appends T - D tokens, then the last D tokens
    one at a time; saves argmax and logits of every step.
Also saves the token ids, so both sides are known to have read the same input,
and the kernel library this process actually mapped.
"""
import argparse
import json

import numpy as np
import onnxruntime_genai as og


def _generator(model, n_tokens):
    p = og.GeneratorParams(model)
    p.set_search_options(max_length=n_tokens + 1, do_sample=False)
    return og.Generator(model, p)


def _logits(gen):
    lg = np.asarray(gen.get_output("logits"))
    return lg.reshape(-1, lg.shape[-1])


def _mapped(substr):
    with open("/proc/self/maps") as f:
        paths = {line.split()[-1] for line in f if substr in line}
    return sorted(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--texts", required=True, help="json: [{\"file\": path, \"len\": T}, ...]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=64)
    ap.add_argument("--decode-steps", type=int, default=32)
    a = ap.parse_args()

    model = og.Model(a.model)
    tok = og.Tokenizer(model)
    with open(a.texts) as f:
        texts = json.load(f)
    arrays = {}
    for i, t in enumerate(texts):
        with open(t["file"]) as f:
            ids = np.asarray(tok.encode(f.read()), dtype=np.int32)
        T, D = t["len"], a.decode_steps
        if len(ids) < T:
            raise SystemExit(f"{t['file']} has {len(ids)} tokens, need {T}")
        ids = ids[:T]

        gen = _generator(model, T)
        gen.append_tokens(ids)
        lg = _logits(gen)
        if lg.shape[0] != T:
            raise SystemExit(f"model returned logits for {lg.shape[0]} of {T} positions")
        pos = np.linspace(0, T - 1, a.sample).round().astype(np.int64)
        arrays.update({f"ids_{i}": ids, f"argmax_{i}": lg.argmax(-1).astype(np.int32),
                       f"pos_{i}": pos, f"logits_{i}": lg[pos].astype(np.float16)})
        del gen, lg

        gen = _generator(model, T)
        gen.append_tokens(ids[:T - D])
        steps = []
        for tkn in ids[T - D:]:
            gen.append_tokens(np.asarray([tkn], dtype=np.int32))
            steps.append(_logits(gen)[-1])
        dl = np.stack(steps)
        arrays.update({f"dargmax_{i}": dl.argmax(-1).astype(np.int32),
                       f"dlogits_{i}": dl.astype(np.float16)})
        del gen
        print(f"[parity_dump] text {i}: {T} tokens, vocab {dl.shape[-1]}", flush=True)

    libs = _mapped("libcustom_kernels")
    arrays["kernel_lib"] = np.array(libs[0] if len(libs) == 1 else ";".join(libs))
    np.savez(a.out, **arrays)
    print(f"[parity_dump] kernel lib {arrays['kernel_lib']} -> {a.out}")


if __name__ == "__main__":
    main()
