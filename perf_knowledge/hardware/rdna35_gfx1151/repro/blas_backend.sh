#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Which BLAS library does torch.mm call on this part? Traces one fp16 GEMM with rocprofv3 and
# looks the dispatched kernel name up in the rocBLAS and hipBLASLt code-object libraries.
# preferred_blas_library() only reports a preference; the kernel name is what actually ran.
set -uo pipefail
command -v rocprofv3 >/dev/null 2>&1 || { echo "error: rocprofv3 not on PATH" >&2; exit 1; }
PY="${PYTHON:-python3}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
timeout 120 rocprofv3 --kernel-trace --stats --output-format csv -d "$OUT" -- "$PY" -c '
import torch
a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
for _ in range(5):
    a @ a
torch.cuda.synchronize()
print("preferred_blas_library:", torch.backends.cuda.preferred_blas_library())' 2>/dev/null | grep preferred
name="$("$PY" - "$OUT" <<'PY'
import csv, glob, sys
rows = [r for f in glob.glob(sys.argv[1] + "/**/*kernel_stats.csv", recursive=True) for r in csv.DictReader(open(f))]
rows.sort(key=lambda r: -float(r["TotalDurationNs"]))
print(rows[0]["Name"] if rows else "")
PY
)"
[ -n "$name" ] || { echo "error: no kernel was traced" >&2; exit 1; }
echo "hottest kernel: ${name:0:90}..."
ROCM="${ROCM_PATH:-/opt/rocm}"
for lib in rocblas hipblaslt; do
  hit="$(grep -l -a -F "$name" "$ROCM"/lib/"$lib"/library/* 2>/dev/null | head -1)"
  echo "  in $lib: ${hit:-no}"
done
