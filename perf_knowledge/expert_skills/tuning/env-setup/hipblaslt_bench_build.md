# Building `hipblaslt-bench` when your container doesn't ship it

`hipblaslt-bench` is the solution-racing tool for hipBLASLt. It is **not available as a
package** — `hipblaslt-dev` gives you headers, and there is no `hipblaslt-clients` package
in the ROCm apt repo. If you need it, you build it.

The good news: you do **not** need to rebuild the hipBLASLt library or generate Tensile
kernels. You build the *clients* only, linking against the `libhipblaslt.so` already in
the container. On a 32-core build that is minutes, not hours.

## Verified recipe

Validated 2026-07-27 inside `primussafe/vllm-openai-rocm:v0.21.0-rocm720-profilerfix`
(ROCm 7.2.2) on gfx942, producing a working binary that raced 1231 solutions.

### 1. Match the source branch to the container's ROCm version

This is the step that is easy to get wrong. Building the library's `develop` HEAD against
an older runtime invites ABI and API skew. Check the runtime first, then check out the
matching release branch.

```bash
cat /opt/rocm/.info/version          # e.g. 7.2.2  -> use release/rocm-rel-7.2
```

hipBLASLt lives in the `rocm-libraries` monorepo. A worktree keeps the branch separate
from whatever else you have checked out:

```bash
cd <rocm-libraries checkout>
git worktree add ../hipblaslt-rel72 release/rocm-rel-7.2
```

Note this is a large checkout (~5 GB) and can take several minutes to materialize.

### 2. Install build dependencies

The stock inference images carry none of these. Each missing one fails at a *different*
stage, so install them all up front:

```bash
apt-get update
apt-get install -y cmake python3-pip git \
    libgtest-dev libgmock-dev \
    libboost-filesystem-dev libboost-program-options-dev libboost-system-dev
pip install invoke
```

Why each is needed — all three failures come from a **tests** subdirectory that gets
configured even when you only want the bench binary:

| Missing | Symptom |
| --- | --- |
| `invoke` | `install.sh` is a thin wrapper over `invoke build`; without it nothing runs |
| `libgtest-dev` | configure: `Could NOT find GTest (missing: GTEST_LIBRARY ...)` |
| boost filesystem/program-options | configure: `Could not find ... "boost_filesystem"` |
| `libgmock-dev` | compile at ~98%: `fatal error: 'gmock/gmock.h' file not found` — Ubuntu's `libgtest-dev` does **not** include gmock headers |

### 3. Build clients only

```bash
cd <worktree>/projects/hipblaslt
./install.sh -c -n -a gfx942 -j 32
```

- `-c` / `--clients` — build the client apps (this is what produces `hipblaslt-bench`)
- `-n` / `--client-only` — **build without Tensile**; this is the flag that saves hours
- `-a gfx942` — build for your arch only. Use `gfx950` for MI355. Building `all` is a
  large waste of time.
- `-j 32` — caps both cmake jobs and Tensile kernel-generation threads

If a later target fails after the bench binary is already linked, you can resume in place
rather than re-running configure:

```bash
cd build/release && make -j 32
```

The binary lands at:

```
<worktree>/projects/hipblaslt/build/release/clients/hipblaslt-bench
```

### 4. Smoke-test it

```bash
export HIP_VISIBLE_DEVICES=4        # pin an idle GPU
BENCH=<worktree>/projects/hipblaslt/build/release/clients/hipblaslt-bench

$BENCH -m 4096 -n 4096 -k 4096 -r bf16_r --compute_type f32_r \
       --transA N --transB T --algo_method all -j 5 -i 20
```

`--algo_method all` races every applicable solution and prints a `Winner:` block. Expect
~1200 candidates on gfx942 and ~2100 on gfx950 for a common bf16 shape, so give it a generous
timeout.

The `Winner:` block's leading `[N]` is **not** the solution index — it is the enumeration
position in this run's output, and it drifts between identical runs. Add
`--print_kernel_info` and read the `--Solution index:` line, which is the value you carry
forward to deploy. On gfx950 the winner printed as `[1910]` with solution index `441281`.
Full explanation and the replay procedure: `../tuning-hipblaslt/SKILL.md` §3.

## Notes

- The build emits many CMake developer warnings (CMP0077, CMP0167, yaml-cpp). They are
  noise; only `CMake Error` and `error:` lines matter.
- Built binaries are architecture-specific. A `-a gfx942` build is for this box only. The
  `-a gfx950` build was validated on MI355X (job 8545) from the `release/rocm-rel-7.2` branch
  of `rocm-libraries`, took ~10 minutes at `-j 48`, and produced a 4.9 MB binary that races
  2085 solutions and reports `gfx950` / `ISA950` kernel names.
- Keep the worktree — rebuilds after a `git pull` are incremental and fast.
