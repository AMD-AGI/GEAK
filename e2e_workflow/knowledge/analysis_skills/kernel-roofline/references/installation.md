# Install and verify TraceLens

Use this procedure at the start of a new TL analysis to resolve current upstream
`main` and verify the analysis environment. Upstream source and instructions:

- Repository: <https://github.com/AMD-AGI/TraceLens>
- README: <https://github.com/AMD-AGI/TraceLens/blob/main/README.md>

The skill ZIP does not vendor TraceLens. Install it in the environment that will
run TL analysis. Offline trace analysis can use a separate environment from the
GPU workload; GPU trace capture and calibration require the appropriate framework,
GPU runtime, and device access as well.

## 1. Check the selected interpreter

In the intended environment, run:

```bash
python3 -c 'import sys, TraceLens; print(sys.executable); print(TraceLens.__file__)'
python3 -m pip show TraceLens
```

If import succeeds, reuse it only after verifying it matches the main commit
resolved for this analysis. Package version alone is insufficient; record source
commit and import path. If import fails, inspect the exception: a
missing transitive dependency differs from an absent TraceLens package. Use
`python3 -m pip` with the same interpreter rather than an unrelated `pip` command.

Use Python 3.10 or newer for this workflow and check the selected TL revision's
dependency requirements. The inspected package's old minimum-Python metadata is
not a guarantee that every current dependency supports that old interpreter.

## 2. Create an analysis environment when needed

For a new offline analysis environment, run this from the task workspace using
an unused environment path:

```bash
python3 -m venv .venv-tracelens
source .venv-tracelens/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
```

If the task already supplies a suitable virtual environment or container, use
its Python instead. Install inside that container when the analysis runs there;
installing on the host does not install the package in the container. If `venv`
is unavailable, use an existing managed Python environment or install your OS's
matching Python venv package according to its environment setup instructions.

## 3. Resolve latest main, then freeze the commit for this analysis

Prerequisites are Git and network access to GitHub and the package index. For new
analysis, fetch upstream `main`, record its full commit and UTC fetch time, and
keep that commit fixed across all cases and both implementations. Do not upgrade
between baseline and candidate. A user-specified historical reproduction may pin
an older commit; identify it as historical rather than current support evidence.

```bash
TL_SOURCE_DIR="$PWD/.deps/TraceLens"

mkdir -p .deps
git clone --branch main --single-branch https://github.com/AMD-AGI/TraceLens.git "$TL_SOURCE_DIR"
git -C "$TL_SOURCE_DIR" fetch origin main
TL_COMMIT=$(git -C "$TL_SOURCE_DIR" rev-parse refs/remotes/origin/main)
date -u +%Y-%m-%dT%H:%M:%SZ
git -C "$TL_SOURCE_DIR" checkout --detach "$TL_COMMIT"
python3 -m pip install "$TL_SOURCE_DIR"

git -C "$TL_SOURCE_DIR" rev-parse HEAD
git -C "$TL_SOURCE_DIR" status --short
python3 -m pip show TraceLens
python3 -m pip check
```

Choose a fresh source directory if `.deps/TraceLens` already exists, or inspect
and reuse its checkout. Do not reset someone else's working tree. The ordinary
package install resolves TL's declared dependencies; roofline reporting does not
need the optional development or comparative-agent extras. `pip check` should be
clean before relying on the analysis environment.

For installation without a retained source checkout, resolve main first and
install that exact commit:

```bash
TL_COMMIT=$(git ls-remote https://github.com/AMD-AGI/TraceLens.git refs/heads/main | awk '{print $1}')
test -n "$TL_COMMIT" || exit 1
python3 -m pip install "git+https://github.com/AMD-AGI/TraceLens.git@$TL_COMMIT"
```

These are alternative installation methods, not two required consecutive installs.
Use the source checkout method when reviewing model code or tracking local changes.
In `tl-only`, keep model code unmodified; any local model change must be documented.
Keep the install command/log with the commit, and verify that `TraceLens.__file__`
does not resolve to a different checkout through `PYTHONPATH` or the working
directory. If main cannot be fetched, record that limitation; an available older
report can still be explained, but do not call it latest or infer current upstream
support from it. Never reset or overwrite someone else's modified checkout.

## 4. Verify model imports and the native report entry point

Run these checks outside the source checkout to verify the installed package:

```bash
python3 - <<'PY'
import sys
import importlib.metadata
import TraceLens
from TraceLens.PerfModel.perf_model import GEMM

print("Python:", sys.executable)
print("TraceLens version:", importlib.metadata.version("TraceLens"))
print("TraceLens import:", TraceLens.__file__)
flops = GEMM.flops_func(128, 4096, 4096, False)
nbytes = GEMM.bytes_func(128, 4096, 4096, False, 2, 2, 2, 2)
assert flops == 4294967296
assert nbytes == 35651584
print("TraceLens GEMM model import and arithmetic: OK")
PY

python3 -m TraceLens.Reporting.generate_perf_report_pytorch --help
```

The model check is CPU-only and does not benchmark a kernel. Confirm that the
native CLI supports the trace, architecture, and output flags used in
[tl-only.md](tl-only.md); inspect the chosen version's help if they differ.

If the import path points at an unexpected checkout, inspect the working directory
and `PYTHONPATH`; remove unintended import precedence for this analysis before
reporting provenance. If Python 3.10 reports missing `strenum`/`StrEnum`, or TL's
package import reports another missing dependency, confirm that installation ran
with this interpreter and completed dependency resolution. The inspected revision
declares these dependencies; prefer repairing the installation over bypassing them
with import stubs or `--no-deps`.

## 5. Select the hardware reference and continue

A successful installation does not guarantee a matching GPU architecture file.
Inspect the selected version's `TraceLens/Agent/Analysis/utils/arch/` resources
or use the task's existing TL-compatible architecture file with recorded provenance.
Keep that file fixed across baseline/candidate runs. Do not use this skill's
`assets/example_arch.json` for actual GPU results: it contains synthetic values.

If calibration is needed, inspect the installed TL tool before running it on the
target GPU:

```bash
python3 -m TraceLens.PerfModel.benchmarking.microbench --help
```

Follow the installed tool's device/output options and preserve the calibration
artifact. Installing TraceLens does not install or validate a complete ROCm/CUDA
benchmark environment. Return to the chosen mode after import/CLI verification;
report any unresolved setup issue without switching `tl-only` to a substitute model.

## 6. rocprof-compute (for `mode=rocprof`)

TraceLens is not needed for the rocprof leg alone, and rocprof-compute is not needed
for `tl-only`.

For rocprof-compute 3.4.0, use `pandas<3` in its analysis environment. Hardware
validation found that pandas 3's string dtype bypasses this profiler's `object`
check for `Agent_Id`, causing CSV conversion to fail with a string/integer merge
error even after counter collection succeeds. Install the tool's requirements
plus this version constraint in an isolated environment, and record the resolved
versions. This constraint is specific to the affected profiler, not TraceLens.

When running aiter UTs as a non-root container user, set `AITER_JIT_DIR` to an
owned writable cache directory. Its fallback can attempt to copy the image's
entire JIT tree, including unreadable prebuilt caches. Keep a valid passwd entry
for the container UID; compilation tools can use `getpwuid` to find the user.

```bash
rocprof-compute --version        # record this next to every number
rocm-smi --showhw | head         # record the arch (gfx942 / gfx950 / ...)
```

**Calibration on gfx95x needs independent verification.** Old microbenchmarks
can issue CDNA3 MFMA on CDNA4 and understate peaks. A compatible older profiler
may use a verified frozen CSV; the current driver requires an existing file and
checks scaled-sibling calibration consistency without replacing peaks. Record
both collector version and peak provenance; see [rocprof.md](rocprof.md).

Upstream development moved into `ROCm/rocm-systems` under
`projects/rocprofiler-compute`; the standalone `rocprofiler-compute` repository is
frozen at 3.4.0. A checkout of the new location is what
`scripts/new_mibench.py --bench-src <checkout>/src` wants. Note that `src/vendored/`
is empty in a plain checkout, so a small `pyyaml` shim on `sys.path` is needed
before the microbench module imports.

If the container's rocprof-compute is 3.4.0 and cannot be replaced, backport the
gfx950 HBM read-byte fix instead:

```bash
python3 scripts/rocprof_hbm_patch.py          # writes rocprof_compute_hbmfix/ + rocprof_hbmfix_bin/
export PATH=$PWD/rocprof_hbmfix_bin:$PATH
```

That patches the read-byte numerator only. It does **not** fix the compute peaks --
those still need either 3.6.0 or a frozen `roofline.csv` generated from a 3.6.0
checkout via `new_mibench.py`, applied offline with `scripts/repeak.py`.

Freeze the denominator before profiling anything you intend to compare:

```bash
export ROOFLINE_MIBENCH_CACHE=/path/to/frozen/roofline.csv
```

Verify it took effect: the profile log contains no microbenchmark run, and the peak
columns are bit-identical across cases.

A working `rocprof-compute` still does not guarantee a working profiling setup --
counter collection needs the right permissions and an uncontended GPU. Confirm with
one small case end to end before launching a batch.
