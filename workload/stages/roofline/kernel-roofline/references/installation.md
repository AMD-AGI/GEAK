# Install and verify TraceLens

Use this procedure when the selected analysis environment does not have a working
TraceLens installation. Upstream source and installation instructions:

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

If import succeeds, reuse the installation when its version supports the task;
record the version and import path. If import fails, inspect the exception: a
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

## 3. Download and install a pinned source revision

Prerequisites are Git and network access to GitHub and the package index. The
following pin is the historical version used by this skill's September comparison,
not a claim that it is the latest version. Use a user/project-specified revision
when provided and record the resolved full commit. A newer revision may be needed
for additional operator support; keep it fixed across the comparison.

```bash
TL_REF=a59a9c165bb64c7c416fd7cf79149803d552e43c
TL_SOURCE_DIR="$PWD/.deps/TraceLens"

mkdir -p .deps
git clone https://github.com/AMD-AGI/TraceLens.git "$TL_SOURCE_DIR"
git -C "$TL_SOURCE_DIR" checkout --detach "$TL_REF"
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

For installation without a retained source checkout, the upstream Git install
can also be pinned directly:

```bash
python3 -m pip install \
  'git+https://github.com/AMD-AGI/TraceLens.git@a59a9c165bb64c7c416fd7cf79149803d552e43c'
```

These are alternative installation methods, not two required consecutive installs.
Use the source checkout method when reviewing model code or tracking local changes.
In `tl-only`, keep model code unmodified; any local model change must be documented.

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

## 6. rocprof-compute (for `mode=rocprof` and `mode=combination`)

TraceLens is not needed for the rocprof leg alone, and rocprof-compute is not needed
for `tl-only`. `combination` needs both.

```bash
rocprof-compute --version        # record this next to every number
rocm-smi --showhw | head         # record the arch (gfx942 / gfx950 / ...)
```

**Version gate: on gfx95x, rocprof-compute must be >= 3.6.0.** Below that, the
microbenchmark issues CDNA3 MFMA on a CDNA4 part and the BF16/FP16/INT8 empirical
peaks read 2-4x low; see [rocprof.md](rocprof.md) section 1 for the measured table
and for what to do when you cannot upgrade (score on the datasheet column). Record
the version, do not assume it.

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
