#!/usr/bin/env python3
"""Server-less SGLang-Diffusion benchmark for the e2e_workflow `sglang_diffusion` adapter.

Driven by scripts/adapters/sglang_diffusion.sh; never imported. Reads its whole configuration
from the environment (the contract bench_e2e.sh's adapter layer uses) and appends ONE canonical
result line to $RESULT_JSONL, so bench_e2e.sh's backend-independent summarizer medians it exactly
like a text-LLM run. The canonical throughput key is `output_throughput`; for diffusion its unit
is **img/s**, not tok/s (bench_e2e.sh is metric-neutral and records `metric_basis`).

Why server-less: sglang-diffusion's HTTP server adds a request hop and no batching benefit for a
saturating image workload, and the in-process `DiffGenerator` (local_mode) is the configuration
whose baseline was already validated on this box. It also gives the torch profiler a clean
per-request window, which the diffusion HTTP server does not expose (no /start_profile).

Three passes, in order:
  1. warmup  (SGLD_WARMUP_CALLS)  untimed — absorbs aiter/Triton JIT, torch.compile, graph capture.
  2. timed   (SGLD_NUM_ITERATIONS) the throughput measurement. save_output=False so PNG encoding
     stays out of the number.
  3. quality (SGLD_QUALITY_NUM_PROMPTS) untimed, ONLY when a parity reference is configured — one
     image per fixed prompt at a fixed seed, either establishing the reference set (baseline) or
     written for scripts/image_parity.py to compare against it (candidate).

Structure and several hard-won details (the stall watchdog for dead SP workers, the sys.argv swap
around ServerArgs.get_provided_args(), per-GPU port derivation) follow the validated harness at
Hyperloom/src/hyperloom/inference_optimizer/assets/bench_scripts/sglang_diffusion_bench.py. GEAK
keeps its own copy so a run has no dependency on that tree.

Env contract
------------
  MODEL / MODEL_PATH   model dir                       TP        num-gpus for ONE engine
  GPU                  comma id list (ROCR pinning)    CONC      default timed image count
  ISL / OSL            -> image width / height         RESULT_JSONL  canonical result sink (append)
  OUT_DIR              full report + images land here  EXTRA_SGLANG_DIFFUSION_ARGS  free-form flags
  PROFILE=1            torch-profile the timed pass    PROFILE_DIR   where the trace lands
  SGLD_*               generation knobs (below)        GEAK_PARITY_REF / _REF_WRITE  parity set
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
# The fixed prompt set: used for the timed loop (cycled) and for the parity image set.
# This file and the prompt set live UNDER adapters/ on purpose: roles/director.md sets an eval dir
# up with `cp -r "$SKILL_DIR/scripts/adapters" "$EVAL_DIR/adapters"`, so anything the adapter needs
# at run time has to travel inside that tree or the copied bench_e2e.sh cannot find it.
_DEFAULT_PROMPTS_FILE = _HERE / "prompts_flux15.txt"


# ── env helpers ────────────────────────────────────────────────────────────
def _env(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default) or "").strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _load_prompts(path: Path) -> list[str]:
    """Return the ordered, de-duplicated prompt list."""
    prompts: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line in seen:
            continue
        seen.add(line)
        prompts.append(line)
    if not prompts:
        raise ValueError(f"no prompts in {path}")
    return prompts


def _ref_slot(template: str, index: int) -> Path:
    """A parity reference is a SET; the configured path is a template for slot N."""
    p = Path(template)
    return p.parent / f"{p.stem}_{index:02d}{p.suffix or '.png'}"


# ── engine ─────────────────────────────────────────────────────────────────
def _free_port(start: int, limit: int = 200) -> int:
    import socket

    for port in range(start, start + limit):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"no free port in [{start}, {start + limit})")


def _port_args() -> list[str]:
    """Per-run port flags so concurrent benches cannot collide.

    sglang-diffusion defaults --master-port to a fixed 30005 (and --port to 30000), so two
    benches on disjoint GPU leases would fight over the same torch.distributed rendezvous and
    take each other down (EOFError / connection refused). Derive the base from the first pinned
    GPU — concurrent leases hold disjoint GPU sets, so their ranges cannot overlap — then probe
    each port for real, which also covers unrelated processes already holding it.
    """
    devices = _env("ROCR_VISIBLE_DEVICES") or _env("GPU")
    try:
        first_gpu = int(str(devices).split(",")[0].strip())
    except (TypeError, ValueError):
        first_gpu = 0
    base = _env_int("SGLD_PORT_BASE", 31000) + (first_gpu * 200)
    return [
        "--master-port", str(_free_port(base)),
        "--scheduler-port", str(_free_port(base + 60)),
        "--port", str(_free_port(base + 120)),
    ]


def _build_server_args(model: str, num_gpus: int, extra_args: str):
    """Parse EXTRA_SGLANG_DIFFUSION_ARGS through sglang's OWN CLI parser.

    Reusing add_multimodal_gen_generate_args (the parser behind `sglang generate`) means every
    flag the installed sglang accepts is searchable by the Config Tuner without this script
    enumerating any of them.
    """
    import shlex

    from sglang.multimodal_gen.runtime.entrypoints.cli.generate import (
        add_multimodal_gen_generate_args,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs
    from sglang.multimodal_gen.utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(description="geak sglang-diffusion bench")
    add_multimodal_gen_generate_args(parser)

    # Ports before extra_args so a caller/variant flag still wins (argparse takes the last).
    argv = ["--model-path", model, "--num-gpus", str(num_gpus)]
    argv += _port_args()
    argv += shlex.split(extra_args)

    # ServerArgs.get_provided_args() distinguishes "user supplied this" from "argparse default"
    # by scanning sys.argv directly, so it must see the constructed argv rather than this
    # script's real command line (which has no flags — everything arrives via env). Without the
    # swap every value, model_path included, is treated as a default and dropped.
    saved_argv = sys.argv
    try:
        sys.argv = ["diffusion_bench", *argv]
        args, unknown = parser.parse_known_args(argv)
        args.request_id = "geak_bench"
        if unknown:
            print(f"[bench] passing through unrecognized args: {unknown}", flush=True)
        server_args = ServerArgs.from_cli_args(args, unknown)
        # Data parallelism cannot be set from the CLI in sglang 0.5.12: --dp-size /
        # --data-parallel-size / --dp all land in the argparse dest `data_parallel_size`, while the
        # dataclass field is `dp_size`, and from_dict() copies BY FIELD NAME — so the value is
        # silently dropped and you get dp_size=1 (verified). Set the field directly. The serve path
        # uses `--config {"dp_size": N}`, which reaches the field for the same reason.
        dp = _env_int("SGLD_DP_SIZE", 0)
        if dp > 0:
            server_args.dp_size = dp
            # sp_degree is derived from num_gpus/(dp*tp) only when left unspecified; if the caller
            # pinned one, respect it and let sglang's own validation catch a bad combination.
            if not getattr(args, "sp_degree", None):
                server_args.sp_degree = max(1, int(server_args.num_gpus) // dp)
                server_args.ulysses_degree = server_args.sp_degree
            print(f"[bench] dp_size={server_args.dp_size} sp_degree={server_args.sp_degree} "
                  f"(set directly — the CLI flag does not reach the field)", flush=True)
        return server_args, argv
    finally:
        sys.argv = saved_argv


class _StallWatchdog:
    """Fail fast when the engine stops making progress.

    sglang's sequence-parallel workers can die (seen with --enable-torch-compile at TP=2: both
    ranks go <defunct>) while the parent stays blocked in poll/kfd_wait forever. Without this the
    variant burns its entire timeout holding the serving GPUs and reports nothing.
    """

    def __init__(self, result_path: Path, report: dict[str, Any], timeout_s: int) -> None:
        self._result_path = result_path
        self._report = report
        self._timeout = timeout_s
        self._last = time.monotonic()
        self._stage = "startup"
        self._stop = threading.Event()

    def beat(self, stage: str) -> None:
        self._last = time.monotonic()
        self._stage = stage

    def start(self) -> None:
        if self._timeout > 0:
            threading.Thread(target=self._run, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(15.0):
            idle = time.monotonic() - self._last
            if idle < self._timeout:
                continue
            msg = (f"stalled: no progress for {idle:.0f}s during '{self._stage}' "
                   f"(limit {self._timeout}s); sglang workers likely died with the parent blocked.")
            print(f"[bench] {msg}", file=sys.stderr, flush=True)
            self._report.update({"success": False, "error": msg, "stalled": True})
            try:
                self._result_path.write_text(json.dumps(self._report, indent=2), encoding="utf-8")
            except OSError:
                pass
            # Hard-exit: the main thread is in a C-level wait no exception can interrupt.
            os._exit(1)


def _generate_once(engine, params: dict[str, Any]) -> None:
    from sglang.multimodal_gen.configs.sample.sampling_params import generate_request_id

    kwargs = dict(params)
    kwargs["request_id"] = generate_request_id()
    result = engine.generate(sampling_params_kwargs=kwargs)
    if result is None:
        raise RuntimeError(f"generation returned None for prompt={params.get('prompt')!r}")


# ── main ───────────────────────────────────────────────────────────────────
def main() -> int:
    out_dir = Path(_env("OUT_DIR") or ".").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{_env('RESULT_FILENAME', 'diffusion_bench')}.json"
    result_jsonl = _env("RESULT_JSONL")

    model = _env("MODEL") or _env("MODEL_PATH")
    tp = _env_int("TP", 1)
    conc = _env_int("CONC", 64)

    # Diffusion has no sequence lengths: the workflow's ISL/OSL ARE the image dimensions.
    width = _env_int("SGLD_WIDTH", _env_int("ISL", 1024))
    height = _env_int("SGLD_HEIGHT", _env_int("OSL", 1024))
    steps = _env_int("SGLD_NUM_STEPS", 28)
    guidance = _env_float("SGLD_GUIDANCE_SCALE", 3.5)
    seed = _env_int("SGLD_SEED", 42)
    warmups = _env_int("SGLD_WARMUP_CALLS", 3)
    timed_n = _env_int("SGLD_NUM_ITERATIONS", conc)
    profile = _env_bool("PROFILE", False)

    prompts_file = Path(_env("SGLD_PROMPTS_FILE") or _DEFAULT_PROMPTS_FILE)
    prompts = _load_prompts(prompts_file)

    ref_read = _env("GEAK_PARITY_REF")
    ref_write = _env("GEAK_PARITY_REF_WRITE")
    # The quality pass costs one image per prompt; run it ONLY when a parity reference is
    # actually configured (the Integrator sets one), so a plain throughput bench stays cheap.
    quality_n = 0
    if ref_read or ref_write:
        quality_n = min(_env_int("SGLD_QUALITY_NUM_PROMPTS", len(prompts)), len(prompts))

    report: dict[str, Any] = {
        "framework": "sglang_diffusion",
        "model": model,
        "throughput_unit": "img/s",
        "success": False,
        "config": {
            "tp_num_gpus": tp,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
            "warmup_calls": warmups,
            "timed_images": timed_n,
            "quality_prompts": quality_n,
            "profile": profile,
            "extra_args": _env("EXTRA_SGLANG_DIFFUSION_ARGS"),
            "rocr_visible_devices": _env("ROCR_VISIBLE_DEVICES"),
            "overlay_pythonpath": _env("OVERLAY_PYTHONPATH"),
        },
    }

    watchdog = _StallWatchdog(report_path, report, _env_int("SGLD_STALL_TIMEOUT_SEC", 1800))
    watchdog.start()

    engine = None
    try:
        if not model:
            raise ValueError("MODEL/MODEL_PATH is empty")
        if not Path(model).is_dir():
            raise ValueError(f"model path is not a directory: {model}")

        from sglang.multimodal_gen import DiffGenerator

        server_args, argv = _build_server_args(model, tp, _env("EXTRA_SGLANG_DIFFUSION_ARGS"))
        report["config"]["resolved_argv"] = " ".join(argv)
        load_start = time.perf_counter()
        engine = DiffGenerator.from_server_args(server_args, local_mode=True)
        report["engine_load_s"] = round(time.perf_counter() - load_start, 3)
        watchdog.beat("engine_loaded")

        base_params: dict[str, Any] = {
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
            "save_output": False,
            "return_frames": False,
        }

        # ── 1. warmup (untimed) ──
        for i in range(warmups):
            _generate_once(engine, {**base_params, "prompt": prompts[i % len(prompts)]})
            watchdog.beat(f"warmup {i + 1}/{warmups}")
        print(f"[bench] warmup done ({warmups} calls)", flush=True)

        # ── 2. timed ──
        # Profiling is armed HERE only: warmup would capture JIT noise and the quality pass is
        # not steady state. num_profiled_timesteps bounds the trace to a few denoise steps —
        # every step is the same DiT graph, so more steps add size, not information.
        timed_params = dict(base_params)
        if profile:
            timed_params["profile"] = True
            timed_params["num_profiled_timesteps"] = _env_int("SGLD_PROFILED_TIMESTEPS", 5)
            if _env_bool("SGLD_PROFILE_ALL_STAGES", False):
                timed_params["profile_all_stages"] = True

        latencies_ms: list[float] = []
        wall_start = time.perf_counter()
        for i in range(timed_n):
            t0 = time.perf_counter()
            _generate_once(engine, {**timed_params, "prompt": prompts[i % len(prompts)]})
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            watchdog.beat(f"timed {i + 1}/{timed_n}")
        wall_s = time.perf_counter() - wall_start
        print(f"[bench] timed {timed_n} images in {wall_s:.2f}s", flush=True)

        if not latencies_ms or wall_s <= 0:
            raise RuntimeError("timed pass produced no measurable work")

        ordered = sorted(latencies_ms)
        p99 = ordered[min(len(ordered) - 1, int(round(0.99 * (len(ordered) - 1))))]

        # ── 3. quality / parity image set (untimed) ──
        produced: list[str] = []
        if quality_n:
            img_dir = out_dir / "parity_images"
            img_dir.mkdir(parents=True, exist_ok=True)
            for i in range(quality_n):
                dest = _ref_slot(ref_write, i) if ref_write else img_dir / f"parity_{i:02d}.png"
                dest.parent.mkdir(parents=True, exist_ok=True)
                _generate_once(engine, {**base_params, "prompt": prompts[i],
                                        "save_output": True,
                                        "output_path": str(dest.parent),
                                        "output_file_name": dest.name})
                produced.append(str(dest))
                watchdog.beat(f"quality {i + 1}/{quality_n}")
            print(f"[bench] parity pass wrote {len(produced)} images", flush=True)

        report.update({
            "success": True,
            # CANONICAL key bench_e2e.sh medians. Unit is img/s for this backend.
            "output_throughput": timed_n / wall_s,
            "request_throughput": timed_n / wall_s,
            "completed": timed_n,
            "images_generated": timed_n,
            "duration": wall_s,
            "latency_s": wall_s / timed_n,
            # bench_e2e.sh's canonical latency keys: for diffusion there is no TTFT/TPOT, so the
            # per-image e2e latency is reported as ttft (time to the one and only output) and the
            # per-denoise-step time as tpot. Both are recorded under their diffusion meaning in
            # mean/median_e2el_ms and ms_per_step below; these two exist so the summarizer's
            # latency columns are populated with something true rather than null.
            "median_ttft_ms": statistics.median(latencies_ms),
            "median_tpot_ms": statistics.median(latencies_ms) / max(1, steps),
            "mean_e2el_ms": statistics.fmean(latencies_ms),
            "median_e2el_ms": statistics.median(latencies_ms),
            "p99_e2el_ms": p99,
            "std_e2el_ms": statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
            "ms_per_step": statistics.median(latencies_ms) / max(1, steps),
            "parity_images": produced,
            "parity_ref_written": bool(ref_write),
            "parity_ref": ref_read or ref_write or "",
        })
        if profile:
            report["profile_dir"] = (_env("SGLANG_DIFFUSION_TORCH_PROFILER_DIR")
                                     or _env("SGLANG_TORCH_PROFILER_DIR") or "./logs")

    except Exception as exc:  # noqa: BLE001 — must always emit a report
        report["success"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()[-4000:]
        print(report["traceback"], file=sys.stderr, flush=True)
    finally:
        watchdog.stop()
        if engine is not None:
            try:
                engine.shutdown()
            except Exception:  # noqa: BLE001 — teardown must not mask the result
                pass
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        # The canonical line bench_e2e.sh's summarizer reads (one JSON object per bench).
        # NEVER append a PROFILED run: torch.profiler inflates latency several-fold (11.4 s vs 3.1 s
        # per image, measured), and bench_e2e.sh runs its profile pass through adapter_bench AFTER the
        # timed repeats into the SAME sink — so appending it would drag the median down and blow the
        # spread. A profiled pass is a trace, not a measurement.
        if result_jsonl and report.get("success") and not profile:
            try:
                with open(result_jsonl, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(report) + "\n")
            except OSError as exc:
                print(f"[bench] WARNING: could not append to {result_jsonl}: {exc}", flush=True)
        print(f"[bench] wrote {report_path}", flush=True)
        if report.get("success"):
            print(f"[bench] output_throughput={report['output_throughput']:.4f} img/s "
                  f"median_e2el={report['median_e2el_ms']:.1f}ms "
                  f"ms_per_step={report['ms_per_step']:.1f}", flush=True)

    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
