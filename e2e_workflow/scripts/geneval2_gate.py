#!/usr/bin/env python3
"""GenEval2 prompt-fidelity probe for the sglang_diffusion backend — the diffusion analogue of
the gsm8k accuracy gate.

Image parity (scripts/image_parity.py) answers "did the pixels move?"; it cannot answer "does the
image still match the prompt?", because it only ever compares a candidate to the baseline's OWN
images. GenEval2 Soft-TIFA closes that hole: a Qwen3-VL VQA model answers per-prompt atomic
questions ("Is the backpack green?", "How many pigs are in the image?") and the geometric mean is
the headline score, 0-100.

It is the EXPENSIVE gate (generation dominates; scoring ~40 s for 64 prompts), so the workflow
runs it at the endpoints — TRUE baseline and final winner — not per candidate.

This is a thin wrapper over the user's existing harness at /home/aditysin/PROJECTS/flux-geneval2
(evaluate_flux.py -> GenEval2/evaluation.py). It exists to pin the mandatory FLUX flags, pin the
GPU set, and emit one machine-readable line the Director/Architect can read.

Usage
  geneval2_gate.py --label baseline --limit 64 --gpus 0,1,2,3,4,5,6,7 --num-gpus 8 \
                   --out-dir <dir> [--extra-arg "--vae-precision bf16"]...
  geneval2_gate.py --label final --limit 64 --gpus 6,7 --num-gpus 2 --out-dir <dir>

Prints:  GENEVAL2 <label> score=<0-100> prompts=<n> method=soft_tifa_gm
Exit 0 on success. With --baseline-score X --tol T, exits 1 when score < X - T (a fidelity
regression), so it can be used as a hard gate rather than a report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_HARNESS = Path("/home/aditysin/PROJECTS/flux-geneval2")
MODEL_PATH = "/home/aditysin/PROJECTS/models/FLUX-1-dev"

# Same mandatory hygiene as the bench adapter: without --model-id the native FLUX pipeline is not
# resolved from a path spelled FLUX-1-dev and sglang silently serves the ~17x slower diffusers
# fallback — which would also change the images being scored.
BASE_ARGS = ["--model-id", "FLUX.1-dev",
             "--dit-cpu-offload", "False",
             "--text-encoder-cpu-offload", "False"]


def normalize_images(out_dir: Path) -> int:
    """Re-encode any JPEG-payload-in-a-.png-filename image to a real PNG.

    `sglang serve` returns JPEG bytes, and evaluate_flux.py saves them under a `.png` name. The
    GenEval2 scorer loads images through transformers -> torchvision `decode_image`, and the
    torchvision build in this container has no libjpeg, so scoring dies with
    `RuntimeError: decode_jpeg: torchvision not compiled with libjpeg support` AFTER the whole
    (expensive) generation pass has completed. Normalizing costs ~1 s per 100 images and makes the
    scorer's input format independent of which generation backend produced it.
    """
    from PIL import Image

    fixed = 0
    for path in sorted(out_dir.rglob("*.png")):
        try:
            with Image.open(path) as im:
                if im.format == "PNG":
                    continue
                rgb = im.convert("RGB")
            rgb.save(path, "PNG")
            fixed += 1
        except Exception as exc:  # noqa: BLE001 — a bad file should not kill the gate
            print(f"[geneval2] WARNING: could not normalize {path}: {exc}", flush=True)
    if fixed:
        print(f"[geneval2] re-encoded {fixed} JPEG-payload images to real PNG "
              f"(torchvision here has no libjpeg)", flush=True)
    return fixed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="baseline | final | <variant name>")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--harness-dir", default=str(DEFAULT_HARNESS))
    ap.add_argument("--model-path", default=MODEL_PATH)
    ap.add_argument("--limit", type=int, default=64,
                    help="leading N of the 800 GenEval2 prompts; 0 or negative = ALL 800. "
                         "Use the full set for any accept/reject decision — at 64 prompts the "
                         "seed alone moves the score by ~4.4 points (55.44 at seed 42 vs 51.04 at "
                         "seed 43), which swamps the effect you are measuring.")
    ap.add_argument("--gpus", default="", help="ROCR_VISIBLE_DEVICES for generation + scoring")
    ap.add_argument("--num-gpus", type=int, default=1, help="sglang --num-gpus for generation")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-inference-steps", type=int, default=28)
    ap.add_argument("--guidance-scale", type=float, default=3.5)
    ap.add_argument("--gen-backend", default="serve", choices=["serve", "cli"])
    ap.add_argument("--port", type=int, default=31500)
    ap.add_argument("--method", default="soft_tifa_gm")
    ap.add_argument("--extra-arg", action="append", default=[],
                    help="extra sglang flag, repeatable; quote the whole flag+value")
    ap.add_argument("--baseline-score", type=float, default=None,
                    help="if set, gate: fail when score < baseline - tol")
    ap.add_argument("--tol", type=float, default=1.0, help="allowed absolute score drop")
    args = ap.parse_args()

    harness = Path(args.harness_dir)
    evaluate = harness / "evaluate_flux.py"
    if not evaluate.is_file():
        print(f"!!! GenEval2 harness not found at {evaluate}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_extra: list[str] = list(BASE_ARGS) + ["--num-gpus", str(args.num_gpus)]
    for raw in args.extra_arg:
        gen_extra += raw.split()

    cmd = [sys.executable, str(evaluate),
           "--geneval2-dir", str(harness / "GenEval2"),
           "--models", args.model_path,
           "--out-dir", str(out_dir),
           "--gen-backend", args.gen_backend,
           "--width", str(args.width), "--height", str(args.height),
           "--seed", str(args.seed),
           "--num-inference-steps", str(args.num_inference_steps),
           "--guidance-scale", str(args.guidance_scale),
           "--method", args.method,
           "--port", str(args.port)]
    # limit<=0 means the full 800: evaluate_flux.py takes "all" as the ABSENCE of --limit
    # (it applies prompts[:limit] whenever the value is not None, so --limit 0 would score nothing).
    if args.limit > 0:
        cmd += ["--limit", str(args.limit)]
    # `--gen-extra-arg --model-id` makes argparse read the VALUE as the next option and fail with
    # "expected one argument"; the single-token `--gen-extra-arg=--model-id` form is unambiguous.
    for flag in gen_extra:
        cmd.append(f"--gen-extra-arg={flag}")

    env = dict(os.environ)
    env.pop("HIP_VISIBLE_DEVICES", None)   # ROCm stack: this can break torch.cuda.is_available()
    if args.gpus:
        env["ROCR_VISIBLE_DEVICES"] = args.gpus

    log_path = out_dir / f"geneval2_{args.label}.log"
    print(f"[geneval2] ROCR_VISIBLE_DEVICES={env.get('ROCR_VISIBLE_DEVICES', '<inherited>')}")
    print(f"[geneval2] log -> {log_path}")

    def _run(extra: list[str], mode: str) -> subprocess.CompletedProcess:
        full = cmd + extra
        print(f"[geneval2] {mode}: {' '.join(full)}")
        p = subprocess.run(full, env=env, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"\n===== {mode} (exit {p.returncode}) =====\n{p.stdout or ''}")
        return p

    # Generation and scoring run as two calls with an image-normalization step between them, because
    # the serve backend emits JPEG bytes under .png names and this container's torchvision cannot
    # decode JPEG (see normalize_images). Doing it in one call means the whole generation pass is
    # thrown away when the scorer trips over the first image.
    log_path.write_text("", encoding="utf-8")
    proc = _run(["--skip-scoring"], "generate")
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-25:])
        print(f"!!! evaluate_flux.py generation failed (exit {proc.returncode}). Tail:\n{tail}",
              file=sys.stderr)
        return 2

    normalize_images(out_dir)

    proc = _run(["--skip-generation"], "score")
    tail = "\n".join((proc.stdout or "").splitlines()[-25:])
    if proc.returncode != 0:
        print(f"!!! evaluate_flux.py scoring failed (exit {proc.returncode}). Tail:\n{tail}",
              file=sys.stderr)
        return 2

    score = None
    summary = out_dir / "summary.json"
    if summary.is_file():
        try:
            rows = json.loads(summary.read_text(encoding="utf-8"))
            if rows:
                score = float(rows[-1].get("score"))
        except Exception:  # noqa: BLE001 — fall through to the log scrape
            score = None
    if score is None:
        m = re.findall(r"=\s*([0-9]+\.[0-9]+)\s*$", proc.stdout or "", re.M)
        score = float(m[-1]) if m else None
    if score is None:
        print(f"!!! could not parse a score. Tail:\n{tail}", file=sys.stderr)
        return 2

    verdict = {"label": args.label, "score": score, "prompts": (args.limit if args.limit > 0 else 800),
               "method": args.method, "out_dir": str(out_dir),
               "extra_args": args.extra_arg}
    if args.baseline_score is not None:
        verdict["baseline_score"] = args.baseline_score
        verdict["tol"] = args.tol
        verdict["passed"] = score >= args.baseline_score - args.tol
    (out_dir / f"geneval2_{args.label}.json").write_text(json.dumps(verdict, indent=2),
                                                         encoding="utf-8")
    print(f"GENEVAL2 {args.label} score={score:.2f} prompts={args.limit} method={args.method}"
          + (f" baseline={args.baseline_score:.2f} passed={verdict['passed']}"
             if args.baseline_score is not None else ""))
    if args.baseline_score is not None and not verdict["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
