#!/usr/bin/env python3
"""Image-parity gate for the sglang_diffusion backend — the diffusion analogue of greedy
token parity.

A text run gates a candidate on "same prompts, temp=0, same tokens out". A diffusion run cannot:
there are no tokens, and a numerically-different-but-correct kernel shifts the last bits of a
latent, which shifts a few pixels. So the bar is a TOLERANCE on the image itself, at a fixed
seed, against the baseline's own images:

    LPIPS <= 0.15   (perceptual;  soft dependency — skipped if weights are unavailable offline)
    SSIM  >= 0.85   (structural)
    MSE   <= 0.006  (raw)

Reported WORST-CASE over the prompt set, so one corrupted image cannot hide behind good averages.
On this box an identical config reproduces the reference to ~1e-10 MSE, while an 8-step run (the
classic way to fake a diffusion speedup: 3x "faster") fails every metric — the gate has been
verified to bite in both directions.

Fails CLOSED: if no reference image can be found, or a shape differs, the verdict is FAIL. A
missing correctness signal is not a pass.

Usage
  image_parity.py --candidate-dir DIR --reference REF_TEMPLATE [--out verdict.json]
  image_parity.py --candidate-dir DIR --reference-dir DIR      [--out verdict.json]

REF_TEMPLATE is the path the bench was given (e.g. .../baseline_ref/ref.png); slot N is
<parent>/<stem>_NN<suffix>, matching how diffusion_bench.py writes the set.

Exit code 0 = parity PASS, 1 = FAIL, 2 = could not evaluate. Stdlib + numpy/PIL/skimage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_rgb(path: Path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def _ref_slot(template: str, index: int) -> Path:
    p = Path(template)
    return p.parent / f"{p.stem}_{index:02d}{p.suffix or '.png'}"


def _lpips_scorer():
    """Return an LPIPS callable, or None when the pretrained weights are unavailable.

    LPIPS pulls AlexNet weights on first use. That download is a SOFT dependency: without it the
    gate still runs on SSIM + MSE rather than failing the candidate for an offline box.
    """
    try:
        import lpips
        import torch

        net = lpips.LPIPS(net="alex", verbose=False).eval()

        def score(a, b) -> float:
            ta = torch.from_numpy(a).permute(2, 0, 1)[None] * 2.0 - 1.0
            tb = torch.from_numpy(b).permute(2, 0, 1)[None] * 2.0 - 1.0
            with torch.no_grad():
                return float(net(ta, tb).item())

        return score
    except Exception as exc:  # noqa: BLE001 — optional metric, never fatal
        print(f"[parity] LPIPS unavailable ({type(exc).__name__}: {exc}); "
              f"gating on SSIM+MSE only", flush=True)
        return None


def compare(candidates: list[Path], ref_for: callable, thresholds: dict[str, float]) -> dict[str, Any]:
    import numpy as np
    from skimage.metrics import structural_similarity as ssim

    lpips_score = _lpips_scorer()
    worst_lpips = worst_ssim = worst_mse = None
    worst_at = {}
    per_image = []
    missing: list[str] = []

    for i, img_path in enumerate(candidates):
        ref_path = ref_for(i)
        if ref_path is None or not Path(ref_path).is_file():
            missing.append(str(ref_path))
            continue
        a = _load_rgb(img_path)
        b = _load_rgb(Path(ref_path))
        if a.shape != b.shape:
            return {"passed": False, "compared": len(per_image),
                    "error": f"shape mismatch at slot {i}: {a.shape} vs {b.shape}"}
        mse = float(np.mean((a - b) ** 2))
        s = float(ssim(a, b, channel_axis=-1, data_range=1.0))
        lp = lpips_score(a, b) if lpips_score is not None else None
        per_image.append({"slot": i, "candidate": str(img_path), "reference": str(ref_path),
                          "mse": mse, "ssim": s, "lpips": lp})
        if worst_mse is None or mse > worst_mse:
            worst_mse = mse; worst_at["mse"] = i
        if worst_ssim is None or s < worst_ssim:
            worst_ssim = s; worst_at["ssim"] = i
        if lp is not None and (worst_lpips is None or lp > worst_lpips):
            worst_lpips = lp; worst_at["lpips"] = i

    if not per_image:
        # Fail closed: no comparison happened, so there is no correctness evidence.
        return {"passed": False, "compared": 0,
                "error": f"no reference images found (looked for {missing[:3]})"}

    passed = (worst_ssim >= thresholds["ssim_min"]) and (worst_mse <= thresholds["mse_max"])
    if worst_lpips is not None:
        passed = passed and worst_lpips <= thresholds["lpips_max"]

    verdict: dict[str, Any] = {
        "passed": bool(passed),
        "lpips": worst_lpips,
        "ssim": worst_ssim,
        "mse": worst_mse,
        "compared": len(per_image),
        "worst_slot": worst_at,
        "thresholds": thresholds,
        "aggregation": "worst_case_over_prompts",
        "lpips_available": worst_lpips is not None,
        "per_image": per_image,
    }
    if missing:
        verdict["missing_references"] = missing[:5]
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-dir", required=True, help="dir of candidate images (sorted by name)")
    ap.add_argument("--reference", default="", help="reference path TEMPLATE (<stem>_NN<suffix>)")
    ap.add_argument("--reference-dir", default="", help="dir of reference images (sorted by name)")
    ap.add_argument("--glob", default="*.png")
    ap.add_argument("--lpips-max", type=float, default=0.15)
    ap.add_argument("--ssim-min", type=float, default=0.85)
    ap.add_argument("--mse-max", type=float, default=0.006)
    ap.add_argument("--out", default="", help="write the verdict JSON here")
    args = ap.parse_args()

    cand_dir = Path(args.candidate_dir)
    candidates = sorted(cand_dir.glob(args.glob))
    if not candidates:
        verdict = {"passed": False, "compared": 0,
                   "error": f"no candidate images matching {args.glob} in {cand_dir}"}
    else:
        if args.reference_dir:
            refs = sorted(Path(args.reference_dir).glob(args.glob))
            ref_for = (lambda i: refs[i] if i < len(refs) else None)
        elif args.reference:
            ref_for = (lambda i: _ref_slot(args.reference, i))
        else:
            print("!!! one of --reference / --reference-dir is required", file=sys.stderr)
            return 2
        verdict = compare(candidates, ref_for,
                          {"lpips_max": args.lpips_max, "ssim_min": args.ssim_min,
                           "mse_max": args.mse_max})

    text = json.dumps(verdict, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    # One machine-readable line for a shell/agent caller, mirroring bench_e2e.sh's E2E_SUMMARY.
    print(text)
    print(f"IMAGE_PARITY {'pass' if verdict.get('passed') else 'fail'} "
          f"lpips={verdict.get('lpips')} ssim={verdict.get('ssim')} mse={verdict.get('mse')} "
          f"compared={verdict.get('compared')}")
    if "error" in verdict:
        return 2 if not verdict.get("compared") else 1
    return 0 if verdict.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
