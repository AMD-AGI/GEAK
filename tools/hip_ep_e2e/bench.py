#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""End-to-end timing of an OGA model on hip-ep packages, and the A/B gate.

Each measurement is one `model_benchmark` process:
    --prompt_file <L tokens of data/parity_text.txt> -g <gen> -r <reps> -w 1 -b 1
A prompt file skips model_benchmark's untimed token-by-token prompt generation
(what `-l` alone does). --use_random_tokens would too, but on Qwen3.8-27B
(qwen3_5_text) it intermittently fails with "Not enough image_grid_thw data for
image tokens". The run's TTFT and decode numbers are its p50 over the repetitions. A package's
number for a length is the median of its runs. Packages alternate ABBA across
rounds so slow drift (thermals) lands on both sides.

    python3 bench.py ab --model M --base A --cand B --out DIR [--lens 1024,2048]
"""
import argparse
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hipep_env  # noqa: E402

TEXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "parity_text.txt")

# A section header ("Prompt processing (time to first token):") is followed by indented
# "stat (unit): value" lines. The gate reads each section's p50, in microseconds.
_SECTIONS = {"Prompt processing": "ttft_us_p50", "Token generation": "decode_us_p50"}
_P50_RE = re.compile(r"^\s+p50 \(us\):\s*([0-9.eE+-]+)")
_HEAD_RE = re.compile(r"^Batch size: \d+, prompt tokens: (\d+), tokens to generate: (\d+)")


def parse_model_benchmark(text):
    """TTFT and decode p50 (us) and the token counts of model_benchmark's stdout. Raises
    ValueError if either p50 is missing."""
    out, key = {}, None
    for line in text.splitlines():
        m = _HEAD_RE.match(line)
        if m:
            out["prompt_tokens"], out["gen_tokens"] = map(int, m.groups())
        elif not line.startswith((" ", "\t")):
            key = next((v for k, v in _SECTIONS.items() if line.startswith(k)), None)
        elif key and (m := _P50_RE.match(line)):
            out[key] = float(m.group(1))
    for name, key in _SECTIONS.items():
        if key not in out:
            raise ValueError(f"model_benchmark output has no '{name}' p50")
    return out


def prompt_file(model, length, out_dir):
    """The first `length` tokens of the fixed text, cut with the model's tokenizer.json."""
    path = os.path.join(out_dir, f"prompt_{length}.txt")
    if not os.path.exists(path):
        import tokenizers
        tok = tokenizers.Tokenizer.from_file(os.path.join(model, "tokenizer.json"))
        with open(TEXT) as f:
            ids = tok.encode(f.read()).ids
        if len(ids) < length:
            raise ValueError(f"{TEXT} has {len(ids)} tokens, need {length}")
        with open(path, "w") as f:
            f.write(tok.decode(ids[:length]))
    return path


def run_once(env, model, length, gen, reps, log_path, timeout=3600):
    """One model_benchmark process under the serving lock. Returns the parse."""
    exe = os.path.join(env.pkg, "bin", "model_benchmark")
    out_dir = os.path.dirname(os.path.abspath(log_path))
    prompt = prompt_file(model, length, out_dir)
    args = ["-i", model, "--prompt_file", prompt, "-g", str(gen), "-r", str(reps),
            "-w", "1", "-b", "1"]
    p = hipep_env.run_logged("model_benchmark", env, hipep_env.pkg_argv(env, exe, args),
                             hipep_env.pkg_home(out_dir, env.pkg), log_path, timeout=timeout)
    res = parse_model_benchmark(p.stdout)
    if abs(res.get("prompt_tokens", 0) - length) > 8:
        raise RuntimeError(f"prompt has {res.get('prompt_tokens')} tokens, expected about "
                           f"{length}; log: {log_path}")
    return res


def run_order(rounds):
    """ABBA: round r runs (a, b) when r is even, (b, a) when odd."""
    return [("a", "b") if r % 2 == 0 else ("b", "a") for r in range(rounds)]


def gate(base, cand, noise, lens=(1024, 2048)):
    """A/B verdict. base/cand: {L: {"ttft": [us per run], "decode": [us per run]}};
    noise: {"ttft": band, "decode": band}. Accepted when the candidate wins one metric
    beyond its band at every length and holds the other within its band."""
    per_len = {}
    for L in lens:
        tb, tc = statistics.median(base[L]["ttft"]), statistics.median(cand[L]["ttft"])
        db, dc = statistics.median(base[L]["decode"]), statistics.median(cand[L]["decode"])
        row = {"ttft_base": tb, "ttft_cand": tc, "ttft_gain": (tb - tc) / tb,
               "decode_base": db, "decode_cand": dc, "decode_delta": (dc - db) / db}
        row["ttft_win"] = row["ttft_gain"] > noise["ttft"]
        row["ttft_held"] = row["ttft_gain"] >= -noise["ttft"]
        row["decode_win"] = -row["decode_delta"] > noise["decode"]
        row["decode_held"] = row["decode_delta"] <= noise["decode"]
        per_len[L] = row
    rows = per_len.values()
    won = [m for m, held in (("ttft", "decode_held"), ("decode", "ttft_held"))
           if all(r[f"{m}_win"] and r[held] for r in rows)]
    return {"accepted": bool(won), "won": won, "per_len": per_len, "noise": noise}


def collect(env_by_side, model, lens, gen, reps, rounds, out_dir):
    """Run every side at every length, ABBA across rounds. Returns {side: {L: {...}}}."""
    os.makedirs(out_dir, exist_ok=True)
    samples = {s: {L: {"ttft": [], "decode": [], "runs": []} for L in lens} for s in env_by_side}
    for r, order in enumerate(run_order(rounds)):
        for side in order:
            if side not in env_by_side:
                continue
            for L in lens:
                log = os.path.join(out_dir, f"r{r}_{side}_l{L}.log")
                res = run_once(env_by_side[side], model, L, gen, reps, log)
                print(f"[bench] round {r} {side} L={L}: ttft p50 {res['ttft_us_p50'] / 1e3:.1f} ms, "
                      f"decode p50 {res['decode_us_p50'] / 1e3:.2f} ms/token", flush=True)
                s = samples[side][L]
                s["ttft"].append(res["ttft_us_p50"])
                s["decode"].append(res["decode_us_p50"])
                s["runs"].append(res)
    return samples


def _lens(s):
    lens = [int(x) for x in s.split(",") if x.strip()]
    if not lens or min(lens) < 1:
        raise argparse.ArgumentTypeError(f"expected positive lengths like 1024,2048, got {s!r}")
    return lens


def _positive(s):
    if int(s) < 1:
        raise argparse.ArgumentTypeError(f"expected at least 1, got {s}")
    return int(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p = ap.add_subparsers(dest="cmd", required=True).add_parser("ab")
    p.add_argument("--model", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--cand", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lens", type=_lens, default="1024,2048")
    p.add_argument("--gen", type=_positive, default=128)
    p.add_argument("--reps", type=_positive, default=3)
    p.add_argument("--rounds", type=_positive, default=2)
    a = ap.parse_args()
    path = hipep_env.fresh(os.path.join(a.out, "ab.json"))
    envs = {"a": hipep_env.Env.from_environ(a.base), "b": hipep_env.Env.from_environ(a.cand)}
    noise = {"ttft": 0.01, "decode": 0.01}
    s = collect(envs, a.model, a.lens, a.gen, a.reps, a.rounds, a.out)
    g = gate(s["a"], s["b"], noise, lens=a.lens)
    res = {"model": a.model, "base": hipep_env.pkg_info(envs["a"].pkg), "cand": hipep_env.pkg_info(envs["b"].pkg),
           "gen": a.gen, "reps": a.reps, "rounds": a.rounds, "gate": g, "samples": s}
    for L, r in g["per_len"].items():
        print(f"[ab] L={L}: TTFT {r['ttft_base'] / 1e3:.1f} -> {r['ttft_cand'] / 1e3:.1f} ms "
              f"({100 * r['ttft_gain']:+.2f}%), decode {r['decode_base'] / 1e3:.2f} -> "
              f"{r['decode_cand'] / 1e3:.2f} ms ({100 * r['decode_delta']:+.2f}%)")
    print(f"[ab] noise band TTFT {100 * noise['ttft']:.2f}%, decode {100 * noise['decode']:.2f}%: "
          f"{'ACCEPTED (' + ' and '.join(g['won']) + ')' if g['accepted'] else 'REJECTED'}")
    hipep_env.write_json(path, res)
    print(f"[bench] wrote {path}")


if __name__ == "__main__":
    main()
