#!/usr/bin/env python3
"""Lightweight gsm8k accuracy eval against an OpenAI-compatible /v1 server (greedy, exact_match).

Mirrors the SemiAnalysis InferenceX gsm8k method (lm-eval gsm8k.yaml): 5-shot, do_sample=false /
temperature=0, answer on the last line as "#### <number>", scored by exact_match on the extracted
number (strict "#### N" + flexible "last number" fallback). Samples a SUBSET (--limit) with a FIXED
seed so the baseline and candidate servers see the IDENTICAL problems (apples-to-apples).

Self-contained (requests + datasets only) so there is no lm-eval dependency to install/break.

Usage:
  python3 gsm8k_eval.py --base-url http://127.0.0.1:30000/v1 --model <path> --limit 200 \
      --out <dir>/gsm8k.json [--fewshot 5] [--max-tokens 512] [--concurrency 32]
Prints a final line:  GSM8K_EXACT_MATCH=<0..1>
"""
import argparse, json, os, re, sys, random, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset

ANS_RE = re.compile(r"####\s*(-?[0-9][0-9,]*\.?[0-9]*)")          # strict: the "#### N" line
NUM_RE = re.compile(r"(-?[$0-9][0-9,]*\.?[0-9]*)")                # flexible: any number

def gold_answer(a):
    m = ANS_RE.search(a)
    return _norm(m.group(1)) if m else None

def _norm(s):
    return s.replace(",", "").replace("$", "").rstrip(".").strip()

def extract_pred(text):
    # strict-match first (the "#### N" the prompt asks for), else flexible last-number
    ms = ANS_RE.findall(text)
    if ms:
        return _norm(ms[-1])
    ms = NUM_RE.findall(text)
    return _norm(ms[-1]) if ms else None

def build_fewshot(train, k):
    shots = []
    for i in range(k):
        q = train[i]["question"].strip()
        a = train[i]["answer"].strip()  # includes reasoning + "#### N"
        shots.append(f"Question: {q}\nEnd your response with the answer on the last line, formatted as: #### [number]\nAnswer: {a}")
    return "\n\n".join(shots)

def ask(base_url, model, prompt, max_tokens, timeout=1800):
    r = requests.post(base_url.rstrip("/") + "/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.0, "top_p": 1.0, "max_tokens": max_tokens, "seed": 0},
        timeout=timeout)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    # MiniMax-M3 is a REASONING model: the chat response splits into `content` (final answer) and
    # `reasoning_content` (the CoT), and `content` is often None when the answer falls inside the reasoning
    # or the response is truncated. Combine BOTH (never None) so the "#### N" is found wherever it lands —
    # this fixes the TypeError crash on None content and the truncation-driven accuracy decay.
    # reasoning FIRST, content LAST: the final answer is in `content`, so the flexible "last number"
    # fallback in extract_pred() must see content at the end.
    return ((msg.get("reasoning_content") or "") + "\n" + (msg.get("content") or "")).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)            # e.g. http://127.0.0.1:30000/v1
    ap.add_argument("--model", required=True)               # served model name/path
    ap.add_argument("--limit", type=int, default=200)       # subset size (InferenceX-style --limit)
    ap.add_argument("--fewshot", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=1024)   # reasoning model needs room to finish CoT + answer
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ds = load_dataset("openai/gsm8k", "main")
    train, test = ds["train"], ds["test"]
    idx = list(range(len(test)))
    random.Random(a.seed).shuffle(idx)            # FIXED seed -> same subset for baseline & cand
    idx = idx[: a.limit]
    fewshot = build_fewshot(train, a.fewshot)

    def one(i):
        q = test[i]["question"].strip()
        gold = gold_answer(test[i]["answer"])
        prompt = fewshot + "\n\n" + f"Question: {q}\nEnd your response with the answer on the last line, formatted as: #### [number]\nAnswer:"
        try:
            out = ask(a.base_url, a.model, prompt, a.max_tokens)
        except Exception as e:
            return i, gold, None, f"ERR:{str(e)[:80]}"
        return i, gold, extract_pred(out), out[-200:]

    results, correct = [], 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = [ex.submit(one, i) for i in idx]
        for n, f in enumerate(as_completed(futs), 1):
            i, gold, pred, tail = f.result()
            ok = (gold is not None and pred is not None and gold == pred)
            correct += int(ok)
            results.append({"idx": i, "gold": gold, "pred": pred, "ok": ok})
            if n % 25 == 0:
                print(f"  [gsm8k] {n}/{len(idx)} running acc={correct/n:.4f}", file=sys.stderr)
    score = correct / len(idx) if idx else 0.0
    summary = {"task": "gsm8k", "exact_match": score, "n": len(idx), "correct": correct,
               "fewshot": a.fewshot, "greedy": True, "seed": a.seed, "limit": a.limit,
               "base_url": a.base_url, "elapsed_s": round(time.time() - t0, 1)}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=1)
    print(json.dumps(summary))
    print(f"GSM8K_EXACT_MATCH={score:.4f}")

if __name__ == "__main__":
    main()
