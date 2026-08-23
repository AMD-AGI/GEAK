#!/usr/bin/env bash
# gsm8k correctness gate against an already-running server. INSIDE the container.
#
#   ./run_eval.sh                  # 5-shot gsm8k, 1319 problems
#   TAG=my_change ./run_eval.sh
#
# A throughput win that changes the answers is not a win. This is the gate, run it on any
# configuration you intend to report -- and note that the reference session recorded no accuracy
# number for this model, so your first run establishes the reference. Record it in FINDINGS.md.
#
# Three settings below are what separate a real accuracy number from a meaningless one on a model
# that reasons before answering, and each was learned by getting it wrong first:
#
#   * max_tokens=9216. lm-eval's default generation budget is 256 tokens, which truncates the
#     reasoning so the answer never arrives. On this stack the default scored 0.0318 strict-match,
#     which reads as a broken model rather than as a broken measurement. The budget here is the
#     served context (11264) less room for the 5-shot prompt.
#   * the sitecustomize patch. When a server puts the text in `reasoning_content` and leaves
#     `content` empty, lm-eval 0.4.12 substitutes a placeholder and warns, and the response is lost.
#     The patch falls back to `reasoning_content`. (0.4.12 no longer needs the chat-template half of
#     the equivalent patch in the sibling bundles; it applies the template the same way now.)
#   * temperature=0, top_p=1, fixed seeds. A gate that moves on its own cannot gate anything.
#
# This matches the invocation the sibling bundles' accuracy references were produced with, so the
# numbers are comparable to theirs.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-43102}"
MODEL="${MODEL:-/shared_nfs/hyperloom/models/Qwen3-14B-FP8}"
TASK_DIR="${TASK_DIR:-$HERE/../eval}"
TAG="${TAG:-run}"
OUT="${OUT:-$HERE/../eval_results/${TAG}_$(date +%Y%m%d_%H%M%S)}"

if ! curl -sf -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[eval] no healthy server on port ${PORT} -- start it first" >&2; exit 1
fi
if ! python3 -c "import lm_eval" 2>/dev/null; then
    echo "[eval] lm_eval is not installed in this container. Install it in a venv of its OWN:" >&2
    echo "[eval]   python3 -m venv /tmp/lmeval_venv" >&2
    echo "[eval]   /tmp/lmeval_venv/bin/pip install 'lm-eval[api]==0.4.12'" >&2
    echo "[eval]   PATH=/tmp/lmeval_venv/bin:\$PATH ./run_eval.sh" >&2
    echo "[eval] Not into the serving environment, and not an older lm-eval:" >&2
    echo "[eval]   * 0.4.9.2 cannot import against the transformers 5.x in this image" >&2
    echo "[eval]     (it reads transformers.AutoModelForVision2Seq, which no longer exists)" >&2
    echo "[eval]   * in 0.4.12 torch and transformers are optional extras, so [api] pulls neither" >&2
    echo "[eval]     and cannot shadow the framework's own build. This eval talks HTTP; it does" >&2
    echo "[eval]     not need torch at all." >&2
    exit 1
fi

mkdir -p "$OUT"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export HF_HUB_TRUST_REMOTE_CODE=1

# Patched in-process rather than by editing the installed package, so the eval venv stays disposable.
PATCH_DIR="$(mktemp -d)"
cat > "$PATCH_DIR/sitecustomize.py" <<'PYPATCH'
from lm_eval.models.openai_completions import LocalChatCompletion as _LCC


def _parse_generations(outputs, **kwargs):
    """Fall back to reasoning_content when content is empty."""
    res = []
    if not isinstance(outputs, list):
        outputs = [outputs]
    for out in outputs or []:
        try:
            choices = out.get("choices", [])
            tmp = ["" for _ in choices]
            for choice in choices:
                msg = choice.get("message") or {}
                content = msg.get("content")
                if content in (None, "", []):
                    content = msg.get("reasoning_content") or ""
                tmp[choice.get("index", 0)] = content
        except Exception:
            tmp = [""]
        res.extend(tmp)
    return res


_LCC.parse_generations = staticmethod(_parse_generations)
PYPATCH
export PYTHONPATH="${PATCH_DIR}:${PYTHONPATH:-}"

python3 -m lm_eval \
    --model local-chat-completions \
    --model_args "model=${MODEL},base_url=http://127.0.0.1:${PORT}/v1/chat/completions,api_key=EMPTY,eos_string=</s>,max_retries=5,num_concurrent=64,timeout=1800,tokenized_requests=false,max_length=11264" \
    --tasks gsm8k \
    --include_path "$TASK_DIR" \
    --num_fewshot 5 \
    --apply_chat_template \
    --batch_size 1 \
    --seed 0,1234,1234,1234 \
    --gen_kwargs max_tokens=9216,temperature=0,top_p=1 \
    --output_path "$OUT" \
    --log_samples \
    2>&1 | tee "$OUT/lm_eval_stdout.log"

echo
echo "result -> $OUT"
echo "no reference accuracy exists for this model; treat your first clean run as the reference"
