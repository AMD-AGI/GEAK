import sys, json, urllib.request

base_url = sys.argv[1]
out_file = sys.argv[2]
model = "/models/Qwen3-14B-FP8"

prompts = [
    "The capital of France is",
    "Explain in one sentence why the sky appears blue.",
    "List the first five prime numbers:",
    "Write a haiku about the ocean.",
    "2 + 2 * 3 equals",
    "The quick brown fox",
    "In 2020, the most populous country in the world was",
    "Translate 'good morning' into Spanish:",
    "Complete the sequence: 1, 1, 2, 3, 5, 8,",
    "Define the word 'ephemeral' briefly.",
]

results = []
for p in prompts:
    body = json.dumps({
        "text": p,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 64, "top_p": 1.0},
    }).encode()
    req = urllib.request.Request(base_url + "/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    txt = d["text"] if isinstance(d, dict) else d[0]["text"]
    results.append({"prompt": p, "output": txt})

with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print("wrote", out_file, "n=", len(results))
