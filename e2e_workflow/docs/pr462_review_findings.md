# PR #462 review findings (swappable agent backends)

Findings from reviewing `feat/swappable-agent-backends-v3`. All are fixed on the branch. This note
records *why*, so none of the changes is re-introduced by accident.

---

## Finding 1 — `serving_weighted_speedup` amortized prefill over OSL twice (REVERTED)

**Introduced by** `35d8f491` "fix(harness): amortize one-time prefill/TTFT over OSL in
serving_weighted_speedup" (a +16-line addition, never on `main`).
**Fix**: reverted — `serving_weighted_speedup` in `e2e_workflow/scripts/harness_lib.py` is now
byte-identical to `main`.

### Problem

The commit divided every non-decode bucket's weight by OSL:

```python
r["weight"] = (b or 0.0) * r.get("calls", 1)
if _osl is not None and r["regime"] and r["regime"] != "decode":
    r["weight"] = r["weight"] / _osl   # amortize one-time prefill/TTFT over OSL tokens
```

The stated intent is right — a prefill bucket is paid once per request and should be amortized
over the OSL tokens it precedes. But that amortization was **already there**, carried by the
analytic call model, so applying it again divides by OSL a second time.

### Root cause

`weighted` is a *ratio*, so only the relative weight of the buckets matters. The analytic call
model (`serving_weight_model.analytic_calls`) already puts both regimes on one per-wave basis:

| | `calls` | weight on `main` | correct per-token basis |
|---|---|---|---|
| decode | `OSL` | `b_d · OSL` | `b_d` |
| prefill | `CONC·ceil(ISL/chunk)` | `b_p · calls_p` | `b_p · calls_p / OSL` |

Both columns give the same **ratio** `b_d·OSL : b_p·calls_p`. `main` is already on the correct
basis. Dividing prefill by OSL once more makes the ratio `b_d·OSL² : b_p·calls_p`, i.e. prefill is
under-weighted by a factor of OSL — typically 1024×.

### Impact

Prefill regressions become invisible to the PRIMARY gate. Worked example (OSL=1024, CONC=64,
ISL=4096, chunk=2048 → `analytic_calls = {prefill: 128, decode: 1024}`; prefill 0.98x, decode 1.02x):

| | prefill weight share | decode weight share | `weighted` |
|---|---|---|---|
| `main` (correct) | 95.24 % | 4.76 % | **0.9818** → regression |
| `35d8f491` | 1.92 % | 98.08 % | **1.0192** → "win" |

A sign flip on the gate metric: a run that made the GPU-time-dominant prefill 2 % slower and
decode 2 % faster is reported as a 1.02x win. The same distortion propagates into the downstream
Amdahl ceiling.

### Note for whoever revisits this

If a future workload really does need prefill amortized differently, the change belongs in
`_analytic_calls_from_meta` / `attribute_weights` (which owns the per-wave basis), **not** as a
second division inside the weighting loop. Any such change must ship with a mixed prefill+decode
regression test — the 16 lines above passed the entire suite (327 tests in
`e2e_workflow/scripts/tests/`) both before and after removal, because nothing covered them.

---

## Finding 2 — `invoke_workflow` test doubles drifted out of sync with the call site (FIXED)

**Introduced by** `39ae37a3` "feat(interface/runtime): swappable code-agent backends for GEAK",
which added a fourth argument at the call site and did not update the doubles.

### Problem

`invoke_workflow` gained a `ps_args` parameter:

```python
def invoke_workflow(prompt, timeout_s, eval_dir=None, ps_args=None):   # interface/run_e2e.py
...
wf = invoke_workflow(prompt, timeout_s, ps_args["eval_dir"], ps_args=ps_args)   # :6503
```

Ten monkeypatched doubles across two test files still declared `(prompt, t, ed)`, so every call
raised `TypeError: ... got an unexpected keyword argument 'ps_args'`.

### Why it was not all red

`run_e2e.py:6504` wraps the call in a deliberately wide `except Exception` (it must recover from a
crashed/timed-out agent by scraping disk). That catch swallowed the `TypeError` too, so most of
the affected tests still "passed" — while exercising the crash-recovery path instead of the path
they were written for. Concretely:

- 4 doubles broke loudly: 3 tests in `test_run_e2e_dispatch.py`, plus
  `test_emit_timeout_still_writes_journey`, which asserted `status == "timeout"` and got `"error"`
- the other 6 silently tested nothing, e.g. `test_workflow_failure_with_a_recoverable_disk_win`
  intended to raise `WorkflowParseError` but raised `TypeError`, reaching the same assertion by a
  different route

### Fix

`4b92e93d` ("Fix CI on the codex backend branch") repaired the 4 loud ones. This change repairs
the remaining 6 silent ones — 5 in `interface/test_run_e2e_recovery.py`, 1 in
`interface/test_run_e2e_dispatch.py` — so all ten doubles now read
`(prompt, timeout_s, eval_dir, ps_args=None)`. No production code changed.

Because the 6 were already green-but-vacuous, the suite looks identical before and after; the
difference is that each test now reaches the code path named in its own docstring.

Note the doubles *accept and ignore* `ps_args`; this restores the original coverage but does not
assert the value passed is correct. That gap is separately covered by
`interface/test_run_e2e_runtime_dispatch.py`, added on this branch.

### Guard against recurrence

Most doubles in these files use `lambda *a, **k:`, which is signature-agnostic and survived the
change. The ones that broke are the named `def` doubles. When adding a parameter to a function
that tests monkeypatch, grep for the named doubles:

```
grep -rn '"invoke_workflow"' --include=*.py .
```

---

## Finding 3 — the shipped `codex-home/config.toml` made codex refuse to start (FIXED by deletion)

### Problem

`interface/runtime/codex-home/config.toml` (PR-added; no such file on `main`) defined
`[model_providers.openai]`. codex 0.146.1 rejects that outright:

```
Error loading config.toml: model_providers contains reserved built-in provider IDs:
`openai`. Built-in providers cannot be overridden.
```

codex refuses to load the **whole file**, so it never starts — independent of which provider is
selected. Anyone following `setup.sh` (which exported `CODEX_HOME` to that directory) got a hard
failure. Measured against the AMD gateway with a real key, deterministic 2/2, and reproduced on the
pre-rewrite revision `5bdbea3c^` as well, so it was broken from the moment the file was introduced.

### Why it went unnoticed

The default path never reads it. With `GEAK_CODEX_AUTOCONFIG=1` the runtime passes the provider as
`-c model_providers.geak_auto.*` overrides and codex needs no `config.toml` and no `CODEX_HOME` at
all. `setup.sh` was optional, so only someone who actually sourced it hit the bug. Same probe with
`CODEX_HOME` unset or pointed at an empty directory: 4/4 clean round-trips on the AMD gateway.

### Fix

Deleted the whole SaFE-era shim chain, which is what `config.toml` existed to configure:
`codex-home/`, `responses_shim.mjs`, `setup.sh`, the `127.0.0.1`/`localhost` autoconfig
short-circuit in `config.mjs`, and the matching `.gitignore` rules. Only the AMD gateway and an
OpenAI API key are supported, and neither needs a shim or a config file.

A side effect worth knowing: that short-circuit used to *skip* autoconfig for any localhost
`base_url`, so pointing `OPENAI_BASE_URL` at a local OpenAI-compatible server (vLLM, llama.cpp)
left codex with no provider at all. It is now autoconfigured like any other endpoint; `selftest.mjs`
locks that in.

---

## Open items (not addressed in this PR)

Runtime-internal, only reachable on the codex path:

- `interface/runtime/engine/backends/base.mjs:71` — `child.stdin.write(prompt)` is wrapped in `try/catch`,
  but an EPIPE from a CLI that exits before reading the prompt arrives as an **asynchronous
  `'error'` event on the stream**, which `try/catch` cannot intercept. With no `child.stdin.on
  ('error', ...)` listener that is still an uncaught exception: the runtime dies instead of
  surfacing the CLI's own error message.
- `interface/runtime/engine/experiment.mjs:83` — `spawn('node', ...)` with no `'error'` listener, so an
  ENOENT (no `node` on PATH) is likewise uncaught rather than reported.
- Schema-validation retry re-runs the whole agent turn rather than just re-asking for the JSON.
- `registry.json` pins the `claude` agent to `--allowedTools Bash Read Write`; the baseline
  workflow's WebSearch/WebFetch are unavailable under the runtime path.

Gateway-side, not ours: the AMD gateway intermittently answers a codex request with
`{"error":"Missing required header","message":"The 'user' header with a valid User NTID is mandatory
for application 'GEAK_GROUP'..."}`. Seen twice in ~11 live runs, each time as the FIRST request of a
session, with 3/3 and 6/6 clean immediately afterwards under an identical environment — an identical
request succeeding seconds later is what makes this flakiness on their side rather than a missing
config. Worth knowing because the failure text names a header and reads like a config bug: retry once
before changing anything.

(An earlier version of this note argued the case differently — that codex has no way to send a `user`
header at all. That is wrong. `env_http_headers` in `provider_autoselect` is exactly that mechanism:
it maps a header name to an env var name, and `config.mjs:218` emits each pair as
`-c model_providers.<P>.env_http_headers.<h>=<env var>`. Today the AMD entry maps only
`Ocp-Apim-Subscription-Key`, so if this ever stops being intermittent, the fix is a second pair —
`"user": "GEAK_AMD_NTID"` — not a code change. The flakiness conclusion stands on the retry evidence
above; it never needed the impossibility claim.)

Backend selection itself was reviewed and is **working as designed**: with only `GEAK_AMDKEY` and/or
`OPENAI_API_KEY` set the run goes to codex; with neither it goes to baseline. One caveat worth
knowing — the exclusion rule in `_derive_agent_from_env` means that if the image also bakes in any
`ANTHROPIC_*` variable, auto-selection declines and the run silently falls back to `default_profile`
(baseline) even though a gateway key is present.

---

## Finding 4 — the registry advertised three backends nobody had verified (FIXED by removal)

`registry.json` shipped five agents (`claude`, `qwen`, `codex`, `kimi`, `cursor`) and seven
profiles. Only `claude` and `codex` were ever exercised: the `qwen` entry's own note dates its last
verification to 2026-07-31 **against the now-decommissioned SaFE gateway**, and `kimi` / `cursor`
carry "verify … at bring-up" instructions rather than results. Since the supported surface is now the
AMD gateway plus an OpenAI API key, and neither `qwen` nor `kimi` can attach the
`Ocp-Apim-Subscription-Key` header the AMD gateway requires, those entries could not have worked
against the one gateway that matters.

A registry entry is not documentation of an experiment — it is an offer. Leaving three unverified
CLIs in it meant `--agent kimi` resolved cleanly and then failed somewhere deep in a subprocess.
Removed: the three agents, their three profiles, and the `--profile qwen` / `npm i -g
@qwen-code/qwen-code` / `@moonshotai/kimi-code` instructions in `interface/run_e2e.md`.

Nothing in the runtime was specific to them — the removal is data plus comments. `selftest.mjs`'s
fixture registry kept the same coverage by renaming its second agent to a clearly synthetic `stub`
(it exercises stdin delivery + an approve flag + a model endpoint, which the two real agents do not
combine), and gained an assertion that a retired name now **throws** rather than resolving, so a
stale `--agent qwen` in someone's script fails loudly instead of quietly running on claude.
