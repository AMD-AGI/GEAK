# Setup — running GEAK on the codex CLI

> The runtime in this directory is **self-contained and has zero npm dependencies** (Node built-ins
> only). After a `git pull` you only need the CLI installed and a couple of environment variables.

The orchestration engine that runs GEAK's `.js` workflows lives in `interface/runtime/engine/`.
There is nothing to install or configure beyond the CLI and one key — no `config.toml`, no
`CODEX_HOME`, no setup script. All commands below assume you are at the **repo root**.

---

## How a key selects codex and configures its provider

codex's provider is configured **automatically** — no hand-written `config.toml`, no `CODEX_HOME`,
no provider to pick. Setting the key also **selects codex as the backend**, so you do not even need
`GEAK_AGENT_BACKEND=codex`. When launching codex the runtime resolves, first match wins, and emits
`-c model_providers.geak_auto.*` overrides:

1. An explicit **`OPENAI_BASE_URL`** (or the selected model's `base_url`) → use it as-is (any
   OpenAI-compatible gateway).
2. Otherwise **pick by which key is non-empty**: `GEAK_AMDKEY` → AMD gateway, `OPENAI_API_KEY` → official
   OpenAI.

The auto-selected provider carries its own **`default_model`** (used when `GEAK_CODEX_MODEL` is
unset); both are currently `gpt-5.6-sol`. Endpoint and model id live in the same entry on purpose: an
id is only valid on its own endpoint.

The AMD gateway authenticates with the `Ocp-Apim-Subscription-Key` header — **only** that header; a
bare Bearer token gets a 401. The runtime attaches it for you.

### Why the AMD key is `GEAK_AMDKEY` and the OpenAI ones are not prefixed

GEAK's usual caller is hyperloom, whose standard way to configure a deployment is a `.env` file — and
that file is filtered. `common/env_safety.py` admits a name only if it is in `DOTENV_EXACT_ALLOWLIST`
or starts with a `DOTENV_PREFIX_ALLOWLIST` prefix. `"GEAK_"` is one of those prefixes, so any name
GEAK invents for itself gets through without asking hyperloom to change anything. A bare `AMDKEY` is
in neither list: it is dropped with a single line of stderr (`Preflight: WARNING — ignoring
unsupported .env key AMDKEY`), while `GEAK_AGENT_BACKEND=codex` beside it *is* admitted. The result
is the worst shape available — codex selected, no key, no provider overrides emitted, and a failure
that only surfaces at the first agent call.

`OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_CUSTOM_HEADERS` are left alone on purpose. They are
already in that exact allowlist, so a prefix buys nothing; they are ecosystem-standard names; and
renaming them would silently move every environment that already exports `OPENAI_API_KEY` back onto
claude.

None of this applies to a plain `export` in the launching shell. Both of hyperloom's file-based
loaders only fill a name that is *absent* from `os.environ`, so an ambient variable is never filtered
— the allowlist governs the `.env` route alone.

### The selection rule is about the shape of the whole credential environment

Not "is this key present". A key selects its backend only while **no other backend's credentials are
also set**:

| Environment | Runs |
| --- | --- |
| only `GEAK_AMDKEY` or `OPENAI_API_KEY` | **codex** |
| only Anthropic-side (any of `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` / `CLAUDE_CODE_OAUTH_TOKEN`) | claude |
| **both sides set** | claude (`default_profile`) |
| nothing set | claude |

Declining to guess when both are set protects a running claude deployment: otherwise anyone who
exports `GEAK_AMDKEY` silently replaces the existing claude path, which is painful to diagnose. Falling
back to claude when nothing is set is deliberate too — in that environment the claude CLI is probably
authenticated some other way (an already-logged-in CLI, Bedrock). This matches hyperloom
`common/llm_config.py`'s `is_openai_only()` / `is_anthropic_only()`.

To force codex in an environment that has both, pass `--agent codex` explicitly or set
`GEAK_AGENT_BACKEND=codex`. `GEAK_AGENT_AUTO=0` switches key-based selection off entirely (falling
back to the registry's `default_profile`, i.e. claude).

---

## Install the codex CLI

Do not assume it is already present.

```bash
# 1) Node.js v20+ (codex needs it)
node -v        # no node, or < 20: install Node 20+ (nvm / system package manager / nodejs.org)

# 2) install the codex CLI -- pin 0.146.1 (0.147 is incompatible with gateways)
npm i -g @openai/codex@0.146.1
#   no write access to /usr/local? use a user-level prefix:
#   npm config set prefix "$HOME/.npm-global"
#   export PATH="$HOME/.npm-global/bin:$PATH"      # worth persisting in ~/.bashrc
#   npm i -g @openai/codex@0.146.1

# 3) verify
codex --version        # expect 0.146.1
```

## Step 1 — pick a provider by setting its key

```bash
# AMD gateway (adds the Ocp-Apim-Subscription-Key header + llm-api.amd.com/Unified)
export GEAK_AMDKEY="<32-hex subscription key>"
# its certificate is publicly trusted -- no SSL_CERT_FILE needed

# or -- official OpenAI (public CA; no SSL_CERT_FILE needed)
# export OPENAI_API_KEY="sk-....."
```

`GEAK_CODEX_MODEL` is **optional**: unset, the provider's `default_model` is used (`gpt-5.6-sol` on
both). When overriding it, remember an id is only valid on its own endpoint — the AMD gateway serves
`gpt-5.6-sol` / `-terra` / `-luna` but **not** the suffixless `gpt-5.6`, and which ids an official
account can use depends on its entitlement.

To use official OpenAI's **suffixless `gpt-5.6`** (which exists only on that endpoint), pin the
profile: `--profile codex-gpt56` for a single kernel, `GEAK_AGENT_PROFILE=codex-gpt56` for e2e. A
pinned model brings its own `base_url` and `OPENAI_API_KEY` and outranks key-based auto-selection, so
a stray `GEAK_AMDKEY` in the environment will not move the run onto the gateway. This combination is
**untested here** (no official key on hand); if your account returns 404/400, fall back to
`--profile codex-openai` with `GEAK_CODEX_MODEL=<an id you can use>`.

> On the AMD gateway the **gpt family works over both protocols** (`/v1/responses` including
> streaming, and `/v1/chat/completions` — both measured 200), but the **claude family does not answer
> over the OpenAI protocol**: `claude-opus-4-8` / `-4-1` / `Claude-Sonnet-4.5` / `claude-sonnet-5` all
> return 500 on `/v1/chat/completions` and fail to complete on `/v1/responses` (re-measured
> 2026-09-15). That is why the registry pins no claude model.
>
> This is a *protocol* limit, not a model one, and the asymmetry is worth knowing: the same
> `claude-opus-4-8` answers 200 over the gateway's native **Anthropic** `/v1/messages` endpoint. So a
> claude model on this gateway is reachable from GEAK's baseline path (Claude Code, configured with
> `ANTHROPIC_BASE_URL` + `ANTHROPIC_CUSTOM_HEADERS`) but not from this runtime, which drives codex
> over the OpenAI protocol. Use gpt ids here.

## Step 2 — run

The key from step 1 has already selected codex, so just run it.

```bash
# only needed to override auto-selection, e.g. to go back to claude:
# export GEAK_AGENT_BACKEND=claude

# e2e (whole-model throughput): a JSON describes the run. run_e2e.py takes the path as its first
# argument and never hardcodes a name -- its usage string calls it a handoff.
# Fields and a full example: interface/run_e2e.md
python3 interface/run_e2e.py run_spec.json result.json

# single kernel:
node interface/runtime/engine/run_workflow.mjs kernel_workflow/kernel_workflow.js --agent codex \
  --args '{"kernel_path":"/abs/kernel","workflow_dir":"'"$PWD"'/kernel_workflow","budget":6}'
```

## Knobs

| Variable | Default | What it does |
| --- | --- | --- |
| `GEAK_CODEX_MODEL` | the provider's `default_model` (`gpt-5.6-sol`) | Model id. Endpoint-specific — see step 1. |
| `GEAK_CODEX_EFFORT` | `xhigh` | Thinking level. |
| `GEAK_AGENT_PROFILE` / `--profile` | — | Pin an `(agent, model)` combo including its endpoint. |
| `GEAK_AGENT_BACKEND` / `--agent` | — | Pin the agent only; the model still resolves by key. |
| `GEAK_AGENT_AUTO` | `1` | `0` disables key-based backend selection. |
| `GEAK_CODEX_AUTOCONFIG` | `1` | `0` disables provider auto-config (codex then falls back to its own `~/.codex/config.toml`). |
| `GEAK_CODEX_EXTRA_ARGS` | — | Raw `-c key=value` overrides passed to codex; wins over auto-config. |
| `OPENAI_BASE_URL` | — | Any OpenAI-compatible gateway; wins over key-based selection. |

**Thinking level is maxed out by default.** codex's own scale is `none` / `low` / `medium` / `high` /
`xhigh` and has **no `max`** — `xhigh` *is* its top setting, so the runtime emits
`-c model_reasoning_effort=xhigh`. `GEAK_CODEX_EFFORT=max` is still accepted and translates to
`xhigh` (the same mapping as hyperloom's `resolve_codex_reasoning_effort`); any other off-scale value
is rejected up front rather than passed through to codex. To pin it explicitly instead, use
`GEAK_CODEX_EXTRA_ARGS="-c model_reasoning_effort=high"`.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `401` | key empty or invalid |
| `404` on the model | `GEAK_CODEX_MODEL` not served by that endpoint, or not Responses-API-capable |
| TLS error | a private intranet gateway needs `SSL_CERT_FILE` (neither official OpenAI nor the AMD gateway does) |

## Verifying

```bash
# is the runtime itself sound? no network, no GPU, no key required
node interface/runtime/engine/selftest.mjs                          # expect 105/105

# can codex actually drive GEAK, and has GEAK stayed inside the contract?
node interface/runtime/engine/conformance.mjs --profile codex
#   --fake         self-check the harness with no CLI at all
#   --audit-only   run only the static contract-drift audit (no CLI needed)
#   --quick        skip the concurrency probe
#   --geak-root D  point the audit at another tree
```

A failing *probe* names the requirement it violated. The **R-items** are what the backend contract
asks of any CLI — the probes and `registry.json` refer to them by name:

| # | Requirement |
|---|---|
| R1 | **Structured output.** Nearly every `agent()` call carries a schema. A *native* JSON/schema mode is used where it exists; otherwise `schema.mjs` extract + retry, whose failure rate has to be measured. |
| R2 | **Headless one-shot.** One command runs the full agentic loop, exits, and leaves the final answer cleanly on stdout. |
| R3 | **Auto-approval + sandbox.** Roles write outside cwd and run `hipcc` / `rocprof` / `git`; codex's default sandbox blocks both. |
| R4 | **Per-command timeout.** A build or bench runs minutes to hours, so any built-in cap must be raisable. |
| R5 | **Context window.** Largest single prompt is role + knowledge + source, ≈16K tokens and up (63KB `kernel_extractor` is the worst case). |
| R6 | **Provider auth / endpoint.** Env-nameable base_url and key — a same-model-different-CLI comparison needs both pointed at one endpoint. |
| R7 | **cwd / absolute paths.** The CLI honours cwd and can do FS work on absolute paths outside it (tied to R3). |

R1 and R3 are the blocking pair: unsolved, no CLI finishes a single round.

A failing *audit* is the drift detector, not a backend problem: GEAK has grown past what this runtime
supports. Implement the missing capability first, *then* move the baseline constant in
`conformance.mjs` — never the other way round, or parity silently stops meaning anything.

## What each file is

Two files, then a directory:

- **`SETUP.md`** — this file: how to get codex running.
- **`../run_e2e.py`** — the programmatic entry point; picks native Claude Code vs this runtime from
  the environment.
- **`engine/`** — the orchestration engine. See [`engine/README.md`](engine/README.md) for what each
  file in it does.
