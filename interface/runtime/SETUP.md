# Setup — running GEAK on the codex CLI

> The runtime in this directory is **self-contained and has zero npm dependencies** (Node built-ins
> only). After a `git pull` you only need the CLI installed and a couple of environment variables.
> For the design and architecture, see [`DESIGN.md`](DESIGN.md).

This layer lives in `interface/runtime/`. All commands below assume you are at the **repo root**.

---

## How a key selects codex and configures its provider

codex's provider is configured **automatically** — no hand-written `config.toml`, no `setup.sh`, no
provider to pick. Setting the key also **selects codex as the backend**, so you do not even need
`GEAK_AGENT_BACKEND=codex`. When launching codex the runtime resolves, first match wins, and emits
`-c model_providers.geak_auto.*` overrides:

1. An explicit **`OPENAI_BASE_URL`** (or the selected model's `base_url`) → use it as-is (any
   OpenAI-compatible gateway).
2. Otherwise **pick by which key is non-empty**: `AMDKEY` → AMD gateway, `OPENAI_API_KEY` → official
   OpenAI.

The auto-selected provider carries its own **`default_model`** (used when `GEAK_CODEX_MODEL` is
unset); both are currently `gpt-5.6-sol`. Endpoint and model id live in the same entry on purpose: an
id is only valid on its own endpoint.

The AMD gateway authenticates with the `Ocp-Apim-Subscription-Key` header — **only** that header; a
bare Bearer token gets a 401. The runtime attaches it for you.

### The selection rule is about the shape of the whole credential environment

Not "is this key present". A key selects its backend only while **no other backend's credentials are
also set**:

| Environment | Runs |
| --- | --- |
| only `AMDKEY` or `OPENAI_API_KEY` | **codex** |
| only Anthropic-side (any of `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` / `CLAUDE_CODE_OAUTH_TOKEN`) | claude |
| **both sides set** | claude (`default_profile`) |
| nothing set | claude |

Declining to guess when both are set protects a running claude deployment: otherwise anyone who
exports `AMDKEY` silently replaces the existing claude path, which is painful to diagnose. Falling
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
export AMDKEY="<32-hex subscription key>"
# its certificate is publicly trusted -- no SSL_CERT_FILE needed

# or -- official OpenAI (public CA; no shim, no SSL_CERT_FILE, no config.toml)
# export OPENAI_API_KEY="sk-....."
```

`GEAK_CODEX_MODEL` is **optional**: unset, the provider's `default_model` is used (`gpt-5.6-sol` on
both). When overriding it, remember an id is only valid on its own endpoint — the AMD gateway serves
`gpt-5.6-sol` / `-terra` / `-luna` but **not** the suffixless `gpt-5.6`, and which ids an official
account can use depends on its entitlement.

To use official OpenAI's **suffixless `gpt-5.6`** (which exists only on that endpoint), pin the
profile: `--profile codex-gpt56` for a single kernel, `GEAK_AGENT_PROFILE=codex-gpt56` for e2e. A
pinned model brings its own `base_url` and `OPENAI_API_KEY` and outranks key-based auto-selection, so
a stray `AMDKEY` in the environment will not move the run onto the gateway. This combination is
**untested here** (no official key on hand); if your account returns 404/400, fall back to
`--profile codex-openai` with `GEAK_CODEX_MODEL=<an id you can use>`.

> On the AMD gateway the **gpt family works over both protocols** (`/v1/responses` including
> streaming, and `/v1/chat/completions` — both measured 200), but the **claude family mostly does not
> answer** (Opus all 500, Sonnet-5 504; only `Claude-Sonnet-4.5` succeeded). That is why the registry
> pins no claude model.

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
node interface/runtime/run_workflow.mjs kernel_workflow/kernel_workflow.js --agent codex \
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
| `GEAK_CODEX_AUTOCONFIG` | `1` | `0` disables provider auto-config (falls back to `codex-home/config.toml`). |
| `GEAK_CODEX_EXTRA_ARGS` | — | Raw `-c key=value` overrides passed to codex; wins over auto-config. |
| `OPENAI_BASE_URL` | — | Any OpenAI-compatible gateway; wins over key-based selection. |

**Thinking level is maxed out by default.** codex's own scale is `none` / `low` / `medium` / `high` /
`xhigh` and has **no `max`** — `xhigh` *is* its top setting, so the runtime emits
`-c model_reasoning_effort=xhigh`. `GEAK_CODEX_EFFORT=max` is still accepted and translates to
`xhigh` (the same mapping as hyperloom's `resolve_codex_reasoning_effort`); any other off-scale value
is rejected up front rather than passed through to codex. To pin it explicitly instead, use
`GEAK_CODEX_EXTRA_ARGS="-c model_reasoning_effort=high"`.

One special case: when `base_url` points at `127.0.0.1` / `localhost` (i.e. the local shim), the
runtime does **not** auto-override it, preserving the `safe_shim` path from `config.toml`.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `401` | key empty or invalid |
| `404` on the model | `GEAK_CODEX_MODEL` not served by that endpoint, or not Responses-API-capable |
| TLS error | a private intranet gateway needs `SSL_CERT_FILE` (neither official OpenAI nor the AMD gateway does) |

To check the runtime itself is not broken — no network, no GPU, no key required:

```bash
node interface/runtime/selftest.mjs      # expect 50/50
```

## What each file is

- `run_workflow.mjs` — runtime core · `config.mjs` + `registry.json` — backend/model configuration
- `backends/` — the backend contract and its generic implementation · `schema.mjs` — structured output
- `responses_shim.mjs` — de-streaming proxy · `setup.sh` — one-shot environment bring-up
- `codex-home/config.toml` — in-repo `CODEX_HOME` (providers: `safe_shim` default, `openai` official)
