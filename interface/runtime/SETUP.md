# 快速上手:在你自己的机器上跑 GEAK 可切换后端(codex / cursor)

> 这个分支的 runtime **自包含、零 npm 依赖**(只用 Node 内置模块)。`git pull` 后,只需装好 CLI、设几个环境变量,即可跑。设计/架构见 `BACKEND_DESIGN.md`。

本层目录:`interface/runtime/`。以下命令假设在**仓库根**执行。

---

## A. codex(经 SaFE 网关跑 claude-opus-4-8 / 直连跑 gpt)

### 前提(你自己的环境提供)
1. **能访问 SaFE 网关** `https://global.primus-safe.amd.com`(在别的集群先确认网络可达;不可达则整条不通)。
2. `node` 在 PATH 上(v20+)。
3. codex CLI:`npm i -g @openai/codex@0.146.1`(0.147 与网关不兼容;安装细节见 A2 "前置")。
4. **API key**:SaFE `ak-` key。
5. **CA 证书**:连网关 TLS 用的 CA bundle(你机器本地路径)。

### 步骤
```bash
# 1) 设置你的密钥与 CA(不要提交到 git)
export OPENAI_API_KEY="ak-....."          # 或已有 ANTHROPIC_API_KEY 也行
export SSL_CERT_FILE="/path/to/your/ca.pem"

# 2) source 启动脚本:起 shim、设好 CODEX_HOME(仓库内)、qwen 认证等
.  interface/runtime/setup.sh

# 3) 跑一个 workflow
node interface/runtime/run_workflow.mjs <workflow.js> --profile codex-opus48   # claude-opus-4-8(经 shim)
# 或
node interface/runtime/run_workflow.mjs <workflow.js> --profile codex-gpt54    # gpt-5.4(可直连)
```

### 换成别的网关?
- 改 `interface/runtime/registry.json` 里 `models[].base_url`;
- 并给 shim 设 `export GEAK_GW_BASE="https://你的网关/.../v1"` 后再 `. setup.sh`;
- `codex-home/config.toml` 里 `safe_direct` 的 base_url 也相应改(仅 gpt 直连用)。

### 排错
- `codex-opus48` 报连接错 → shim 没起:看 `interface/runtime/shim.log`,或重跑 `. setup.sh`。
- 401 → `OPENAI_API_KEY` 空或无效。
- TLS 错 → `SSL_CERT_FILE` 没设或 CA 不对。
- 端口冲突 → `export SHIM_PORT=8899` 再 `. setup.sh`,并把 `codex-home/config.toml` 的 `safe_shim` base_url 端口同步改掉。

---

## A2. codex 自动按 key 选网关(OpenAI 官方 / AMD / SaFE)

codex 的 provider **自动配置**,无需手写 config.toml、无需 `setup.sh`、无需选 provider。
runtime 在启动 codex 时按下面顺序解析(第一个命中),生成 `-c model_providers.geak_auto.*` 覆盖:

1. **显式 `OPENAI_BASE_URL`** → 直接用它(任意 OpenAI 兼容网关)。
2. 否则**按"哪个 key 非空"自动选**:`AMDKEY`→AMD、`SAFE_API_KEY`→SaFE、`OPENAI_API_KEY`→OpenAI 官方。

### 前置:安装 codex CLI(不要假设已装好)
```bash
# 1) Node.js v20+(codex 依赖)
node -v        # 无 node 或 <20:先装 Node 20+(nvm / 系统包管理器 / nodejs.org)

# 2) 安装 codex CLI —— 务必 pin 0.146.1(0.147 与网关不兼容)
npm i -g @openai/codex@0.146.1
#   若没有 /usr/local 写权限,用用户级 prefix:
#   npm config set prefix "$HOME/.npm-global"
#   export PATH="$HOME/.npm-global/bin:$PATH"      # 建议写进 ~/.bashrc
#   npm i -g @openai/codex@0.146.1

# 3) 验证
codex --version        # 期望 0.146.1
```

### 第 1 步:选 provider —— 设对应的 key(三选一)
```bash
# 官方 OpenAI(最简单:公网 CA,无需 shim / SSL_CERT_FILE / config.toml)
export OPENAI_API_KEY="sk-....."
export GEAK_CODEX_MODEL="<你的-OpenAI-模型id>"   # 必填:官方模型 id(如 gpt-5.x);勿用 AMD 的 gpt-5.6-sol(官方端点 404)

# 或 —— AMD 网关(自动加 Ocp-Apim header + llm-api.amd.com)
# export AMDKEY="<32位hex 订阅 key>"
# export SSL_CERT_FILE="/path/to/amd-ca.pem"      # 内网 CA
# export GEAK_CODEX_MODEL="gpt-5.6-sol"           # 例:AMD 网关上的模型 id

# 或 —— SaFE 网关(gpt 直连)
# export SAFE_API_KEY="ak-....."
# export SSL_CERT_FILE="/path/to/safe-ca.pem"     # 内网 CA
# export GEAK_CODEX_MODEL="gpt-5.6"               # 例:SaFE 网关上的模型 id
```
> claude-via-SaFE 需要 de-stream shim(见 A 节:`. setup.sh` + `--profile codex-opus48`);上面 SaFE 的自动选择走的是 gpt 直连。

### 第 2 步:运行(设 `GEAK_AGENT_BACKEND=codex`,再选 e2e 或单核)
```bash
export GEAK_AGENT_BACKEND=codex

# e2e(整模型吞吐):用 handoff.json 描述任务(字段/示例见 interface/run_e2e.md)
python3 interface/run_e2e.py <handoff.json> <result.json>

# 单核:
node interface/runtime/run_workflow.mjs kernel_workflow/kernel_workflow.js --agent codex \
  --args '{"kernel_path":"/abs/kernel","workflow_dir":"'"$PWD"'/kernel_workflow","budget":6}'
```

### 覆盖 / 关闭 / 排错
- **thinking level(reasoning effort)默认 `max`**;改用 `GEAK_CODEX_EFFORT`(`low`/`medium`/`high`/`xhigh`/`max`),如 `export GEAK_CODEX_EFFORT=xhigh`;或用 `GEAK_CODEX_EXTRA_ARGS="-c model_reasoning_effort=xhigh"` 显式钉(优先)。
- 任意网关:`export OPENAI_BASE_URL=https://你的网关/v1`(+ 对应 key)——优先于 key 自动选。
- 关闭自动配置:`export GEAK_CODEX_AUTOCONFIG=0`(回落到 `codex-home/config.toml`)。
- 手动指定 provider:`export GEAK_CODEX_EXTRA_ARGS="-c model_provider=safe_direct"`(优先于自动)。
- base_url 指向 `127.0.0.1`/`localhost`(即本地 shim)时**不会**自动覆盖,保留 config.toml 的 `safe_shim` 路径。
- 401 → key 空/无效;404 model → `GEAK_CODEX_MODEL` 不可用或不支持 Responses API;TLS 错 → 内网网关需 `SSL_CERT_FILE`(官方 OpenAI 不需要)。

---

## B. cursor(注意:走 Cursor 私有云,**不经** SaFE 网关)

cursor 与 shim/网关无关,不需要 `setup.sh`。

### 前提
1. `cursor-agent` CLI 装好。
2. **你自己的 Cursor Team 账号**:`cursor-agent login`(登录态存 `~/.config/cursor/auth.json`)。别人的账号带不过去;需要你有对应 Team 的访问权。或设 `export CURSOR_API_KEY=...`。

### 步骤
```bash
cursor-agent login          # 首次
# 可选:选模型(Cursor 侧 id,如 composer-2.5 / sonnet-4-thinking)
export GEAK_CURSOR_MODEL="composer-2.5"
node interface/runtime/run_workflow.mjs <workflow.js> --profile cursor
```

> 提醒:cursor 的请求和代码会**外发到 Cursor 云**,且模型是 Cursor 侧模型 —— 因此它**不能**和 codex/qwen 做"同网关同模型"的严格对照。

---

## 验证 runtime 本身没坏(可选,不需网络/GPU)
```bash
node interface/runtime/selftest.mjs      # 期望 50/50
```

## 各文件是什么
- `run_workflow.mjs` runtime 核心 · `config.mjs`+`registry.json` 后端/模型配置
- `backends/` 后端契约与 generic 实现 · `schema.mjs` 结构化输出
- `responses_shim.mjs` codex+claude 的 de-stream 代理 · `setup.sh` 一键起环境
- `codex-home/config.toml` 仓库内 CODEX_HOME(providers:`safe_shim` 默认 / `safe_direct` / `openai` 官方)
