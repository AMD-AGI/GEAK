# 快速上手:在你自己的机器上跑 GEAK 可切换后端(codex / cursor)

> 这个分支的 runtime **自包含、零 npm 依赖**(只用 Node 内置模块)。`git pull` 后,只需装好 CLI、设几个环境变量,即可跑。设计/架构见 `BACKEND_DESIGN.md`。

本层目录:`interface/runtime/`。以下命令假设在**仓库根**执行。

---

## A. codex(经 SaFE 网关跑 claude-opus-4-8 / 直连跑 gpt)

### 前提(你自己的环境提供)
1. **能访问 SaFE 网关** `https://global.primus-safe.amd.com`(在别的集群先确认网络可达;不可达则整条不通)。
2. `node` 在 PATH 上(v20+)。
3. codex CLI:`npm i -g @openai/codex`(或你的共享 prefix)。
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
node interface/runtime/selftest.mjs      # 期望 36/36
```

## 各文件是什么
- `run_workflow.mjs` runtime 核心 · `config.mjs`+`registry.json` 后端/模型配置
- `backends/` 后端契约与 generic 实现 · `schema.mjs` 结构化输出
- `responses_shim.mjs` codex+claude 的 de-stream 代理 · `setup.sh` 一键起环境
- `codex-home/config.toml` 仓库内 CODEX_HOME(provider 指向 shim)
