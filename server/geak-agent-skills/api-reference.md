# GEAK API Reference

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `PUT` | `/api/v1/config/model` | Set model config |
| `GET` | `/api/v1/config/model` | Get model config |
| `DELETE` | `/api/v1/config/model` | Delete model config |
| `POST` | `/api/v1/tasks` | Create task |
| `GET` | `/api/v1/tasks` | List tasks |
| `GET` | `/api/v1/tasks/{id}` | Get task details |
| `POST` | `/api/v1/tasks/{id}/submit` | Submit task |
| `POST` | `/api/v1/tasks/{id}/cancel` | Cancel task |
| `GET` | `/api/v1/tasks/{id}/outputs` | List output files |
| `GET` | `/api/v1/tasks/{id}/download?path=` | Download file |

## Request / Response Details

### PUT /api/v1/config/model

Set LLM model configuration (one-time setup per user).

```json
{
  "model_class": "litellm",
  "model_name": "openai/claude-opus-4.5",
  "model_kwargs": {
    "api_base": "http://litellm-service:4000/v1",
    "api_key": "sk-xxx",
    "temperature": 0.0,
    "max_tokens": 16000
  }
}
```

### POST /api/v1/tasks — File Input

Upload HIP kernel file(s) for optimization.

```json
{
  "input_type": "file",
  "files": [
    {"filename": "silu.hip", "content": "...kernel source..."}
  ],
  "prompt": "Optimize for MI300X",
  "config": {"agent": {"step_limit": 20}},
  "runtime": {"gpu_count": 1}
}
```

### POST /api/v1/tasks — Repo Input

Optimize kernels from a git repository.

```json
{
  "input_type": "repo",
  "repo": {
    "url": "https://github.com/org/hip-kernels.git",
    "branch": "main"
  },
  "prompt": "Optimize all HIP kernels. Run tests to verify.",
  "config": {"agent": {"step_limit": 30}}
}
```

### GET /api/v1/tasks/{id}

Returns task details including status.

**Task statuses:** `pending`, `running`, `completed`, `failed`, `cancelled`

### GET /api/v1/tasks/{id}/outputs

Returns list of output files:

```json
{
  "files": [
    {"path": "silu.hip", "size": 5678},
    {"path": "execution.log", "size": 12345}
  ]
}
```

### GET /api/v1/tasks/{id}/download?path=FILE

Download a specific output file. Returns binary content.

## Authentication

All requests require `Authorization: Bearer <GEAK_API_KEY>` header.

## MCP Mode

Alternative to REST API. Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "geak": {
      "url": "https://your-geak-server.com/mcp",
      "headers": {
        "Authorization": "Bearer ak-your-api-key"
      }
    }
  }
}
```

Then use GEAK tools directly in Cursor chat — no script needed.
