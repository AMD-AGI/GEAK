# Scripts

## run-docker.sh

Build and run the GEAK-agent Docker container. Run tests and examples **inside** the container. The repo is mounted at `/workspace`, so edits on the host are visible without rebuilding.

Requires `AMD_LLM_API_KEY`.

```bash
./scripts/run-docker.sh          # build if needed, start container, exec into bash
./scripts/run-docker.sh --rebuild # rebuild image (no cache) and start
```

Inside the container you land in `/workspace`. Run e.g. `pytest tests/ -v` or `python examples/resolve_kernel_url/resolve_kernel_url.py <spec>`.

