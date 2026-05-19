ARG BASE_IMAGE=lmsysorg/sglang:v0.5.11-rocm720-mi35x
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y git make && rm -rf /var/lib/apt/lists/*

# Copy installable sources first (cache-friendly — changes here rebuild pip install)
WORKDIR /workspace
COPY pyproject.toml README.md LICENSE.md Makefile ./
COPY src/ src/
COPY mcp_tools/ mcp_tools/

RUN make install

# Verify core imports
RUN python3 -c "from profiler_mcp.server import profile_kernel; from metrix import Metrix; print('Core imports verified')"

# Runtime assets (not needed for install; changes only rebuild cheap COPY layers)
COPY subagents/ subagents/
COPY skills/ skills/
COPY docs/ docs/
COPY scripts/ scripts/
COPY entrypoint.sh ./

RUN chmod +x /workspace/entrypoint.sh
ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["tail", "-f", "/dev/null"]
