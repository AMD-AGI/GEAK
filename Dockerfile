FROM lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x

# Install git if not present
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy and install GEAK-agent
WORKDIR /workspace
COPY . .
RUN pip install -e .

# Install MCP dependencies
RUN pip install fastmcp

# Install Metrix from AMD intellikit (required by metrix-mcp)
RUN git clone https://github.com/AMDResearch/intellikit.git /tmp/intellikit \
    && cd /tmp/intellikit/metrix \
    && pip install -e . \
    && cd /workspace

# Install MCP tools (metrix-mcp needs metrix installed first)
RUN pip install -e mcp_tools/mcp-client/ && \
    pip install -e mcp_tools/metrix-mcp/ && \
    pip install -e mcp_tools/automated-test-discovery/ && \
    pip install -e mcp_tools/profiler-mcp/

# Verify core imports (metrix is ROCm runtime dependency)
RUN python3 -c "from mcp_client import MCPClient; from profiler_mcp.server import profile_kernel; print('✅ Core imports verified')"

# Verify metrix is available
RUN python3 -c "from metrix import Metrix; print('✅ Metrix installed')" || echo "⚠️  Metrix not available (will be needed for profiling)"

# Add entrypoint script for runtime configuration and health checks
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["tail", "-f", "/dev/null"]
