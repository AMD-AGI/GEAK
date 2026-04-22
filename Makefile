.PHONY: install install-dev install-full index

# Production install (used by Dockerfile): core + MCP tools
install:
	pip install '.[langchain]'
	pip install mcp_tools/automated-test-discovery/ \
	            mcp_tools/metrix-mcp/ \
	            mcp_tools/profiler-mcp/ \
	            mcp_tools/cross-session-memory-mcp/ \
	            mcp_tools/rag-mcp/

# Full install: core + MCP tools + dev + swe-rex
install-full:
	pip install '.[dev,langchain]' 'swe-rex>=1.4.0'
	pip install mcp_tools/automated-test-discovery/ \
	            mcp_tools/metrix-mcp/ \
	            mcp_tools/profiler-mcp/ \
	            mcp_tools/cross-session-memory-mcp/ \
	            mcp_tools/rag-mcp/

# Editable full install (for developers)
install-dev:
	pip install -e '.[dev,langchain]' 'swe-rex>=1.4.0'
	pip install -e mcp_tools/automated-test-discovery/ \
	            -e mcp_tools/metrix-mcp/ \
	            -e mcp_tools/profiler-mcp/ \
	            -e mcp_tools/cross-session-memory-mcp/ \
	            -e mcp_tools/rag-mcp/

# Build the RAG semantic-search index (opt-in; mini.py auto-builds on first
# use when tools.rag is enabled, so explicit invocation is only needed for
# offline pre-baking).
index:
	python scripts/build_index.py --force