.PHONY: install install-dev

# Production install (used by Dockerfile)
install:
	pip install .
	pip install mcp_tools/automated-test-discovery/ \
	            mcp_tools/metrix-mcp/ \
	            mcp_tools/profiler-mcp/

# Editable install (used by developers / GEAK_EDITABLE=1)
install-dev:
	pip install -e .
	pip install -e mcp_tools/automated-test-discovery/ \
	            -e mcp_tools/metrix-mcp/ \
	            -e mcp_tools/profiler-mcp/
