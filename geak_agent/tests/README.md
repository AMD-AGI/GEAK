# GEAK Agent Tests

Unit tests for GEAK agent MCP tools (MetrixTool) and scripts (resolve_kernel_url).

## Running Tests

```bash
# Run all tests
pytest geak_agent/tests/

# Run with verbose output
pytest geak_agent/tests/ -v

# Run only MetrixTool tests
pytest geak_agent/tests/test_metrix.py -v

# Run only resolve_kernel_url tests
pytest geak_agent/tests/test_resolve_kernel_url.py -v

# Run specific test
pytest geak_agent/tests/test_metrix.py::TestMetrixTool::test_classify_bottleneck

# Run with coverage
pytest geak_agent/tests/ --cov=geak_agent.mcp_tools.metrix --cov=geak_agent.resolve_kernel_url
```

## Test Structure

- `test_metrix.py` - Unit tests for MetrixTool (geak_agent.mcp_tools.metrix)
- `test_resolve_kernel_url.py` - Unit tests for resolve_kernel_url (geak_agent.resolve_kernel_url)

## Requirements

```bash
pip install pytest pytest-mock
```
