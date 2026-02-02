# MetrixTool Tests

Minimal unit tests for the MetrixTool MCP.

## Running Tests

```bash
# Run all tests
pytest mini_kernel/tests/

# Run with verbose output
pytest mini_kernel/tests/ -v

# Run specific test
pytest mini_kernel/tests/test_metrix.py::TestMetrixTool::test_classify_bottleneck

# Run with coverage
pytest mini_kernel/tests/ --cov=mini_kernel.mcp_tools.metrix
```

## Test Structure

- `test_metrix.py` - Unit tests for MetrixTool class
  - Initialization and setup
  - Kernel filtering logic
  - Metric pattern matching
  - Bottleneck classification
  - Suggestion generation
  - MCP tool definition

## Requirements

```bash
pip install pytest pytest-mock
```
