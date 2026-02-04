# MetrixTool Tests

Minimal unit tests for the MetrixTool MCP.

## Running Tests

```bash
# Run all tests
pytest geak_agent/tests/

# Run with verbose output
pytest geak_agent/tests/ -v

# Run specific test
pytest geak_agent/tests/test_metrix.py::TestMetrixTool::test_classify_bottleneck

# Run with coverage
pytest geak_agent/tests/ --cov=geak_agent.mcp_tools.metrix
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
