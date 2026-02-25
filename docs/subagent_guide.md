# RAG Filter Sub-Agent

This module provides a reusable sub-agent pattern for filtering and summarizing RAG (Retrieval-Augmented Generation) database results.

## Overview

The RAG filter sub-agent processes retrieved chunks from RAG queries by:
1. Evaluating chunk relevance to the original query
2. Removing duplicates and highly similar content
3. Summarizing key points into concise, actionable information

## Usage

### 1. Standalone Usage

```python
from minisweagent.utils.subagent import create_rag_filter_subagent

# Create sub-agent
subagent = create_rag_filter_subagent(
    model_name="claude-opus-4.5",
    api_key="your-api-key",
    enabled=True,
)

# Process RAG results
rag_chunks = """
Chunk 1: Some relevant information...
Chunk 2: More details...
"""

filtered_result = subagent.process(rag_chunks, query="your query")
print(filtered_result)
```

### 2. Integrated in MCP Environment

The sub-agent is automatically integrated into `MCPEnabledEnvironment` and processes results from RAG-based tools:

```python
from minisweagent.mcp_integration.mcp_environment import MCPEnabledEnvironment

# Create environment with sub-agent enabled
env = MCPEnabledEnvironment(
    enable_rag_subagent=True,
    rag_subagent_model="claude-opus-4.5",
    rag_subagent_api_key="your-api-key",
)

# Execute MCP tool - result is automatically filtered by sub-agent
result = env.execute('@amd:query {"topic": "HIP optimization"}')
# Result is now filtered and summarized
```

### 3. Configuration Options

```python
from minisweagent.utils.subagent import SubAgentConfig, RAGFilterSubAgent

config = SubAgentConfig(
    model_name="claude-opus-4.5",      # LLM model to use
    api_key="your-api-key",             # API key (or use env vars)
    system_prompt="custom prompt...",   # Custom system prompt (optional)
    enabled=True,                       # Enable/disable sub-agent
    model_kwargs={},                    # Additional model parameters
)

subagent = RAGFilterSubAgent(config)
```

## Supported MCP Tools

The sub-agent automatically processes results from these MCP tools:
- `query` / `query_knowledge` - Knowledge base queries
- `example` / `get_code_example` - Code example retrieval
- `optimize` / `suggest_optimization` - Optimization suggestions
- `troubleshoot` - Error troubleshooting

## Disabling the Sub-Agent

To disable the sub-agent (pass-through mode):

```python
# Option 1: In configuration
env = MCPEnabledEnvironment(
    enable_rag_subagent=False,
)

# Option 2: In SubAgentConfig
subagent = create_rag_filter_subagent(enabled=False)
```

## Custom System Prompts

You can customize the filtering behavior:

```python
custom_prompt = """
You are a specialized sub-agent for processing GPU programming information.
Focus on extracting:
1. Performance optimization techniques
2. Code examples
3. Common pitfalls and solutions
Output format: Bullet points with clear categories.
"""

subagent = create_rag_filter_subagent(
    system_prompt=custom_prompt,
)
```

## Example Script

See `examples/test_subagent.py` for complete examples:

```bash
cd /home/ethany/three_projects_clean/gpu-agent-workspace/mini-swe-agent
python examples/test_subagent.py
```

## Architecture

```
┌─────────────────┐
│  MCP Tool Call  │
│  (RAG Query)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Raw RAG        │
│  Chunks Result  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RAG Filter     │◄─── SubAgentConfig
│  Sub-Agent      │      - model_name
└────────┬────────┘      - api_key
         │               - enabled
         ▼
┌─────────────────┐
│  Filtered &     │
│  Summarized     │
│  Result         │
└─────────────────┘
```

## Creating Additional Sub-Agents

The pattern is designed to be extensible. To create new sub-agents:

```python
from minisweagent.utils.subagent import SubAgentConfig
from minisweagent.models.amd_llm import AmdLlmModel

class MyCustomSubAgent:
    DEFAULT_SYSTEM_PROMPT = "Your custom prompt..."
    
    def __init__(self, config: SubAgentConfig):
        self.config = config
        self._model = None
    
    @property
    def model(self) -> AmdLlmModel:
        if self._model is None:
            self._model = AmdLlmModel(
                model_name=self.config.model_name,
                api_key=self.config.api_key,
            )
        return self._model
    
    def process(self, input_data: str) -> str:
        if not self.config.enabled:
            return input_data
        
        response = self.model.query([
            {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": input_data}
        ])
        
        return response["content"]
```

## Environment Variables

The sub-agent respects these environment variables:
- `AMD_LLM_API_KEY` - API key for AMD LLM Gateway
- `LLM_GATEWAY_KEY` - Alternative API key variable

## Notes

- The sub-agent uses lazy initialization for efficiency
- Model costs are tracked via `GLOBAL_MODEL_STATS`
- Logging is available via the `minisweagent.utils.subagent` logger
- Sub-agent processing adds latency but improves result quality

