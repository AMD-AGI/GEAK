---
myst:
    html_meta:
        "description": "Use the GEAK RAG filter sub-agent to evaluate, deduplicate, and summarize retrieval results from MCP tools. Covers configuration, integration, custom prompts, and extension patterns."
        "keywords": "GEAK, RAG, sub-agent, filter, MCP, retrieval-augmented generation, MCPEnabledEnvironment"
---

# RAG filter sub-agent

This module provides a reusable sub-agent pattern for filtering and summarizing RAG (Retrieval-Augmented Generation) database results.

## Overview


The RAG filter sub-agent processes retrieved chunks from RAG queries by:
1. Evaluating chunk relevance to the original query
2. Removing duplicates and highly similar content
3. Summarizing key points into concise, actionable information

## Usage

The sub-agent can be used standalone, integrated into an MCP environment, or configured with a custom system prompt. The following examples cover each approach.

### Standalone usage

```python
from minisweagent.utils.subagent import create_rag_filter_subagent

# Create sub-agent
subagent = create_rag_filter_subagent(
    model_name="claude-opus-4.6",
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

### Integrated in MCP environment

The sub-agent is automatically integrated into `MCPEnabledEnvironment` and processes results from RAG-based tools:

```python
from minisweagent.mcp_integration.mcp_environment import MCPEnabledEnvironment

# Create environment with sub-agent enabled
env = MCPEnabledEnvironment(
    enable_rag_subagent=True,
    rag_subagent_model="claude-opus-4.6",
    rag_subagent_api_key="your-api-key",
)

# Execute MCP tool - result is automatically filtered by sub-agent
result = env.execute('@amd:query {"topic": "HIP optimization"}')
# Result is now filtered and summarized
```

### Configuration options

```python
from minisweagent.utils.subagent import SubAgentConfig, RAGFilterSubAgent

config = SubAgentConfig(
    model_name="claude-opus-4.6",      # LLM model to use
    api_key="your-api-key",             # API key (or use env vars)
    system_prompt="custom prompt...",   # Custom system prompt (optional)
    enabled=True,                       # Enable/disable sub-agent
    model_kwargs={},                    # Additional model parameters
)

subagent = RAGFilterSubAgent(config)
```

## Supported MCP tools

The sub-agent automatically processes results from these MCP tools:
- `query` / `query_knowledge` - Knowledge base queries
- `example` / `get_code_example` - Code example retrieval
- `optimize` / `suggest_optimization` - Optimization suggestions
- `troubleshoot` - Error troubleshooting

## Disabling the sub-agent

To disable the sub-agent (pass-through mode):

```python
# Option 1: In configuration
env = MCPEnabledEnvironment(
    enable_rag_subagent=False,
)

# Option 2: In SubAgentConfig
subagent = create_rag_filter_subagent(enabled=False)
```

## Custom system prompts

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

## Example script

See `examples/test_subagent.py` for complete examples:

```bash
python examples/test_subagent.py
```

## Architecture

The diagram below shows how a RAG tool call flows through the sub-agent before returning a result to the optimization agent.

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

## Creating additional sub-agents

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

## Environment variables

The sub-agent respects these environment variables:
- `AMD_LLM_API_KEY` - API key for AMD LLM Gateway
- `LLM_GATEWAY_KEY` - Alternative API key variable

## Notes

Keep the following in mind when deploying or extending the sub-agent.

- The sub-agent uses lazy initialization for efficiency
- Model costs are tracked via `GLOBAL_MODEL_STATS`
- Logging is available via the `minisweagent.utils.subagent` logger
- Sub-agent processing adds latency but improves result quality

## Related topics

- [GEAK agent loop](../docs/conceptual/geak-pipeline.md) — how the knowledge base and MCP tools fit into the optimization pipeline.
- [API reference](../docs/reference/api-reference.md) — environment variables for configuring the RAG sub-agent.
- [Model configuration](model-config.md) — configure the LLM backend used by the sub-agent.

