# Developer guide

Documentation for contributors extending GEAK behavior: prompts, tools, and configuration.

| Topic | Description |
|-------|-------------|
| [System & instance prompts](prompts.md) | Where prompts live, how YAML merges, Jinja variables |
| [MCP tools](mcp-tools.md) | Adding a FastMCP server under `mcp_tools/`, discovery, debugging |
| [AVO + Supervisor design](avo_design.md) | Minimally-invasive AVO (Agentic Variation Operators) + supervisor on top of GEAK |
| [AVO usage README](../../src/minisweagent/run/avo/README.md) | How to run `geak-avo`: CLI, config, outputs, troubleshooting |

For branching, CI, and reviews, see [Contribution guidelines](contribution_guidelines.md).

For RAG sub-agent internals, see [RAG sub-agent guide](../subagent_guide.md).
