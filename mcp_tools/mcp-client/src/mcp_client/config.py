"""
MCP server configuration registry.
"""

from pathlib import Path
from typing import Dict, Any, Optional

# Get the msa root directory (mcp-client is in mcp_tools/mcp-client)
MSA_ROOT = Path(__file__).parent.parent.parent.parent.parent

# MCP server configurations
MCP_SERVERS = {
    "openevolve-mcp": {
        "command": ["python3", "-m", "openevolve_mcp.server"],
        "cwd": str(MSA_ROOT / "mcp_tools" / "openevolve-mcp"),
        "env": {
            "PYTHONPATH": str(MSA_ROOT / "mcp_tools" / "openevolve-mcp" / "src")
        },
        "tools": ["optimize_kernel"],
        "description": "OpenEvolve optimizer - LLM-guided kernel evolution"
    },
    "kernel-profiler": {
        "command": ["python3", "-m", "kernel_profiler.server"],
        "cwd": str(MSA_ROOT / "mcp_tools" / "kernel-profiler"),
        "env": {
            "PYTHONPATH": str(MSA_ROOT / "mcp_tools" / "kernel-profiler" / "src")
        },
        "tools": ["profile_kernel", "profile_kernel_with_rocprof"],
        "description": "GPU kernel profiler using rocprof-compute"
    },
    "kernel-evolve": {
        "command": ["python3", "-m", "kernel_evolve.server"],
        "cwd": str(MSA_ROOT / "mcp_tools" / "kernel-evolve"),
        "env": {
            "PYTHONPATH": str(MSA_ROOT / "mcp_tools" / "kernel-evolve" / "src")
        },
        "tools": ["mutate_kernel", "crossover_kernels"],
        "description": "LLM-based kernel mutation and crossover"
    },
    "kernel-ercs": {
        "command": ["python3", "-m", "kernel_ercs.server"],
        "cwd": str(MSA_ROOT / "mcp_tools" / "kernel-ercs"),
        "env": {
            "PYTHONPATH": str(MSA_ROOT / "mcp_tools" / "kernel-ercs" / "src")
        },
        "tools": ["evaluate_kernel", "reflect_on_kernel", "check_compatibility", "extract_specs"],
        "description": "Kernel evaluation, reflection, and compatibility checking"
    },
    "automated-test-discovery": {
        "command": ["python3", "-m", "automated_test_discovery.server"],
        "cwd": str(MSA_ROOT / "mcp_tools" / "automated-test-discovery"),
        "env": {
            "PYTHONPATH": str(MSA_ROOT / "mcp_tools" / "automated-test-discovery" / "src")
        },
        "tools": ["discover_tests"],
        "description": "Automated test and benchmark discovery for kernels"
    }
}


def get_server_config(server_name: str) -> Dict[str, Any]:
    """
    Get configuration for an MCP server.
    
    Args:
        server_name: Name of the MCP server
        
    Returns:
        Server configuration dict
        
    Raises:
        KeyError: If server not found
    """
    if server_name not in MCP_SERVERS:
        available = ", ".join(MCP_SERVERS.keys())
        raise KeyError(
            f"Unknown MCP server: {server_name}. "
            f"Available servers: {available}"
        )
    
    return MCP_SERVERS[server_name]


def list_servers() -> Dict[str, str]:
    """
    List all available MCP servers with descriptions.
    
    Returns:
        Dict mapping server names to descriptions
    """
    return {
        name: config.get("description", "No description")
        for name, config in MCP_SERVERS.items()
    }
