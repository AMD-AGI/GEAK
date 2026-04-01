#!/usr/bin/env python3
"""MCP Server Tests for GEAK Online Service.

Tests the MCP HTTP endpoints and tool functionality.
"""

import httpx
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# 测试配置
BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("TEST_API_KEY")

MCP_BASE = f"{BASE_URL}/mcp"
MCP_ENDPOINT = f"{MCP_BASE}/"  # JSON-RPC endpoint
MCP_TOOLS = f"{MCP_BASE}/tools"  # Tools endpoint


def print_step(step: str):
    """Print step header."""
    print(f"\n{'=' * 60}")
    print(step)
    print("=" * 60)


def print_result(success: bool, message: str):
    """Print result."""
    icon = "✅" if success else "❌"
    print(f"   {icon} {message}")


def test_mcp_discovery():
    """Test MCP server discovery endpoints."""
    print_step("1. MCP Discovery")
    
    with httpx.Client(timeout=30.0) as client:
        # Test info endpoint
        print("\n1.1 GET /mcp/info (server info)")
        response = client.get(f"{MCP_BASE}/info")
        assert response.status_code == 200
        data = response.json()
        print_result(True, f"Server: {data['name']} v{data['version']}")
        
        # Test tools list
        print("\n1.2 GET /mcp/tools (list tools)")
        response = client.get(MCP_TOOLS)
        assert response.status_code == 200
        data = response.json()
        tools = data["tools"]
        print_result(True, f"Found {len(tools)} tools")
        for tool in tools:
            print(f"      - {tool['name']}")


def test_mcp_jsonrpc_initialize():
    """Test MCP JSON-RPC initialize."""
    print_step("2. MCP JSON-RPC Initialize")
    
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            MCP_ENDPOINT,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize"
            }
        )
        assert response.status_code == 200
        data = response.json()
        
        assert data["result"]["protocolVersion"] == "2024-11-05"
        print_result(True, f"Protocol: {data['result']['protocolVersion']}")
        print_result(True, f"Server: {data['result']['serverInfo']['name']}")


def test_mcp_jsonrpc_tools_list():
    """Test MCP JSON-RPC tools/list."""
    print_step("3. MCP JSON-RPC tools/list")
    
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            MCP_ENDPOINT,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list"
            }
        )
        assert response.status_code == 200
        data = response.json()
        
        tools = data["result"]["tools"]
        print_result(True, f"Found {len(tools)} tools")


def test_mcp_jsonrpc_tools_call():
    """Test MCP JSON-RPC tools/call."""
    print_step("4. MCP JSON-RPC tools/call")
    
    if not API_KEY:
        print_result(False, "TEST_API_KEY not set")
        return
    
    with httpx.Client(timeout=30.0) as client:
        # Test without auth (should fail)
        print("\n4.1 Call without API key (should fail)")
        response = client.post(
            MCP_ENDPOINT,
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "geak_list_tasks",
                    "arguments": {}
                }
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("error") is not None
        print_result(True, f"Error: {data['error']['message'][:50]}...")
        
        # Test with auth (should succeed)
        print("\n4.2 Call with API key (Authorization header)")
        response = client.post(
            MCP_ENDPOINT,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "geak_list_tasks",
                    "arguments": {"limit": 2}
                }
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("result") is not None
        content = json.loads(data["result"]["content"][0]["text"])
        print_result(True, f"Found {len(content.get('tasks', []))} tasks")
        
        # Test with X-API-Key header
        print("\n4.3 Call with X-API-Key header")
        response = client.post(
            MCP_ENDPOINT,
            headers={"X-API-Key": API_KEY},
            json={
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "geak_get_model_config",
                    "arguments": {}
                }
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("result") is not None
        print_result(True, "Model config retrieved")


def test_mcp_rest_tools():
    """Test MCP REST-style tool endpoints."""
    print_step("5. MCP REST-style Tool Endpoints")
    
    if not API_KEY:
        print_result(False, "TEST_API_KEY not set")
        return
    
    with httpx.Client(timeout=30.0) as client:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        
        # Test list tasks
        print("\n5.1 POST /mcp/tools/geak_list_tasks")
        response = client.post(
            f"{MCP_TOOLS}/geak_list_tasks",
            headers=headers,
            json={"limit": 3}
        )
        assert response.status_code == 200
        data = response.json()
        print_result(True, f"Total tasks: {data.get('total', len(data.get('tasks', [])))}")
        
        # Test get model config
        print("\n5.2 POST /mcp/tools/geak_get_model_config")
        response = client.post(
            f"{MCP_TOOLS}/geak_get_model_config",
            headers=headers,
        )
        if response.status_code == 200:
            data = response.json()
            if "error" in data:
                print_result(True, f"No config set: {data.get('error', 'Not found')}")
            else:
                print_result(True, f"Config: {data.get('config', {}).get('model_name', 'unknown')}")
        else:
            print_result(True, "No config set (404)")


def test_mcp_model_config_workflow():
    """Test MCP model config workflow."""
    print_step("6. MCP Model Config Workflow")
    
    if not API_KEY:
        print_result(False, "TEST_API_KEY not set")
        return
    
    with httpx.Client(timeout=30.0) as client:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        
        # Set model config
        print("\n6.1 Set model config")
        response = client.post(
            f"{MCP_TOOLS}/geak_set_model_config",
            headers=headers,
            json={
                "model_class": "litellm",
                "model_name": "openai/gpt-4o",
                "model_kwargs": {
                    "api_base": "http://litellm.example.com/v1",
                    "api_key": "test-key",
                    "max_tokens": 4096
                }
            }
        )
        assert response.status_code == 200
        data = response.json()
        print_result(True, f"Config saved for user: {data.get('user_id', 'unknown')[:8]}...")
        
        # Get model config
        print("\n6.2 Get model config")
        response = client.post(
            f"{MCP_TOOLS}/geak_get_model_config",
            headers=headers,
        )
        assert response.status_code == 200
        data = response.json()
        config = data.get("config", {})
        print_result(True, f"Model: {config.get('model_name')}")
        
        # Delete model config
        print("\n6.3 Delete model config")
        response = client.post(
            f"{MCP_TOOLS}/geak_delete_model_config",
            headers=headers,
        )
        assert response.status_code == 200
        data = response.json()
        print_result(True, f"Config deleted: {data.get('success', False)}")


def test_mcp_task_workflow():
    """Test MCP task creation workflow."""
    print_step("7. MCP Task Workflow")
    
    if not API_KEY:
        print_result(False, "TEST_API_KEY not set")
        return
    
    with httpx.Client(timeout=60.0) as client:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        
        # Create a simple task with multiple files
        print("\n7.1 Create task")
        response = client.post(
            f"{MCP_TOOLS}/geak_create_task",
            headers=headers,
            json={
                "input_type": "file",
                "files": [
                    {"filename": "test_mcp.hip", "content": "// Test HIP kernel\n__global__ void test() {}"},
                ],
                "prompt": "Optimize this kernel",
                "step_limit": 3
            }
        )
        assert response.status_code == 200
        data = response.json()
        
        if "error" in data:
            print_result(False, f"Error: {data['error']}")
            return
        
        task_id = data.get("id")
        print_result(True, f"Task created: {task_id}")
        
        # Get task details
        print("\n7.2 Get task details")
        response = client.post(
            f"{MCP_TOOLS}/geak_get_task",
            headers=headers,
            json={"task_id": task_id}
        )
        assert response.status_code == 200
        data = response.json()
        print_result(True, f"Status: {data.get('status')}")
        
        # Get outputs (will be empty for pending task)
        print("\n7.3 Get task outputs")
        response = client.post(
            f"{MCP_TOOLS}/geak_get_outputs",
            headers=headers,
            json={"task_id": task_id}
        )
        assert response.status_code == 200
        data = response.json()
        files = data.get("files", [])
        print_result(True, f"Output files: {len(files)}")


def run_all_tests():
    """Run all MCP tests."""
    print("=" * 60)
    print("GEAK MCP Server Tests")
    print("=" * 60)
    print(f"Base URL: {MCP_BASE}")
    print(f"API Key: {API_KEY[:20] if API_KEY else 'Not set'}...")
    
    try:
        test_mcp_discovery()
        test_mcp_jsonrpc_initialize()
        test_mcp_jsonrpc_tools_list()
        test_mcp_jsonrpc_tools_call()
        test_mcp_rest_tools()
        test_mcp_model_config_workflow()
        test_mcp_task_workflow()
        
        print("\n" + "=" * 60)
        print("All MCP Tests Passed! ✅")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
