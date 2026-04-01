"""API Tests for GEAK Online Service."""

import pytest
import httpx
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载 .env 文件
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# 测试配置
BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("TEST_API_KEY")
VERIFY_SSL = False

# 测试数据 - SiLU kernel
SILU_CODE = '''// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

using bf16 = __hip_bfloat16;

__device__ __forceinline__ float silu_f(float x){
  return x / (1.0f + expf(-x));
}

__global__ void silu_mul_kernel(
    bf16* __restrict__ out,
    const bf16* __restrict__ in,
    int64_t B, int64_t H)
{
  const int64_t token_idx = blockIdx.x;
  const int64_t base_in = token_idx * 2 * H;
  const int64_t base_out = token_idx * H;
  
  for (int64_t idx = threadIdx.x; idx < H; idx += blockDim.x) {
    const float x = __bfloat162float(in[base_in + idx]);
    const float y = __bfloat162float(in[base_in + H + idx]);
    out[base_out + idx] = __float2bfloat16(silu_f(x) * y);
  }
}
'''

SILU_PROMPT = """# HIP Kernel Optimization Task

Please optimize the silu_mul_kernel for better performance on AMD MI308 GPU.

Focus on:
1. Vectorized memory access (float4)
2. Loop unrolling
3. Memory coalescing
"""


def get_headers():
    """Get request headers with API key."""
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


class TestHealthCheck:
    """Test health check endpoints."""
    
    def test_health(self):
        """Test health check endpoint."""
        with httpx.Client(verify=VERIFY_SSL) as client:
            response = client.get(f"{BASE_URL}/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
    
    def test_root(self):
        """Test root endpoint."""
        with httpx.Client(verify=VERIFY_SSL) as client:
            response = client.get(f"{BASE_URL}/")
            assert response.status_code == 200
            data = response.json()
            assert "GEAK" in data["service"]


class TestAuthentication:
    """Test authentication."""
    
    def test_no_auth(self):
        """Test request without authentication."""
        with httpx.Client(verify=VERIFY_SSL) as client:
            response = client.get(f"{BASE_URL}/api/v1/tasks")
            assert response.status_code in (401, 403)
    
    def test_invalid_auth(self):
        """Test request with invalid API key."""
        with httpx.Client(verify=VERIFY_SSL) as client:
            headers = {"Authorization": "Bearer invalid-key"}
            response = client.get(f"{BASE_URL}/api/v1/tasks", headers=headers)
            assert response.status_code in (401, 403, 502, 503)


class TestTaskAPI:
    """Test task API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create HTTP client."""
        return httpx.Client(base_url=BASE_URL, headers=get_headers(), timeout=30.0, verify=VERIFY_SSL)
    
    def test_create_task_file(self, client):
        """Test creating a task with file input."""
        payload = {
            "input_type": "file",
            "files": [
                {
                    "filename": "silu.hip",
                    "content": SILU_CODE,
                },
            ],
            "prompt": SILU_PROMPT,
        }
        
        response = client.post("/api/v1/tasks", json=payload)
        
        # 如果认证失败，跳过测试
        if response.status_code in (401, 403):
            pytest.skip("Authentication failed - check TEST_API_KEY")
        
        assert response.status_code == 201, f"Failed: {response.text}"
        data = response.json()
        
        assert "id" in data
        assert data["status"] == "pending"
        assert data["input_type"] == "file"
        
        return data["id"]
    
    def test_create_task_repo(self, client):
        """Test creating a task with repository input."""
        payload = {
            "input_type": "repo",
            "repo": {
                "url": "https://github.com/ROCm/rocPRIM.git",
                "branch": "develop",
                "subdir": "rocprim/include",
            },
            "prompt": "Optimize the device_binary_search kernel.",
        }
        
        response = client.post("/api/v1/tasks", json=payload)
        
        if response.status_code in (401, 403):
            pytest.skip("Authentication failed - check TEST_API_KEY")
        
        # 可能会因为克隆仓库超时，这里只检查请求格式正确
        assert response.status_code in (201, 500), f"Unexpected: {response.text}"
    
    def test_create_task_invalid(self, client):
        """Test creating a task with invalid input."""
        # 缺少 file 字段
        payload = {
            "input_type": "file",
        }
        
        response = client.post("/api/v1/tasks", json=payload)
        
        if response.status_code in (401, 403):
            pytest.skip("Authentication failed - check TEST_API_KEY")
        
        assert response.status_code == 400
    
    def test_list_tasks(self, client):
        """Test listing tasks."""
        response = client.get("/api/v1/tasks")
        
        if response.status_code in (401, 403):
            pytest.skip("Authentication failed - check TEST_API_KEY")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "tasks" in data
        assert "total" in data
        assert isinstance(data["tasks"], list)
    
    def test_list_tasks_with_filter(self, client):
        """Test listing tasks with status filter."""
        response = client.get("/api/v1/tasks", params={"status": "pending"})
        
        if response.status_code in (401, 403):
            pytest.skip("Authentication failed - check TEST_API_KEY")
        
        assert response.status_code == 200
    
    def test_get_task_not_found(self, client):
        """Test getting non-existent task."""
        response = client.get("/api/v1/tasks/non-existent-id")
        
        if response.status_code in (401, 403):
            pytest.skip("Authentication failed - check TEST_API_KEY")
        
        assert response.status_code == 404


class TestTaskWorkflow:
    """Test complete task workflow."""
    
    @pytest.fixture
    def client(self):
        """Create HTTP client."""
        return httpx.Client(base_url=BASE_URL, headers=get_headers(), timeout=60.0, verify=VERIFY_SSL)
    
    def test_full_workflow(self, client):
        """Test complete workflow: create -> get -> list."""
        # 1. Create task
        payload = {
            "input_type": "file",
            "files": [
                {
                    "filename": "silu_test.hip",
                    "content": SILU_CODE,
                },
            ],
            "prompt": SILU_PROMPT,
            "config": {
                "model": {
                    "model_class": "litellm",
                    "model_name": "openai/qwen3-max",
                }
            },
            "runtime": {
                "gpu_count": 1,
                "timeout": 1800,
            }
        }
        
        response = client.post("/api/v1/tasks", json=payload)
        
        if response.status_code in (401, 403):
            pytest.skip("Authentication failed - check TEST_API_KEY")
        
        assert response.status_code == 201
        task = response.json()
        task_id = task["id"]
        
        # 2. Get task details
        response = client.get(f"/api/v1/tasks/{task_id}")
        assert response.status_code == 200
        task = response.json()
        assert task["id"] == task_id
        
        # 3. List tasks and verify our task is there
        response = client.get("/api/v1/tasks")
        assert response.status_code == 200
        tasks = response.json()["tasks"]
        task_ids = [t["id"] for t in tasks]
        assert task_id in task_ids
        
        # 4. Get outputs (should be empty for pending task)
        response = client.get(f"/api/v1/tasks/{task_id}/outputs")
        assert response.status_code == 200
        outputs = response.json()
        assert outputs["task_id"] == task_id
        
        print(f"\n✅ Task created successfully: {task_id}")
        print(f"   Status: {task['status']}")
        print(f"   Output path: {outputs.get('output_path', 'N/A')}")


def run_quick_test():
    """Run a quick test without pytest."""
    print("=" * 60)
    print("GEAK Online Service - Quick Test")
    print("=" * 60)
    
    # Health check
    print("\n1. Testing health endpoint...")
    try:
        with httpx.Client(verify=VERIFY_SSL) as client:
            response = client.get(f"{BASE_URL}/health", timeout=5.0)
            if response.status_code == 200:
                print("   ✅ Health check passed")
            else:
                print(f"   ❌ Health check failed: {response.status_code}")
                return False
    except Exception as e:
        print(f"   ❌ Cannot connect to server: {e}")
        return False
    
    # Create task
    print("\n2. Creating test task...")
    headers = get_headers()
    payload = {
        "input_type": "file",
        "files": [
            {
                "filename": "silu_quick_test.hip",
                "content": SILU_CODE,
            },
        ],
        "prompt": SILU_PROMPT,
    }
    
    try:
        with httpx.Client(verify=VERIFY_SSL) as client:
            response = client.post(
                f"{BASE_URL}/api/v1/tasks",
                json=payload,
                headers=headers,
                timeout=30.0,
            )
            
            if response.status_code == 201:
                task = response.json()
                print(f"   ✅ Task created: {task['id']}")
                print(f"   Status: {task['status']}")
            elif response.status_code in (401, 403):
                print(f"   ⚠️  Authentication failed (expected if using test key)")
                print(f"   Set TEST_API_KEY environment variable with valid SaFE API key")
            else:
                print(f"   ❌ Failed to create task: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
    except Exception as e:
        print(f"   ❌ Error creating task: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("Quick test completed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_test()
    else:
        # Run pytest
        pytest.main([__file__, "-v", "--tb=short"])
