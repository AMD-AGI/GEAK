#!/usr/bin/env python3
"""Full Flow Tests for GEAK Online Service.

Tests the complete workflow:
1. User saves default model configuration
2. User creates a task (uses saved config)
3. Submit task for execution
4. Monitor task status
5. Get outputs and download files
"""

import httpx
import os
import sys
import time
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
API_BASE = os.getenv("TEST_LLM_BASE", "http://litellm-service.primus-safe.svc.cluster.local:4000/v1")
LLM_API_KEY = os.getenv("TEST_LLM_KEY", "")
VERIFY_SSL = False

# 读取测试文件
GEAK_V3_DIR = Path(__file__).parent.parent.parent / "geak_v3"


def load_file(path: str) -> str:
    """Load file content."""
    file_path = GEAK_V3_DIR / path
    if file_path.exists():
        return file_path.read_text()
    raise FileNotFoundError(f"File not found: {file_path}")


def get_headers():
    """Get request headers with API key."""
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def print_step(step_num: int, title: str):
    """Print step header."""
    print(f"\n{'=' * 60}")
    print(f"Step {step_num}: {title}")
    print("=" * 60)


def print_result(success: bool, message: str):
    """Print result."""
    icon = "✅" if success else "❌"
    print(f"   {icon} {message}")


# =============================================================================
# Step 1: User Model Configuration
# =============================================================================

def test_get_user_config(client: httpx.Client) -> dict | None:
    """Get user's saved model configuration."""
    response = client.get("/api/v1/config/model")
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        return None
    else:
        print_result(False, f"Failed to get config: {response.status_code}")
        return None


def test_save_user_config(client: httpx.Client, config: dict) -> dict | None:
    """Save user's default model configuration."""
    response = client.put("/api/v1/config/model", json=config)
    if response.status_code == 200:
        return response.json()
    else:
        print_result(False, f"Failed to save config: {response.status_code} - {response.text}")
        return None


def test_delete_user_config(client: httpx.Client) -> bool:
    """Delete user's model configuration."""
    response = client.delete("/api/v1/config/model")
    return response.status_code == 204


# =============================================================================
# Step 2: Task Creation
# =============================================================================

def test_create_task(client: httpx.Client, payload: dict) -> dict | None:
    """Create a new task."""
    response = client.post("/api/v1/tasks", json=payload)
    if response.status_code == 201:
        return response.json()
    else:
        print_result(False, f"Failed to create task: {response.status_code} - {response.text[:200]}")
        return None


def test_get_task(client: httpx.Client, task_id: str) -> dict | None:
    """Get task details."""
    response = client.get(f"/api/v1/tasks/{task_id}")
    if response.status_code == 200:
        return response.json()
    else:
        print_result(False, f"Failed to get task: {response.status_code}")
        return None


# =============================================================================
# Step 3: Task Submission
# =============================================================================

def test_submit_task(client: httpx.Client, task_id: str) -> dict | None:
    """Submit task for execution."""
    response = client.post(f"/api/v1/tasks/{task_id}/submit")
    if response.status_code == 200:
        return response.json()
    else:
        print_result(False, f"Failed to submit task: {response.status_code} - {response.text[:200]}")
        return None


# =============================================================================
# Step 4: Monitor Execution
# =============================================================================

def monitor_task_execution(task_id: str, user_id: str, max_wait: int = 300, interval: int = 10) -> bool:
    """Monitor task execution until completion or timeout.
    
    Uses both log file monitoring and API status polling to detect
    early failures (e.g. workload crashes before execution.log is created).
    """
    nfs_base = os.getenv("NFS_BASE_PATH", "/shared_nfs/geak")
    output_dir = Path(f"{nfs_base}/tasks/{user_id}/{task_id}/output")
    log_path = output_dir / "execution.log"
    
    start_time = time.time()
    last_step = 0
    
    while time.time() - start_time < max_wait:
        elapsed = int(time.time() - start_time)
        
        if log_path.exists():
            content = log_path.read_text()
            
            if "completed successfully" in content:
                print(f"   ✅ Task completed after {elapsed} seconds")
                return True
            
            if "FAILED" in content or "Error" in content.split('\n')[-5:]:
                print(f"   ⚠️ Task may have failed after {elapsed} seconds")
                return True
            
            import re
            steps = re.findall(r'step (\d+)', content.lower())
            if steps:
                current_step = int(steps[-1])
                if current_step > last_step:
                    last_step = current_step
                    print(f"   [{elapsed}s] Executing step {current_step}...")
        else:
            if elapsed % 30 == 0:
                print(f"   [{elapsed}s] Waiting for execution to start...")
            
            # Poll API status when log doesn't exist yet (detect early failures)
            if elapsed > 0 and elapsed % 30 == 0:
                try:
                    with httpx.Client(base_url=BASE_URL, headers=get_headers(), timeout=10.0, verify=VERIFY_SSL) as client:
                        resp = client.get(f"/api/v1/tasks/{task_id}")
                        if resp.status_code == 200:
                            status = resp.json().get("status", "")
                            if status == "failed":
                                print(f"   ❌ Task failed after {elapsed} seconds (workload error)")
                                return False
                            if status == "completed":
                                print(f"   ✅ Task completed after {elapsed} seconds (API)")
                                return True
                except Exception:
                    pass
        
        time.sleep(interval)
    
    print(f"   ⚠️ Timeout after {max_wait} seconds")
    return False


# =============================================================================
# Step 5: Get Outputs
# =============================================================================

def test_get_outputs(client: httpx.Client, task_id: str) -> dict | None:
    """Get task outputs."""
    response = client.get(f"/api/v1/tasks/{task_id}/outputs")
    if response.status_code == 200:
        return response.json()
    else:
        print_result(False, f"Failed to get outputs: {response.status_code}")
        return None


def test_download_file(client: httpx.Client, task_id: str, file_path: str) -> bytes | None:
    """Download a file from task outputs."""
    response = client.get(f"/api/v1/tasks/{task_id}/download", params={"path": file_path})
    if response.status_code == 200:
        return response.content
    elif response.status_code == 404:
        return None
    else:
        print_result(False, f"Failed to download {file_path}: {response.status_code}")
        return None


# =============================================================================
# Full Flow Test
# =============================================================================

def run_full_flow_test(skip_submit: bool = False):
    """Run the complete flow test."""
    print("=" * 60)
    print("GEAK Online Service - Full Flow Test")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print(f"API Key: {API_KEY[:20] if API_KEY else 'Not set'}...")
    print(f"LLM API Base: {API_BASE}")
    
    if not API_KEY:
        print("\n❌ TEST_API_KEY not set in .env")
        return False
    
    with httpx.Client(base_url=BASE_URL, headers=get_headers(), timeout=120.0, verify=VERIFY_SSL) as client:
        
        # =================================================================
        # Step 1: User Model Configuration
        # =================================================================
        print_step(1, "User Model Configuration")
        
        # 1.1 Check current config
        print("\n1.1 Checking current user config...")
        current_config = test_get_user_config(client)
        if current_config:
            print_result(True, f"Found existing config: {current_config['config'].get('model_name')}")
        else:
            print_result(True, "No existing config (will create new)")
        
        # 1.2 Save new config
        # Use litellm model class for remote execution (openai_compatible is local only)
        print("\n1.2 Saving user default model config...")
        new_config = {
            "model_class": "litellm",
            "model_name": "openai/claude-opus-4.5",  # litellm format: provider/model
            "model_kwargs": {
                "api_base": API_BASE,
                "api_key": LLM_API_KEY,
                "max_tokens": 16000,
                "temperature": 0.0
            }
        }
        saved_config = test_save_user_config(client, new_config)
        if not saved_config:
            return False
        print_result(True, f"Saved config: model_class={saved_config['config']['model_class']}, model_name={saved_config['config']['model_name']}")
        
        # 1.3 Verify config was saved
        print("\n1.3 Verifying saved config...")
        verified_config = test_get_user_config(client)
        if verified_config:
            assert verified_config['config']['model_name'] == new_config['model_name']
            print_result(True, "Config verified successfully")
        else:
            print_result(False, "Failed to verify config")
            return False
        
        # =================================================================
        # Step 2: Create Task (without model config - should use saved)
        # =================================================================
        print_step(2, "Create Task (Using Saved Config)")
        
        # Load test data
        try:
            silu_code = load_file("test_silu/silu.hip")
            silu_prompt = load_file("test_prompts/prompt_silu.md")
        except FileNotFoundError as e:
            print_result(False, f"Test file not found: {e}")
            return False
        
        print(f"   Loaded silu.hip ({len(silu_code)} bytes)")
        print(f"   Loaded prompt_silu.md ({len(silu_prompt)} bytes)")
        
        # Create task WITHOUT model config - should use user's saved config
        payload = {
            "input_type": "file",
            "files": [
                {
                    "filename": "silu.hip",
                    "content": silu_code,
                },
            ],
            "prompt": silu_prompt,
            # No config.model - should use saved user config
            "config": {
                "agent": {
                    "step_limit": 10,  # Limit steps for testing
                    "cost_limit": 5.0
                }
            },
            "runtime": {
                "gpu_count": 1,
                "timeout": 3600,
            }
        }
        
        print("\n2.1 Creating task...")
        task = test_create_task(client, payload)
        if not task:
            return False
        
        task_id = task["id"]
        user_id = task["user_id"]
        print_result(True, f"Task created: {task_id}")
        
        # Verify task uses saved model config
        print("\n2.2 Verifying task uses saved model config...")
        task_config = task.get("config", {}).get("model", {})
        if task_config.get("model_name") == new_config["model_name"]:
            print_result(True, f"Task using saved config: {task_config.get('model_name')}")
        else:
            print_result(False, f"Task not using saved config: {task_config.get('model_name')}")
        
        # Verify api_key is in the config
        if task_config.get("model_kwargs", {}).get("api_key"):
            print_result(True, "API key present in task config")
        else:
            print_result(False, "API key missing from task config")
        
        # =================================================================
        # Step 3: Get Task Details
        # =================================================================
        print_step(3, "Get Task Details")
        
        task_details = test_get_task(client, task_id)
        if task_details:
            print_result(True, f"Status: {task_details['status']}")
            print_result(True, f"Input Type: {task_details['input_type']}")
            print_result(True, f"Input Path: {task_details.get('input_path', 'N/A')}")
        else:
            return False
        
        if skip_submit:
            print("\n⚠️ Skipping submit step (--no-submit flag)")
            print_step(4, "Cleanup")
            
            # Delete config at end
            print("\n4.1 Deleting user config...")
            if test_delete_user_config(client):
                print_result(True, "User config deleted")
            
            print("\n" + "=" * 60)
            print("Flow Test Completed (without submission)")
            print("=" * 60)
            print(f"\nTask ID: {task_id}")
            print(f"To submit manually:")
            print(f"  curl -X POST {BASE_URL}/api/v1/tasks/{task_id}/submit \\")
            print(f"       -H 'Authorization: Bearer {API_KEY}'")
            return True
        
        # =================================================================
        # Step 4: Submit Task
        # =================================================================
        print_step(4, "Submit Task for Execution")
        
        print("\n4.1 Submitting task to SaFE platform...")
        submitted_task = test_submit_task(client, task_id)
        if not submitted_task:
            return False
        
        print_result(True, f"Task submitted! Status: {submitted_task['status']}")
        print_result(True, f"Workload ID: {submitted_task.get('safe_workload_id', 'N/A')}")
        
        # =================================================================
        # Step 5: Monitor Execution
        # =================================================================
        print_step(5, "Monitor Task Execution")
        
        print(f"\n   Monitoring task (max 5 minutes)...")
        completed = monitor_task_execution(task_id, user_id, max_wait=300, interval=10)
        
        # =================================================================
        # Step 6: Get Outputs
        # =================================================================
        print_step(6, "Get Task Outputs")
        
        outputs = test_get_outputs(client, task_id)
        if outputs:
            print_result(True, f"Task ID: {outputs.get('task_id', task_id)}")
            files = outputs.get('files', [])
            print_result(True, f"Total files: {len(files)}")
            for f in files:
                print(f"      - {f['path']} ({f['size']} bytes)")
        
        # =================================================================
        # Step 7: Download Files
        # =================================================================
        print_step(7, "Download Output Files")
        
        # Download execution.log
        print("\n7.1 Downloading execution.log...")
        log_content = test_download_file(client, task_id, "execution.log")
        if log_content:
            print_result(True, f"Downloaded execution.log ({len(log_content)} bytes)")
            # Show last 10 lines
            lines = log_content.decode('utf-8', errors='replace').strip().split('\n')
            print("\n   --- Last 10 lines of execution.log ---")
            for line in lines[-10:]:
                print(f"   {line[:100]}")
        else:
            print_result(False, "execution.log not found")
        
        # Download optimized kernel
        print("\n7.2 Downloading silu.hip (optimized)...")
        hip_content = test_download_file(client, task_id, "silu.hip")
        if hip_content:
            print_result(True, f"Downloaded silu.hip ({len(hip_content)} bytes)")
        else:
            print_result(False, "silu.hip not found (may not have been copied)")
        
        # =================================================================
        # Step 8: Cleanup (Optional)
        # =================================================================
        print_step(8, "Cleanup")
        
        # Optionally delete user config
        print("\n8.1 Keeping user config for future tests")
        # Uncomment to delete:
        # if test_delete_user_config(client):
        #     print_result(True, "User config deleted")
        
        # =================================================================
        # Summary
        # =================================================================
        print("\n" + "=" * 60)
        print("Full Flow Test Completed!")
        print("=" * 60)
        print(f"\nTask ID: {task_id}")
        print(f"Status: {'Completed' if completed else 'In Progress/Timeout'}")
        
        return True


def run_config_only_test():
    """Test only the user configuration API."""
    print("=" * 60)
    print("GEAK Online Service - Config API Test")
    print("=" * 60)
    
    if not API_KEY:
        print("\n❌ TEST_API_KEY not set in .env")
        return False
    
    with httpx.Client(base_url=BASE_URL, headers=get_headers(), timeout=30.0, verify=VERIFY_SSL) as client:
        
        # Test 1: Get (should be 404 or existing)
        print("\n1. GET /api/v1/config/model")
        config = test_get_user_config(client)
        print(f"   Result: {config if config else '404 Not Found'}")
        
        # Test 2: PUT (create/update)
        print("\n2. PUT /api/v1/config/model")
        new_config = {
            "model_class": "litellm",
            "model_name": "openai/gpt-5.2",
            "model_kwargs": {
                "api_base": API_BASE,
                "api_key": LLM_API_KEY,
                "max_tokens": 8000
            }
        }
        saved = test_save_user_config(client, new_config)
        print(f"   Result: {saved}")
        
        # Test 3: GET (should return saved)
        print("\n3. GET /api/v1/config/model (verify)")
        verified = test_get_user_config(client)
        print(f"   Result: {verified}")
        
        # Test 4: PUT (update)
        print("\n4. PUT /api/v1/config/model (update)")
        updated_config = {
            "model_class": "litellm",
            "model_name": "openai/claude-opus-4.5",
            "model_kwargs": {
                "api_base": API_BASE,
                "api_key": LLM_API_KEY,
                "max_tokens": 16000
            }
        }
        updated = test_save_user_config(client, updated_config)
        print(f"   Result: model_name changed to {updated['config']['model_name']}")
        
        # Test 5: DELETE
        print("\n5. DELETE /api/v1/config/model")
        deleted = test_delete_user_config(client)
        print(f"   Result: {'204 No Content' if deleted else 'Failed'}")
        
        # Test 6: GET (should be 404)
        print("\n6. GET /api/v1/config/model (after delete)")
        final = test_get_user_config(client)
        print(f"   Result: {final if final else '404 Not Found'}")
        
        print("\n" + "=" * 60)
        print("Config API Test Completed!")
        print("=" * 60)
        
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GEAK Full Flow Test")
    parser.add_argument("--config-only", action="store_true", help="Test only config API")
    parser.add_argument("--no-submit", action="store_true", help="Skip task submission")
    args = parser.parse_args()
    
    if args.config_only:
        success = run_config_only_test()
    else:
        success = run_full_flow_test(skip_submit=args.no_submit)
    
    sys.exit(0 if success else 1)
