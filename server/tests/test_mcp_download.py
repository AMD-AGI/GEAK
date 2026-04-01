#!/usr/bin/env python3
"""Test script for MCP download_file functionality."""

import asyncio
import os
import tempfile
from pathlib import Path

# Add server to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # Set minimal required env vars for testing
    os.environ.setdefault("SAFE_API_BASE", "http://localhost:8080")
    os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")

from server.mcp.tools import GEAKTools


async def test_download_file():
    """Test the download_file function."""
    
    # Use the API key from environment or config
    api_key = os.getenv("SAFE_SYSTEM_API_KEY", os.getenv("GEAK_API_KEY", "ak-test"))
    
    # Known task ID with outputs
    task_id = "af4a2cee-9104-4fc8-acc7-ed3f455c5550"
    
    print(f"Testing download_file:")
    print(f"  task_id: {task_id}")
    print()
    
    # Initialize tools
    tools = GEAKTools(default_api_key=api_key)
    
    # Test get_outputs first
    print("1. Testing get_outputs...")
    outputs = await tools.get_outputs(api_key, task_id)
    if "error" in outputs:
        print(f"   ERROR: {outputs}")
        return
    print(f"   Found {len(outputs.get('files', []))} files:")
    for f in outputs.get("files", []):
        print(f"     - {f.get('path')} ({f.get('size', 0):,} bytes)")
    print()
    
    # Test download text file
    print("2. Testing download text file (execution.log)...")
    result = await tools.download_file(api_key, task_id, "execution.log")
    if "error" in result:
        print(f"   ERROR: {result}")
    else:
        print(f"   download_url: {result.get('download_url')}")
        print(f"   size: {result.get('size'):,} bytes")
        print(f"   message: {result.get('message')}")
        content = result.get('content', '')
        if content:
            print(f"   content included: YES ({len(content):,} chars)")
            print(f"   content preview:\n{content[:300]}...")
        else:
            print(f"   content included: NO")
    print()
    
    # Test download binary file (tar.gz)
    print("3. Testing download binary file (modified_repo.tar.gz)...")
    result = await tools.download_file(api_key, task_id, "modified_repo.tar.gz")
    if "error" in result:
        print(f"   ERROR: {result}")
    else:
        print(f"   download_url: {result.get('download_url')}")
        print(f"   size: {result.get('size'):,} bytes")
        print(f"   message: {result.get('message')}")
        print(f"   curl_command: {result.get('curl_command')}")
        
        # Verify URL works with curl
        import subprocess
        download_url = result.get('download_url')
        print(f"\n   Testing download URL with curl...")
        cmd = f'curl -s -H "Authorization: Bearer {api_key}" "{download_url}" -o /tmp/test_download.tar.gz -w "%{{http_code}}"'
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        http_code = proc.stdout.strip()
        print(f"   HTTP status: {http_code}")
        
        if http_code == "200":
            import tarfile
            size = os.path.getsize("/tmp/test_download.tar.gz")
            print(f"   File downloaded: {size:,} bytes")
            try:
                with tarfile.open("/tmp/test_download.tar.gz", "r:gz") as tar:
                    print(f"   Tar.gz valid: YES ({len(tar.getnames())} files)")
            except Exception as e:
                print(f"   Tar.gz valid: NO ({e})")
            os.unlink("/tmp/test_download.tar.gz")
        else:
            print(f"   Download failed: {proc.stderr}")
    
    print()
    print("=" * 60)
    print("Test completed!")


if __name__ == "__main__":
    asyncio.run(test_download_file())
