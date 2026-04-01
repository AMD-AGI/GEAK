"""SaFE Platform API Client."""

import httpx
import base64
from typing import Any

from server.config import get_settings


class SaFEClient:
    """Client for interacting with SaFE platform API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.settings = get_settings()
        self.base_url = self.settings.safe_api_base
    
    def _headers(self) -> dict:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    async def get_workspaces(self) -> list[dict]:
        """Get list of workspaces for the user.
        
        Returns:
            List of workspace dictionaries.
        """
        url = f"{self.base_url}/api/v1/workspaces"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self._headers(), timeout=10.0)
            response.raise_for_status()
            data = response.json()
            # Handle both list and paginated response
            if isinstance(data, list):
                return data
            return data.get("items", data.get("workspaces", []))
    
    async def get_default_workspace_id(self) -> str:
        """Get the default (first) workspace ID.
        
        Returns:
            Workspace ID string.
        
        Raises:
            Exception: If no workspaces found.
        """
        workspaces = await self.get_workspaces()
        if not workspaces:
            raise Exception("No workspaces found for user")
        return workspaces[0].get("workspaceId") or workspaces[0].get("id")
    
    async def create_workload(
        self,
        workspace_id: str,
        name: str,
        image: str,
        command: str,
        gpu_count: int = 1,
        cpu: int = 4,
        memory: str = "16Gi",
        env_vars: dict[str, str] | None = None,
        volumes: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Create a PyTorchJob workload on SaFE platform.
        
        Args:
            workspace_id: Target workspace ID.
            name: Workload name.
            image: Docker image to use.
            command: Command to run (will be base64 encoded).
            gpu_count: Number of GPUs.
            cpu: Number of CPU cores.
            memory: Memory limit.
            env_vars: Environment variables.
            volumes: Volume mounts.
        
        Returns:
            Created workload information.
        """
        # API endpoint is /api/v1/workloads, workspaceId goes in the body
        url = f"{self.base_url}/api/v1/workloads"
        
        # Base64 encode the entry point command
        entry_point = base64.b64encode(command.encode()).decode()
        
        # Build workload payload according to SaFE API spec
        payload = {
            "displayName": name,
            "description": f"GEAK optimization task: {name}",
            "workspaceId": workspace_id,
            "groupVersionKind": {
                "kind": "PyTorchJob",
                "version": "v1"
            },
            "resources": [
                {
                    "cpu": str(cpu),
                    "gpu": str(gpu_count),
                    "memory": memory,
                    "ephemeralStorage": "50Gi",
                    "replica": 1
                }
            ],
            "images": [image],
            "entryPoints": [entry_point],
            "priority": 1, 
            "maxRetry": 0,
            "ttlSecondsAfterFinished": 0,
        }
        
        # Add environment variables
        if env_vars:
            payload["env"] = env_vars
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=self._headers(),
                json=payload,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()
    
    async def get_workload(self, workload_id: str) -> dict[str, Any]:
        """Get workload status.
        
        Args:
            workload_id: Workload ID.
        
        Returns:
            Workload information.
        """
        url = f"{self.base_url}/api/v1/workloads/{workload_id}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self._headers(), timeout=10.0)
            response.raise_for_status()
            return response.json()
    
    async def stop_workload(self, workload_id: str) -> bool:
        """Stop a running workload.
        
        Args:
            workload_id: Workload ID.
        
        Returns:
            True if stopped successfully.
        
        Raises:
            Exception: If stop request fails.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        url = f"{self.base_url}/api/v1/workloads/{workload_id}/stop"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=self._headers(), timeout=10.0)
            if response.status_code not in (200, 204):
                body = response.text
                logger.error("Failed to stop workload %s: status=%d, body=%s", workload_id, response.status_code, body)
                raise Exception(f"Failed to stop workload {workload_id}: {response.status_code} {body}")
            return True
    
