"""GEAK MCP Tools Implementation.

Provides the actual tool implementations that call GEAK API endpoints.
"""

import httpx
import logging
from typing import Any

from server.config import get_settings

logger = logging.getLogger(__name__)


class GEAKTools:
    """GEAK tool implementations for MCP server."""
    
    def __init__(self, default_api_key: str | None = None, base_url: str | None = None):
        """Initialize GEAK tools.
        
        Args:
            default_api_key: Default API key for authentication.
            base_url: Base URL for GEAK API. Defaults to localhost:8000.
        """
        self.default_api_key = default_api_key
        settings = get_settings()
        # Internal base URL for API calls
        self.base_url = base_url or f"http://{settings.host}:{settings.port}"
        # External base URL for download links (used in responses to users)
        self.external_base_url = (settings.external_base_url or f"http://localhost:{settings.port}").rstrip("/")
    
    def _get_headers(self, api_key: str) -> dict[str, str]:
        """Get request headers with authentication."""
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    
    async def _request(
        self,
        method: str,
        path: str,
        api_key: str,
        json: dict | None = None,
        params: dict | None = None,
    ) -> dict[str, Any]:
        """Make HTTP request to GEAK API."""
        async with httpx.AsyncClient(base_url=self.base_url, timeout=120.0) as client:
            response = await client.request(
                method=method,
                url=path,
                headers=self._get_headers(api_key),
                json=json,
                params=params,
            )
            
            if response.status_code == 204:
                return {"success": True}
            
            if response.status_code >= 400:
                try:
                    error = response.json()
                except Exception:
                    error = {"detail": response.text}
                return {
                    "error": error.get("detail", str(error)),
                    "status_code": response.status_code
                }
            
            return response.json()
    
    # =========================================================================
    # User Configuration Tools
    # =========================================================================
    
    async def get_model_config(self, api_key: str) -> dict[str, Any]:
        """Get user's default model configuration."""
        return await self._request("GET", "/api/v1/config/model", api_key)
    
    async def set_model_config(
        self,
        api_key: str,
        model_class: str,
        model_name: str,
        model_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Set user's default model configuration."""
        payload = {
            "model_class": model_class,
            "model_name": model_name,
            "model_kwargs": model_kwargs,
        }
        return await self._request("PUT", "/api/v1/config/model", api_key, json=payload)
    
    async def delete_model_config(self, api_key: str) -> dict[str, Any]:
        """Delete user's default model configuration."""
        return await self._request("DELETE", "/api/v1/config/model", api_key)
    
    # =========================================================================
    # Task Management Tools
    # =========================================================================
    
    async def create_task(
        self,
        api_key: str,
        input_type: str,
        files: list[dict[str, str]] | None = None,
        repo_url: str | None = None,
        repo_branch: str | None = None,
        prompt: str | None = None,
        step_limit: int | None = None,
        gpu_count: int | None = None,
        image: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a new optimization task.
        
        Args:
            api_key: API key for authentication.
            input_type: Type of input - 'file' or 'repo'.
            files: List of files for file input, each with 'filename' and 'content'.
            repo_url: Git repository URL for repo input.
            repo_branch: Git branch name for repo input.
            prompt: Optimization instructions.
            step_limit: Maximum agent steps.
            gpu_count: Number of GPUs.
            image: Docker image to use. If not provided, uses server default.
            workspace_id: SaFE workspace ID. If not provided, uses user's first workspace.
        
        Returns:
            Created task information.
        """
        payload: dict[str, Any] = {"input_type": input_type}
        
        # Handle file input
        if input_type == "file":
            if not files:
                return {"error": "files list is required for file input"}
            payload["files"] = files
        
        # Handle repo input
        elif input_type == "repo":
            if not repo_url:
                return {"error": "repo_url is required for repo input"}
            payload["repo"] = {
                "url": repo_url,
            }
            if repo_branch:
                payload["repo"]["branch"] = repo_branch
        
        # Add optional workspace
        if workspace_id:
            payload["workspace_id"] = workspace_id
        
        # Add optional prompt
        if prompt:
            payload["prompt"] = prompt
        
        # Add config overrides
        config: dict[str, Any] = {}
        if step_limit:
            config["agent"] = {"step_limit": step_limit}
        if config:
            payload["config"] = config
        
        # Add runtime config
        runtime: dict[str, Any] = {}
        if gpu_count is not None:
            runtime["gpu_count"] = gpu_count
        if image is not None:
            runtime["image"] = image
        if runtime:
            payload["runtime"] = runtime
        
        return await self._request("POST", "/api/v1/tasks", api_key, json=payload)
    
    async def get_task(self, api_key: str, task_id: str) -> dict[str, Any]:
        """Get task details."""
        return await self._request("GET", f"/api/v1/tasks/{task_id}", api_key)
    
    async def list_tasks(
        self,
        api_key: str,
        status: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """List user's tasks."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return await self._request("GET", "/api/v1/tasks", api_key, params=params)
    
    async def submit_task(self, api_key: str, task_id: str) -> dict[str, Any]:
        """Submit task for execution."""
        return await self._request("POST", f"/api/v1/tasks/{task_id}/submit", api_key)
    
    async def cancel_task(self, api_key: str, task_id: str) -> dict[str, Any]:
        """Cancel a running task."""
        return await self._request("POST", f"/api/v1/tasks/{task_id}/cancel", api_key)
    
    # =========================================================================
    # Task Output Tools
    # =========================================================================
    
    async def get_outputs(self, api_key: str, task_id: str) -> dict[str, Any]:
        """Get task outputs information."""
        return await self._request("GET", f"/api/v1/tasks/{task_id}/outputs", api_key)
    
    async def download_file(
        self,
        api_key: str,
        task_id: str,
        file_path: str,
    ) -> dict[str, Any]:
        """Get download information for a file from task outputs.
        
        Args:
            api_key: API key for authentication.
            task_id: Task ID.
            file_path: Path to file within task outputs (e.g., execution.log, modified_repo.tar.gz).
        
        Returns:
            Download URL and file information. For text files under 1MB, also includes content.
        """
        # First, check file exists via outputs API
        outputs = await self.get_outputs(api_key, task_id)
        if "error" in outputs:
            return outputs
        
        # Find the file in outputs
        file_info = None
        for f in outputs.get("files", []):
            if f.get("path") == file_path:
                file_info = f
                break
        
        if not file_info:
            return {"error": f"File '{file_path}' not found in task outputs"}
        
        file_size = file_info.get("size", 0)
        
        # Determine if binary
        binary_extensions = (".tar.gz", ".zip", ".gz", ".tar", ".bin", ".so", ".a", ".o", ".exe", ".dll", ".png", ".jpg", ".jpeg", ".gif", ".pdf")
        is_binary = file_path.endswith(binary_extensions)
        
        # Generate download URL with token for direct access
        download_url = f"{self.external_base_url}/api/v1/tasks/{task_id}/download?path={file_path}&token={api_key}"
        
        result = {
            "file_path": file_path,
            "size": file_size,
            "download_url": download_url,
        }
        
        # For small text files, also include content directly
        TEXT_SIZE_LIMIT = 1 * 1024 * 1024  # 1MB
        
        if not is_binary and file_size <= TEXT_SIZE_LIMIT:
            try:
                async with httpx.AsyncClient(base_url=self.base_url, timeout=60.0) as client:
                    response = await client.get(
                        f"/api/v1/tasks/{task_id}/download",
                        headers=self._get_headers(api_key),
                        params={"path": file_path},
                    )
                    
                    if response.status_code < 400:
                        content = response.content
                        try:
                            text_content = content.decode("utf-8")
                        except UnicodeDecodeError:
                            text_content = content.decode("utf-8", errors="replace")
                        
                        result["content"] = text_content
                        result["message"] = f"Text file ({file_size:,} bytes). Content included below."
                        return result
            except Exception as e:
                logger.warning(f"Failed to fetch text content: {e}")
        
        # For binary or large files, just return the download URL
        result["message"] = f"Download URL for {file_path} ({file_size:,} bytes). Click the link to download directly."
        
        return result
