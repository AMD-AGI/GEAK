"""Authentication service for GEAK Online Service."""

import httpx
from typing import Any
from fastapi import HTTPException, status

from server.config import get_settings


class AuthService:
    """Service for user authentication via SaFE platform."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.settings = get_settings()
        self._user_info: dict | None = None
    
    async def get_user_info(self) -> dict[str, Any]:
        """Get user info from SaFE platform using the API key.

        In local mode (GEAK_LOCAL=true), skips SaFE and returns a synthetic
        local user derived from the API key.

        Returns:
            User info dictionary containing at least 'id', 'name', 'email'.

        Raises:
            HTTPException: If authentication fails.
        """
        import os
        if self._user_info:
            return self._user_info

        # Local mode: skip SaFE auth, return synthetic user
        if os.getenv("GEAK_LOCAL", "false").lower() == "true":
            self._user_info = {
                "id": f"local-{self.api_key[:8]}",
                "userId": f"local-{self.api_key[:8]}",
                "name": "local-user",
                "email": "local@localhost",
            }
            return self._user_info

        url = f"{self.settings.safe_api_base}/api/v1/users/self"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=10.0)
                
                if response.status_code == 401:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid API key",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                elif response.status_code == 403:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access forbidden",
                    )
                elif response.status_code != 200:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=f"SaFE platform error: {response.status_code}",
                    )
                
                self._user_info = response.json()
                return self._user_info
                
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Cannot connect to SaFE platform: {str(e)}",
            )
    
    @property
    def user_id(self) -> str:
        """Get user ID. Must call get_user_info first."""
        if not self._user_info:
            raise RuntimeError("User info not loaded. Call get_user_info first.")
        return self._user_info.get("id") or self._user_info.get("userId")
    
    @property
    def user_name(self) -> str:
        """Get user name. Must call get_user_info first."""
        if not self._user_info:
            raise RuntimeError("User info not loaded. Call get_user_info first.")
        return self._user_info.get("name") or self._user_info.get("username", "unknown")


async def verify_api_key(api_key: str) -> dict[str, Any]:
    """Verify API key and return user info.
    
    Args:
        api_key: Bearer token from Authorization header.
    
    Returns:
        User info dictionary.
    """
    auth = AuthService(api_key)
    return await auth.get_user_info()
