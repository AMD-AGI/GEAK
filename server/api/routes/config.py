"""User Configuration Routes for GEAK API."""

from fastapi import APIRouter, HTTPException, status

from server.api.deps import CurrentUser
from server.api.schemas.config import UserModelConfigRequest, UserModelConfigResponse
from server.database import UserConfigDB

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/model", response_model=UserModelConfigResponse)
async def get_model_config(user: CurrentUser):
    """Get current user's default model configuration.
    
    Returns the user's saved model configuration, or 404 if not set.
    """
    user_id = user.get("id") or user.get("userId")
    config = await UserConfigDB.get(user_id)
    
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No default model configuration found. Use PUT to create one."
        )
    
    return UserModelConfigResponse(
        user_id=user_id,
        config=config.get("model_config", {}),
        created_at=config.get("created_at", ""),
        updated_at=config.get("updated_at", ""),
    )


@router.put("/model", response_model=UserModelConfigResponse, status_code=status.HTTP_200_OK)
async def update_model_config(user: CurrentUser, request: UserModelConfigRequest):
    """Create or update user's default model configuration.
    
    This configuration will be used when creating tasks without specifying model config.
    
    Example:
    ```json
    {
        "model_class": "openai_compatible",
        "model_name": "claude-opus-4.5",
        "model_kwargs": {
            "api_base": "http://litellm-service.primus-safe.svc.cluster.local:4000/v1",
            "api_key": "sk-xxx",
            "max_tokens": 16000,
            "temperature": 0.0
        }
    }
    ```
    """
    user_id = user.get("id") or user.get("userId")
    
    # Build model config in the format expected by mini-swe-agent
    model_config = {
        "model_class": request.model_class,
        "model_name": request.model_name,
        "model_kwargs": request.model_kwargs,
    }
    
    config = await UserConfigDB.upsert(user_id, model_config)
    
    return UserModelConfigResponse(
        user_id=user_id,
        config=config.get("model_config", {}),
        created_at=config.get("created_at", ""),
        updated_at=config.get("updated_at", ""),
    )


@router.delete("/model", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model_config(user: CurrentUser):
    """Delete user's default model configuration.
    
    After deletion, task creation will use system default configuration.
    """
    user_id = user.get("id") or user.get("userId")
    deleted = await UserConfigDB.delete(user_id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No default model configuration found."
        )
