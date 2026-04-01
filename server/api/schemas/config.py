"""Configuration schemas for GEAK API."""

from pydantic import BaseModel, Field
from typing import Any


class ModelConfig(BaseModel):
    """Model configuration for the agent."""
    model_class: str | None = Field(default=None, description="Model class: litellm, amd_llm, etc.")
    model_name: str | None = Field(default=None, description="Model name, e.g., openai/qwen3-max")
    model_kwargs: dict[str, Any] | None = Field(default=None, description="Additional model parameters")


class AgentConfig(BaseModel):
    """Agent configuration that can be customized by users.
    
    All fields are optional. If not provided, defaults will be used.
    Users can override any field from the default config.
    """
    model: ModelConfig | None = Field(default=None, description="Model configuration")
    working_dir: str | None = Field(default=None, description="Working directory for the agent")
    
    # Additional config sections that map to mini-swe-agent config
    tools: dict[str, Any] | None = Field(default=None, description="Tool configurations")
    agent: dict[str, Any] | None = Field(default=None, description="Agent behavior settings")
    env: dict[str, Any] | None = Field(default=None, description="Environment settings")
    
    class Config:
        extra = "allow"  # Allow additional fields for flexibility


class RuntimeConfig(BaseModel):
    """Runtime configuration for task execution."""
    image: str | None = Field(default=None, description="Docker image to use")
    gpu_count: int | None = Field(default=None, ge=0, le=8, description="Number of GPUs")
    cpu: int | None = Field(default=None, ge=1, le=32, description="Number of CPU cores")
    memory: str | None = Field(default=None, description="Memory limit, e.g., 16Gi")
    timeout: int | None = Field(default=None, ge=60, le=86400, description="Timeout in seconds")


class UserModelConfigRequest(BaseModel):
    """Request schema for user's default model configuration."""
    model_class: str = Field(..., description="Model class: openai_compatible, amd_llm, litellm, etc.")
    model_name: str = Field(..., description="Model name, e.g., claude-opus-4.5, gpt-5.2")
    model_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Model parameters including api_base, api_key, temperature, max_tokens, etc."
    )


class UserModelConfigResponse(BaseModel):
    """Response schema for user's default model configuration."""
    user_id: str = Field(..., description="User ID")
    config: dict[str, Any] = Field(..., description="Model configuration")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
