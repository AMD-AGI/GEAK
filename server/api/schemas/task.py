"""Task schemas for GEAK API."""

from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime

from server.api.schemas.config import AgentConfig, RuntimeConfig


class FileInput(BaseModel):
    """Single file input."""
    filename: str = Field(..., description="File name, e.g., silu.hip")
    content: str = Field(..., description="File content")


class RepoInput(BaseModel):
    """Git repository input."""
    url: str = Field(..., description="Git repository URL")
    branch: str | None = Field(default=None, description="Branch name, defaults to main/master")
    subdir: str | None = Field(default=None, description="Subdirectory to focus on")


class TaskCreate(BaseModel):
    """Request schema for creating a new task."""
    
    # Input source - either files or repo
    input_type: Literal["file", "repo"] = Field(..., description="Type of input: file or repo")
    files: list[FileInput] | None = Field(default=None, description="List of file inputs (required if input_type=file). Can include .hip, Makefile, header files, etc.")
    repo: RepoInput | None = Field(default=None, description="Repository input (required if input_type=repo)")
    
    # User prompt - optional, defaults to system template
    prompt: str | None = Field(default=None, description="User prompt for optimization task")
    
    # Workspace - optional, defaults to user's first workspace
    workspace_id: str | None = Field(default=None, description="SaFE workspace ID to run in (defaults to user's first workspace)")
    
    # Configuration - optional, merged with defaults
    config: AgentConfig | None = Field(default=None, description="Agent configuration overrides")
    
    # Runtime - optional, uses platform defaults
    runtime: RuntimeConfig | None = Field(default=None, description="Runtime configuration")


class TaskResponse(BaseModel):
    """Response schema for task."""
    id: str = Field(..., description="Task ID")
    user_id: str = Field(..., description="User ID")
    status: str = Field(..., description="Task status: pending, running, completed, failed, cancelled")
    input_type: str = Field(..., description="Input type: file or repo")
    input_path: str | None = Field(default=None, description="Input path on storage")
    prompt: str | None = Field(default=None, description="User prompt")
    config: dict | None = Field(default=None, description="Agent configuration")
    runtime_config: dict | None = Field(default=None, description="Runtime configuration")
    safe_workload_id: str | None = Field(default=None, description="SaFE workload ID")
    output_path: str | None = Field(default=None, description="Output path on storage")
    error_message: str | None = Field(default=None, description="Error message if failed")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")


class TaskListResponse(BaseModel):
    """Response schema for task list."""
    tasks: list[TaskResponse] = Field(..., description="List of tasks")
    total: int = Field(..., description="Total number of tasks")
    limit: int = Field(..., description="Limit per page")
    offset: int = Field(..., description="Offset")


class OutputFile(BaseModel):
    """Output file information."""
    path: str = Field(..., description="Relative path within output directory")
    size: int = Field(..., description="File size in bytes")
    modified_at: str = Field(..., description="Last modified timestamp")


class TaskOutputsResponse(BaseModel):
    """Response schema for task outputs."""
    task_id: str = Field(..., description="Task ID")
    files: list[OutputFile] = Field(..., description="List of output files")
