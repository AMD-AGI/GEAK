"""Task API routes for GEAK Online Service."""

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse
from typing import Annotated

from server.api.deps import CurrentUser, CurrentTaskManager
from server.api.schemas.task import (
    TaskCreate,
    TaskResponse,
    TaskListResponse,
    TaskOutputsResponse,
)

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(
    request: TaskCreate,
    task_manager: CurrentTaskManager,
):
    """Create a new optimization task.
    
    Creates a task with the provided input (file or repository), prompt, and configuration.
    The task will be in 'pending' status until submitted for execution.
    """
    # Validate input
    if request.input_type == "file" and not request.files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="files list is required when input_type is 'file'",
        )
    if request.input_type == "repo" and not request.repo:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="repo is required when input_type is 'repo'",
        )
    
    try:
        task = await task_manager.create_task(request)
        return TaskResponse(**task)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/{task_id}/submit", response_model=TaskResponse)
async def submit_task(
    task_id: str,
    task_manager: CurrentTaskManager,
):
    """Submit a task for execution on SaFE platform.
    
    This will create a GPU workload on the SaFE platform and start the optimization.
    """
    try:
        task = await task_manager.submit_task(task_id)
        return TaskResponse(**task)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("", response_model=TaskListResponse)
async def list_tasks(
    task_manager: CurrentTaskManager,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List tasks for the current user.
    
    Supports filtering by status and pagination.
    """
    tasks, total = await task_manager.list_tasks(
        status=status_filter,
        limit=limit,
        offset=offset,
    )
    return TaskListResponse(
        tasks=[TaskResponse(**t) for t in tasks],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str,
    task_manager: CurrentTaskManager,
):
    """Get task details by ID."""
    task = await task_manager.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task not found: {task_id}",
        )
    return TaskResponse(**task)


@router.post("/{task_id}/cancel", response_model=TaskResponse)
async def cancel_task(
    task_id: str,
    task_manager: CurrentTaskManager,
):
    """Cancel a running task.
    
    This will delete the workload on SaFE platform if running.
    """
    try:
        task = await task_manager.cancel_task(task_id)
        return TaskResponse(**task)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )


@router.get("/{task_id}/outputs", response_model=TaskOutputsResponse)
async def get_task_outputs(
    task_id: str,
    task_manager: CurrentTaskManager,
):
    """Get list of output files for a task.
    
    Returns file information for all output files.
    """
    try:
        outputs = await task_manager.get_outputs(task_id)
        return TaskOutputsResponse(**outputs)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )


@router.get("/{task_id}/download")
async def download_file(
    task_id: str,
    path: str,
    task_manager: CurrentTaskManager,
):
    """Download a specific file from task outputs.
    
    Args:
        task_id: Task ID
        path: Relative path to the file within the output directory
    """
    try:
        file_path, filename = await task_manager.get_file(task_id, path)
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/octet-stream",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )
