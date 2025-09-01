from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated, List, Optional

from batchgrids.api.routers.auth import get_current_user
from batchgrids.database import get_async_db
from batchgrids.models.user import User

router = APIRouter()


class ToolResponse(BaseModel):
    id: int
    canonical_name: str
    notes: Optional[str] = None
    dims_mm: Optional[dict] = None
    
    class Config:
        from_attributes = True


class ToolUpdate(BaseModel):
    canonical_name: Optional[str] = None
    notes: Optional[str] = None


@router.post("/")
async def create_tool(
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """Create a new tool record."""
    # TODO: Implement tool creation (usually auto-created from image processing)
    return {"message": "Tool creation not implemented yet"}


@router.get("/", response_model=List[ToolResponse])
async def get_tools(
    skip: int = 0,
    limit: int = 100,
    search: Optional[str] = None,
    class_filter: Optional[str] = None,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """Get tools with optional search and filtering."""
    # TODO: Implement tool listing with search and filters
    return []


@router.get("/{tool_id}", response_model=ToolResponse)
async def get_tool(
    tool_id: int,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """Get a specific tool by ID."""
    # TODO: Implement tool retrieval
    raise HTTPException(status_code=404, detail="Tool not found")


@router.patch("/{tool_id}", response_model=ToolResponse)
async def update_tool(
    tool_id: int,
    tool_update: ToolUpdate,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """Update tool metadata."""
    # TODO: Implement tool updates
    raise HTTPException(status_code=404, detail="Tool not found")