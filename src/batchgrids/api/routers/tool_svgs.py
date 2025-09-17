from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from batchgrids.api.routers.auth import get_current_user
from batchgrids.database import get_async_db
from batchgrids.models.tool_svg import ToolSvg
from batchgrids.models.user import User

router = APIRouter()


class ToolSvgCreate(BaseModel):
    name: str = Field(..., max_length=200, description="Display name for the stored SVG")
    svg: str = Field(..., description="SVG markup for the tool outline")
    ai_description: str = Field(..., description="AI generated description for the tool")
    tags: List[str] = Field(default_factory=list, description="Searchable tags or AI hints")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional structured metadata such as dimensions or px/mm",
    )


class ToolSvgResponse(BaseModel):
    id: int
    name: str
    svg: str
    ai_description: str
    tags: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


@router.post("/", response_model=ToolSvgResponse, status_code=status.HTTP_201_CREATED)
async def create_tool_svg(
    payload: ToolSvgCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_async_db),
) -> ToolSvg:
    """Persist a generated SVG and its AI description for later reuse."""

    tags = [tag.strip() for tag in payload.tags if tag and tag.strip()]
    tool_svg = ToolSvg(
        user_id=current_user.id,
        name=payload.name.strip() or "Tool Outline",
        svg=payload.svg,
        ai_description=payload.ai_description.strip(),
        tags=tags,
        metadata=payload.metadata or None,
    )
    db.add(tool_svg)
    await db.commit()
    await db.refresh(tool_svg)
    return tool_svg


@router.get("/", response_model=List[ToolSvgResponse])
async def list_tool_svgs(
    search: Optional[str] = None,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_async_db),
) -> List[ToolSvg]:
    """Return stored SVGs for the current user, optionally filtered by a search term."""

    stmt = select(ToolSvg).where(ToolSvg.user_id == current_user.id).order_by(ToolSvg.created_at.desc())
    result = await db.execute(stmt)
    items = list(result.scalars().all())

    if search:
        query = search.strip().lower()
        if query:
            filtered: List[ToolSvg] = []
            for item in items:
                haystack = f"{(item.name or '').lower()} {(item.ai_description or '').lower()}"
                tags = item.tags or []
                if query in haystack or any(query in (tag or '').lower() for tag in tags):
                    filtered.append(item)
            items = filtered
    return items


@router.get("/{svg_id}", response_model=ToolSvgResponse)
async def get_tool_svg(
    svg_id: int,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_async_db),
) -> ToolSvg:
    """Fetch a single stored SVG by identifier."""

    result = await db.execute(
        select(ToolSvg).where(ToolSvg.id == svg_id, ToolSvg.user_id == current_user.id)
    )
    item = result.scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="SVG not found")
    return item
