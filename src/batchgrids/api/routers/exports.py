from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated, List

from batchgrids.api.routers.auth import get_current_user
from batchgrids.database import get_async_db
from batchgrids.models.user import User

router = APIRouter()


class ExportResponse(BaseModel):
    id: int
    bin_id: int
    format: str
    uri: str
    version: int
    file_size_bytes: int
    generation_time_ms: int
    created_at: str
    
    class Config:
        from_attributes = True


@router.get("/bins/{bin_id}/exports", response_model=List[ExportResponse])
async def get_bin_exports(
    bin_id: int,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """Get export history for a bin."""
    # TODO: Implement export history retrieval
    return []


@router.get("/{export_id}/download")
async def download_export(
    export_id: int,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """Download an exported file."""
    # TODO: Implement file download with S3 pre-signed URLs
    raise HTTPException(status_code=404, detail="Export not found")