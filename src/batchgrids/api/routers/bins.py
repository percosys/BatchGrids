from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated, Optional

from batchgrids.api.routers.auth import get_current_user
from batchgrids.database import get_async_db
from batchgrids.models.user import User

router = APIRouter()


class BinCreate(BaseModel):
    name: str
    x_units: int
    y_units: int
    z_units: int
    inner_clearance_mm: Optional[float] = 40.0
    wall_mm: Optional[float] = 1.0


class BinResponse(BaseModel):
    id: int
    name: str
    x_units: int
    y_units: int
    z_units: int
    inner_clearance_mm: float
    wall_mm: float
    label_code: Optional[str] = None
    
    class Config:
        from_attributes = True


@router.post("/", response_model=BinResponse)
async def create_bin(
    bin_data: BinCreate,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """Create a new Gridfinity bin."""
    # TODO: Implement bin creation
    return {"message": "Bin creation not implemented yet"}


@router.get("/{bin_id}", response_model=BinResponse)
async def get_bin(
    bin_id: int,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """Get bin details."""
    # TODO: Implement bin retrieval
    raise HTTPException(status_code=404, detail="Bin not found")


@router.post("/{bin_id}/autofill")
async def autofill_bin(
    bin_id: int,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """Auto-pack tools into bin."""
    # TODO: Implement auto-packing algorithm
    return {"message": "Auto-fill not implemented yet", "bin_id": bin_id}


@router.post("/{bin_id}/export")
async def export_bin(
    bin_id: int,
    format: str,  # svg, dxf, stl, step, 3mf
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """Export bin in specified format."""
    # TODO: Queue export generation job
    return {"message": "Export not implemented yet", "bin_id": bin_id, "format": format}