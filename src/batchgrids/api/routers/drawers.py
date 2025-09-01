from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated

from batchgrids.api.routers.auth import get_current_user
from batchgrids.database import get_async_db
from batchgrids.models.user import User

router = APIRouter()


class DrawerCreate(BaseModel):
    name: str
    label_code: str


class DrawerResponse(BaseModel):
    id: int
    name: str
    label_code: str
    
    class Config:
        from_attributes = True


@router.post("/", response_model=DrawerResponse)
async def create_drawer(
    drawer_data: DrawerCreate,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """Create a new drawer for organization."""
    # TODO: Implement drawer creation
    return {"message": "Drawer creation not implemented yet"}


@router.get("/{drawer_id}", response_model=DrawerResponse)
async def get_drawer(
    drawer_id: int,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """Get drawer details."""
    # TODO: Implement drawer retrieval
    raise HTTPException(status_code=404, detail="Drawer not found")