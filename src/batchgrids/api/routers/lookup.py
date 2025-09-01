from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated, List, Optional

from batchgrids.api.routers.auth import get_current_user
from batchgrids.database import get_async_db
from batchgrids.models.user import User

router = APIRouter()


class LocationResult(BaseModel):
    tool_id: int
    tool_name: str
    drawer_code: str
    drawer_name: str
    tile_code: Optional[str] = None
    bin_id: Optional[int] = None


@router.get("/", response_model=List[LocationResult])
async def lookup_tools(
    query: str,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """Look up tool locations by name, class, or description."""
    # TODO: Implement tool location lookup
    # Search tools by name/class/description
    # Return drawer and tile locations
    return []