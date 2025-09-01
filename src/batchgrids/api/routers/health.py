from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from batchgrids.database import get_async_db

router = APIRouter()


@router.get("/")
async def health_check(db: AsyncSession = Depends(get_async_db)):
    """Health check endpoint."""
    try:
        # Test database connection
        await db.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "0.1.0",
        "database": db_status,
    }