import os
from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

# Skip database imports if external services are disabled
skip_external = os.getenv('SKIP_EXTERNAL_SERVICES', '').lower() in ('true', '1', 'yes')

if not skip_external:
    from batchgrids.database import get_async_db
else:
    get_async_db = None

router = APIRouter()


if skip_external or get_async_db is None:
    # Standalone mode health endpoint
    @router.get("/")
    async def health_check():
        """Health check endpoint - standalone mode."""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "version": "0.1.0",
            "mode": "standalone",
            "database": "disabled",
        }
else:
    # Database mode health endpoint
    @router.get("/")
    async def health_check(db: AsyncSession = Depends(get_async_db)):
        """Health check endpoint - database mode."""
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