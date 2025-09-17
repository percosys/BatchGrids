from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import os

# Skip database-dependent imports if external services are disabled
skip_external = os.getenv('SKIP_EXTERNAL_SERVICES', '').lower() in ('true', '1', 'yes')

if not skip_external:
    from batchgrids.api.routers import auth, bins, drawers, exports, health, images, lookup, tools, web
    from batchgrids.database import async_engine
    from batchgrids.models import Base
else:
    # Only import vision-related routers for standalone mode
    from batchgrids.api.routers import health, web
    async_engine = None
    Base = None

from fastapi.staticfiles import StaticFiles
from pathlib import Path
from batchgrids.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    if not skip_external and async_engine and settings.is_development:
        # Create tables in development mode with retry to avoid race with DB readiness
        max_attempts = 3  # Reduced attempts for faster startup
        attempt = 0
        while attempt < max_attempts:
            try:
                async with async_engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                print("Database tables created successfully")
                break
            except Exception as e:
                print(f"Database connection attempt {attempt + 1} failed: {e}")
                attempt += 1
                if attempt >= max_attempts:
                    # Give up but don't crash app; DB-dependent endpoints will report unhealthy
                    print("Database connection failed - continuing without database (vision features still available)")
                    break
                await asyncio.sleep(0.5 * attempt)
    elif skip_external:
        print("Skipping external services - running in standalone mode")
    
    yield
    
    # Shutdown
    if not skip_external and async_engine:
        try:
            await async_engine.dispose()
        except Exception:
            pass  # Ignore errors during shutdown


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="BatchGrids API",
        description="Tool management and organization application with computer vision",
        version="0.1.0",
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(web.router, prefix="/web", tags=["web"])
    
    if not skip_external:
        # Include database-dependent routers only when external services are available
        app.include_router(auth.router, prefix="/auth", tags=["auth"])
        app.include_router(images.router, prefix="/images", tags=["images"])
        app.include_router(tools.router, prefix="/tools", tags=["tools"])
        app.include_router(bins.router, prefix="/bins", tags=["bins"])
        app.include_router(drawers.router, prefix="/drawers", tags=["drawers"])
        app.include_router(lookup.router, prefix="/lookup", tags=["lookup"])
        app.include_router(exports.router, prefix="/exports", tags=["exports"])

    # Serve a simple frontend (if present)
    # Serve from project root /web (parents[3] from src/batchgrids/api/main.py)
    static_dir = Path(__file__).resolve().parents[3] / "web"
    try:
        if static_dir.exists():
            app.mount("/app", StaticFiles(directory=str(static_dir), html=True), name="app")
    except Exception:
        pass

    return app


app = create_app()
