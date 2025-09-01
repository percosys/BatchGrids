from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from batchgrids.api.routers import auth, bins, drawers, exports, health, images, lookup, tools
from batchgrids.config import settings
from batchgrids.database import async_engine
from batchgrids.models import Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    if settings.is_development:
        # Create tables in development mode
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    yield
    
    # Shutdown
    await async_engine.dispose()


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
    app.include_router(auth.router, prefix="/auth", tags=["auth"])
    app.include_router(images.router, prefix="/images", tags=["images"])
    app.include_router(tools.router, prefix="/tools", tags=["tools"])
    app.include_router(bins.router, prefix="/bins", tags=["bins"])
    app.include_router(drawers.router, prefix="/drawers", tags=["drawers"])
    app.include_router(lookup.router, prefix="/lookup", tags=["lookup"])
    app.include_router(exports.router, prefix="/exports", tags=["exports"])

    return app


app = create_app()