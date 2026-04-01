"""GEAK Online Service - Main Entry Point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.config import get_settings
from server.database import init_db
from server.api.routes.tasks import router as tasks_router
from server.api.routes.config import router as config_router
from server.mcp.http_server import create_mcp_http_app
from server.services.task_scheduler import start_scheduler, stop_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    await init_db()
    await start_scheduler()  # Start background task scheduler
    yield
    # Shutdown
    await stop_scheduler()  # Stop background task scheduler


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="GEAK Online Service",
        description="GPU/HIP Kernel Optimization Service powered by mini-swe-agent",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(tasks_router, prefix="/api/v1")
    app.include_router(config_router, prefix="/api/v1")
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "geak-online"}
    
    @app.get("/")
    async def root():
        """Root endpoint with service info."""
        return {
            "service": "GEAK Online Service",
            "version": "1.0.0",
            "docs": "/docs",
            "mcp": "/mcp",
        }
    
    # Mount MCP HTTP server
    mcp_app = create_mcp_http_app()
    app.mount("/mcp", mcp_app)
    
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "server.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
