import time
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager

import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.database import init_db
from src.api.routers import customers

# Initialize logger and config
logger = get_logger(__name__)
config = get_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for the FastAPI application."""
    # Startup
    logger.info("Starting E-commerce Analytics API")
    
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down E-commerce Analytics API")


# Create FastAPI app
app = FastAPI(
    title=config.get("app.name", "E-commerce Analytics Platform"),
    description="API for e-commerce analytics and machine learning services",
    version=config.get("app.version", "0.1.0"),
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and their processing time."""
    start_time = time.time()
    
    # Process the request
    response = await call_next(request)
    
    # Log request details
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} "
        f"- Status: {response.status_code} "
        f"- Time: {process_time:.4f}s"
    )
    
    return response


# Root route
@app.get("/")
async def root():
    """Root endpoint to check if API is running."""
    return {
        "name": config.get("app.name", "E-commerce Analytics Platform"),
        "version": config.get("app.version", "0.1.0"),
        "status": "running"
    }


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# Include routers
app.include_router(customers.router, prefix="/customers", tags=["customers"])


# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom documentation information
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


if __name__ == "__main__":
    """Run the API server when executed directly."""
    import uvicorn
    
    host = config.get("api.host", "0.0.0.0")
    port = config.get("api.port", 8000)
    
    uvicorn.run(
        "main:app",
        host=host, 
        port=port,
        reload=config.get("app.debug", True),
        workers=config.get("api.workers", 1)
    ) 