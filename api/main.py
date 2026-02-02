"""
Main FastAPI application for the Server Management API.

This module initializes the FastAPI application, registers all routes,
sets up middleware, and handles application lifecycle events.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from api.config import APIConfig
from api.database import get_database
from api.routes import training, system, models, datasets
from api.services.system_monitor import get_system_monitor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events for the API.
    """
    # Startup
    logger.info("Starting Server Management API...")
    
    try:
        # Validate configuration
        APIConfig.validate_config()
        logger.info("Configuration validated successfully")
        
        # Initialize database
        db = get_database()
        logger.info(f"Database initialized at {db.db_path}")
        
        # Start system monitor
        system_monitor = get_system_monitor()
        system_monitor.start_monitoring()
        logger.info("System monitor started")
        
        logger.info("Server Management API started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down Server Management API...")
    
    try:
        # Stop system monitor
        system_monitor = get_system_monitor()
        system_monitor.stop_monitoring()
        logger.info("System monitor stopped")
        
        # Close database
        db = get_database()
        db.close()
        logger.info("Database closed")
        
        logger.info("Server Management API shut down successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title="Server Management API",
    description="RESTful API for managing PyTorch model training server",
    version="1.0.0",
    lifespan=lifespan
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler for consistent error responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Handle all unhandled exceptions with consistent error format.
    
    Args:
        request: The request that caused the exception
        exc: The exception that was raised
        
    Returns:
        JSON error response
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "details": str(exc) if APIConfig.HOST == "0.0.0.0" else None
            }
        }
    )


# Register routers
app.include_router(training.router)
app.include_router(system.router)
app.include_router(models.router)
app.include_router(datasets.router)


# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "service": "Server Management API",
        "version": "1.0.0"
    }


# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        API information
    """
    return {
        "name": "Server Management API",
        "version": "1.0.0",
        "description": "RESTful API for managing PyTorch model training server",
        "docs": "/docs",
        "health": "/health"
    }
