"""
AI Payment Intelligence API - Modular Architecture
Main application entry point with clean separation of concerns
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import configuration and database setup
from config.settings import settings
from database.connection import db_manager
from database.indexes import ensure_database_indexes

# Import route modules
from routes.health import router as health_router
from routes.predictions import router as predictions_router
from routes.webhooks import router as webhooks_router
from routes.risk import router as risk_router
from routes.rdr import router as rdr_router

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    # Validate configuration
    settings.validate_config()
    
    # Initialize database indexes
    ensure_database_indexes()
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.API_TITLE,
        version=settings.API_VERSION,
        description=settings.API_DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health_router)
    app.include_router(predictions_router)
    app.include_router(webhooks_router)
    app.include_router(risk_router)
    app.include_router(rdr_router)  # RDR (Rapid Dispute Resolution)
    
    return app

# Create the application instance
app = create_app()

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    print(f"{settings.API_TITLE} v{settings.API_VERSION} starting up...")
    print(f"Database: {settings.DATABASE_NAME}")
    print(f"Model path: {settings.MODEL_PATH}")
    print("Application startup completed")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    print("Application shutting down...")
    db_manager.close()
    print("Application shutdown completed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8010,
        reload=True,
        log_level="info"
    )