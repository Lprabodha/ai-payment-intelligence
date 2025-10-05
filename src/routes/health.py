"""
Health check and system status routes
"""
from fastapi import APIRouter
from datetime import datetime
from models.schemas import HealthResponse
from config.settings import settings
from database.connection import db_manager

router = APIRouter(tags=["health"])

@router.get("/", response_model=HealthResponse)
def root():
    """Root endpoint with basic API information"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=settings.API_VERSION,
        database="connected" if db_manager.client else "disconnected",
        models={
            "fraud_detection": "available",
            "chargeback_prediction": "available",
            "smart_routing": "available",
            "revenue_forecasting": "available"
        }
    )

@router.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint"""
    try:
        # Test database connection
        db_manager.client.admin.command('ping')
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return HealthResponse(
        status="healthy" if db_status == "connected" else "unhealthy",
        timestamp=datetime.utcnow(),
        version=settings.API_VERSION,
        database=db_status,
        models={
            "fraud_detection": "available",
            "chargeback_prediction": "available", 
            "smart_routing": "available",
            "revenue_forecasting": "available"
        }
    )
