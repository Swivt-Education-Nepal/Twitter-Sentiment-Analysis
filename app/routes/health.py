from fastapi import APIRouter
from app.models import HealthResponse
from app.core.config import settings
from app.core.model_loader import get_model
from app.core.twitter_client import get_twitter_client

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        model = get_model()
        model_loaded = model.model is not None
    except:
        model_loaded = False
    
    try:
        client = get_twitter_client()
        twitter_configured = client.client is not None
    except:
        twitter_configured = False
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        message="API is running",
        model_loaded=model_loaded,
        twitter_configured=twitter_configured
    )