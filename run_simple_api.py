#!/usr/bin/env python3
"""
Simple API runner without complex model loading
"""

from fastapi import FastAPI
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Twitter Sentiment Analysis API",
    description="Real-time Twitter sentiment analysis",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "Twitter Sentiment Analysis API is running!"}

@app.get("/api/v1/health")
async def health():
    return {"status": "healthy", "models_loaded": False, "twitter_connected": False}

@app.post("/api/v1/sentiment")
async def analyze_sentiment(request: dict):
    text = request.get("text", "")
    # Simple mock sentiment analysis
    text_lower = text.lower()
    if any(word in text_lower for word in ['good', 'great', 'amazing', 'love', 'excellent']):
        return {"sentiment": "positive", "confidence": 0.85, "text": text}
    elif any(word in text_lower for word in ['bad', 'terrible', 'awful', 'hate', 'worst']):
        return {"sentiment": "negative", "confidence": 0.85, "text": text}
    else:
        return {"sentiment": "neutral", "confidence": 0.5, "text": text}

if __name__ == "__main__":
    logger.info("Starting simple API server...")
    uvicorn.run(
        "run_simple_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )