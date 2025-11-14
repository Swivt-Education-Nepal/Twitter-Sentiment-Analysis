from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio

router = APIRouter()

# Request models
class TextRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: List[str]

class TwitterRequest(BaseModel):
    query: str
    max_results: int = 5

@router.post("/analyze")
async def analyze_text(request: Request, text_request: TextRequest):
    """Analyze single text"""
    model_loader = request.app.model_loader
    if not model_loader:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    result = model_loader.predict_sentiment(text_request.text)
    return result

@router.post("/analyze-batch")
async def analyze_batch(request: Request, batch_request: BatchRequest):
    """Analyze multiple texts"""
    model_loader = request.app.model_loader
    if not model_loader:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    results = model_loader.analyze_batch(batch_request.texts)
    
    # Calculate statistics
    sentiment_counts = {}
    for result in results:
        sentiment = result['sentiment']
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
    
    return {
        "results": results,
        "statistics": {
            "total": len(results),
            "sentiment_distribution": sentiment_counts,
            "dominant_sentiment": max(sentiment_counts, key=sentiment_counts.get) if sentiment_counts else "Unknown"
        }
    }

@router.post("/analyze-twitter")
async def analyze_twitter(request: Request, twitter_request: TwitterRequest):
    """Analyze tweets from Twitter"""
    model_loader = request.app.model_loader
    twitter_client = request.app.twitter_client
    
    if not model_loader:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    if not twitter_client:
        raise HTTPException(status_code=503, detail="Twitter client not ready")
    
    # Fetch tweets
    tweets, message = await twitter_client.search_tweets(twitter_request.query, twitter_request.max_results)
    
    # Analyze tweets
    tweet_texts = [tweet['text'] for tweet in tweets]
    analysis_results = model_loader.analyze_batch(tweet_texts)
    
    # Combine results
    for tweet, analysis in zip(tweets, analysis_results):
        tweet.update(analysis)
    
    # Calculate statistics
    sentiment_counts = {}
    for result in analysis_results:
        sentiment = result['sentiment']
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
    
    return {
        "tweets": tweets,
        "statistics": {
            "total_tweets": len(tweets),
            "sentiment_distribution": sentiment_counts,
            "dominant_sentiment": max(sentiment_counts, key=sentiment_counts.get) if sentiment_counts else "Unknown",
            "message": message
        },
        "query": twitter_request.query
    }

@router.get("/health")
async def health_check(request: Request):
    model_loader = request.app.model_loader
    twitter_client = request.app.twitter_client
    
    return {
        "status": "healthy",
        "model_loaded": model_loader is not None,
        "model_type": "real" if hasattr(model_loader, 'model_loaded') and model_loader.model_loaded else "simple",
        "twitter_ready": twitter_client is not None
    }