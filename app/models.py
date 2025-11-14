from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class TextInput(BaseModel):
    text: str = Field(..., description="Text to analyze")

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze")

class SentimentResponse(BaseModel):
    text: str
    predicted_class: int
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]

class TwitterAnalysisRequest(BaseModel):
    query: str = Field(..., description="Username (without @) or hashtag (with or without #)")
    query_type: str = Field(..., description="Type of query: 'user' or 'hashtag'")
    max_results: int = Field(5, ge=1, le=10, description="Maximum number of tweets to analyze")

class TweetSentiment(BaseModel):
    tweet_id: str
    text: str
    created_at: str
    likes: int
    retweets: int
    replies: int
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]

class TwitterAnalysisResponse(BaseModel):
    query: str
    query_type: str
    total_tweets: int
    tweets: List[TweetSentiment]
    summary: Dict[str, int]
    average_sentiment_score: float

class HealthResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool
    twitter_configured: bool