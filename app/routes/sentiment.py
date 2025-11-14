from fastapi import APIRouter, HTTPException
from app.models import TextInput, BatchTextInput, SentimentResponse
from app.core.model_loader import get_model
from typing import List

router = APIRouter()

@router.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(input_data: TextInput):
    """Analyze sentiment of a single text"""
    try:
        model = get_model()
        result = model.predict(input_data.text)
        return SentimentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")

@router.post("/sentiment/batch", response_model=List[SentimentResponse])
async def analyze_sentiment_batch(input_data: BatchTextInput):
    """Analyze sentiment of multiple texts"""
    try:
        model = get_model()
        results = model.predict_batch(input_data.texts)
        return [SentimentResponse(**result) for result in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiments: {str(e)}")