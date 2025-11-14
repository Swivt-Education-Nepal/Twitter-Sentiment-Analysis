from fastapi import APIRouter, HTTPException
from app.models import TwitterAnalysisRequest, TwitterAnalysisResponse, TweetSentiment
from app.core.model_loader import get_model
from app.core.twitter_client import get_twitter_client
from app.core.config import settings

router = APIRouter()

@router.post("/analyze/twitter", response_model=TwitterAnalysisResponse)
async def analyze_twitter(request: TwitterAnalysisRequest):
    """Analyze sentiment of recent tweets from a user or hashtag"""
    try:
        # Get Twitter client and model
        twitter_client = get_twitter_client()
        model = get_model()
        
        # Fetch tweets based on query type
        if request.query_type.lower() == "user":
            tweets = twitter_client.get_user_tweets(request.query, request.max_results)
        elif request.query_type.lower() == "hashtag":
            tweets = twitter_client.get_hashtag_tweets(request.query, request.max_results)
        else:
            raise HTTPException(status_code=400, detail="query_type must be 'user' or 'hashtag'")
        
        if not tweets:
            raise HTTPException(status_code=404, detail=f"No tweets found for {request.query}")
        
        # Analyze sentiment for each tweet
        analyzed_tweets = []
        sentiment_counts = {label: 0 for label in settings.SENTIMENT_LABELS.values()}
        sentiment_scores = []
        
        for tweet in tweets:
            sentiment_result = model.predict(tweet['text'])
            
            analyzed_tweets.append(TweetSentiment(
                tweet_id=str(tweet['id']),
                text=tweet['text'],
                created_at=tweet['created_at'],
                likes=tweet['likes'],
                retweets=tweet['retweets'],
                replies=tweet['replies'],
                sentiment=sentiment_result['sentiment'],
                confidence=sentiment_result['confidence'],
                probabilities=sentiment_result['probabilities']
            ))
            
            sentiment_counts[sentiment_result['sentiment']] += 1
            sentiment_scores.append(sentiment_result['predicted_class'])
        
                    # Calculate average sentiment score (0-4 scale)
        avg_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 2.0
        
        # Calculate average positive probability
        avg_positive_prob = sum(
            model.predict(tweet['text'])['positive_probability'] 
            for tweet in tweets
        ) / len(tweets) if tweets else 0.5
        
        return TwitterAnalysisResponse(
            query=request.query,
            query_type=request.query_type,
            total_tweets=len(analyzed_tweets),
            tweets=analyzed_tweets,
            summary=sentiment_counts,
            average_sentiment_score=avg_positive_prob  # Use positive probability as score
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing tweets: {str(e)}")