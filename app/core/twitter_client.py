import tweepy
from app.core.config import settings
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class TwitterClient:
    def __init__(self):
        self.client = None
        self.init_client()
    
    def init_client(self):
        """Initialize Twitter API client"""
        try:
            if settings.TWITTER_BEARER_TOKEN:
                self.client = tweepy.Client(bearer_token=settings.TWITTER_BEARER_TOKEN)
                logger.info("Twitter client initialized successfully")
            else:
                logger.warning("Twitter Bearer Token not found")
        except Exception as e:
            logger.error(f"Error initializing Twitter client: {str(e)}")
            raise
    
    def get_user_tweets(self, username: str, max_results: int = 5) -> List[Dict]:
        """Fetch recent tweets from a user"""
        try:
            if not self.client:
                raise Exception("Twitter client not initialized")
            
            # Get user ID
            user = self.client.get_user(username=username)
            if not user.data:
                raise Exception(f"User @{username} not found")
            
            user_id = user.data.id
            
            # Get tweets
            tweets = self.client.get_users_tweets(
                id=user_id,
                max_results=min(max_results, 100),
                tweet_fields=['created_at', 'public_metrics', 'lang']
            )
            
            if not tweets.data:
                return []
            
            results = []
            for tweet in tweets.data[:max_results]:
                results.append({
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': str(tweet.created_at),
                    'likes': tweet.public_metrics['like_count'],
                    'retweets': tweet.public_metrics['retweet_count'],
                    'replies': tweet.public_metrics['reply_count']
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching user tweets: {str(e)}")
            raise
    
    def get_hashtag_tweets(self, hashtag: str, max_results: int = 5) -> List[Dict]:
        """Fetch recent tweets with a hashtag"""
        try:
            if not self.client:
                raise Exception("Twitter client not initialized")
            
            # Remove # if present
            hashtag = hashtag.lstrip('#')
            
            # Search tweets
            tweets = self.client.search_recent_tweets(
                query=f"#{hashtag} -is:retweet lang:en",
                max_results=min(max_results, 100),
                tweet_fields=['created_at', 'public_metrics', 'author_id']
            )
            
            if not tweets.data:
                return []
            
            results = []
            for tweet in tweets.data[:max_results]:
                results.append({
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': str(tweet.created_at),
                    'likes': tweet.public_metrics['like_count'],
                    'retweets': tweet.public_metrics['retweet_count'],
                    'replies': tweet.public_metrics['reply_count']
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching hashtag tweets: {str(e)}")
            raise

# Global client instance
twitter_client = None

def get_twitter_client():
    """Get or create the global Twitter client instance"""
    global twitter_client
    if twitter_client is None:
        twitter_client = TwitterClient()
    return twitter_client