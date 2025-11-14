from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_TITLE: str = "Twitter Sentiment Analysis"
    API_VERSION: str = "1.0.0"
    
    # Twitter API Keys
    TWITTER_API_KEY: Optional[str] = None
    TWITTER_API_SECRET: Optional[str] = None
    TWITTER_ACCESS_TOKEN: Optional[str] = None
    TWITTER_ACCESS_TOKEN_SECRET: Optional[str] = None
    TWITTER_BEARER_TOKEN: Optional[str] = None
    
    # Model Settings
    MODEL_PATH_2CLASS: str = "models/distilbert_sentiment_2class_final.pt"
    MODEL_NAME: str = "distilbert-base-uncased"
    MAX_LENGTH: int = 128
    NUM_CLASSES: int = 2  # Using 2-class model
    
    # Sentiment Labels (mapped from probability)
    SENTIMENT_LABELS: dict = {
        0: "Strongly Negative",
        1: "Negative",
        2: "Neutral",
        3: "Positive",
        4: "Strongly Positive"
    }
    
    # Probability ranges for 5-class mapping
    # Based on positive class probability (class 1)
    PROBABILITY_RANGES: dict = {
        "Strongly Positive": (0.80, 1.00),   # 80-100%
        "Positive": (0.55, 0.80),             # 55-80%
        "Neutral": (0.45, 0.55),              # 45-55%
        "Negative": (0.20, 0.45),             # 20-45%
        "Strongly Negative": (0.00, 0.20)    # 0-20%
    }
    
    # Twitter Settings
    MAX_TWEETS: int = 5
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()