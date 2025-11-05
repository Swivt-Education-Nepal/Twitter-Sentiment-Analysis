# twitter_sentiment_analysis/configs/__init__.py
from .logger import setup_logger

# Initialize global logger for the project
logger = setup_logger("twitter_sentiment_analysis")

def get_logger(name: str = "twitter_sentiment_analysis"):
    """
    Provides a logger instance for any module.
    """
    return setup_logger(name)

__all__ = ["logger", "get_logger"]
