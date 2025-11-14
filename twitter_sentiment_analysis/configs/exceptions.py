class SentimentAnalysisError(Exception):
    """Base exception for sentiment analysis errors"""
    pass

class ModelLoadingError(SentimentAnalysisError):
    """Raised when model fails to load"""
    pass

class PredictionError(SentimentAnalysisError):
    """Raised when prediction fails"""
    pass

class DataProcessingError(SentimentAnalysisError):
    """Raised when data processing fails"""
    pass