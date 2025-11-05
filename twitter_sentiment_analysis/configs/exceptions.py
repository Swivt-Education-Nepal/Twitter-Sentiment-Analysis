# twitter_sentiment_analysis/configs/exceptions.py

class TwitterSentimentError(Exception):
    """Base class for all custom exceptions."""
    pass


class DataLoadError(TwitterSentimentError):
    """Raised when loading dataset fails."""
    def __init__(self, message="Error loading dataset. Check file path or format."):
        super().__init__(message)


class DataProcessingError(TwitterSentimentError):
    """Raised when data cleaning or preprocessing fails."""
    def __init__(self, message="Error during data processing step."):
        super().__init__(message)


class ModelTrainingError(TwitterSentimentError):
    """Raised when training process fails."""
    def __init__(self, message="Model training failed. Check logs for details."):
        super().__init__(message)


class PredictionError(TwitterSentimentError):
    """Raised when model prediction fails."""
    def __init__(self, message="Prediction failed. Check input data or model path."):
        super().__init__(message)
