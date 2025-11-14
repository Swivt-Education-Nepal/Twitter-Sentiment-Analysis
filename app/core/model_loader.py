import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class SentimentModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the 2-class sentiment model"""
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = DistilBertTokenizer.from_pretrained(settings.MODEL_NAME)
            
            logger.info("Loading model...")
            self.model = DistilBertForSequenceClassification.from_pretrained(
                settings.MODEL_NAME,
                num_labels=settings.NUM_CLASSES
            )
            
            # Load trained weights
            logger.info(f"Loading weights from {settings.MODEL_PATH_2CLASS}")
            state_dict = torch.load(settings.MODEL_PATH_2CLASS, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def map_probability_to_sentiment(self, positive_prob: float) -> tuple:
        """
        Map positive class probability to 5-class sentiment
        
        Args:
            positive_prob: Probability of positive class (0-1)
            
        Returns:
            tuple: (sentiment_label, predicted_class_index)
        """
        # Check each probability range
        if positive_prob >= 0.80:  # 80-100%
            return "Strongly Positive", 4
        elif positive_prob >= 0.55:  # 55-80%
            return "Positive", 3
        elif positive_prob >= 0.45:  # 45-55%
            return "Neutral", 2
        elif positive_prob >= 0.20:  # 20-45%
            return "Negative", 1
        else:  # 0-20%
            return "Strongly Negative", 0
    
    def predict(self, text: str) -> dict:
        """Predict sentiment for a single text"""
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=settings.MAX_LENGTH,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
            
            # Get probabilities for both classes
            probs = probabilities[0].cpu().numpy()
            negative_prob = float(probs[0])  # Class 0: Negative
            positive_prob = float(probs[1])  # Class 1: Positive
            
            # Map to 5-class sentiment based on positive probability
            sentiment, predicted_class = self.map_probability_to_sentiment(positive_prob)
            
            # Calculate confidence as the distance from neutral (0.5)
            confidence = abs(positive_prob - 0.5) * 2  # Scale to 0-1
            
            # Create 5-class probability distribution
            five_class_probs = {
                "Strongly Negative": negative_prob if positive_prob < 0.20 else 0.0,
                "Negative": negative_prob if 0.20 <= positive_prob < 0.45 else 0.0,
                "Neutral": 1.0 if 0.45 <= positive_prob <= 0.55 else 0.0,
                "Positive": positive_prob if 0.55 <= positive_prob < 0.80 else 0.0,
                "Strongly Positive": positive_prob if positive_prob >= 0.80 else 0.0
            }
            
            # Adjust probabilities to show distribution
            if sentiment == "Strongly Negative":
                five_class_probs["Strongly Negative"] = negative_prob
                five_class_probs["Negative"] = positive_prob * 0.5
            elif sentiment == "Negative":
                five_class_probs["Negative"] = negative_prob
                five_class_probs["Strongly Negative"] = max(0, (0.45 - positive_prob) / 0.25)
                five_class_probs["Neutral"] = max(0, (positive_prob - 0.20) / 0.25)
            elif sentiment == "Neutral":
                five_class_probs["Neutral"] = 1.0 - confidence
                five_class_probs["Positive"] = positive_prob - 0.45
                five_class_probs["Negative"] = 0.55 - positive_prob
            elif sentiment == "Positive":
                five_class_probs["Positive"] = positive_prob
                five_class_probs["Neutral"] = max(0, (0.80 - positive_prob) / 0.25)
                five_class_probs["Strongly Positive"] = max(0, (positive_prob - 0.55) / 0.25)
            else:  # Strongly Positive
                five_class_probs["Strongly Positive"] = positive_prob
                five_class_probs["Positive"] = negative_prob * 0.5
            
            return {
                "text": text,
                "predicted_class": predicted_class,
                "sentiment": sentiment,
                "confidence": float(confidence),
                "positive_probability": positive_prob,
                "negative_probability": negative_prob,
                "probabilities": five_class_probs
            }
        except Exception as e:
            logger.error(f"Error predicting sentiment: {str(e)}")
            raise
    
    def predict_batch(self, texts: list) -> list:
        """Predict sentiment for multiple texts"""
        return [self.predict(text) for text in texts]

# Global model instance
sentiment_model = None

def get_model():
    """Get or create the global model instance"""
    global sentiment_model
    if sentiment_model is None:
        sentiment_model = SentimentModel()
    return sentiment_model