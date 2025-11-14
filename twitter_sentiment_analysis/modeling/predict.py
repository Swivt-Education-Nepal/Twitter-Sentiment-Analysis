import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict, Any
from ..dataset import PredictionDataset
from ..configs.logger import logger

class SentimentPredictor:
    """Handles batch predictions with the trained model"""
    
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def predict_batch(self, texts: List[str], batch_size: int = 32, 
                     clean_function=None) -> List[Dict[str, Any]]:
        """Predict sentiment for a batch of texts"""
        
        # Clean texts if function provided
        if clean_function:
            texts = [clean_function(text) for text in texts]
        
        # Create dataset and dataloader
        dataset = PredictionDataset(texts, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Get predictions
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_classes = torch.argmax(probs, dim=1)
                confidences = torch.max(probs, dim=1).values
                
                # Convert to Python types
                batch_preds = predicted_classes.cpu().numpy()
                batch_confs = confidences.cpu().numpy()
                
                for pred, conf in zip(batch_preds, batch_confs):
                    predictions.append({
                        'predicted_class': int(pred),
                        'confidence': float(conf)
                    })
        
        return predictions
    
    def predict_dataframe(self, df: pd.DataFrame, text_column: str = 'text',
                         batch_size: int = 32, clean_function=None) -> pd.DataFrame:
        """Predict sentiment for a DataFrame and return enriched DataFrame"""
        
        texts = df[text_column].tolist()
        predictions = self.predict_batch(texts, batch_size, clean_function)
        
        # Add predictions to dataframe
        result_df = df.copy()
        result_df['predicted_sentiment'] = [p['predicted_class'] for p in predictions]
        result_df['confidence'] = [p['confidence'] for p in predictions]
        
        # Map class IDs to sentiment labels
        sentiment_map = {0: 'negative', 1: 'positive'}
        result_df['sentiment_label'] = result_df['predicted_sentiment'].map(sentiment_map)
        
        return result_df