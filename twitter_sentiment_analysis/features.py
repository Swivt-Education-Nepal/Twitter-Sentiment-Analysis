import re
import contractions
import emoji
import pandas as pd
from typing import Callable

def clean_text(text: str) -> str:
    """
    Clean text by removing URLs, mentions, emojis, etc.
    """
    if not isinstance(text, str):
        return ""
    
    # Fix contractions
    text = contractions.fix(text)
    
    # Remove emojis
    text = emoji.replace_emoji(text, replace="")
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # Remove mentions
    text = re.sub(r"@\w+", "", text)
    
    # Remove hashtags but keep text
    text = re.sub(r"#(\w+)", r"\1", text)
    
    # Remove special characters except alphanumeric and spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip().lower()
    
    return text

def preprocess_dataframe(df: pd.DataFrame, text_column: str = 'text', 
                        target_column: str = 'target', clean_func: Callable = None) -> pd.DataFrame:
    """
    Preprocess a DataFrame for sentiment analysis
    """
    if clean_func is None:
        clean_func = clean_text
    
    # Create clean text column
    df_clean = df.copy()
    df_clean['clean_text'] = df_clean[text_column].astype(str).apply(clean_func)
    
    # Remove empty texts
    df_clean = df_clean[df_clean['clean_text'].str.len() > 0]
    df_clean = df_clean.dropna(subset=['clean_text']).reset_index(drop=True)
    
    return df_clean

def create_text_cleaner(**kwargs):
    """Factory function to create a text cleaner with custom parameters"""
    def custom_cleaner(text):
        # Start with basic cleaning
        text = clean_text(text)
        
        # Apply additional custom cleaning if needed
        if kwargs.get('remove_numbers', False):
            text = re.sub(r'\d+', '', text)
            
        if kwargs.get('min_length', 0) > 0:
            if len(text.split()) < kwargs['min_length']:
                text = ""
                
        return text.strip()
    
    return custom_cleaner