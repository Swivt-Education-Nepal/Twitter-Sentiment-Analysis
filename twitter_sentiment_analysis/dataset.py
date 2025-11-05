# twitter_sentiment_analysis/dataset.py

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

from twitter_sentiment_analysis.configs import get_logger
from twitter_sentiment_analysis.configs.exceptions import DataLoadError, DataProcessingError

# Initialize logger for this module
log = get_logger(__name__)

# ================================================================
# 🧹 Text Cleaning
# ================================================================
def clean_tweet(text: str) -> str:
    """
    Basic text cleaning for tweets:
    - Converts to lowercase
    - Removes URLs, mentions, hashtags, special chars
    - Collapses multiple spaces
    """
    try:
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
        text = re.sub(r"@\w+", "", text)                    # remove mentions
        text = re.sub(r"#", "", text)                       # remove hashtags
        text = re.sub(r"[^a-z\s]", "", text)                # remove punctuation, numbers
        text = re.sub(r"\s+", " ", text).strip()            # normalize whitespace
        return text
    except Exception as e:
        log.error(f"Text cleaning failed: {e}")
        raise DataProcessingError(str(e))


# ================================================================
# 📂 Load Dataset
# ================================================================
def load_dataset(
    path: str,
    text_col: str = "text",
    label_col: str = "sentiment",
    dropna: bool = True
) -> pd.DataFrame:
    """
    Load and clean a CSV dataset for sentiment analysis.

    Args:
        path (str): Path to dataset file.
        text_col (str): Name of the text column.
        label_col (str): Name of the label column.
        dropna (bool): Whether to drop rows with missing values.

    Returns:
        pd.DataFrame: Cleaned dataframe with [text, sentiment]
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        log.info(f"Loading dataset from: {path}")
        df = pd.read_csv(path)

        if text_col not in df.columns or label_col not in df.columns:
            raise DataLoadError(
                f"Columns '{text_col}' and '{label_col}' must exist in dataset. "
                f"Found columns: {list(df.columns)}"
            )

        # Drop missing values
        if dropna:
            missing = df[[text_col, label_col]].isna().sum().sum()
            if missing > 0:
                log.warning(f"Dropping {missing} missing values.")
                df = df.dropna(subset=[text_col, label_col])

        # Clean the text column
        log.info("Cleaning text column...")
        df[text_col] = df[text_col].astype(str).apply(clean_tweet)

        log.info(f"Dataset loaded successfully: {len(df)} rows")
        return df[[text_col, label_col]]

    except Exception as e:
        log.error(f"Dataset loading failed: {e}")
        raise DataLoadError(str(e))


# ================================================================
# ✂️ Split Dataset
# ================================================================
def split_dataset(
    df: pd.DataFrame,
    label_col: str = "sentiment",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Split dataset into train and test sets with stratification.

    Args:
        df (pd.DataFrame): Input dataframe.
        label_col (str): Target column for stratification.
        test_size (float): Fraction for test set.
        random_state (int): Reproducibility seed.

    Returns:
        (pd.DataFrame, pd.DataFrame): (train_df, test_df)
    """
    try:
        log.info(f"Splitting dataset (test size = {test_size})...")
        train_df, test_df = train_test_split(
            df, test_size=test_size, stratify=df[label_col], random_state=random_state
        )

        log.info(f"Split complete → Train: {len(train_df)}, Test: {len(test_df)}")
        return train_df, test_df

    except Exception as e:
        log.error(f"Dataset split failed: {e}")
        raise DataProcessingError(str(e))


# ================================================================
# 🧪 Run Standalone (for debugging)
# ================================================================
if __name__ == "__main__":
    sample_path = "data/raw/tweets.csv"
    try:
        df = load_dataset(sample_path)
        train_df, test_df = split_dataset(df)
        log.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    except Exception as e:
        log.error(f"Dataset module test failed: {e}")
