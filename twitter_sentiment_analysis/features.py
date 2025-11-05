# twitter_sentiment_analysis/features.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast

from twitter_sentiment_analysis.configs import get_logger
from twitter_sentiment_analysis.configs.exceptions import DataProcessingError

# Initialize logger
log = get_logger(__name__)


# ================================================================
# 🧠 Custom Dataset Class
# ================================================================
class TweetDataset(Dataset):
    """
    Custom PyTorch dataset for tweets with tokenization support.
    Each sample returns a dictionary: {input_ids, attention_mask, labels}.
    """

    def __init__(
        self,
        texts,
        labels,
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 128
    ):
        try:
            log.info(f"Initializing tokenizer: {tokenizer_name}")
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_name)
            self.texts = texts.tolist() if hasattr(texts, "tolist") else list(texts)
            self.labels = labels.tolist() if hasattr(labels, "tolist") else list(labels)
            self.max_length = max_length
            log.info(f"Loaded {len(self.texts)} samples for tokenization.")
        except Exception as e:
            log.error(f"Tokenizer initialization failed: {e}")
            raise DataProcessingError(f"Tokenizer initialization failed: {e}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        """
        Tokenizes a single tweet and returns encoded inputs and label.
        """
        try:
            text = self.texts[idx]
            label = self.labels[idx]
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )

            item = {key: val.squeeze(0) for key, val in encoding.items()}
            item["labels"] = torch.tensor(label, dtype=torch.long)
            return item

        except Exception as e:
            log.error(f"Error tokenizing sample index {idx}: {e}")
            raise DataProcessingError(f"Tokenization failed for sample {idx}: {e}")


# ================================================================
# ⚙️ DataLoader Creation Function
# ================================================================
def create_dataloader(
    df,
    text_col: str = "text",
    label_col: str = "sentiment",
    tokenizer_name: str = "distilbert-base-uncased",
    max_length: int = 128,
    batch_size: int = 16,
    shuffle: bool = True
) -> DataLoader:
    """
    Converts a cleaned DataFrame into a PyTorch DataLoader.

    Args:
        df (pd.DataFrame): Dataset with text and sentiment columns.
        text_col (str): Column containing the tweet text.
        label_col (str): Column containing sentiment labels.
        tokenizer_name (str): Hugging Face tokenizer name.
        max_length (int): Maximum token length for each text.
        batch_size (int): Dataloader batch size.
        shuffle (bool): Whether to shuffle samples.

    Returns:
        torch.utils.data.DataLoader: Ready-to-use DataLoader.
    """
    try:
        if text_col not in df.columns or label_col not in df.columns:
            raise DataProcessingError(
                f"Required columns '{text_col}' and '{label_col}' not found in DataFrame."
            )

        log.info(
            f"Creating DataLoader with batch_size={batch_size}, shuffle={shuffle}, max_length={max_length}"
        )

        dataset = TweetDataset(
            texts=df[text_col],
            labels=df[label_col],
            tokenizer_name=tokenizer_name,
            max_length=max_length
        )

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        log.info(f"DataLoader ready with {len(dataloader)} batches.")
        return dataloader

    except Exception as e:
        log.error(f"Failed to create DataLoader: {e}")
        raise DataProcessingError(str(e))


# ================================================================
# 🧪 Run Standalone (Debug Mode)
# ================================================================
if __name__ == "__main__":
    import pandas as pd

    try:
        log.info("Testing feature module...")
        df = pd.DataFrame({
            "text": ["I love this!", "This is bad...", "Totally neutral today."],
            "sentiment": [2, 0, 1]
        })
        loader = create_dataloader(df)
        for batch in loader:
            log.info(f"Sample batch: {batch['input_ids'].shape}")
            break
    except Exception as e:
        log.error(f"Feature module test failed: {e}")
