import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_cosine_schedule_with_warmup
)
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from ..config import TrainingConfig
from ..dataset import SentimentDataset
from ..features import preprocess_dataframe, train_val_split

def set_seeds(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class DistilBertTrainer:
    """Main trainer class for DistilBERT sentiment analysis"""
    
    def __init__(self, config=None):
        self.config = config or TrainingConfig()
        set_seeds(self.config.SEED)
        
        # Load data directly from dataset (which handles MongoDB)
        print("📊 Loading and preprocessing data...")
        raw_dataset = SentimentDataset(use_mongodb=True)
        raw_df = raw_dataset.get_dataframe()
        
        # Preprocess data
        self.df = preprocess_dataframe(raw_df)
        self.train_df, self.val_df = train_val_split(
            self.df, test_size=0.2, random_state=self.config.SEED
        )
        
        print(f"✅ Final dataset: {self.df.shape}")
        print(f"📊 Class distribution:\n{self.df['target'].value_counts().sort_index()}")
        
        # Initialize model components
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.config.MODEL_NAME)
        self.model = self._initialize_model()
        self.optimizer, self.scheduler, self.scaler = self._initialize_optimizer()
        
        # Data loaders
        self.train_loader, self.val_loader = self._create_data_loaders()
        
    def _initialize_model(self):
        """Initialize the model"""
        print("🧠 Initializing model...")
        model = DistilBertForSequenceClassification.from_pretrained(
            self.config.MODEL_NAME, 
            num_labels=5
        ).to(self.config.device)
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Trainable parameters: {total_params:,}")
        
        return model
    
    def _initialize_optimizer(self):
        """Initialize optimizer, scheduler and scaler"""
        optimizer = AdamW(self.model.parameters(), lr=self.config.LEARNING_RATE)
        
        total_steps = len(self.train_loader) * self.config.EPOCHS // self.config.ACCUMULATION_STEPS
        warmup_steps = int(self.config.WARMUP_PROPORTION * total_steps)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        scaler = GradScaler()
        
        return optimizer, scheduler, scaler
    
    def _create_data_loaders(self):
        """Create data loaders for training and validation"""
        print("📦 Creating data loaders...")
        
        # Use clean_text column from preprocessing
        train_dataset = SentimentDataset(
            df=self.train_df, 
            tokenizer=self.tokenizer, 
            max_len=self.config.MAX_LEN,
            use_mongodb=False,  # We already have the DataFrame
            text_column='clean_text'
        )
        val_dataset = SentimentDataset(
            df=self.val_df, 
            tokenizer=self.tokenizer, 
            max_len=self.config.MAX_LEN,
            use_mongodb=False,  # We already have the DataFrame
            text_column='clean_text'
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True
        )
        
        print(f"✅ Train: {len(self.train_df)}, Val: {len(self.val_df)}")
        return train_loader, val_loader
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for step, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch} Training")):
            ids = batch['input_ids'].to(self.config.device)
            masks = batch['attention_mask'].to(self.config.device)
            labels = batch['labels'].to(self.config.device)
            
            with autocast():
                out = self.model(input_ids=ids, attention_mask=masks, labels=labels)
                loss = out.loss / self.config.ACCUMULATION_STEPS
            
            self.scaler.scale(loss).backward()
            
            if (step + 1) % self.config.ACCUMULATION_STEPS == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
            
            train_loss += loss.item() * self.config.ACCUMULATION_STEPS * ids.size(0)
            preds = torch.argmax(out.logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        avg_train_loss = train_loss / len(self.train_loader.dataset)
        train_acc = train_correct / train_total
        
        return avg_train_loss, train_acc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                ids = batch['input_ids'].to(self.config.device)
                masks = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                
                out = self.model(input_ids=ids, attention_mask=masks, labels=labels)
                val_loss += out.loss.item() * ids.size(0)
                preds = torch.argmax(out.logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        avg_val_loss = val_loss / len(self.val_loader.dataset)
        val_acc = val_correct / val_total
        
        return avg_val_loss, val_acc
    
    def train(self):
        """Main training loop"""
        best_val_loss = float('inf')
        epochs_no_improve = 0
        metrics_log = []
        
        print(f"🟩 Using device: {self.config.device}")
        print("🚀 Starting training...")
        
        for epoch in range(1, self.config.EPOCHS + 1):
            print(f"\n==================== EPOCH {epoch}/{self.config.EPOCHS} ====================")
            start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_acc = self.validate_epoch()
            
            epoch_time = time.time() - start_time
            
            print(f"⏱️ Epoch Time: {epoch_time:.2f}s | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            
            # Log metrics
            metrics_log.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "epoch_time_sec": epoch_time
            })
            
            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), "models/best_model.pt")
                print("💾 Best model saved!")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"🛑 Early stopping triggered after {epoch} epochs.")
                    break
        
        # Save metrics and final model
        self._save_results(metrics_log)
        
        return metrics_log
    
    def _save_results(self, metrics_log):
        """Save training results and metrics"""
        # Ensure directories exist
        os.makedirs("reports", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        metrics_df = pd.DataFrame(metrics_log)
        metrics_df.to_csv("reports/train_val_metrics_5class.csv", index=False)
        print(f"✅ Metrics saved to reports/train_val_metrics_5class.csv")
        
        # Save final model
        torch.save(self.model.state_dict(), "models/distilbert_sentiment_5class_final.pt")
        print("✅ Final model saved as models/distilbert_sentiment_5class_final.pt")

def main():
    """Main function to run training"""
    trainer = DistilBertTrainer()
    metrics = trainer.train()
    print("🎉 Training completed!")
    return metrics

if __name__ == "__main__":
    main()