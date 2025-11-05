# twitter_sentiment_analysis/modeling/train.py
import os
import time
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW

from twitter_sentiment_analysis.configs import get_logger
from twitter_sentiment_analysis.configs.exceptions import ModelTrainingError
from twitter_sentiment_analysis.dataset import load_dataset, split_dataset
from twitter_sentiment_analysis.features import create_dataloader

log = get_logger(__name__)

# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_artifacts(model, tokenizer, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "best_model.pt")
    tokenizer_path = os.path.join(save_dir, "tokenizer")
    torch.save(model.state_dict(), model_path)
    tokenizer.save_pretrained(tokenizer_path)
    log.info(f"Saved model to {model_path} and tokenizer to {tokenizer_path}")


# -------------------------
# Training function
# -------------------------
def train(
    data_path: str,
    model_name: str = "distilbert-base-uncased",
    max_len: int = 256,
    batch_size: int = 32,
    accumulation_steps: int = 8,
    epochs: int = 10,
    learning_rate: float = 2e-5,
    warmup_proportion: float = 0.1,
    sample_size: int = None,
    seed: int = 42,
    early_stopping_patience: int = 2,
    device: str = None,
    output_dir: str = "models",
):
    try:
        set_seed(seed)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {device}")

        # -------------------------
        # Load & prepare data
        # -------------------------
        log.info("Loading dataset...")
        df = load_dataset(data_path, text_col="text", label_col="sentiment", dropna=True)

        if sample_size is not None and sample_size > 0 and len(df) > sample_size:
            log.info(f"Sampling {sample_size} rows from dataset")
            df = df.sample(sample_size, random_state=seed).reset_index(drop=True)

        texts = df["text"].tolist()
        labels = df["sentiment"].astype(int).tolist()

        # stratified splits using sklearn (works with Python lists by converting inside)
        train_texts, temp_texts, train_labels, temp_labels = \
            __import__("sklearn.model_selection").model_selection.train_test_split(
                texts, labels, test_size=0.2, stratify=labels, random_state=seed
            )
        val_texts, test_texts, val_labels, test_labels = \
            __import__("sklearn.model_selection").model_selection.train_test_split(
                temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=seed
            )

        # create dataframes (create_dataloader expects a DataFrame)
        train_df = pd.DataFrame({"text": train_texts, "sentiment": train_labels})
        val_df = pd.DataFrame({"text": val_texts, "sentiment": val_labels})
        test_df = pd.DataFrame({"text": test_texts, "sentiment": test_labels})

        train_loader = create_dataloader(train_df, max_length=max_len, batch_size=batch_size, shuffle=True)
        val_loader = create_dataloader(val_df, max_length=max_len, batch_size=batch_size, shuffle=False)
        test_loader = create_dataloader(test_df, max_length=max_len, batch_size=batch_size, shuffle=False)

        # -------------------------
        # Model, optimizer, scheduler
        # -------------------------
        log.info(f"Loading model: {model_name}")
        model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

        optimizer = AdamW(model.parameters(), lr=learning_rate)

        total_steps = (len(train_loader) * epochs) // accumulation_steps
        num_warmup_steps = int(warmup_proportion * total_steps) if total_steps > 0 else 0
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, total_steps) if total_steps > 0 else None

        scaler = GradScaler()
        criterion = nn.CrossEntropyLoss()

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Trainable parameters: {total_params:,}")

        # -------------------------
        # Training loop
        # -------------------------
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            log.info(f"========== Epoch {epoch}/{epochs} ==========")
            epoch_start = time.time()

            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} Train")
            for step, batch in pbar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_batch = batch["labels"].to(device)

                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
                    loss = outputs.loss / accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()

                # accumulate stats (scale back up)
                running_loss += (loss.item() * accumulation_steps) * input_ids.size(0)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels_batch).sum().item()
                total += labels_batch.size(0)

                pbar.set_postfix({"loss": f"{running_loss/total:.4f}", "acc": f"{correct/total:.4f}"})

            avg_train_loss = running_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0.0
            train_acc = correct / total if total > 0 else 0.0

            # -------------------------
            # Validation
            # -------------------------
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch} Val"):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels_batch = batch["labels"].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
                    val_loss += outputs.loss.item() * input_ids.size(0)
                    preds = torch.argmax(outputs.logits, dim=1)
                    val_correct += (preds == labels_batch).sum().item()
                    val_total += labels_batch.size(0)

            avg_val_loss = val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0.0
            val_acc = val_correct / val_total if val_total > 0 else 0.0

            epoch_time = time.time() - epoch_start
            log.info(
                f"Epoch {epoch} done in {epoch_time:.1f}s | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
            )

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                save_artifacts(model, tokenizer, output_dir)
            else:
                epochs_no_improve += 1
                log.info(f"No improvement count: {epochs_no_improve}/{early_stopping_patience}")
                if epochs_no_improve >= early_stopping_patience:
                    log.info("Early stopping triggered.")
                    break

        log.info("Training finished.")

        # Evaluation on test set
        log.info("Running final evaluation on test set...")
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_batch = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels_batch.cpu().tolist())

        report = classification_report(all_labels, all_preds, digits=4)
        log.info("Test classification report:\n" + report)

    except Exception as e:
        log.exception("Training failed.")
        raise ModelTrainingError(str(e))


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train DistilBERT sentiment classifier")
    p.add_argument("--data", type=str, required=True, help="Path to CSV dataset (cleaned columns: text, sentiment)")
    p.add_argument("--model_name", default="distilbert-base-uncased")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--accumulation_steps", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_prop", type=float, default=0.1)
    p.add_argument("--sample_size", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early_stop", type=int, default=2)
    p.add_argument("--output_dir", type=str, default="models")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sample_size = args.sample_size if args.sample_size > 0 else None
    train(
        data_path=args.data,
        model_name=args.model_name,
        max_len=args.max_len,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        epochs=args.epochs,
        learning_rate=args.lr,
        warmup_proportion=args.warmup_prop,
        sample_size=sample_size,
        seed=args.seed,
        early_stopping_patience=args.early_stop,
        output_dir=args.output_dir,
    )
