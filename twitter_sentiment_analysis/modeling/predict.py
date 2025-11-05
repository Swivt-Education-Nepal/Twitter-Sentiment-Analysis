# twitter_sentiment_analysis/modeling/predict.py
import os
import argparse
from typing import List

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from twitter_sentiment_analysis.configs import get_logger
from twitter_sentiment_analysis.configs.exceptions import PredictionError

log = get_logger(__name__)


def load_model_and_tokenizer(model_state_path: str, tokenizer_dir: str, model_name: str = None, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Loading model to device: {device}")

    # If a tokenizer_dir is given, prefer it. Otherwise fallback to model_name.
    if tokenizer_dir and os.path.isdir(tokenizer_dir):
        tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_dir)
    elif model_name:
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    else:
        raise PredictionError("Tokenizer not found. Provide tokenizer_dir or model_name.")

    # instantiate model from model_name (weights will be overwritten by state_dict load)
    model_source = model_name if model_name else tokenizer_dir
    model = DistilBertForSequenceClassification.from_pretrained(model_source, num_labels=2)
    model.load_state_dict(torch.load(model_state_path, map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer, device


def predict_texts(texts: List[str], model, tokenizer, device: str = None, max_length: int = 256, batch_size: int = 32):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().tolist()
        confs = probs.max(dim=1).values.cpu().tolist()

    return preds, confs


def parse_args():
    p = argparse.ArgumentParser(description="Predict using a trained DistilBERT model")
    p.add_argument("--model_state", required=True, help="Path to saved model state (best_model.pt)")
    p.add_argument("--tokenizer_dir", required=True, help="Path to saved tokenizer directory")
    p.add_argument("--model_name", default=None, help="Fallback model name if tokenizer_dir not provided")
    p.add_argument("--text", type=str, default=None, help="Single text to predict (quote it)")
    p.add_argument("--max_len", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        model, tokenizer, device = load_model_and_tokenizer(args.model_state, args.tokenizer_dir, args.model_name)
        if args.text:
            preds, confs = predict_texts([args.text], model, tokenizer, device, max_length=args.max_len)
            print(f"Pred: {preds[0]}, Conf: {confs[0]:.4f}")
        else:
            # Example interactive prompt
            while True:
                text = input("Enter text (or 'quit' to exit): ").strip()
                if text.lower() in ("quit", "exit"):
                    break
                preds, confs = predict_texts([text], model, tokenizer, device, max_length=args.max_len)
                print(f"Pred: {preds[0]}, Conf: {confs[0]:.4f}")

    except Exception as e:
        log.exception("Prediction failed.")
        raise PredictionError(str(e))
