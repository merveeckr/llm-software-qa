import os
import random
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Heavy deps kept explicit to match project style
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Reuse architecture and shared constants from Turkish module
from bil import (
    BertBiLSTMCNN,
    LABEL_COLS,
    MAX_LEN as TR_MAX_LEN,
    DEVICE as TR_DEVICE,
    THRESH as TR_THRESH,
)


# Settings (overridable via environment)
EN_MODEL_NAME: str = os.environ.get("EN_BERT_MODEL", "C:/a/bert")
EN_WEIGHTS_PATH: str = os.environ.get("EN_WEIGHTS_PATH", "best_model_en.pt")
EN_MAX_LEN: int = int(os.environ.get("EN_MAX_LEN", str(TR_MAX_LEN)))
EN_THRESH: float = float(os.environ.get("EN_THRESH", str(TR_THRESH)))
DEVICE = TR_DEVICE
EN_CSV_PATH: str = os.environ.get("EN_CSV_PATH", "C:/a/eng.csv")
EN_TEXT_COL: str = os.environ.get("EN_TEXT_COL", "Requirement_EN")
EN_BATCH_SIZE: int = int(os.environ.get("EN_BATCH_SIZE", "8"))
EN_EPOCHS: int = int(os.environ.get("EN_EPOCHS", "1"))
EN_LR: float = float(os.environ.get("EN_LR", "1e-5"))
EN_WEIGHT_DECAY: float = float(os.environ.get("EN_WEIGHT_DECAY", "1e-2"))
GRAD_CLIP: float = float(os.environ.get("EN_GRAD_CLIP", "1.0"))
RANDOM_SEED = 42


# Reproducibility
def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# Dataset (English)
class ReqDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_len: int = 128):
        self.texts = texts
        clean = np.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
        self.labels = clean.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }
        return item


def collate_fn(batch):
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# Metrics
def evaluate_model(model: BertBiLSTMCNN, dataloader: DataLoader, device, threshold: float) -> Dict[str, float]:
    from sklearn.metrics import f1_score, accuracy_score, hamming_loss

    model.eval()
    all_labels, all_preds = [], []
    total_loss = 0.0
    loss_fn = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).float()
            labels = torch.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
            labels = labels.clamp(0.0, 1.0)

            logits = model(input_ids, attention_mask)
            if not torch.isfinite(logits).all():
                continue
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                continue
            total_loss += loss.item() * input_ids.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds) if all_preds else np.zeros((0, len(LABEL_COLS)))
    all_labels = np.vstack(all_labels) if all_labels else np.zeros((0, len(LABEL_COLS)))
    if all_preds.size == 0 or all_labels.size == 0:
        return {"loss": 0.0, "f1_macro": 0.0, "accuracy": 0.0, "hamming": 1.0}
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    ham_loss = hamming_loss(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader.dataset)
    return {"loss": avg_loss, "f1_macro": f1_macro, "accuracy": accuracy, "hamming": ham_loss}


# Training loop (saves to EN_WEIGHTS_PATH)
def train_loop(
    model: BertBiLSTMCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device,
    epochs: int,
    lr: float,
    weight_decay: float,
    threshold: float,
):
    model.to(device)
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    total_steps = len(train_loader) * max(1, epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(0.10*total_steps)), num_training_steps=total_steps)
    loss_fn = nn.BCEWithLogitsLoss()

    best_f1 = -1.0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for step, batch in enumerate(loop):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).float()
            labels = torch.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
            labels = labels.clamp(0.0, 1.0)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask)
            if not torch.isfinite(logits).all():
                continue
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * input_ids.size(0)
            loop.set_postfix(loss=loss.item())

        val_metrics = evaluate_model(model, val_loader, device, threshold=threshold)
        print(
            f"Epoch {epoch+1} - TrainLoss: {epoch_loss/len(train_loader.dataset):.4f} | "
            f"ValLoss: {val_metrics['loss']:.4f} | F1_macro: {val_metrics['f1_macro']:.4f} | "
            f"Hamming: {val_metrics['hamming']:.4f}"
        )
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            torch.save(model.state_dict(), EN_WEIGHTS_PATH)
            print(f"Saved best EN model to {EN_WEIGHTS_PATH}.")

    return model


def load_english_model_and_tokenizer(
    model_name: Optional[str] = None,
    weights_path: Optional[str] = None,
) -> Tuple[BertBiLSTMCNN, AutoTokenizer]:
    """
    Load English BERT + BiLSTM + CNN model and tokenizer.

    - model_name defaults to EN_MODEL_NAME
    - if weights_path exists, it will be loaded (expects same architecture)
    """
    model_name = model_name or EN_MODEL_NAME
    weights_path = weights_path or EN_WEIGHTS_PATH

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertBiLSTMCNN(bert_model_name=model_name, num_labels=len(LABEL_COLS))
    model.to(DEVICE)
    model.eval()

    if weights_path and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
    return model, tokenizer


def batch_predict_english(
    model: BertBiLSTMCNN,
    tokenizer,
    texts: List[str],
    max_len: Optional[int] = None,
    threshold: Optional[float] = None,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Predict multi-label qualities for a list of English requirement sentences.
    Returns array of shape (N, len(LABEL_COLS)) with 0/1.
    """
    if not texts:
        return np.zeros((0, len(LABEL_COLS)), dtype=int)
    max_len = int(max_len or EN_MAX_LEN)
    threshold = float(threshold if threshold is not None else EN_THRESH)

    predictions: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        chunk = [str(t) for t in texts[i:i+batch_size]]
        enc = tokenizer(
            chunk,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)
        predictions.append(preds)
    return np.vstack(predictions)


def predict_english_as_dicts(
    texts: List[str],
    model_name: Optional[str] = None,
    weights_path: Optional[str] = None,
    max_len: Optional[int] = None,
    threshold: Optional[float] = None,
) -> List[Dict[str, int]]:
    """
    Convenience: load model/tokenizer and return a list of dict[label->0/1].
    """
    model, tokenizer = load_english_model_and_tokenizer(model_name, weights_path)
    preds = batch_predict_english(model, tokenizer, texts, max_len=max_len, threshold=threshold)
    out: List[Dict[str, int]] = []
    for row in preds:
        out.append({LABEL_COLS[j]: int(row[j]) for j in range(len(LABEL_COLS))})
    return out


def find_missing_features_from_dicts(pred_dicts: List[Dict[str, int]]) -> List[List[str]]:
    """
    For each prediction dict, return the list of labels predicted as 0 (missing).
    """
    missing: List[List[str]] = []
    for pdict in pred_dicts:
        missing.append([lab for lab in LABEL_COLS if int(pdict.get(lab, 0)) == 0])
    return missing


def analyze_english_requirements(
    texts: List[str],
    model_name: Optional[str] = None,
    weights_path: Optional[str] = None,
    max_len: Optional[int] = None,
    threshold: Optional[float] = None,
) -> List[Dict[str, object]]:
    """
    End-to-end analysis for English requirement sentences.
    Returns list of items: { 'text', 'pred', 'missing' }.
    """
    preds = predict_english_as_dicts(
        texts=texts,
        model_name=model_name,
        weights_path=weights_path,
        max_len=max_len,
        threshold=threshold,
    )
    missing = find_missing_features_from_dicts(preds)
    results: List[Dict[str, object]] = []
    for t, p, m in zip(texts, preds, missing):
        results.append({"text": t, "pred": p, "missing": m})
    return results


def compare_tr_en_predictions(
    turkish_preds: np.ndarray,
    english_preds: np.ndarray,
) -> Dict[str, object]:
    """
    Compare Turkish vs English predictions for aligned sentences.
    Expects arrays of shape (N, L) with same label order.
    Returns per-label agreement and overall agreement.
    """
    if turkish_preds.shape != english_preds.shape:
        raise ValueError("turkish_preds and english_preds must have same shape")
    if turkish_preds.shape[1] != len(LABEL_COLS):
        raise ValueError("prediction second dimension must equal number of labels")

    per_label: Dict[str, Dict[str, float]] = {}
    eq = (turkish_preds == english_preds)
    overall_agreement = float(eq.mean()) if turkish_preds.size else 0.0
    for j, lab in enumerate(LABEL_COLS):
        col = eq[:, j]
        tr_col = turkish_preds[:, j]
        en_col = english_preds[:, j]
        per_label[lab] = {
            "agreement": float(col.mean()) if col.size else 0.0,
            "tr_only": int(((tr_col == 1) & (en_col == 0)).sum()),
            "en_only": int(((tr_col == 0) & (en_col == 1)).sum()),
            "both": int(((tr_col == 1) & (en_col == 1)).sum()),
            "neither": int(((tr_col == 0) & (en_col == 0)).sum()),
        }
    return {"per_label": per_label, "overall_agreement": overall_agreement}


def english_missing_from_texts(texts: List[str]) -> List[List[str]]:
    """
    Shortcut: return only missing features per English sentence using default settings.
    """
    results = analyze_english_requirements(texts)
    return [r["missing"] for r in results]


if __name__ == "__main__":
    mode = os.environ.get("EN_MODE", "infer").strip().lower()
    if mode == "train":
        # Training entrypoint (like bil.py but for English)
        if not os.path.exists(EN_CSV_PATH):
            raise FileNotFoundError(f"CSV not found at {EN_CSV_PATH}")
        df = pd.read_csv(EN_CSV_PATH)
        text_col = EN_TEXT_COL if EN_TEXT_COL in df.columns else ("Requirement" if "Requirement" in df.columns else None)
        if text_col is None:
            raise ValueError(f"Text column '{EN_TEXT_COL}' or 'Requirement' not found in CSV")
        missing_labels = [c for c in LABEL_COLS if c not in df.columns]
        if missing_labels:
            raise ValueError(f"Label columns missing in CSV: {missing_labels}")

        texts = df[text_col].fillna("").astype(str).tolist()
        labels = df[LABEL_COLS].values

        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.15, random_state=RANDOM_SEED, shuffle=True)

        tokenizer = AutoTokenizer.from_pretrained(EN_MODEL_NAME)
        train_dataset = ReqDataset(X_train, np.array(y_train), tokenizer, max_len=EN_MAX_LEN)
        val_dataset = ReqDataset(X_val, np.array(y_val), tokenizer, max_len=EN_MAX_LEN)
        train_loader = DataLoader(train_dataset, batch_size=EN_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=EN_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        model = BertBiLSTMCNN(bert_model_name=EN_MODEL_NAME, num_labels=len(LABEL_COLS))
        model.to(DEVICE)
        trained_model = train_loop(
            model,
            train_loader,
            val_loader,
            DEVICE,
            epochs=EN_EPOCHS,
            lr=EN_LR,
            weight_decay=EN_WEIGHT_DECAY,
            threshold=EN_THRESH,
        )
        metrics = evaluate_model(trained_model, val_loader, DEVICE, threshold=EN_THRESH)
        print("Final metrics (EN):", metrics)  # noqa: T201
    else:
        # Minimal CLI demo for inference
        raw = os.environ.get("EN_SENTENCES", "").strip()
        if raw:
            inputs = [s.strip() for s in raw.split("|") if s.strip()]
        else:
            inputs = [
                "The system shall log all failed login attempts with timestamp and user ID.",
                "The UI should be nice.",
            ]
        analysis = analyze_english_requirements(inputs)
        for item in analysis:
            print("REQ:", item["text"])  # noqa: T201
            print("PRED:", item["pred"])  # noqa: T201
            print("MISSING:", item["missing"])  # noqa: T201
            print("---")  # noqa: T201
