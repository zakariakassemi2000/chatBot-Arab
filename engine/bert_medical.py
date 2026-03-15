# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI — Medical BERT Engine
  Fine-tuned AraBERT for Arabic medical text classification
  
  Supports:
    - Medical text classification (5 intents)
    - Drug interaction detection (DDI)
    - Medical entity recognition
  
  Models:
    - aubmindlab/bert-base-arabertv2 (Arabic)
    - CAMeL-Lab/bert-base-arabic-camelbert-mix (Arabic biomedical)
═══════════════════════════════════════════════════════════════════════
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from utils.logger import get_logger

logger = get_logger("shifa.bert_medical")

# ─── Default Configuration ───────────────────────────────────────
DEFAULT_MODEL_NAME = "aubmindlab/bert-base-arabertv2"
MAX_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Medical intent labels (aligned with existing classifier.py)
MEDICAL_INTENTS = [
    "وصف_أعراض",       # Symptom description
    "طلب_معلومات",     # Information request
    "طلب_علاج",        # Treatment request
    "استشارة_طارئة",   # Emergency consultation
    "طلب_توجيه",       # Guidance request
]


# ═══════════════════════════════════════════════════════════════════
#  Dataset Classes
# ═══════════════════════════════════════════════════════════════════

class MedicalTextDataset(Dataset):
    """PyTorch Dataset for medical text classification."""

    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


class DDIDataset(Dataset):
    """Dataset for Drug-Drug Interaction detection."""

    def __init__(self, drug_pairs, labels, tokenizer, max_length=MAX_LENGTH):
        """
        Args:
            drug_pairs: list of (drug1_text, drug2_text) tuples
            labels: interaction type labels
        """
        self.drug_pairs = drug_pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.drug_pairs)

    def __getitem__(self, idx):
        drug1, drug2 = self.drug_pairs[idx]
        label = self.labels[idx]

        # Encode pair with [SEP] separator
        encoding = self.tokenizer(
            drug1, drug2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════════════
#  Custom BERT Model with Attention
# ═══════════════════════════════════════════════════════════════════

class MedicalBERTClassifier(nn.Module):
    """
    BERT + Custom Attention Head for medical text classification.
    Enhanced with self-attention pooling instead of simple [CLS] token.
    """

    def __init__(self, model_name, num_classes, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768 for base

        # Self-attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # Attention-weighted pooling
        attn_weights = self.attention(hidden_states)  # (batch, seq_len, 1)
        attn_weights = attn_weights.squeeze(-1)       # (batch, seq_len)

        # Mask padding tokens
        attn_weights = attn_weights.masked_fill(
            attention_mask == 0, float("-inf")
        )
        attn_weights = torch.softmax(attn_weights, dim=1)  # (batch, seq_len)

        # Weighted sum of hidden states
        weighted = torch.bmm(
            attn_weights.unsqueeze(1), hidden_states
        ).squeeze(1)  # (batch, hidden)

        logits = self.classifier(weighted)
        return logits


# ═══════════════════════════════════════════════════════════════════
#  Main Engine Class
# ═══════════════════════════════════════════════════════════════════

class MedicalBERT:
    """
    Medical BERT engine for SHIFA AI.
    Handles fine-tuning, prediction, and model management.
    """

    MODEL_DIR = "models/bert_medical"

    def __init__(self, model_name=DEFAULT_MODEL_NAME, num_classes=None):
        self.model_name = model_name
        self.device = DEVICE
        self.tokenizer = None
        self.model = None
        self.num_classes = num_classes or len(MEDICAL_INTENTS)
        self.label_map = {label: i for i, label in enumerate(MEDICAL_INTENTS)}
        self.inv_label_map = {i: label for label, i in self.label_map.items()}
        self.load_error = None
        self._initialized = False

        logger.info("MedicalBERT initialized (device=%s, model=%s)", self.device, model_name)

    def _ensure_tokenizer(self):
        """Lazy-load tokenizer."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self.tokenizer

    def _build_model(self):
        """Build the custom BERT model."""
        model = MedicalBERTClassifier(
            model_name=self.model_name,
            num_classes=self.num_classes,
            dropout=0.3,
        )
        return model.to(self.device)

    # ─── Fine-Tuning ─────────────────────────────────────────────

    def fine_tune(
        self,
        texts: list,
        labels: list,
        val_texts: list = None,
        val_labels: list = None,
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        val_split: float = 0.15,
        verbose: bool = True,
    ) -> dict:
        """
        Fine-tune AraBERT on medical text data.

        Args:
            texts: Training texts
            labels: Training labels (strings from MEDICAL_INTENTS)
            val_texts: Validation texts (optional, auto-split if None)
            val_labels: Validation labels
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Peak learning rate for AdamW
            warmup_ratio: Proportion of steps for warmup
            val_split: Validation split ratio (if val_texts is None)
            verbose: Print progress

        Returns:
            dict with training history (loss, accuracy per epoch)
        """
        self._ensure_tokenizer()

        # Encode string labels to integers
        encoded_labels = [self.label_map[l] for l in labels]

        # Auto-split if no validation set provided
        if val_texts is None:
            from sklearn.model_selection import train_test_split
            texts, val_texts, encoded_labels, val_labels_enc = train_test_split(
                texts, encoded_labels, test_size=val_split,
                random_state=42, stratify=encoded_labels
            )
        else:
            val_labels_enc = [self.label_map[l] for l in val_labels]

        # Create datasets
        train_dataset = MedicalTextDataset(texts, encoded_labels, self.tokenizer)
        val_dataset = MedicalTextDataset(val_texts, val_labels_enc, self.tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Build model
        self.model = self._build_model()

        # Optimizer: different LR for BERT base vs classification head
        optimizer = torch.optim.AdamW([
            {"params": self.model.bert.parameters(), "lr": learning_rate},
            {"params": self.model.attention.parameters(), "lr": learning_rate * 10},
            {"params": self.model.classifier.parameters(), "lr": learning_rate * 10},
        ], weight_decay=0.01)

        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        loss_fn = nn.CrossEntropyLoss()

        # Training loop
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        if verbose:
            print(f"\n{'═' * 60}")
            print(f"  🧠 Fine-tuning AraBERT — {self.model_name}")
            print(f"  📊 Train: {len(train_dataset)} | Val: {len(val_dataset)}")
            print(f"  ⚙️  Epochs: {epochs} | Batch: {batch_size} | LR: {learning_rate}")
            print(f"  🖥️  Device: {self.device}")
            print(f"{'═' * 60}\n")

        best_val_acc = 0.0

        for epoch in range(epochs):
            # ── Train ──
            self.model.train()
            total_loss, correct, total = 0, 0, 0

            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["label"].to(self.device)

                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = loss_fn(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

            train_loss = total_loss / len(train_loader)
            train_acc = correct / total

            # ── Validate ──
            val_loss, val_acc = self._evaluate(val_loader, loss_fn)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if verbose:
                print(
                    f"  Epoch {epoch+1}/{epochs} │ "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} │ "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
                )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_checkpoint("best")
                if verbose:
                    print(f"    ✅ New best model saved (Val Acc: {val_acc:.4f})")

        if verbose:
            print(f"\n  🏆 Best Validation Accuracy: {best_val_acc:.4f}\n")

        self._initialized = True
        return history

    def _evaluate(self, loader, loss_fn):
        """Evaluate model on a dataloader."""
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["label"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = loss_fn(logits, targets)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        return total_loss / len(loader), correct / total

    # ─── Prediction ──────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """
        Predict medical intent from Arabic text.

        Returns:
            dict with keys: intent, confidence, all_probabilities
        """
        if self.model is None:
            return {"intent": "طلب_معلومات", "confidence": 0.0, "all_probabilities": {}}

        self._ensure_tokenizer()
        self.model.eval()

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        intent = self.inv_label_map[pred_idx]
        confidence = float(probs[pred_idx])

        all_probs = {
            self.inv_label_map[i]: float(probs[i])
            for i in range(len(probs))
        }

        return {
            "intent": intent,
            "confidence": confidence,
            "all_probabilities": all_probs,
        }

    def predict_batch(self, texts: list) -> list:
        """Predict intents for a batch of texts."""
        return [self.predict(t) for t in texts]

    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Extract BERT embeddings for a text (useful for downstream tasks).
        Returns (768,) embedding vector.
        """
        self._ensure_tokenizer()

        if self.model is None:
            # Fallback: use pretrained BERT directly
            model = AutoModel.from_pretrained(self.model_name).to(self.device)
        else:
            model = self.model.bert

        model.eval()

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Use [CLS] token embedding
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

        return cls_embedding

    # ─── Detailed Evaluation ─────────────────────────────────────

    def evaluate_detailed(self, texts: list, labels: list) -> dict:
        """
        Comprehensive evaluation with sklearn metrics.

        Returns:
            dict with accuracy, precision, recall, f1, confusion_matrix,
                  classification_report
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, confusion_matrix, classification_report,
        )

        predictions = self.predict_batch(texts)
        pred_labels = [p["intent"] for p in predictions]

        acc = accuracy_score(labels, pred_labels)
        prec = precision_score(labels, pred_labels, average="weighted", zero_division=0)
        rec = recall_score(labels, pred_labels, average="weighted", zero_division=0)
        f1 = f1_score(labels, pred_labels, average="weighted", zero_division=0)
        cm = confusion_matrix(labels, pred_labels, labels=MEDICAL_INTENTS)
        report = classification_report(
            labels, pred_labels, target_names=MEDICAL_INTENTS, zero_division=0
        )

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

    # ─── Save / Load ─────────────────────────────────────────────

    def _save_checkpoint(self, tag="best"):
        """Save model checkpoint."""
        save_dir = os.path.join(self.MODEL_DIR, tag)
        os.makedirs(save_dir, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(save_dir, "model.pt"))

        meta = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "label_map": self.label_map,
            "inv_label_map": {str(k): v for k, v in self.inv_label_map.items()},
        }
        with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info("Checkpoint saved: %s", save_dir)

    def save(self, path: str = None):
        """Save the complete model."""
        path = path or self.MODEL_DIR
        self._save_checkpoint("final")
        self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
        logger.info("Model saved to: %s", path)

    def load(self, path: str = None, tag: str = "best") -> bool:
        """Load a saved model. Returns True on success."""
        path = path or self.MODEL_DIR
        checkpoint_dir = os.path.join(path, tag)

        model_file = os.path.join(checkpoint_dir, "model.pt")
        meta_file = os.path.join(checkpoint_dir, "meta.json")

        if not os.path.exists(model_file):
            self.load_error = f"Model checkpoint not found: {model_file}"
            logger.warning(self.load_error)
            return False

        try:
            # Load metadata
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.model_name = meta["model_name"]
            self.num_classes = meta["num_classes"]
            self.label_map = meta["label_map"]
            self.inv_label_map = {int(k): v for k, v in meta["inv_label_map"].items()}

            # Load tokenizer
            tokenizer_path = os.path.join(path, "tokenizer")
            if os.path.exists(tokenizer_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            else:
                self._ensure_tokenizer()

            # Build and load model
            self.model = self._build_model()
            state_dict = torch.load(model_file, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()

            self._initialized = True
            self.load_error = None
            logger.info("MedicalBERT loaded from: %s/%s", path, tag)
            return True

        except Exception as e:
            self.load_error = str(e)
            logger.error("Failed to load MedicalBERT: %s", e, exc_info=True)
            return False

    @property
    def is_ready(self) -> bool:
        return self._initialized and self.model is not None
