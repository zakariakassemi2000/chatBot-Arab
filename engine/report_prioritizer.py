# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI — Radiology Report Prioritizer
  Architecture: BERT + Self-Attention + Bidirectional GRU
  
  Classifies radiology reports into priority levels:
    - 🟢 Routine      (Priority 0) : Normal findings, follow-up
    - 🟠 Semi-urgent  (Priority 1) : Abnormal findings, needs attention
    - 🔴 Urgent       (Priority 2) : Critical findings, immediate action
  
  The model combines:
    1. BERT encoder for contextualized text representations
    2. Multi-head self-attention for capturing key medical entities
    3. Bidirectional GRU for sequential pattern recognition
    4. Classification head for priority prediction
═══════════════════════════════════════════════════════════════════════
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from utils.logger import get_logger

logger = get_logger("shifa.report_prioritizer")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Priority Levels ─────────────────────────────────────────────
PRIORITY_LEVELS = {
    0: {
        "label_en": "Routine",
        "label_ar": "روتيني",
        "icon": "🟢",
        "color": "#28A745",
        "max_days": 30,
        "description_ar": "نتائج طبيعية أو تغييرات حميدة. المتابعة الروتينية كافية.",
    },
    1: {
        "label_en": "Semi-urgent",
        "label_ar": "شبه طارئ",
        "icon": "🟠",
        "color": "#FF9800",
        "max_days": 7,
        "description_ar": "نتائج غير طبيعية تستدعي اهتماماً خلال أسبوع.",
    },
    2: {
        "label_en": "Urgent",
        "label_ar": "طارئ",
        "icon": "🔴",
        "color": "#DC3545",
        "max_days": 1,
        "description_ar": "نتائج حرجة تستدعي تدخلاً طبياً فورياً.",
    },
}

# ─── Medical Keywords for Rule-Based Fallback ─────────────────────
URGENT_KEYWORDS = [
    "mass", "tumor", "malignant", "metastasis", "metastatic", "carcinoma",
    "hemorrhage", "bleeding", "fracture", "pneumothorax", "embolism",
    "stroke", "infarction", "dissection", "perforation", "obstruction",
    "كتلة", "ورم", "خبيث", "انتشار", "سرطان", "نزيف", "كسر",
    "استرواح", "انصمام", "سكتة", "احتشاء", "تسلخ", "انثقاب", "انسداد",
]

SEMI_URGENT_KEYWORDS = [
    "suspicious", "nodule", "opacity", "consolidation", "effusion",
    "enlarged", "abnormal", "lesion", "infiltrate", "atelectasis",
    "مشبوه", "عقدة", "عتامة", "تكثف", "انصباب",
    "تضخم", "غير طبيعي", "آفة", "ارتشاح", "انخماص",
]


# ═══════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════

class RadiologyReportDataset(Dataset):
    """Dataset for radiology report priority classification."""

    def __init__(self, texts, labels, tokenizer, max_length=512):
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


# ═══════════════════════════════════════════════════════════════════
#  BERT + Attention + GRU Model
# ═══════════════════════════════════════════════════════════════════

class BERTAttentionGRU(nn.Module):
    """
    BERT + Self-Attention + Bidirectional GRU for report prioritization.
    
    Architecture:
        Input Text → BERT Encoder → Multi-Head Self-Attention → BiGRU → Classifier
    
    The self-attention layer captures relationships between medical entities
    in the report, while the GRU captures sequential patterns and context.
    """

    def __init__(
        self,
        bert_model_name: str,
        num_classes: int = 3,
        gru_hidden_dim: int = 256,
        gru_num_layers: int = 2,
        attention_heads: int = 8,
        dropout: float = 0.3,
        freeze_bert_layers: int = 0,
    ):
        super().__init__()

        # ── BERT Encoder ──
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size  # 768

        # Optionally freeze early BERT layers
        if freeze_bert_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for i, layer in enumerate(self.bert.encoder.layer):
                if i < freeze_bert_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        # ── Multi-Head Self-Attention ──
        self.self_attention = nn.MultiheadAttention(
            embed_dim=bert_hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_layer_norm = nn.LayerNorm(bert_hidden_size)
        self.attn_dropout = nn.Dropout(dropout)

        # ── Bidirectional GRU ──
        self.gru = nn.GRU(
            input_size=bert_hidden_size,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_num_layers > 1 else 0,
        )

        # ── Attention Pooling over GRU outputs ──
        gru_output_dim = gru_hidden_dim * 2  # bidirectional
        self.pool_attention = nn.Sequential(
            nn.Linear(gru_output_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # ── Classification Head ──
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        # ── Step 1: BERT Encoding ──
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = bert_output.last_hidden_state  # (batch, seq_len, 768)

        # ── Step 2: Self-Attention ──
        # Key padding mask: True for positions to ignore
        key_padding_mask = (attention_mask == 0)
        attn_output, attn_weights = self.self_attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=key_padding_mask,
        )
        # Residual connection + layer norm
        hidden_states = self.attn_layer_norm(hidden_states + self.attn_dropout(attn_output))

        # ── Step 3: GRU Encoding ──
        gru_output, gru_hidden = self.gru(hidden_states)
        # gru_output: (batch, seq_len, gru_hidden*2)

        # ── Step 4: Attention Pooling ──
        attn_scores = self.pool_attention(gru_output).squeeze(-1)  # (batch, seq_len)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))
        attn_weights_pool = torch.softmax(attn_scores, dim=1)     # (batch, seq_len)

        weighted_output = torch.bmm(
            attn_weights_pool.unsqueeze(1), gru_output
        ).squeeze(1)  # (batch, gru_hidden*2)

        # ── Step 5: Classification ──
        logits = self.classifier(weighted_output)
        return logits

    def get_attention_weights(self, input_ids, attention_mask):
        """Extract attention weights for interpretability."""
        self.eval()
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden = bert_output.last_hidden_state
            key_padding_mask = (attention_mask == 0)
            _, attn_weights = self.self_attention(
                hidden, hidden, hidden, key_padding_mask=key_padding_mask
            )
        return attn_weights


# ═══════════════════════════════════════════════════════════════════
#  Report Prioritizer Engine
# ═══════════════════════════════════════════════════════════════════

class ReportPrioritizer:
    """
    Main engine for radiology report prioritization.
    Combines BERT+Attention+GRU model with rule-based fallback.
    """

    MODEL_DIR = "models/report_prioritizer"

    def __init__(self, model_name="aubmindlab/bert-base-arabertv2"):
        self.model_name = model_name
        self.device = DEVICE
        self.tokenizer = None
        self.model = None
        self.num_classes = len(PRIORITY_LEVELS)
        self.load_error = None
        self._initialized = False
        self.max_length = 512

        logger.info("ReportPrioritizer initialized (device=%s)", self.device)

    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self.tokenizer

    def _build_model(self):
        model = BERTAttentionGRU(
            bert_model_name=self.model_name,
            num_classes=self.num_classes,
            gru_hidden_dim=256,
            gru_num_layers=2,
            attention_heads=8,
            dropout=0.3,
            freeze_bert_layers=6,  # Freeze first 6 BERT layers
        )
        return model.to(self.device)

    # ─── Rule-Based Prioritization ───────────────────────────────

    def rule_based_prioritize(self, text: str) -> dict:
        """
        Fallback prioritization using keyword matching.
        Used when BERT model is not available.
        """
        text_lower = text.lower()

        # Check for urgent keywords
        urgent_score = sum(1 for kw in URGENT_KEYWORDS if kw in text_lower)
        semi_urgent_score = sum(1 for kw in SEMI_URGENT_KEYWORDS if kw in text_lower)

        if urgent_score >= 2:
            priority = 2
            confidence = min(0.7 + urgent_score * 0.05, 0.95)
        elif urgent_score >= 1 or semi_urgent_score >= 2:
            priority = 1
            confidence = min(0.6 + (urgent_score + semi_urgent_score) * 0.05, 0.90)
        else:
            priority = 0
            confidence = 0.6 + (1 - min(semi_urgent_score * 0.1, 0.3))

        level = PRIORITY_LEVELS[priority]
        return {
            "priority": priority,
            "label_en": level["label_en"],
            "label_ar": level["label_ar"],
            "icon": level["icon"],
            "color": level["color"],
            "confidence": confidence,
            "max_days": level["max_days"],
            "description": level["description_ar"],
            "source": "rule_based",
            "keywords_found": {
                "urgent": [kw for kw in URGENT_KEYWORDS if kw in text_lower],
                "semi_urgent": [kw for kw in SEMI_URGENT_KEYWORDS if kw in text_lower],
            },
        }

    # ─── BERT Prediction ────────────────────────────────────────

    def predict(self, report_text: str) -> dict:
        """
        Prioritize a radiology report.
        Uses BERT model if available, falls back to rule-based.
        """
        if self.model is not None and self._initialized:
            return self._bert_predict(report_text)
        return self.rule_based_prioritize(report_text)

    def _bert_predict(self, text: str) -> dict:
        """Use BERT+Attention+GRU for prediction."""
        self._ensure_tokenizer()
        self.model.eval()

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
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
        level = PRIORITY_LEVELS[pred_idx]

        return {
            "priority": pred_idx,
            "label_en": level["label_en"],
            "label_ar": level["label_ar"],
            "icon": level["icon"],
            "color": level["color"],
            "confidence": float(probs[pred_idx]),
            "max_days": level["max_days"],
            "description": level["description_ar"],
            "all_probabilities": {
                PRIORITY_LEVELS[i]["label_ar"]: float(probs[i])
                for i in range(len(probs))
            },
            "source": "bert_attention_gru",
        }

    def predict_batch(self, reports: list) -> list:
        """Prioritize multiple reports and sort by priority."""
        results = [self.predict(r) for r in reports]
        # Sort: urgent first, then semi-urgent, then routine
        results_sorted = sorted(results, key=lambda x: (-x["priority"], -x["confidence"]))
        return results_sorted

    # ─── Fine-Tuning ─────────────────────────────────────────────

    def fine_tune(
        self,
        texts: list,
        labels: list,
        epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        verbose: bool = True,
    ) -> dict:
        """
        Fine-tune the BERT+Attention+GRU model on labeled reports.
        """
        self._ensure_tokenizer()
        self.model = self._build_model()

        from sklearn.model_selection import train_test_split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.15, random_state=42, stratify=labels
        )

        train_dataset = RadiologyReportDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = RadiologyReportDataset(val_texts, val_labels, self.tokenizer, self.max_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Differential learning rates
        optimizer = torch.optim.AdamW([
            {"params": self.model.bert.parameters(), "lr": learning_rate},
            {"params": self.model.self_attention.parameters(), "lr": learning_rate * 5},
            {"params": self.model.gru.parameters(), "lr": learning_rate * 10},
            {"params": self.model.pool_attention.parameters(), "lr": learning_rate * 10},
            {"params": self.model.classifier.parameters(), "lr": learning_rate * 10},
        ], weight_decay=0.01)

        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )

        loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 2.0, 3.0]).to(self.device)  # Higher weight for urgent
        )

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_acc = 0

        if verbose:
            print(f"\n{'═' * 60}")
            print(f"  📋 Fine-tuning BERT+Attention+GRU Report Prioritizer")
            print(f"  📊 Train: {len(train_dataset)} | Val: {len(val_dataset)}")
            print(f"  ⚙️  Epochs: {epochs} | Batch: {batch_size} | LR: {learning_rate}")
            print(f"{'═' * 60}\n")

        for epoch in range(epochs):
            self.model.train()
            total_loss, correct, total = 0, 0, 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attn_mask = batch["attention_mask"].to(self.device)
                targets = batch["label"].to(self.device)

                optimizer.zero_grad()
                logits = self.model(input_ids, attn_mask)
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

            # Validate
            self.model.eval()
            v_loss, v_correct, v_total = 0, 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attn_mask = batch["attention_mask"].to(self.device)
                    targets = batch["label"].to(self.device)
                    logits = self.model(input_ids, attn_mask)
                    v_loss += loss_fn(logits, targets).item()
                    v_correct += (torch.argmax(logits, 1) == targets).sum().item()
                    v_total += targets.size(0)

            val_loss = v_loss / len(val_loader)
            val_acc = v_correct / v_total

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if verbose:
                print(f"  Epoch {epoch+1}/{epochs} │ "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} │ "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_checkpoint()
                if verbose:
                    print(f"    ✅ Best model saved (Val Acc: {val_acc:.4f})")

        self._initialized = True
        return history

    # ─── Evaluation ──────────────────────────────────────────────

    def evaluate(self, texts: list, labels: list) -> dict:
        """Comprehensive evaluation with all metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, confusion_matrix, classification_report,
        )

        predictions = [self.predict(t) for t in texts]
        pred_labels = [p["priority"] for p in predictions]

        target_names = [PRIORITY_LEVELS[i]["label_en"] for i in range(self.num_classes)]

        return {
            "accuracy": accuracy_score(labels, pred_labels),
            "precision": precision_score(labels, pred_labels, average="weighted", zero_division=0),
            "recall": recall_score(labels, pred_labels, average="weighted", zero_division=0),
            "f1_score": f1_score(labels, pred_labels, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(labels, pred_labels).tolist(),
            "classification_report": classification_report(
                labels, pred_labels, target_names=target_names, zero_division=0
            ),
        }

    # ─── Save / Load ─────────────────────────────────────────────

    def _save_checkpoint(self):
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.MODEL_DIR, "model.pt"))
        meta = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "max_length": self.max_length,
        }
        with open(os.path.join(self.MODEL_DIR, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Report prioritizer saved to: %s", self.MODEL_DIR)

    def save(self):
        self._save_checkpoint()
        if self.tokenizer:
            self.tokenizer.save_pretrained(os.path.join(self.MODEL_DIR, "tokenizer"))

    def load(self) -> bool:
        model_path = os.path.join(self.MODEL_DIR, "model.pt")
        if not os.path.exists(model_path):
            logger.warning("Report prioritizer not found: %s", model_path)
            return False
        try:
            with open(os.path.join(self.MODEL_DIR, "meta.json")) as f:
                meta = json.load(f)
            self.model_name = meta["model_name"]
            self.num_classes = meta["num_classes"]
            self.max_length = meta.get("max_length", 512)

            tok_path = os.path.join(self.MODEL_DIR, "tokenizer")
            if os.path.exists(tok_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
            else:
                self._ensure_tokenizer()

            self.model = self._build_model()
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            self.model.eval()
            self._initialized = True
            logger.info("Report prioritizer loaded")
            return True
        except Exception as e:
            self.load_error = str(e)
            logger.error("Failed to load report prioritizer: %s", e, exc_info=True)
            return False

    @property
    def is_ready(self) -> bool:
        return True  # Rule-based always available
