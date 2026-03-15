# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI — Drug-Drug Interaction (DDI) Detector
  
  Uses fine-tuned AraBERT to detect interactions between medications.
  
  Interaction Types:
    - mechanism  : The DDI is described in terms of mechanism/PK
    - effect     : The DDI is described in terms of effect/PD  
    - advise     : Advisory information about the DDI
    - int        : Generic interaction detected
    - none       : No interaction detected
  
  Supports both Arabic and English drug names via multilingual approach.
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

logger = get_logger("shifa.ddi")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── DDI Interaction Types ────────────────────────────────────────
DDI_TYPES = {
    0: {"en": "none",      "ar": "لا يوجد تفاعل",    "severity": "safe",    "color": "#28A745"},
    1: {"en": "mechanism", "ar": "تفاعل ميكانيكي",    "severity": "moderate","color": "#FFC107"},
    2: {"en": "effect",    "ar": "تفاعل تأثيري",      "severity": "high",    "color": "#FF9800"},
    3: {"en": "advise",    "ar": "تحذير استشاري",      "severity": "moderate","color": "#17A2B8"},
    4: {"en": "int",       "ar": "تفاعل دوائي عام",   "severity": "high",    "color": "#DC3545"},
}

# ─── Common Drug Interactions Database (Arabic) ──────────────────
# Built-in knowledge base for common drug interactions
KNOWN_INTERACTIONS = {
    ("وارفارين", "أسبرين"): {
        "type": "effect",
        "severity": "high",
        "description_ar": "يزيد الأسبرين من تأثير الوارفارين المضاد للتخثر مما يرفع خطر النزيف بشكل كبير.",
        "recommendation_ar": "تجنب الاستخدام المتزامن. استشر طبيبك فوراً.",
    },
    ("ميتفورمين", "إنسولين"): {
        "type": "mechanism",
        "severity": "moderate",
        "description_ar": "الاستخدام المتزامن قد يزيد من خطر انخفاض سكر الدم.",
        "recommendation_ar": "يجب مراقبة مستوى السكر في الدم بانتظام.",
    },
    ("أملوديبين", "سيمفاستاتين"): {
        "type": "mechanism",
        "severity": "moderate",
        "description_ar": "أملوديبين يزيد من مستوى سيمفاستاتين في الدم مما يرفع خطر الاعتلال العضلي.",
        "recommendation_ar": "لا تتجاوز جرعة 20 ملغ يومياً من سيمفاستاتين.",
    },
    ("أوميبرازول", "كلوبيدوغريل"): {
        "type": "effect",
        "severity": "high",
        "description_ar": "أوميبرازول يقلل من فعالية كلوبيدوغريل كمضاد للصفيحات.",
        "recommendation_ar": "استخدم بانتوبرازول كبديل أو افصل بين الجرعتين بـ 12 ساعة.",
    },
    ("ليثيوم", "إيبوبروفين"): {
        "type": "mechanism",
        "severity": "high",
        "description_ar": "مضادات الالتهاب غير الستيرويدية ترفع مستوى الليثيوم في الدم وقد تسبب تسمماً.",
        "recommendation_ar": "تجنب الاستخدام المتزامن. راقب مستوى الليثيوم في الدم.",
    },
    ("ميترونيدازول", "كحول"): {
        "type": "effect",
        "severity": "high",
        "description_ar": "يسبب الاستخدام المتزامن تفاعلاً شبيهاً بالديسولفيرام: غثيان شديد وتقيؤ وصداع.",
        "recommendation_ar": "تجنب الكحول تماماً أثناء العلاج وحتى 48 ساعة بعد الانتهاء.",
    },
    ("سيبروفلوكساسين", "حديد"): {
        "type": "mechanism",
        "severity": "moderate",
        "description_ar": "مكملات الحديد تقلل امتصاص السيبروفلوكساسين بشكل كبير.",
        "recommendation_ar": "تناول السيبروفلوكساسين قبل ساعتين أو بعد 6 ساعات من مكملات الحديد.",
    },
    ("أتينولول", "فيراباميل"): {
        "type": "effect",
        "severity": "high",
        "description_ar": "الاستخدام المتزامن قد يسبب بطء شديد في القلب وانخفاض ضغط الدم.",
        "recommendation_ar": "تجنب الجمع بين حاصرات بيتا وحاصرات قنوات الكالسيوم من نوع فيراباميل.",
    },
}


# ═══════════════════════════════════════════════════════════════════
#  DDI BERT Model
# ═══════════════════════════════════════════════════════════════════

class DDIBertModel(nn.Module):
    """
    BERT-based model for Drug-Drug Interaction classification.
    Takes a pair of drug descriptions and classifies the interaction type.
    """

    def __init__(self, model_name, num_classes=5, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Interaction-specific attention
        self.drug_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8, batch_first=True, dropout=dropout
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (batch, seq, hidden)

        # Self-attention over token representations
        attn_output, _ = self.drug_attention(hidden, hidden, hidden)

        # Pool: [CLS] token + mean pooling
        cls_token = attn_output[:, 0, :]
        mean_pool = (attn_output * attention_mask.unsqueeze(-1)).sum(1) / \
                    attention_mask.sum(1, keepdim=True)

        combined = torch.cat([cls_token, mean_pool], dim=1)
        logits = self.classifier(combined)
        return logits


# ═══════════════════════════════════════════════════════════════════
#  Drug Interaction Detector
# ═══════════════════════════════════════════════════════════════════

class DrugInteractionDetector:
    """
    Main DDI detection engine for SHIFA AI.
    Combines rule-based lookup with BERT-based classification.
    """

    MODEL_DIR = "models/ddi_model"

    def __init__(self, model_name="aubmindlab/bert-base-arabertv2"):
        self.model_name = model_name
        self.device = DEVICE
        self.tokenizer = None
        self.model = None
        self.num_classes = len(DDI_TYPES)
        self.load_error = None
        self._initialized = False

        logger.info("DDI Detector initialized (device=%s)", self.device)

    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self.tokenizer

    def _build_model(self):
        model = DDIBertModel(
            model_name=self.model_name,
            num_classes=self.num_classes,
        )
        return model.to(self.device)

    # ─── Rule-based Lookup ───────────────────────────────────────

    def check_known_interaction(self, drug1: str, drug2: str) -> dict | None:
        """
        Check the built-in database for known interactions.
        Returns interaction info or None if not found.
        """
        drug1 = drug1.strip()
        drug2 = drug2.strip()

        # Check both orderings
        result = KNOWN_INTERACTIONS.get((drug1, drug2)) or \
                 KNOWN_INTERACTIONS.get((drug2, drug1))

        if result:
            return {
                "drug1": drug1,
                "drug2": drug2,
                "type": result["type"],
                "type_ar": DDI_TYPES[[k for k, v in DDI_TYPES.items() if v["en"] == result["type"]][0]]["ar"],
                "severity": result["severity"],
                "description": result["description_ar"],
                "recommendation": result["recommendation_ar"],
                "source": "knowledge_base",
            }
        return None

    # ─── BERT-based Prediction ───────────────────────────────────

    def predict_interaction(self, drug1: str, drug2: str, context: str = "") -> dict:
        """
        Predict drug interaction using BERT model + knowledge base fallback.

        Args:
            drug1: First drug name
            drug2: Second drug name
            context: Additional context about usage

        Returns:
            dict with interaction details
        """
        # Step 1: Check knowledge base first
        known = self.check_known_interaction(drug1, drug2)
        if known:
            return known

        # Step 2: Use BERT model if available
        if self.model is not None and self._initialized:
            return self._bert_predict(drug1, drug2, context)

        # Step 3: Fallback — unknown interaction
        return {
            "drug1": drug1,
            "drug2": drug2,
            "type": "none",
            "type_ar": "غير معروف",
            "severity": "unknown",
            "description": "لم يتم العثور على معلومات عن تفاعل بين هذين الدوائين في قاعدة البيانات.",
            "recommendation": "يُنصح باستشارة الصيدلي أو الطبيب المختص.",
            "source": "fallback",
        }

    def _bert_predict(self, drug1: str, drug2: str, context: str = "") -> dict:
        """Use BERT model for DDI prediction."""
        self._ensure_tokenizer()
        self.model.eval()

        text1 = f"{drug1} {context}".strip()
        text2 = drug2

        encoding = self.tokenizer(
            text1, text2,
            add_special_tokens=True,
            max_length=256,
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
        ddi_info = DDI_TYPES[pred_idx]

        return {
            "drug1": drug1,
            "drug2": drug2,
            "type": ddi_info["en"],
            "type_ar": ddi_info["ar"],
            "severity": ddi_info["severity"],
            "confidence": float(probs[pred_idx]),
            "all_probabilities": {
                DDI_TYPES[i]["ar"]: float(probs[i]) for i in range(len(probs))
            },
            "description": self._generate_description(drug1, drug2, ddi_info),
            "recommendation": self._generate_recommendation(ddi_info),
            "source": "bert_model",
        }

    def _generate_description(self, drug1, drug2, ddi_info):
        """Generate Arabic description for detected interaction."""
        if ddi_info["en"] == "none":
            return f"لم يُكتشف تفاعل دوائي بين {drug1} و {drug2}."
        elif ddi_info["en"] == "mechanism":
            return f"تم اكتشاف تفاعل ميكانيكي بين {drug1} و {drug2}. قد يؤثر أحدهما على امتصاص أو استقلاب الآخر."
        elif ddi_info["en"] == "effect":
            return f"تم اكتشاف تفاعل تأثيري بين {drug1} و {drug2}. قد يزيد أو يقلل أحدهما من تأثير الآخر."
        elif ddi_info["en"] == "advise":
            return f"هناك تحذير استشاري بخصوص استخدام {drug1} مع {drug2}."
        else:
            return f"تم اكتشاف تفاعل دوائي بين {drug1} و {drug2}. يُنصح بالحذر."

    def _generate_recommendation(self, ddi_info):
        """Generate Arabic recommendation based on severity."""
        if ddi_info["severity"] == "safe":
            return "لا توجد مخاوف من الاستخدام المتزامن."
        elif ddi_info["severity"] == "moderate":
            return "يُنصح بمراقبة الأعراض وإبلاغ الطبيب عن أي تغييرات."
        else:
            return "⚠️ يُنصح بشدة باستشارة الطبيب قبل الاستخدام المتزامن."

    # ─── Batch Check ─────────────────────────────────────────────

    def check_multiple_drugs(self, drugs: list) -> list:
        """
        Check all pairwise interactions for a list of drugs.

        Args:
            drugs: List of drug names

        Returns:
            List of interaction dicts for each pair
        """
        results = []
        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                interaction = self.predict_interaction(drugs[i], drugs[j])
                if interaction["type"] != "none":
                    results.append(interaction)
        return results

    # ─── Fine-tuning ─────────────────────────────────────────────

    def fine_tune(self, drug_pairs, labels, epochs=5, batch_size=16,
                  learning_rate=2e-5, verbose=True) -> dict:
        """
        Fine-tune DDI model on drug interaction dataset.
        """
        self._ensure_tokenizer()
        self.model = self._build_model()

        from sklearn.model_selection import train_test_split
        train_pairs, val_pairs, train_labels, val_labels = train_test_split(
            drug_pairs, labels, test_size=0.15, random_state=42
        )

        train_dataset = DDIDataset(train_pairs, train_labels, self.tokenizer)
        val_dataset = DDIDataset(val_pairs, val_labels, self.tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        loss_fn = nn.CrossEntropyLoss()

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        if verbose:
            print(f"\n{'═' * 60}")
            print(f"  💊 Fine-tuning DDI Model")
            print(f"  📊 Train pairs: {len(train_dataset)} | Val: {len(val_dataset)}")
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

        self._initialized = True
        return history

    # ─── Save / Load ─────────────────────────────────────────────

    def save(self):
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        if self.model:
            torch.save(self.model.state_dict(), os.path.join(self.MODEL_DIR, "ddi_model.pt"))
        meta = {"model_name": self.model_name, "num_classes": self.num_classes}
        with open(os.path.join(self.MODEL_DIR, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        if self.tokenizer:
            self.tokenizer.save_pretrained(os.path.join(self.MODEL_DIR, "tokenizer"))
        logger.info("DDI model saved to: %s", self.MODEL_DIR)

    def load(self) -> bool:
        model_path = os.path.join(self.MODEL_DIR, "ddi_model.pt")
        if not os.path.exists(model_path):
            logger.warning("DDI model not found: %s", model_path)
            return False
        try:
            with open(os.path.join(self.MODEL_DIR, "meta.json")) as f:
                meta = json.load(f)
            self.model_name = meta["model_name"]
            self.num_classes = meta["num_classes"]

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
            logger.info("DDI model loaded from: %s", self.MODEL_DIR)
            return True
        except Exception as e:
            self.load_error = str(e)
            logger.error("Failed to load DDI model: %s", e, exc_info=True)
            return False

    @property
    def is_ready(self) -> bool:
        return True  # Knowledge base always available, BERT is a bonus


# Import DDIDataset from bert_medical if needed
from engine.bert_medical import DDIDataset
