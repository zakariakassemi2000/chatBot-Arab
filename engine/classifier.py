"""
═══════════════════════════════════════════════════════════════════════
  الخطوة 2: المصنّف (المُفهِم)
  Step 2: The Classifier (The Comprehender)

  A lightweight Random Forest classifier trained on sentence
  embeddings. Recognises user intent from Arabic medical queries
  in milliseconds.
═══════════════════════════════════════════════════════════════════════
"""

import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class IntentClassifier:
    """
    Fast intent classifier using Random Forest on sentence embeddings.
    
    Intents:
      - وصف_أعراض   → User describes symptoms
      - طلب_معلومات → User asks for information
      - طلب_علاج    → User asks for treatment
      - استشارة_طارئة → Emergency consultation
      - طلب_توجيه   → User seeks referral / guidance
    """

    MODEL_PATH = "models/intent_classifier.pkl"

    def __init__(self):
        self.model = None
        self.label_map = {}
        self.inv_label_map = {}

    def train(self, embeddings: np.ndarray, intents: list, verbose: bool = True):
        """
        Train the Random Forest classifier on pre-computed embeddings.
        
        Args:
            embeddings: (N, D) array of sentence embeddings
            intents: list of N intent labels
        """
        if verbose:
            print("\n  🧠 تدريب مصنّف النوايا (Random Forest)...")

        # Encode labels
        unique_labels = sorted(set(intents))
        self.label_map = {label: i for i, label in enumerate(unique_labels)}
        self.inv_label_map = {i: label for label, i in self.label_map.items()}
        y = np.array([self.label_map[intent] for intent in intents])

        # Split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, y, test_size=0.15, random_state=42, stratify=y
        )

        # Train Random Forest - fast and light
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
        )
        self.model.fit(X_train, y_train)

        if verbose:
            acc = self.model.score(X_test, y_test)
            print(f"    ✅ دقة المصنّف: {acc:.1%}")
            # Detailed report
            y_pred = self.model.predict(X_test)
            target_names = [self.inv_label_map[i] for i in sorted(self.inv_label_map)]
            print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    def predict(self, embedding: np.ndarray) -> tuple:
        """
        Predict intent from a single query embedding.
        
        Returns:
            (intent_label, confidence) tuple
        """
        if self.model is None:
            return "وصف_أعراض", 0.5

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        proba = self.model.predict_proba(embedding)[0]
        pred_idx = np.argmax(proba)
        confidence = proba[pred_idx]
        intent = self.inv_label_map[pred_idx]

        return intent, float(confidence)

    def predict_top_k(self, embedding: np.ndarray, k: int = 3) -> list:
        """
        Return top-k predicted intents with confidence scores.
        
        Returns:
            List of (intent_label, confidence) tuples sorted by confidence desc.
        """
        if self.model is None:
            return [("وصف_أعراض", 0.5)]

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        proba = self.model.predict_proba(embedding)[0]
        top_k_indices = np.argsort(proba)[-k:][::-1]

        return [
            (self.inv_label_map[idx], float(proba[idx]))
            for idx in top_k_indices
        ]

    def save(self, path: str = None):
        """Save the trained model to disk."""
        path = path or self.MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "label_map": self.label_map,
                "inv_label_map": self.inv_label_map,
            }, f)
        print(f"    💾 تم حفظ المصنّف: {path}")

    def load(self, path: str = None) -> bool:
        """Load a previously trained model. Returns True on success."""
        path = path or self.MODEL_PATH
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.label_map = data["label_map"]
        self.inv_label_map = data["inv_label_map"]
        return True


# ─── Intent-specific response templates ────────────────────────────
INTENT_TEMPLATES = {
    "وصف_أعراض": {
        "prefix": "بناءً على الأعراض التي وصفتها:",
        "suffix": "\n\n💡 **نصيحة:** إذا استمرت الأعراض أكثر من 48 ساعة، يُنصح بزيارة طبيب مختص.",
    },
    "طلب_معلومات": {
        "prefix": "إليك المعلومات المتعلقة بسؤالك:",
        "suffix": "\n\n📚 **ملاحظة:** هذه معلومات للتوعية العامة وليست بديلاً عن الاستشارة الطبية.",
    },
    "طلب_علاج": {
        "prefix": "بخصوص العلاج المناسب:",
        "suffix": "\n\n⚕️ **تنبيه مهم:** لا تأخذ أي دواء دون استشارة طبيب أو صيدلي مؤهل.",
    },
    "استشارة_طارئة": {
        "prefix": "🚨 **حالة قد تستدعي تدخلاً عاجلاً!**",
        "suffix": "\n\n🏥 **يُرجى التوجه فوراً إلى أقرب مستشفى أو الاتصال بخدمة الطوارئ.**",
    },
    "طلب_توجيه": {
        "prefix": "بخصوص التوجيه الطبي المناسب:",
        "suffix": "\n\n🗓️ **نصيحة:** حاول حجز موعد في أقرب وقت ممكن لتجنب تفاقم الحالة.",
    },
}


def format_response(answer: str, intent: str) -> str:
    """Format the response based on detected intent."""
    template = INTENT_TEMPLATES.get(intent, INTENT_TEMPLATES["وصف_أعراض"])
    
    # For emergency, always prepend the urgent prefix
    if intent == "استشارة_طارئة":
        return f"{template['prefix']}\n\n{answer}{template['suffix']}"
    
    return f"{answer}{template['suffix']}"
