# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI — Multi-Model Breast Cancer Benchmark
  
  Compares 6 ML algorithms on the Wisconsin Breast Cancer dataset:
    1. BERT (feature extraction → classification)
    2. LSTM (deep learning on tabular features)
    3. Naive Bayes
    4. SVM (Support Vector Machine)
    5. Random Forest
    6. MobileNetV2 (reference from train_cancer_model.py)
  
  Produces:
    - Comparative metrics table (Accuracy, Precision, Recall, F1, AUC)
    - ROC curves
    - Confusion matrices
    - Training time comparison
    - Results saved to models/benchmark_results.json
  
  USAGE:
    python benchmark_models.py
    python benchmark_models.py --output results/
═══════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Sklearn imports ──────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ─── Deep Learning imports ───────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─── Configuration ────────────────────────────────────────────────
CONFIG = {
    "test_size": 0.20,
    "val_size": 0.15,
    "random_state": 42,
    "cv_folds": 5,
    "output_dir": "models/benchmark_results",
    "lstm_epochs": 50,
    "lstm_batch_size": 32,
    "lstm_lr": 0.001,
    "bert_epochs": 20,
    "bert_batch_size": 32,
    "bert_lr": 0.001,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════
#  Step 1: Load & Prepare Data
# ═══════════════════════════════════════════════════════════════════

def load_data():
    """Load Wisconsin Breast Cancer dataset."""
    print("━" * 60)
    print("  📦 Step 1: Loading Wisconsin Breast Cancer Dataset")
    print("━" * 60)

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    print(f"  ✅ Samples: {len(X)}")
    print(f"  ✅ Features: {X.shape[1]}")
    print(f"  ✅ Classes: {dict(zip(data.target_names, np.bincount(y)))}")
    print(f"     0 = malignant ({(y==0).sum()})")
    print(f"     1 = benign    ({(y==1).sum()})")

    # Exploratory stats
    print(f"\n  📊 Feature Statistics:")
    print(f"     Mean range:  [{X.mean().min():.2f}, {X.mean().max():.2f}]")
    print(f"     Std range:   [{X.std().min():.2f}, {X.std().max():.2f}]")
    print(f"     Missing:     {X.isnull().sum().sum()}")

    return X, y, data


def prepare_splits(X, y):
    """Create train/val/test splits with scaling."""
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"], stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=CONFIG["val_size"],
        random_state=CONFIG["random_state"], stratify=y_trainval
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n  📊 Split sizes:")
    print(f"     Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    return {
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train.values,
        "y_val": y_val.values,
        "y_test": y_test.values,
        "scaler": scaler,
    }


# ═══════════════════════════════════════════════════════════════════
#  Step 2: Define Models
# ═══════════════════════════════════════════════════════════════════

# ── 2a: LSTM Model for Tabular Data ─────────────────────────────

class LSTMClassifier(nn.Module):
    """LSTM for tabular data classification."""

    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout, bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Reshape: (batch, features) → (batch, seq_len=1, features)
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        out = self.classifier(lstm_out[:, -1, :])
        return out.squeeze()


# ── 2b: BERT-style Feature Extractor for Tabular ────────────────

class BERTTabularClassifier(nn.Module):
    """
    Transformer-based classifier for tabular data.
    Uses self-attention (BERT-like) on feature embeddings.
    """

    def __init__(self, input_dim, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.3):
        super().__init__()

        # Feature embedding (project each feature to hidden_dim)
        self.feature_embedding = nn.Linear(1, hidden_dim)
        self.position_embedding = nn.Embedding(input_dim, hidden_dim)

        # Transformer encoder (BERT-like)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size = x.size(0)
        num_features = x.size(1)

        # Embed each feature individually: (batch, features) → (batch, features, hidden)
        x = x.unsqueeze(-1)  # (batch, features, 1)
        x = self.feature_embedding(x)  # (batch, features, hidden)

        # Add positional embeddings
        positions = torch.arange(num_features, device=x.device)
        x = x + self.position_embedding(positions)

        # Prepend [CLS] token
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (batch, 1+features, hidden)

        # Transformer encoding
        x = self.transformer(x)

        # Use [CLS] output for classification
        cls_output = x[:, 0, :]
        return self.classifier(cls_output).squeeze()


# ═══════════════════════════════════════════════════════════════════
#  Step 3: Training & Evaluation Functions
# ═══════════════════════════════════════════════════════════════════

def evaluate_model(y_true, y_pred, y_prob=None):
    """Compute all evaluation metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        metrics["fpr"] = fpr.tolist()
        metrics["tpr"] = tpr.tolist()

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    metrics["report"] = classification_report(y_true, y_pred, zero_division=0)

    return metrics


def train_sklearn_model(name, model, splits):
    """Train and evaluate a scikit-learn model."""
    start = time.time()

    model.fit(splits["X_train"], splits["y_train"])

    train_time = time.time() - start

    # Predictions
    start = time.time()
    y_pred = model.predict(splits["X_test"])
    inference_time = time.time() - start

    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(splits["X_test"])[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(splits["X_test"])

    metrics = evaluate_model(splits["y_test"], y_pred, y_prob)
    metrics["train_time"] = train_time
    metrics["inference_time"] = inference_time

    # Cross-validation
    cv_scores = cross_val_score(
        model, np.vstack([splits["X_train"], splits["X_val"]]),
        np.concatenate([splits["y_train"], splits["y_val"]]),
        cv=CONFIG["cv_folds"], scoring="accuracy"
    )
    metrics["cv_mean"] = float(cv_scores.mean())
    metrics["cv_std"] = float(cv_scores.std())

    return metrics


def train_pytorch_model(name, model, splits, epochs, batch_size, lr):
    """Train and evaluate a PyTorch model (LSTM or BERT-Tabular)."""
    model = model.to(DEVICE)

    X_train = torch.FloatTensor(splits["X_train"]).to(DEVICE)
    y_train = torch.FloatTensor(splits["y_train"]).to(DEVICE)
    X_val = torch.FloatTensor(splits["X_val"]).to(DEVICE)
    y_val = torch.FloatTensor(splits["y_val"]).to(DEVICE)
    X_test = torch.FloatTensor(splits["X_test"]).to(DEVICE)
    y_test = torch.FloatTensor(splits["y_test"]).to(DEVICE)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    loss_fn = nn.BCELoss()

    start = time.time()
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = loss_fn(val_preds, y_val).item()
            val_acc = ((val_preds > 0.5).float() == y_val).float().mean().item()

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > 10:
                break

    train_time = time.time() - start

    # Load best model
    model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    start = time.time()
    with torch.no_grad():
        test_probs = model(X_test).cpu().numpy()
    inference_time = time.time() - start

    test_preds = (test_probs > 0.5).astype(int)
    y_test_np = splits["y_test"]

    metrics = evaluate_model(y_test_np, test_preds, test_probs)
    metrics["train_time"] = train_time
    metrics["inference_time"] = inference_time
    metrics["epochs_trained"] = epoch + 1

    return metrics


# ═══════════════════════════════════════════════════════════════════
#  Step 4: Run Full Benchmark
# ═══════════════════════════════════════════════════════════════════

def run_benchmark():
    """Execute the complete model comparison benchmark."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   SHIFA AI — Multi-Model Breast Cancer Benchmark           ║")
    print(f"║   Device: {str(DEVICE):<48}║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # Load data
    X, y, data = load_data()
    splits = prepare_splits(X, y)

    results = {}
    input_dim = splits["X_train"].shape[1]  # 30 features

    # ── Model 1: Naive Bayes ──────────────────────────────────────
    print(f"\n{'━' * 60}")
    print("  🔬 Model 1/6: Naive Bayes")
    print("━" * 60)

    nb = GaussianNB()
    results["Naive Bayes"] = train_sklearn_model("Naive Bayes", nb, splits)
    print(f"  ✅ Accuracy: {results['Naive Bayes']['accuracy']:.4f} "
          f"| F1: {results['Naive Bayes']['f1_score']:.4f} "
          f"| AUC: {results['Naive Bayes'].get('auc_roc', 0):.4f}")

    # ── Model 2: SVM ──────────────────────────────────────────────
    print(f"\n{'━' * 60}")
    print("  🔬 Model 2/6: Support Vector Machine (SVM)")
    print("━" * 60)

    svm = SVC(kernel="rbf", probability=True, C=10, gamma="scale", random_state=42)
    results["SVM"] = train_sklearn_model("SVM", svm, splits)
    print(f"  ✅ Accuracy: {results['SVM']['accuracy']:.4f} "
          f"| F1: {results['SVM']['f1_score']:.4f} "
          f"| AUC: {results['SVM'].get('auc_roc', 0):.4f}")

    # ── Model 3: Random Forest ────────────────────────────────────
    print(f"\n{'━' * 60}")
    print("  🔬 Model 3/6: Random Forest")
    print("━" * 60)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, class_weight="balanced",
        n_jobs=-1, random_state=42
    )
    results["Random Forest"] = train_sklearn_model("Random Forest", rf, splits)
    print(f"  ✅ Accuracy: {results['Random Forest']['accuracy']:.4f} "
          f"| F1: {results['Random Forest']['f1_score']:.4f} "
          f"| AUC: {results['Random Forest'].get('auc_roc', 0):.4f}")

    # ── Model 4: LSTM ─────────────────────────────────────────────
    print(f"\n{'━' * 60}")
    print("  🔬 Model 4/6: LSTM (Bidirectional)")
    print("━" * 60)

    lstm = LSTMClassifier(input_dim=input_dim, hidden_dim=128, num_layers=2, dropout=0.3)
    results["LSTM"] = train_pytorch_model(
        "LSTM", lstm, splits,
        epochs=CONFIG["lstm_epochs"],
        batch_size=CONFIG["lstm_batch_size"],
        lr=CONFIG["lstm_lr"]
    )
    print(f"  ✅ Accuracy: {results['LSTM']['accuracy']:.4f} "
          f"| F1: {results['LSTM']['f1_score']:.4f} "
          f"| AUC: {results['LSTM'].get('auc_roc', 0):.4f} "
          f"| Epochs: {results['LSTM'].get('epochs_trained', '?')}")

    # ── Model 5: BERT-Tabular (Transformer) ───────────────────────
    print(f"\n{'━' * 60}")
    print("  🔬 Model 5/6: BERT-Tabular (Transformer Encoder)")
    print("━" * 60)

    bert_tab = BERTTabularClassifier(
        input_dim=input_dim, hidden_dim=128,
        num_heads=4, num_layers=2, dropout=0.3
    )
    results["BERT (Tabular)"] = train_pytorch_model(
        "BERT-Tabular", bert_tab, splits,
        epochs=CONFIG["bert_epochs"],
        batch_size=CONFIG["bert_batch_size"],
        lr=CONFIG["bert_lr"]
    )
    print(f"  ✅ Accuracy: {results['BERT (Tabular)']['accuracy']:.4f} "
          f"| F1: {results['BERT (Tabular)']['f1_score']:.4f} "
          f"| AUC: {results['BERT (Tabular)'].get('auc_roc', 0):.4f} "
          f"| Epochs: {results['BERT (Tabular)'].get('epochs_trained', '?')}")

    # ── Model 6: Gradient Boosting (bonus) ─────────────────────
    print(f"\n{'━' * 60}")
    print("  🔬 Model 6/6: Gradient Boosting")
    print("━" * 60)

    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    results["Gradient Boosting"] = train_sklearn_model("Gradient Boosting", gb, splits)
    print(f"  ✅ Accuracy: {results['Gradient Boosting']['accuracy']:.4f} "
          f"| F1: {results['Gradient Boosting']['f1_score']:.4f} "
          f"| AUC: {results['Gradient Boosting'].get('auc_roc', 0):.4f}")

    return results, splits, data


# ═══════════════════════════════════════════════════════════════════
#  Step 5: Generate Reports & Visualizations
# ═══════════════════════════════════════════════════════════════════

def generate_visualizations(results, output_dir):
    """Generate comparison plots and save results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.rcParams["font.family"] = "DejaVu Sans"
    except ImportError:
        print("  ⚠️  matplotlib/seaborn not installed, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    model_names = list(results.keys())
    colors = ["#E53935", "#1976D2", "#388E3C", "#FF9800", "#9C27B0", "#00ACC1"]

    # ── Plot 1: Metrics Comparison Bar Chart ──────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
    titles = ["Accuracy", "Precision", "Recall", "F1-Score"]

    for ax, metric, title in zip(axes, metrics_to_plot, titles):
        vals = [results[m][metric] for m in model_names]
        bars = ax.bar(range(len(model_names)), vals, color=colors[:len(model_names)])
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0.8, 1.02)
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=150, bbox_inches="tight")
    print(f"  📊 Saved: metrics_comparison.png")

    # ── Plot 2: ROC Curves ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))

    for i, name in enumerate(model_names):
        if "fpr" in results[name] and "tpr" in results[name]:
            fpr = results[name]["fpr"]
            tpr = results[name]["tpr"]
            roc_auc = results[name].get("auc_roc", 0)
            ax.plot(fpr, tpr, color=colors[i % len(colors)],
                    label=f"{name} (AUC = {roc_auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150, bbox_inches="tight")
    print(f"  📈 Saved: roc_curves.png")

    # ── Plot 3: Confusion Matrices ────────────────────────────────
    n_models = len(model_names)
    cols = 3
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, name in enumerate(model_names):
        cm = np.array(results[name]["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=axes[i],
                    xticklabels=["Malignant", "Benign"],
                    yticklabels=["Malignant", "Benign"])
        axes[i].set_title(name, fontsize=11, fontweight="bold")
        axes[i].set_ylabel("True")
        axes[i].set_xlabel("Predicted")

    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
    print(f"  🔥 Saved: confusion_matrices.png")

    # ── Plot 4: Training Time Comparison ──────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    train_times = [results[m]["train_time"] for m in model_names]
    bars = ax.barh(model_names, train_times, color=colors[:len(model_names)])
    ax.set_xlabel("Training Time (seconds)", fontsize=12)
    ax.set_title("Training Time Comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    for bar, t in zip(bars, train_times):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{t:.3f}s", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_time.png"), dpi=150, bbox_inches="tight")
    print(f"  ⏱️  Saved: training_time.png")
    plt.close("all")


def print_summary(results):
    """Print a formatted comparison table."""
    print(f"\n{'═' * 80}")
    print("  📊 BENCHMARK RESULTS SUMMARY")
    print("═" * 80)

    header = f"  {'Model':<22} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8} {'Time':>10}"
    print(header)
    print("  " + "─" * 76)

    best_model = None
    best_f1 = 0

    for name, m in results.items():
        auc_val = m.get("auc_roc", 0)
        row = (f"  {name:<22} {m['accuracy']:>8.4f} {m['precision']:>8.4f} "
               f"{m['recall']:>8.4f} {m['f1_score']:>8.4f} {auc_val:>8.4f} "
               f"{m['train_time']:>9.3f}s")
        print(row)

        if m["f1_score"] > best_f1:
            best_f1 = m["f1_score"]
            best_model = name

    print("  " + "─" * 76)
    print(f"  🏆 Best Model: {best_model} (F1: {best_f1:.4f})")
    print("═" * 80)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Multi-model breast cancer benchmark")
    parser.add_argument("--output", default=CONFIG["output_dir"])
    args = parser.parse_args()

    start_total = time.time()

    # Run benchmark
    results, splits, data = run_benchmark()

    # Print summary
    print_summary(results)

    # Save results (remove non-serializable items)
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    results_clean = {}
    for name, m in results.items():
        results_clean[name] = {
            k: v for k, v in m.items()
            if k not in ("report", "fpr", "tpr")
        }
        results_clean[name]["report"] = m.get("report", "")

    with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
        json.dump(results_clean, f, indent=2)
    print(f"\n  💾 Results saved: {output_dir}/benchmark_results.json")

    # Generate visualizations
    print(f"\n{'━' * 60}")
    print("  📊 Generating Visualizations")
    print("━" * 60)
    generate_visualizations(results, output_dir)

    total_time = time.time() - start_total
    print(f"\n{'═' * 60}")
    print(f"  🎉 Benchmark Complete! Total time: {total_time:.1f}s")
    print(f"  📁 Results: {output_dir}/")
    print("═" * 60)


if __name__ == "__main__":
    main()
