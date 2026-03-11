# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║   SHIFA AI — Breast Cancer Detection Model Training Script       ║
║   Architecture: MobileNetV2 Transfer Learning (TF 2.15+)        ║
║   Dataset: Breast Ultrasound Images (Benign/Malignant/Normal)    ║
║   Output: models/breast_cancer_model_v2.keras                   ║
╚══════════════════════════════════════════════════════════════════╝

USAGE:
  1. Download dataset from Kaggle:
     https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
  2. Extract into: data/breast_ultrasound/
     Expected structure:
       data/breast_ultrasound/
         benign/       ← benign ultrasound images
         malignant/    ← malignant ultrasound images
         normal/       ← normal ultrasound images
  3. Run: python train_cancer_model.py
"""

import os
import sys
import numpy as np
import json
import time
from pathlib import Path

# ─── TF / Keras Imports ──────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info/warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, TensorBoard
)

# ─── Configuration ────────────────────────────────────────────────
CONFIG = {
    # Dataset path (relative to project root)
    "data_dir":       "data/breast_ultrasound",

    # Model output
    "model_out":      "models/breast_cancer_model_v2.keras",
    "history_out":    "models/training_history.json",

    # Image dimensions (MobileNetV2 minimum: 96x96)
    "img_size":       (224, 224),
    "img_channels":   3,

    # Training
    "batch_size":     16,
    "epochs_frozen":  15,    # Phase 1: Only train the head
    "epochs_finetune":20,    # Phase 2: Fine-tune last layers

    # Augmentation
    "augment":        True,

    # Class weights (handles class imbalance automatically)
    "use_class_weights": True,

    # Train/Validation split
    "val_split":      0.20,
    "test_split":     0.10,

    # Random seed
    "seed":           42,
}

CLASS_NAMES = ["benign", "malignant", "normal"]
NUM_CLASSES = len(CLASS_NAMES)

print("╔══════════════════════════════════════════════════════════════╗")
print("║   SHIFA AI — Breast Cancer Model Training                   ║")
print(f"║   TensorFlow: {tf.__version__:<46}║")
print(f"║   GPU Available: {str(bool(tf.config.list_physical_devices('GPU'))):<43}║")
print("╚══════════════════════════════════════════════════════════════╝\n")


# ─── Step 0: Validate Dataset ─────────────────────────────────────
def validate_dataset(data_dir: str) -> bool:
    p = Path(data_dir)
    if not p.exists():
        print(f"❌ Dataset folder not found: {data_dir}")
        print("   Please download from:")
        print("   https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset")
        print(f"   And extract to: {data_dir}/")
        print("   Expected structure:")
        print("      data/breast_ultrasound/benign/")
        print("      data/breast_ultrasound/malignant/")
        print("      data/breast_ultrasound/normal/")
        return False

    total = 0
    for cls in CLASS_NAMES:
        cls_path = p / cls
        if not cls_path.exists():
            print(f"❌ Missing class folder: {cls_path}")
            return False
        count = len(list(cls_path.glob("*.png")) + list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.jpeg")))
        print(f"   ✅ {cls:<12}: {count} images")
        total += count

    print(f"   📊 Total: {total} images across {NUM_CLASSES} classes\n")
    return total > 0


# ─── Step 1: Data Pipeline ────────────────────────────────────────
def build_data_generators():
    print("━" * 55)
    print("  📸 Step 1: Building Data Pipeline")
    print("━" * 55)

    # Preprocessing function for MobileNetV2
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    if CONFIG["augment"]:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess,
            validation_split=CONFIG["val_split"],
            rotation_range=15,
            horizontal_flip=True,
            vertical_flip=False,
            zoom_range=0.15,
            width_shift_range=0.10,
            height_shift_range=0.10,
            brightness_range=[0.85, 1.15],
            fill_mode='nearest',
        )
    else:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess,
            validation_split=CONFIG["val_split"],
        )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess,
        validation_split=CONFIG["val_split"],
    )

    train_gen = train_datagen.flow_from_directory(
        CONFIG["data_dir"],
        target_size=CONFIG["img_size"],
        batch_size=CONFIG["batch_size"],
        class_mode="categorical",
        classes=CLASS_NAMES,
        subset="training",
        seed=CONFIG["seed"],
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        CONFIG["data_dir"],
        target_size=CONFIG["img_size"],
        batch_size=CONFIG["batch_size"],
        class_mode="categorical",
        classes=CLASS_NAMES,
        subset="validation",
        seed=CONFIG["seed"],
        shuffle=False,
    )

    print(f"   ✅ Train samples  : {train_gen.samples}")
    print(f"   ✅ Val samples    : {val_gen.samples}")
    print(f"   ✅ Classes        : {train_gen.class_indices}\n")

    return train_gen, val_gen


# ─── Step 2: Compute Class Weights ────────────────────────────────
def compute_class_weights(train_gen):
    from sklearn.utils.class_weight import compute_class_weight
    labels = train_gen.classes
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    cw = dict(zip(classes, weights))
    print(f"   ⚖️  Class weights: {cw}\n")
    return cw


# ─── Step 3: Build Model ──────────────────────────────────────────
def build_model():
    print("━" * 55)
    print("  🧠 Step 2: Building MobileNetV2 Transfer Learning Model")
    print("━" * 55)

    input_shape = (*CONFIG["img_size"], CONFIG["img_channels"])

    # ── Base model (frozen initially) ──
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Freeze all base layers initially

    # ── Custom classification head ──
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="ShifaBreastCancerDetector_v2")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )

    total_params = model.count_params()
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"   ✅ Total params    : {total_params:,}")
    print(f"   ✅ Trainable params: {trainable:,} (head only)\n")

    return model, base_model


# ─── Step 4: Callbacks ────────────────────────────────────────────
def get_callbacks(phase: str):
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    return [
        EarlyStopping(
            monitor="val_auc",
            patience=5,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=f"models/breast_cancer_best_{phase}.keras",
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        TensorBoard(
            log_dir=f"logs/{phase}_{int(time.time())}",
            histogram_freq=0,
        ),
    ]


# ─── Step 5: Training Phase 1 (Head Only) ─────────────────────────
def train_phase1(model, train_gen, val_gen, class_weights):
    print("━" * 55)
    print("  🚀 Step 3: Phase 1 — Training Head (Base Frozen)")
    print("━" * 55)

    history1 = model.fit(
        train_gen,
        epochs=CONFIG["epochs_frozen"],
        validation_data=val_gen,
        class_weight=class_weights if CONFIG["use_class_weights"] else None,
        callbacks=get_callbacks("phase1"),
        verbose=1,
    )

    best_val_auc = max(history1.history.get("val_auc", [0]))
    best_val_acc = max(history1.history.get("val_accuracy", [0]))
    print(f"\n   ✅ Phase 1 Complete — Best Val AUC: {best_val_auc:.3f}, Best Val Acc: {best_val_acc:.3f}\n")
    return history1


# ─── Step 6: Training Phase 2 (Fine-tuning) ──────────────────────
def train_phase2(model, base_model, train_gen, val_gen, class_weights):
    print("━" * 55)
    print("  🔥 Step 4: Phase 2 — Fine-Tuning (Unfreeze last 30 layers)")
    print("━" * 55)

    # Unfreeze the last 30 layers of MobileNetV2
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    trainable_after = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"   ✅ Trainable params after unfreeze: {trainable_after:,}")

    # Recompile with lower LR for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )

    history2 = model.fit(
        train_gen,
        epochs=CONFIG["epochs_finetune"],
        validation_data=val_gen,
        class_weight=class_weights if CONFIG["use_class_weights"] else None,
        callbacks=get_callbacks("phase2"),
        verbose=1,
    )

    best_val_auc = max(history2.history.get("val_auc", [0]))
    best_val_acc = max(history2.history.get("val_accuracy", [0]))
    print(f"\n   ✅ Phase 2 Complete — Best Val AUC: {best_val_auc:.3f}, Best Val Acc: {best_val_acc:.3f}\n")
    return history2


# ─── Step 7: Save Final Model ─────────────────────────────────────
def save_model(model, history1, history2):
    print("━" * 55)
    print("  💾 Step 5: Saving Final Model")
    print("━" * 55)

    os.makedirs("models", exist_ok=True)

    # Save in modern .keras format (zero compatibility issues)
    model.save(CONFIG["model_out"])
    print(f"   ✅ Model saved: {CONFIG['model_out']}")

    # Save training history
    combined_history = {}
    for key in history1.history:
        combined_history[key] = history1.history[key] + history2.history.get(key, [])

    with open(CONFIG["history_out"], "w") as f:
        json.dump(combined_history, f, indent=2)
    print(f"   ✅ History saved: {CONFIG['history_out']}\n")


# ─── Step 8: Final Evaluation ─────────────────────────────────────
def evaluate_model(model, val_gen):
    print("━" * 55)
    print("  📊 Step 6: Final Evaluation on Validation Set")
    print("━" * 55)

    results = model.evaluate(val_gen, verbose=0)
    metric_names = model.metrics_names

    print()
    for name, val in zip(metric_names, results):
        print(f"   {name:<20}: {val:.4f}")

    print()
    print("   📋 Class Mapping:")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"      {i} → {cls}")


# ─── MAIN ─────────────────────────────────────────────────────────
def main():
    start_total = time.time()

    # Validate dataset
    print("━" * 55)
    print("  🗂️  Step 0: Validating Dataset")
    print("━" * 55)
    if not validate_dataset(CONFIG["data_dir"]):
        sys.exit(1)

    # Build data generators
    train_gen, val_gen = build_data_generators()

    # Compute class weights
    class_weights = compute_class_weights(train_gen) if CONFIG["use_class_weights"] else None

    # Build model
    model, base_model = build_model()

    # Phase 1: Train head only
    history1 = train_phase1(model, train_gen, val_gen, class_weights)

    # Phase 2: Fine-tune
    history2 = train_phase2(model, base_model, train_gen, val_gen, class_weights)

    # Save
    save_model(model, history1, history2)

    # Evaluate
    evaluate_model(model, val_gen)

    total_time = time.time() - start_total
    print("═" * 55)
    print(f"  🎉 Training Complete! Total time: {total_time/60:.1f} minutes")
    print(f"  📁 Model: {CONFIG['model_out']}")
    print("═" * 55)
    print()
    print("  Next step: Update engine/cancer_detector.py to use:")
    print(f"  → keras.saving.load_model('{CONFIG['model_out']}')")


if __name__ == "__main__":
    main()
