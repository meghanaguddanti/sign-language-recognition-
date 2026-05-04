import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from config import MODEL_DIR, MODEL_PATH, LABELS_PATH, CURVES_DIR, EPOCHS
from preprocess import get_generators
from model import build_model

os.makedirs(MODEL_DIR, exist_ok=True)

def make_callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=6,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
    ]

def plot_history(history, phase):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["accuracy"],     label="Train")
    ax1.plot(history.history["val_accuracy"], label="Val", linestyle="--")
    ax1.set_title(f"Phase {phase} Accuracy"); ax1.legend(); ax1.grid(True)
    ax2.plot(history.history["loss"],     label="Train")
    ax2.plot(history.history["val_loss"], label="Val", linestyle="--")
    ax2.set_title(f"Phase {phase} Loss"); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    path = os.path.join(CURVES_DIR, f"training_curves_phase{phase}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Curves saved → {path}")

def train():
    print("\n=== Sign Language Recognition — Training ===\n")
    train_gen, val_gen, test_gen, class_names = get_generators()
    num_classes = len(class_names)

    print("\n[Phase 1] Training head (base frozen)...")
    model = build_model(num_classes, fine_tune=False)
    h1 = model.fit(train_gen, validation_data=val_gen,
                   epochs=EPOCHS, callbacks=make_callbacks())
    plot_history(h1, 1)

    print("\n[Phase 2] Fine-tuning top layers...")
    model_ft = build_model(num_classes, fine_tune=True)
    model_ft.load_weights(MODEL_PATH)
    h2 = model_ft.fit(train_gen, validation_data=val_gen,
                      epochs=10, callbacks=make_callbacks())
    plot_history(h2, 2)

    # ── Safe test evaluation ───────────────────────────────────────────────
    if test_gen.samples > 0:
        print("\nEvaluating on test set...")
        loss, acc = model_ft.evaluate(test_gen)
        print(f"\nTest accuracy: {acc*100:.2f}%  |  Loss: {loss:.4f}")
    else:
        print("\nTest folder is empty — skipping evaluation.")
        print("Model trained successfully on train+val data.")

    with open(LABELS_PATH, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"\nDone! Model → {MODEL_PATH}")

if __name__ == "__main__":
    train()