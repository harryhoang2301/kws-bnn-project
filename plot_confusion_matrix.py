import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from sklearn.metrics import confusion_matrix

# -----------------------
# Paths
# -----------------------

TEST_PATH  = Path("data/processed/logmel_test.npz")
STATS_PATH = Path("data/processed/logmel_stats.npz")

# If your model is saved as .h5 use this:
MODEL_PATH = Path("models/cnn_kws.h5")

# If you changed to the new format earlier, use instead:
# MODEL_PATH = Path("models/cnn_kws.keras")

# Where to save the figure
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_PATH = FIG_DIR / "confusion_matrix.png"


# -----------------------
# Load data & model
# -----------------------

print("[INFO] Loading test data...")
test_data  = np.load(TEST_PATH)
x_test, y_test = test_data["x"], test_data["y"]

stats_data = np.load(STATS_PATH, allow_pickle=True)
class_names = stats_data["class_names"]
num_classes = len(class_names)

print(f"[INFO] x_test shape: {x_test.shape}")
print(f"[INFO] y_test shape: {y_test.shape}")
print(f"[INFO] Classes: {class_names}")

# Add channel dimension: (N, 40, 101) -> (N, 40, 101, 1)
x_test = x_test[..., np.newaxis].astype("float32")
y_test = y_test.astype("int32")

print("[INFO] Loading model:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------
# Predict on test set
# -----------------------

print("[INFO] Running predictions...")
y_pred_probs = model.predict(x_test, batch_size=64, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# -----------------------
# Build confusion matrix
# -----------------------

cm = confusion_matrix(y_test, y_pred, labels=range(num_classes))
print("[INFO] Confusion matrix shape:", cm.shape)

# Normalise per row (per true class) to get percentages
cm_norm = cm.astype("float32") / cm.sum(axis=1, keepdims=True)

# -----------------------
# Plot confusion matrix
# -----------------------

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)

ax.figure.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(num_classes),
    yticks=np.arange(num_classes),
    xticklabels=class_names,
    yticklabels=class_names,
    ylabel="True label",
    xlabel="Predicted label",
    title="Confusion Matrix (normalised per class)",
)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Optionally, add numbers inside the cells
thresh = cm_norm.max() / 2.0
for i in range(num_classes):
    for j in range(num_classes):
        value = cm_norm[i, j]
        ax.text(
            j, i, f"{value:.2f}",
            ha="center", va="center",
            color="white" if value > thresh else "black",
            fontsize=7,
        )

fig.tight_layout()

# Save and show
plt.savefig(FIG_PATH, dpi=300)
print(f"[INFO] Saved confusion matrix figure to: {FIG_PATH.resolve()}")

plt.show()