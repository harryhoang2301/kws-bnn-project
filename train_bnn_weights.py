import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from keras import layers, models
from bnn_layers_weight_only import BinaryConv2D

BASE_PRETRAIN_CLASSES = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"
]


def restrict_to_base_classes(x, y, class_names):
    class_names_list = [str(c) for c in class_names.tolist()]
    missing = [name for name in BASE_PRETRAIN_CLASSES if name not in class_names_list]
    if missing:
        raise ValueError(f"Missing required base classes in stats: {missing}")

    base_old_ids = np.array([class_names_list.index(name) for name in BASE_PRETRAIN_CLASSES], dtype=np.int32)
    remap = np.full(len(class_names_list), -1, dtype=np.int32)
    remap[base_old_ids] = np.arange(len(BASE_PRETRAIN_CLASSES), dtype=np.int32)

    keep = remap[y] >= 0
    x_out = x[keep]
    y_out = remap[y[keep]]
    return x_out, y_out

# Load data

TRAIN_PATH = Path("data/processed/logmel_train.npz")
VAL_PATH   = Path("data/processed/logmel_val.npz")
TEST_PATH  = Path("data/processed/logmel_test.npz")
STATS_PATH = Path("data/processed/logmel_stats.npz")

print("[INFO] Loading data...")

train_data = np.load(TRAIN_PATH)
val_data   = np.load(VAL_PATH)
test_data  = np.load(TEST_PATH)
stats_data = np.load(STATS_PATH, allow_pickle=True)

x_train, y_train = train_data["x"], train_data["y"]
x_val, y_val     = val_data["x"], val_data["y"]
x_test, y_test   = test_data["x"], test_data["y"]

class_names = stats_data["class_names"]
num_classes = len(BASE_PRETRAIN_CLASSES)

x_train, y_train = restrict_to_base_classes(x_train, y_train, class_names)
x_val, y_val = restrict_to_base_classes(x_val, y_val, class_names)
x_test, y_test = restrict_to_base_classes(x_test, y_test, class_names)

# Add channel dimension
x_train = x_train[..., np.newaxis].astype("float32")
x_val   = x_val[..., np.newaxis].astype("float32")
x_test  = x_test[..., np.newaxis].astype("float32")

y_train = y_train.astype("int32")
y_val   = y_val.astype("int32")
y_test  = y_test.astype("int32")

print("[INFO] Data loaded successfully.")
print(f"[INFO] Base pretrain classes ({num_classes}): {BASE_PRETRAIN_CLASSES}")
print(f"[INFO] Filtered sizes -> train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")

# ----------------------------------------------
# BNN model (weight-only binarization)
# ----------------------------------------------

def build_bnn_model(num_classes):
    inputs = layers.Input(shape=(40, 101, 1))

      # Block 1 (full-precision conv)
    x = layers.Conv2D(16, (3, 3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 2 (binary conv)
    x = BinaryConv2D(32, (3, 3), padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.AvgPool2D((2, 2))(x)
    x = layers.BatchNormalization()(x)

    # Block 2 (binary conv)
    x = BinaryConv2D(32, (3, 3), padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.AvgPool2D((2, 2))(x)
    x = layers.BatchNormalization()(x)

    # Block 3 (binary conv)
    x = BinaryConv2D(64, (3, 3), padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.AvgPool2D((2, 2))(x)
    x = layers.BatchNormalization()(x)

    # Block 3 (binary conv)
    x = BinaryConv2D(64, (3, 3), padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.AvgPool2D((2, 2))(x)
    x = layers.BatchNormalization()(x)

    # Classifier
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    logits = layers.Dense(num_classes, activation=None, name="trainable_head")(x)
    return models.Model(inputs, logits, name="bnn_weightbin_gap")

# Build and train BNN
model = build_bnn_model(num_classes)
model.summary()

optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-4,
    epsilon=1e-5,   # slightly smaller eps helps BNN stability
)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

batch_size = 64
epochs = 40

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[early_stop],
)

print("[INFO] Evaluating BNN on test set...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"[RESULT] BNN Test accuracy: {test_acc * 100:.2f}%")
print(f"[RESULT] BNN Test loss: {test_loss:.4f}")

# save the BNN
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
model.save_weights(MODEL_DIR / "bnnweightsonly.weights.h5")
print(f"[INFO] Saved upgraded BNN weights to: {(MODEL_DIR / 'bnnweightsonly.weights.h5').resolve()}")

# Plot training curves (BNN)
history_dict = history.history
acc      = history_dict.get("accuracy", [])
val_acc  = history_dict.get("val_accuracy", [])
loss     = history_dict.get("loss", [])
val_loss = history_dict.get("val_loss", [])

epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(10, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Train Accuracy")
plt.plot(epochs_range, val_acc, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("BNN Training vs Validation Accuracy")
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Train Loss")
plt.plot(epochs_range, val_loss, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("BNN Training vs Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
    
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIG_DIR / "bnn_weights_training_curves.png", dpi=300)
print(f"[INFO] Saved BNN training curves to: { (FIG_DIR / 'bnn_weights_training_curves.png').resolve() }")
