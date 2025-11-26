import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path


# 1. Load preprocessed data

# Paths to your .npz files (adjust if needed)
TRAIN_PATH = Path("data/processed/logmel_train.npz")
VAL_PATH   = Path("data/processed/logmel_val.npz")
TEST_PATH  = Path("data/processed/logmel_test.npz")
STATS_PATH = Path("data/processed/logmel_stats.npz")

print("[INFO] Loading data...")

train_data = np.load(TRAIN_PATH)
val_data   = np.load(VAL_PATH)
test_data  = np.load(TEST_PATH)
stats_data = np.load(STATS_PATH, allow_pickle=True)

x_train, y_train = train_data["x"], train_data["y"]  # shapes: (N, 40, 101)
x_val, y_val     = val_data["x"], val_data["y"]
x_test, y_test   = test_data["x"], test_data["y"]

class_names = stats_data["class_names"]
num_classes = len(class_names)

print(f"[INFO] x_train shape: {x_train.shape}")
print(f"[INFO] y_train shape: {y_train.shape}")
print(f"[INFO] Number of classes: {num_classes}")
print(f"[INFO] Classes: {class_names}")

# -----------------------
# 2. Add channel dimension for CNN
#    (40, 101) -> (40, 101, 1)
# -----------------------

x_train = x_train[..., np.newaxis].astype("float32")
x_val   = x_val[..., np.newaxis].astype("float32")
x_test  = x_test[..., np.newaxis].astype("float32")

print(f"[INFO] After adding channel dimension:")
print(f"       x_train: {x_train.shape}")
print(f"       x_val:   {x_val.shape}")
print(f"       x_test:  {x_test.shape}")

# Labels are integers (0..num_classes-1), so we can use sparse_categorical_crossentropy
y_train = y_train.astype("int32")
y_val   = y_val.astype("int32")
y_test  = y_test.astype("int32")

# 3. Build a simple CNN model
# Input:  (40, 101, 1)
# Output: probabilities over num_classes

print("[INFO] Building model...")

model = models.Sequential([
    layers.Input(shape=(40, 101, 1)),

    # Block 1
    layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Block 2
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Block 3
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Classifier
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax"),
])

model.summary()

# 4. Compile the model

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",   # because labels are integers
    metrics=["accuracy"],
)

# 5. Train the model

batch_size = 64
epochs = 20

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,              # stop if val accuracy doesn't improve for 5 epochs
    restore_best_weights=True,
)

print("[INFO] Starting training...")

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[early_stop],
)

# 6. Evaluate on test set

print("[INFO] Evaluating on test set...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"[RESULT] Test accuracy: {test_acc * 100:.2f}%")
print(f"[RESULT] Test loss: {test_loss:.4f}")

# 7. Save the model

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "cnn_kws.h5"

model.save(MODEL_PATH)
print(f"[INFO] Saved model to: {MODEL_PATH.resolve()}")

print("[INFO] Evaluating on test set...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"[RESULT] Test accuracy: {test_acc * 100:.2f}%")
print(f"[RESULT] Test loss: {test_loss:.4f}")

# Plot training curves

import matplotlib.pyplot as plt

history_dict = history.history
acc      = history_dict.get("accuracy", [])
val_acc  = history_dict.get("val_accuracy", [])
loss     = history_dict.get("loss", [])
val_loss = history_dict.get("val_loss", [])

epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(10, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Train Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Train Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
