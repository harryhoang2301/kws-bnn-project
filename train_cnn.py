import numpy as np
import tensorflow as tf
from keras import layers, models
from pathlib import Path

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


# Load preprocessed data
# Paths to .npz files
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
num_classes = len(BASE_PRETRAIN_CLASSES)

x_train, y_train = restrict_to_base_classes(x_train, y_train, class_names)
x_val, y_val = restrict_to_base_classes(x_val, y_val, class_names)
x_test, y_test = restrict_to_base_classes(x_test, y_test, class_names)

print(f" x_train shape: {x_train.shape}")
print(f" y_train shape: {y_train.shape}")
print(f" Number of classes: {num_classes}")
print(f" Classes: {BASE_PRETRAIN_CLASSES}")

# Channel dimension for CNN
#(40, 101, 1)
x_train = x_train[..., np.newaxis].astype("float32")
x_val   = x_val[..., np.newaxis].astype("float32")
x_test  = x_test[..., np.newaxis].astype("float32")

print(f" After adding channel dimension:")
print(f"       x_train: {x_train.shape}")
print(f"       x_val:   {x_val.shape}")
print(f"       x_test:  {x_test.shape}")

# Labels are integers, for sparse_categorical_crossentropy
y_train = y_train.astype("int32")
y_val   = y_val.astype("int32")
y_test  = y_test.astype("int32")

# Building simple CNN model
# Input:  (40, 101, 1)
# Output: probabilities over num_classes

print("Building model...")

model = models.Sequential([
    layers.Input(shape=(40, 101, 1)),
    # Block 1
    layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    # Block 2
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    # Second Block 2 
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    # Block 3
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    # Second Block 3
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    # Classification Layers
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax"),
])

model.summary()

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",   # because labels are integers
    metrics=["accuracy"],
)

# Train the model
batch_size = 64
epochs = 50

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=10,              # stop if val accuracy doesn't improve for 10 epochs
    restore_best_weights=True,
)

print("Starting training...")

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[early_stop],
)

# Evaluate on test set
print("Evaluating on test set...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"[RESULT] Test accuracy: {test_acc * 100:.2f}%")
print(f"[RESULT] Test loss: {test_loss:.4f}")

# Save the model
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "cnn_kws.keras"

model.save(MODEL_PATH)
print(f" Saved model to: {MODEL_PATH.resolve()}")

# Plot training curves
import matplotlib.pyplot as plt

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
