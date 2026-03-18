import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from keras import layers, models
from bnn_layers import BinaryActivation, BinaryConv2D

BASE_PRETRAIN_CLASSES = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"
]

# ----------------------------------------------
# BNN model (weight and activation binarisation)
# ----------------------------------------------

def build_bnn_model(num_classes):
    inputs = layers.Input(shape=(40, 101, 1))

    # Block 1 (full-precision conv)
    x = layers.Conv2D(16, (3, 3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = BinaryActivation()(x)

    # Block 2 (binary conv)
    x = BinaryConv2D(32, (3, 3), padding="same")(x)
    x = BinaryActivation()(x)
    x = layers.AvgPool2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    # Second Block 2 (binary conv)
    x = BinaryConv2D(32, (3, 3), padding="same")(x)
    x = BinaryActivation()(x)
    x = layers.AvgPool2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    # Block 3 (binary conv)
    x = BinaryConv2D(64, (3, 3), padding="same")(x)
    x = BinaryActivation()(x)
    x = layers.AvgPool2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    #  Second Block 3 (binary conv)
    x = BinaryConv2D(64, (3, 3), padding="same")(x)
    x = BinaryActivation()(x)
    x = layers.AvgPool2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    # Classifier
    x = layers.Flatten()(x)
    x = layers.Dense(128 , activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="softmax")(x)
    return models.Model(inputs, outputs)

def load_data(path, keys, allow_pickle=False):
    try:
        with np.load(path, allow_pickle=allow_pickle) as data:
            return tuple(np.array(data[k]) for k in keys)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read '{path}'. The file may be incomplete/corrupted. "
            "Try regenerating processed data with preprocess.py."
        ) from exc


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


def main():
    # Load data
    train_path = Path("data/processed/logmel_train.npz")
    val_path = Path("data/processed/logmel_val.npz")
    test_path = Path("data/processed/logmel_test.npz")
    stats_path = Path("data/processed/logmel_stats.npz")

    print("[INFO] Loading data...")

    x_train, y_train = load_data(train_path, ("x", "y"))
    x_val, y_val = load_data(val_path, ("x", "y"))
    x_test, y_test = load_data(test_path, ("x", "y"))
    (class_names,) = load_data(stats_path, ("class_names",), allow_pickle=True)
    num_classes = len(BASE_PRETRAIN_CLASSES)

    x_train, y_train = restrict_to_base_classes(x_train, y_train, class_names)
    x_val, y_val = restrict_to_base_classes(x_val, y_val, class_names)
    x_test, y_test = restrict_to_base_classes(x_test, y_test, class_names)

    # Add channel dimension
    x_train = x_train[..., np.newaxis].astype("float32")
    x_val = x_val[..., np.newaxis].astype("float32")
    x_test = x_test[..., np.newaxis].astype("float32")

    y_train = y_train.astype("int32")
    y_val = y_val.astype("int32")
    y_test = y_test.astype("int32")

    print("[INFO] Data loaded successfully.")
    print(f"[INFO] Base pretrain classes ({num_classes}): {BASE_PRETRAIN_CLASSES}")
    print(f"[INFO] Filtered sizes -> train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")

    # Build and train BNN
    model = build_bnn_model(num_classes)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=3e-4, 
        epsilon=1e-5,   # slightly smaller eps helps BNN stability
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    batch_size = 64
    epochs = 50

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
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(model_dir / "bnn_kws_v2_registered.weights.h5")
    print(f"[INFO] Saved BNN weights to: {(model_dir / 'bnn_kws_v2_registered.weights.h5').resolve()}")

    # Plot training curves (BNN)
    history_dict = history.history
    acc = history_dict.get("accuracy", [])
    val_acc = history_dict.get("val_accuracy", [])
    loss = history_dict.get("loss", [])
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

    fig_dir = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "bnn_training_curves.png", dpi=300)
    print(f"[INFO] Saved BNN training curves to: {(fig_dir / 'bnn_training_curves.png').resolve()}")


if __name__ == "__main__":
    main()
