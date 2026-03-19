import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers, models

BASE_PRETRAIN_CLASSES = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"
]


def set_global_seed(seed):
    """Set Python, NumPy, and TensorFlow seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


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


def build_cnn_model(num_classes):
    return models.Sequential(
        [
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
        ]
    )


def run_training(seed=0, save_artifacts=True, plot_curves=True, verbose=1):
    # Keep all training choices fixed except the random seed.
    set_global_seed(seed)

    # Paths to .npz files
    train_path = Path("data/processed/logmel_train.npz")
    val_path = Path("data/processed/logmel_val.npz")
    test_path = Path("data/processed/logmel_test.npz")
    stats_path = Path("data/processed/logmel_stats.npz")

    print(f"[INFO] Loading data for CNN (seed={seed})...")
    x_train, y_train = load_data(train_path, ("x", "y"))
    x_val, y_val = load_data(val_path, ("x", "y"))
    x_test, y_test = load_data(test_path, ("x", "y"))
    (class_names,) = load_data(stats_path, ("class_names",), allow_pickle=True)

    num_classes = len(BASE_PRETRAIN_CLASSES)
    x_train, y_train = restrict_to_base_classes(x_train, y_train, class_names)
    x_val, y_val = restrict_to_base_classes(x_val, y_val, class_names)
    x_test, y_test = restrict_to_base_classes(x_test, y_test, class_names)

    print(f"[INFO] x_train shape: {x_train.shape}")
    print(f"[INFO] y_train shape: {y_train.shape}")
    print(f"[INFO] Number of classes: {num_classes}")
    print(f"[INFO] Classes: {BASE_PRETRAIN_CLASSES}")

    # Add channel dimension for CNN.
    x_train = x_train[..., np.newaxis].astype("float32")
    x_val = x_val[..., np.newaxis].astype("float32")
    x_test = x_test[..., np.newaxis].astype("float32")

    y_train = y_train.astype("int32")
    y_val = y_val.astype("int32")
    y_test = y_test.astype("int32")

    print(f"[INFO] Channelized x_train: {x_train.shape}")
    print(f"[INFO] Channelized x_val:   {x_val.shape}")
    print(f"[INFO] Channelized x_test:  {x_test.shape}")

    model = build_cnn_model(num_classes)
    if verbose:
        model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    batch_size = 64
    epochs = 50

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
    )

    print("[INFO] Starting CNN training...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stop],
        verbose=verbose,
    )

    val_eval_loss, val_eval_acc = model.evaluate(x_val, y_val, verbose=0)
    print("[INFO] Evaluating CNN on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[RESULT] CNN Val accuracy: {val_eval_acc * 100:.2f}%")
    print(f"[RESULT] CNN Val loss: {val_eval_loss:.4f}")
    print(f"[RESULT] CNN Test accuracy: {test_acc * 100:.2f}%")
    print(f"[RESULT] CNN Test loss: {test_loss:.4f}")

    if save_artifacts:
        model_dir = Path("models")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "cnn_kws.keras"
        model.save(model_path)
        print(f"[INFO] Saved CNN model to: {model_path.resolve()}")

    history_dict = history.history
    acc = history_dict.get("accuracy", [])
    val_acc_hist = history_dict.get("val_accuracy", [])
    loss = history_dict.get("loss", [])
    val_loss_hist = history_dict.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc_hist, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss_hist, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    fig_dir = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    if plot_curves:
        plt.savefig(fig_dir / "cnn_training_curves.png", dpi=300)
        print(f"[INFO] Saved CNN training curves to: {(fig_dir / 'cnn_training_curves.png').resolve()}")
        plt.show()
    plt.close()

    return {
        "model_name": "cnn",
        "seed": int(seed),
        "validation_accuracy": float(val_eval_acc),
        "validation_loss": float(val_eval_loss),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "parameter_count": int(model.count_params()),
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the main CNN model.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for Python, NumPy, and TensorFlow.")
    parser.add_argument("--no_plot", action="store_true", help="Disable training-curve plotting.")
    parser.add_argument("--no_save", action="store_true", help="Disable saving trained model.")
    args = parser.parse_args()

    run_training(
        seed=args.seed,
        save_artifacts=not args.no_save,
        plot_curves=not args.no_plot,
        verbose=1,
    )


if __name__ == "__main__":
    main()
