import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers, models

from bnn_layers_weight_only import BinaryConv2D

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

    # Second Block 2 (binary conv)
    x = BinaryConv2D(32, (3, 3), padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.AvgPool2D((2, 2))(x)
    x = layers.BatchNormalization()(x)

    # Block 3 (binary conv)
    x = BinaryConv2D(64, (3, 3), padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.AvgPool2D((2, 2))(x)
    x = layers.BatchNormalization()(x)

    # Second Block 3 (binary conv)
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


def run_training(seed=0, save_artifacts=True, plot_curves=True, verbose=1, weights_out="models/bnnweightsonly.weights.h5"):
    # Keep all training choices fixed except the random seed.
    set_global_seed(seed)

    train_path = Path("data/processed/logmel_train.npz")
    val_path = Path("data/processed/logmel_val.npz")
    test_path = Path("data/processed/logmel_test.npz")
    stats_path = Path("data/processed/logmel_stats.npz")

    print(f"[INFO] Loading data for BNN(weight-only) (seed={seed})...")
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
    if verbose:
        model.summary()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4,
        epsilon=1e-5,
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
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stop],
        verbose=verbose,
    )

    val_eval_loss, val_eval_acc = model.evaluate(x_val, y_val, verbose=0)
    print("[INFO] Evaluating BNN(weight-only) on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[RESULT] Val accuracy: {val_eval_acc * 100:.2f}%")
    print(f"[RESULT] Val loss: {val_eval_loss:.4f}")
    print(f"[RESULT] Test accuracy: {test_acc * 100:.2f}%")
    print(f"[RESULT] Test loss: {test_loss:.4f}")

    if save_artifacts:
        weights_path = Path(weights_out)
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(weights_path)
        print(f"[INFO] Saved upgraded BNN weights to: {weights_path.resolve()}")

    history_dict = history.history
    acc = history_dict.get("accuracy", [])
    val_acc_hist = history_dict.get("val_accuracy", [])
    loss = history_dict.get("loss", [])
    val_loss_hist = history_dict.get("val_loss", [])

    epochs_range = range(1, len(acc) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc_hist, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("BNN Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss_hist, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("BNN Training vs Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    fig_dir = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    if plot_curves:
        fig_path = fig_dir / f"bnn_weights_training_curves_seed{seed}.png"
        plt.savefig(fig_path, dpi=300)
        print(f"[INFO] Saved BNN training curves to: {fig_path.resolve()}")
        plt.show()
    plt.close()

    return {
        "model_name": "bnn_weight_only",
        "seed": int(seed),
        "validation_accuracy": float(val_eval_acc),
        "validation_loss": float(val_eval_loss),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "parameter_count": int(model.count_params()),
    }


def parse_seeds(seed_text):
    return [int(x.strip()) for x in seed_text.split(",") if x.strip()]


def write_results_csv(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_name",
        "seed",
        "validation_accuracy",
        "validation_loss",
        "test_accuracy",
        "test_loss",
        "parameter_count",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Saved seed run summary CSV to: {out_path.resolve()}")


def print_seed_summary(rows):
    test_acc = np.array([r["test_accuracy"] for r in rows], dtype=np.float64)
    test_loss = np.array([r["test_loss"] for r in rows], dtype=np.float64)
    std_ddof = 1 if len(rows) > 1 else 0

    print("\n=== BNN Weight-Only Seed Summary ===")
    print(f"Runs: {len(rows)}")
    print(f"Mean test accuracy: {np.mean(test_acc):.6f}")
    print(f"Std test accuracy:  {np.std(test_acc, ddof=std_ddof):.6f}")
    print(f"Mean test loss:     {np.mean(test_loss):.6f}")
    print(f"Std test loss:      {np.std(test_loss, ddof=std_ddof):.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate weight-only BNN.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for Python, NumPy, and TensorFlow.")
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated seeds for multi-seed runs (example: 0,1,2,3,4).",
    )
    parser.add_argument("--weights_out", type=str, default="models/bnnweightsonly.weights.h5", help="Weights output path.")
    parser.add_argument("--no_plot", action="store_true", help="Disable training-curve plotting.")
    parser.add_argument("--no_save", action="store_true", help="Disable saving trained weights.")
    parser.add_argument(
        "--results_csv",
        type=str,
        default="results/bnn_weights_seed_runs_summary.csv",
        help="CSV path for multi-seed results.",
    )
    args = parser.parse_args()

    seed_list = parse_seeds(args.seeds) if args.seeds else [int(args.seed)]

    rows = []
    for seed in seed_list:
        if len(seed_list) > 1:
            # In multi-seed mode, keep each seed's checkpoint separate.
            seed_weights_out = (
                args.weights_out.replace(".weights.h5", f"_seed{seed}.weights.h5")
                if args.weights_out.endswith(".weights.h5")
                else f"{args.weights_out}_seed{seed}"
            )
        else:
            seed_weights_out = args.weights_out

        row = run_training(
            seed=seed,
            save_artifacts=not args.no_save,
            plot_curves=not args.no_plot,
            verbose=1,
            weights_out=seed_weights_out,
        )
        rows.append(row)

    if len(rows) > 1:
        write_results_csv(rows, args.results_csv)
    print_seed_summary(rows)


if __name__ == "__main__":
    main()
