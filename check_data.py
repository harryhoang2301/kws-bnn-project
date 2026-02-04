import os
import time
import numpy as np
from pathlib import Path
import keras  # <-- IMPORTANT: use keras (Keras 3)
from bnn_layers import BinaryActivation, BinaryConv2D, BinaryDense
import tensorflow as tf

CNN_PATH = Path("models/cnn_kws.h5")
BNN_PATH = Path("models/bnn_kws_v2_registered.keras")

# CNN: no custom layers, can load normally
cnn = keras.saving.load_model(CNN_PATH, compile=False)

# BNN: custom layers must be provided
custom_objects = {
    "BinaryActivation": BinaryActivation,
    "BinaryConv2D": BinaryConv2D,
    "BinaryDense": BinaryDense,

    # also add registered names in case Keras stored them that way
    "BNN>BinaryActivation": BinaryActivation,
    "BNN>BinaryConv2D": BinaryConv2D,
    "BNN>BinaryDense": BinaryDense,
}

bnn = keras.saving.load_model(
    BNN_PATH,
    compile=False,
    custom_objects=custom_objects,
    safe_mode=False,
)

print("Loaded CNN + BNN successfully.")
print("CNN params:", cnn.count_params())
print("BNN params:", bnn.count_params())

# -----------------------
# Helpers
# -----------------------
def mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)

def save_both_formats(model, base_name: str):
    """
    Saves model in .keras, .h5, and weights-only .weights.h5
    (without optimizer state) so comparisons are fair.
    """
    # Full model saves
    model.save(f"{base_name}.keras", include_optimizer=False)
    model.save(f"{base_name}.h5", include_optimizer=False)

    # Weights-only (most fair for weight storage comparison)
    model.save_weights(f"{base_name}.weights.h5")

def export_tflite(model, out_path: str):
    """
    Exports model to tflite. If your BNN uses unsupported custom layers/ops,
    this may fail or fall back to float kernels.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)

def bench_keras(model, input_shape, iters=200, warmup=50):
    """
    Simple inference latency benchmark on current machine.
    """
    x = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        _ = model(x, training=False)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x, training=False)
    t1 = time.perf_counter()

    return (t1 - t0) / iters * 1000  # ms

# -----------------------
# 2) Print param counts
# -----------------------
print("=== PARAM COUNTS ===")
print("CNN params:", cnn.count_params())
print("BNN params:", bnn.count_params())
print()

# -----------------------
# 3) Save in SAME formats
# -----------------------
save_both_formats(cnn, "cnn_check")
save_both_formats(bnn, "bnn_check")

print("=== FILE SIZE COMPARISON (MB) ===")
for fname in [
    "cnn_check.keras", "bnn_check.keras",
    "cnn_check.h5", "bnn_check.h5",
    "cnn_check.weights.h5", "bnn_check.weights.h5",
]:
    print(f"{fname:22s}  {mb(fname):.3f} MB")
print()

# -----------------------
# 4) TFLite export + size
# -----------------------
print("=== TFLITE SIZE (MB) ===")
try:
    export_tflite(cnn, "cnn_check.tflite")
    print(f"{'cnn_check.tflite':22s}  {mb('cnn_check.tflite'):.3f} MB")
except Exception as e:
    print("CNN TFLite export failed:", e)

try:
    export_tflite(bnn, "bnn_check.tflite")
    print(f"{'bnn_check.tflite':22s}  {mb('bnn_check.tflite'):.3f} MB")
except Exception as e:
    print("BNN TFLite export failed:", e)

print()

# -----------------------
# 5) Optional: latency benchmark (batch=1)
# -----------------------
# Change input_shape if yours is different.
# Common KWS shape: (1, 40, 101, 1)
input_shape = (1, 40, 101, 1)

print("=== KERAS CPU LATENCY (ms, batch=1) ===")
try:
    cnn_ms = bench_keras(cnn, input_shape)
    bnn_ms = bench_keras(bnn, input_shape)
    print(f"CNN: {cnn_ms:.3f} ms")
    print(f"BNN: {bnn_ms:.3f} ms")
except Exception as e:
    print("Latency benchmark failed:", e)
