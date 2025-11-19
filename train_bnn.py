import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------
# Load data
# -----------------------

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
num_classes = len(class_names)

# Add channel dimension
x_train = x_train[..., np.newaxis].astype("float32")
x_val   = x_val[..., np.newaxis].astype("float32")
x_test  = x_test[..., np.newaxis].astype("float32")

y_train = y_train.astype("int32")
y_val   = y_val.astype("int32")
y_test  = y_test.astype("int32")

print("[INFO] Data loaded successfully.")

# -----------------------
# Binarisation + STE
# -----------------------

@tf.custom_gradient
def binary_activation(x):
    """
    Forward pass: sign(x)
    Backward pass: Straight-Through Estimator (STE)
    """
    y = tf.sign(x)
    # Ensure 0 -> +1 (TensorFlow's sign gives 0 sometimes)
    y = tf.where(tf.equal(y, 0), tf.ones_like(y), y)

    def grad(dy):
        # STE: gradient is passed through unchanged where |x| <= 1
        dx = dy * tf.where(tf.abs(x) <= 1.0, 1.0, 0.0)
        return dx

    return y, grad

class BinaryActivation(layers.Layer):
    def call(self, inputs):
        return binary_activation(inputs)

class BinaryDense(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        # full-precision weights (trainable)
        init = tf.random_normal_initializer(stddev=0.1)
        self.w = self.add_weight(
            shape=(int(input_shape[-1]), self.units),
            initializer=init,
            trainable=True,
            name="w_fp"
        )

    def call(self, inputs):
        # Binarise weights: sign(w_fp)
        w_bin = binary_activation(self.w)
        # Binarise inputs: sign(x)
        x_bin = binary_activation(inputs)

        # Binary matmul
        out = tf.matmul(x_bin, w_bin)
        return out

class BinaryConv2D(layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding="same"):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides
        self.padding = padding.upper()  # "SAME" or "VALID"

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        kh, kw = self.kernel_size

        init = tf.random_normal_initializer(stddev=0.1)
        # full-precision kernel
        self.w = self.add_weight(
            shape=(kh, kw, in_channels, self.filters),
            initializer=init,
            trainable=True,
            name="w_fp"
        )

    def call(self, inputs):
        # Binarise weights and inputs
        w_bin = binary_activation(self.w)
        x_bin = binary_activation(inputs)

        # tf.nn.conv2d wants NHWC: (batch, H, W, C)
        out = tf.nn.conv2d(
            x_bin,
            w_bin,
            strides=(1, self.strides[0], self.strides[1], 1),
            padding=self.padding,
        )
        return out

# -----------------------
# Tiny test BNN model
# -----------------------

def build_tiny_bnn():
    model = models.Sequential([
        layers.Flatten(input_shape=(40, 101, 1)),
        BinaryDense(128),
        layers.BatchNormalization(),
        layers.Activation(binary_activation),
        layers.Dense(num_classes, activation="softmax")  # keep final layer full precision
    ])
    return model

model = build_tiny_bnn()
model.summary()
# -----------------------
# Full BNN model (CNN-like)
# -----------------------

def build_bnn_model():
    inputs = layers.Input(shape=(40, 101, 1))

    # -------------------------
    # Float front-end
    # -------------------------
    x = layers.Conv2D(16, (3, 3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = BinaryActivation()(x)      # first binarisation

    # -------------------------
    # Binary block 1
    # -------------------------
    x = BinaryConv2D(32, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = BinaryActivation()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # -------------------------
    # Binary block 2
    # -------------------------
    x = BinaryConv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = BinaryActivation()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # -------------------------
    # Classifier
    # -------------------------
    x = layers.Flatten()(x)

    x = BinaryDense(64)(x)
    x = layers.BatchNormalization()(x)
    x = BinaryActivation()(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)

# -----------------------
# Build and train BNN
# -----------------------

model = build_bnn_model()
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(5e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

batch_size = 64
epochs = 20

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

