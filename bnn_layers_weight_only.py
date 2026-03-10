import keras
from keras import layers
import tensorflow as tf

# ---------
# STE sign
# ---------
@tf.custom_gradient
def ste_sign(x):
    """
    Forward: sign(x) in {-1,+1} (treat 0 as +1).
    Backward: straight-through estimator with optional mask (|x|<=1).
    """
    y = tf.where(x >= 0.0, tf.ones_like(x), -tf.ones_like(x))

    def grad(dy):
        # Pass gradients only where input is in [-1, 1] (common STE choice)
        mask = tf.cast(tf.abs(x) <= 1.0, dy.dtype)
        return dy * mask

    return y, grad


if hasattr(keras, "saving") and hasattr(keras.saving, "register_keras_serializable"):
    register_keras_serializable = keras.saving.register_keras_serializable
else:
    register_keras_serializable = tf.keras.utils.register_keras_serializable


@register_keras_serializable(package="BNNWeightOnly")
class WeightClip(keras.constraints.Constraint):
    def __init__(self, min_value=-1.0, max_value=1.0):
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

    def get_config(self):
        return {"min_value": self.min_value, "max_value": self.max_value}


@register_keras_serializable(package="BNNWeightOnly")
class BinaryConv2D(layers.Layer):
    """
    Binary-weight Conv2D intended for REAL-VALUED inputs (not binarised activations).

    - Keeps shadow weights w_fp (clipped).
    - Binarises weights with STE: w_b = alpha * sign(w_fp).
    - Optional per-channel input RMS normalisation for stability.
    """
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        use_bias=False,
        input_rms_norm=True,
        eps=1e-6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = tuple(strides)
        self.padding = padding.upper()
        self.use_bias = bool(use_bias)
        self.input_rms_norm = bool(input_rms_norm)
        self.eps = float(eps)

    def build(self, input_shape):
        kh, kw = self.kernel_size
        in_ch = int(input_shape[-1])

        init = tf.random_normal_initializer(stddev=0.1)
        self.w_fp = self.add_weight(
            name="w_fp",
            shape=(kh, kw, in_ch, self.filters),
            initializer=init,
            trainable=True,
            constraint=WeightClip(-1.0, 1.0),
        )

        if self.use_bias:
            self.b = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.b = None

        super().build(input_shape)

    def call(self, inputs, training=None):
        x = tf.convert_to_tensor(inputs)

        # Optional stabilisation for real inputs:
        # Normalise each input channel by its RMS over spatial dims.
        if self.input_rms_norm:
            # rms shape: (B, 1, 1, C)
            rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True) + self.eps)
            x = x / rms

        w_fp = tf.convert_to_tensor(self.w_fp)

        # Per-output-channel scaling factor alpha (no gradient through alpha)
        alpha = tf.stop_gradient(
            tf.reduce_mean(tf.abs(w_fp), axis=(0, 1, 2), keepdims=True)
        )  # shape (1,1,1,filters)

        w_bin = alpha * ste_sign(w_fp)

        y = tf.nn.conv2d(
            x,
            w_bin,
            strides=(1, self.strides[0], self.strides[1], 1),
            padding=self.padding,
        )

        if self.b is not None:
            y = tf.nn.bias_add(y, self.b)

        return y

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "use_bias": self.use_bias,
            "input_rms_norm": self.input_rms_norm,
            "eps": self.eps,
        })
        return cfg