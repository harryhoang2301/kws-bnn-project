import keras
from keras import layers
import tensorflow as tf

if hasattr(keras, "saving") and hasattr(keras.saving, "register_keras_serializable"):
    register_keras_serializable = keras.saving.register_keras_serializable
else:
    register_keras_serializable = tf.keras.utils.register_keras_serializable


@register_keras_serializable(package="BNN")
class WeightClip(keras.constraints.Constraint):
    def __init__(self, min_value=-1.0, max_value=1.0):
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

    def get_config(self):
        return {"min_value": self.min_value, "max_value": self.max_value}


@tf.custom_gradient
def binary_activation(x):
    # I map sign(0) to +1 to keep binary outputs deterministic.
    y = tf.sign(x)
    y = tf.where(tf.equal(y, 0), tf.ones_like(y), y)

    def grad(dy):
        mask = tf.cast(tf.abs(x) <= 1.0, dy.dtype)
        return dy * mask * 0.5

    return y, grad


@register_keras_serializable(package="BNN")
class BinaryActivation(layers.Layer):
    def call(self, inputs):
        return binary_activation(inputs)

    def get_config(self):
        return super().get_config()


@register_keras_serializable(package="BNN")
class BinaryConv2D(layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding="same", **kwargs):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.strides = strides
        self.padding = padding.upper()

    def build(self, input_shape):
        kh, kw = self.kernel_size
        in_ch = int(input_shape[-1])
        init = tf.random_normal_initializer(stddev=0.1)
        self.w_fp = self.add_weight(
            shape=(kh, kw, in_ch, self.filters),
            initializer=init,
            # clip latent weights to keep the binary proxy valid.
            constraint=WeightClip(-1.0, 1.0),
            trainable=True,
            name="w_fp"
        )

    def call(self, inputs):
        w_fp = tf.convert_to_tensor(self.w_fp)
        # stop gradients through alpha so scale estimation does not destabilize learning.
        alpha = tf.stop_gradient(tf.reduce_mean(tf.abs(w_fp), axis=(0, 1, 2), keepdims=True))
        w_bin = alpha * binary_activation(w_fp)
        x_bin = binary_activation(inputs)
        return tf.nn.conv2d(
            x_bin, w_bin,
            strides=(1, self.strides[0], self.strides[1], 1),
            padding=self.padding
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
        })
        return cfg
