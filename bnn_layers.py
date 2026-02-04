import keras
from keras import layers
import tensorflow as tf

@keras.saving.register_keras_serializable(package="BNN")
class BinaryActivation(layers.Layer):
    def call(self, inputs):
        return tf.sign(inputs)

    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable(package="BNN")
class BinaryDense(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(int(input_shape[-1]), self.units),
            initializer="random_normal",
            trainable=True,
            name="w_fp"
        )

    def call(self, inputs):
        w_bin = tf.sign(self.w)
        x_bin = tf.sign(inputs)
        return tf.matmul(x_bin, w_bin)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


@keras.saving.register_keras_serializable(package="BNN")
class BinaryConv2D(layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding="same", **kwargs):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        kh, kw = self.kernel_size
        in_ch = int(input_shape[-1])
        self.w = self.add_weight(
            shape=(kh, kw, in_ch, self.filters),
            initializer="random_normal",
            trainable=True,
            name="w_fp"
        )

    def call(self, inputs):
        w_bin = tf.sign(self.w)
        x_bin = tf.sign(inputs)
        return tf.nn.conv2d(
            x_bin, w_bin,
            strides=(1, self.strides[0], self.strides[1], 1),
            padding=self.padding.upper()  # IMPORTANT
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
