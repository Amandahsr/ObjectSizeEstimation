import tensorflow as tf
from keras.models import Sequential
from keras.layers import (
    Dense,
    Conv2D,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Concatenate,
    Multiply,
)


class CBAM(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.channel_activation_function = "relu"
        self.spatial_activation_function = "sigmoid"

    def build(self, input_shape):
        channels = input_shape[-1]

        # Channel Attention Block
        self.shared_mlp = Sequential(
            [
                Dense(
                    channels // self.reduction_ratio,
                    activation=self.channel_activation_function,
                ),
                Dense(channels),
            ]
        )
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=True)
        self.global_max_pool = GlobalMaxPooling2D(keepdims=True)
        self.sigmoid = tf.keras.activations.sigmoid

        # Spatial Attention Block
        self.conv = Conv2D(
            1,
            kernel_size=self.kernel_size,
            padding="same",
            activation=self.spatial_activation_function,
        )

    def call(self, x):
        # Channel Attention
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)
        channel_attention = self.sigmoid(avg_out + max_out)
        x = Multiply()([x, channel_attention])

        # Spatial Attention
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        spatial_attention = self.conv(concat)
        x = Multiply()([x, spatial_attention])

        return x
