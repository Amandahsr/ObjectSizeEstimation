import tensorflow as tf


class ChannelShuffle(tf.keras.layers.Layer):
    def __init__(self, groups: int = 2):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def call(self, x):
        _, height, width, channels = x.shape
        channels_per_group = channels // self.groups
        x = tf.reshape(x, [-1, height, width, self.groups, channels_per_group])
        x = tf.transpose(x, [0, 1, 2, 4, 3])
        x = tf.reshape(x, [-1, height, width, channels])
        return x
