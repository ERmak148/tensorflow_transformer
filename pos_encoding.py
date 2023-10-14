import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_seq_len, d_model):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = self.positional_encoding(max_seq_len, d_model)

    def positional_encoding(self, max_seq_len, d_model):
        position = tf.range(max_seq_len, dtype=tf.float32)
        div_term = tf.pow(10000.0, 2 * tf.range(d_model, dtype=tf.float32) / d_model)
        div_term = tf.where(tf.math.not_equal(div_term, 0), div_term, tf.ones_like(div_term))
        positional_encoding = position[:, tf.newaxis] / div_term[tf.newaxis, :]
        positional_encoding = tf.expand_dims(positional_encoding, 0)
        return tf.concat([tf.sin(positional_encoding), tf.cos(positional_encoding)], axis=0)

    def call(self, inputs):
        return inputs + self.positional_encoding[:, :tf.shape(inputs)[1], :]

