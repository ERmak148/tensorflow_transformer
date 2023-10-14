from keras.layers import *
from keras.models import *


class TransformerEncoder(Layer):
    def __init__(self, num_heads, key_dim, fflayer1, fflayer2, attn_axes, rate=0.1):
        super(TransformerEncoder, self).__init__()
        # Encoder
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, attention_axes=attn_axes)
        self.ff = Sequential([
            Dense(fflayer1, activation="relu"),
            Dense(fflayer2)
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout = Dropout(rate=rate)

    def call(self, inputs, training):
        attn_out = self.mha(inputs, inputs, inputs)
        attn_out = self.layernorm1(attn_out + inputs)

        out = self.ff(attn_out)
        out = self.dropout(out, training=training)
        out = self.layernorm2(out)
        return out