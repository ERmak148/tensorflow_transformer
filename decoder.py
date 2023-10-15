from keras.layers import *
from keras.models import *


class TransformerDecoder(Layer):
    def __init__(self, num_heads, key_dim, fflayer1, fflayer2, attn_axes=None, rate=0.1):
        super(TransformerDecoder, self).__init__()
        self.mha1 = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, attention_axes=attn_axes)
        self.mha2 = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, attention_axes=attn_axes)

        self.ff = Sequential([
            Dense(fflayer1, activation="relu"),
            Dense(fflayer2)
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate=rate)
        self.dropout2 = Dropout(rate=rate)
        self.dropout3 = Dropout(rate=rate)

        self.softmax = Softmax()

    def call(self, inputs, encoder, training):
        attn_out = self.mha1(inputs, inputs, inputs)
        out1 = self.dropout1(attn_out, training=training)
        out1 = self.layernorm1(out1+inputs)

        attn2_out = self.mha2(encoder, encoder, out1)
        out2 = self.dropout2(attn2_out, training=training)
        out2 = self.layernorm2(out2 + encoder)

        ff_out = self.ff(out2)
        out3 = self.dropout3(ff_out, training=training)
        out3 = self.layernorm3(out3)
        # out3 = self.softmax(out3)
        return out3