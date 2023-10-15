import tensorflow as tf
from keras.models import Model
from transformer import *
from keras.layers import *

max_len = 50
d_model = 32
vocab_size = 10000


input_encoder = Input(shape=(max_len,))
input_decoder = Input(shape=(max_len,))

x = Embedding(vocab_size, d_model)(input_encoder)
x = PositionalEncoding(max_len, d_model)(x)
x = TransformerEncoder(8, d_model, 2048, 1024)(x)
x = TransformerEncoder(8, d_model, 2048, 1024)(x)

x_decoder = Embedding(10000, d_model)(input_decoder)
x_decoder = PositionalEncoding(max_len, d_model)(x_decoder)
x = TransformerDecoder(8, d_model, 2048, 1024)(x_decoder, x)
x = TransformerDecoder(8, d_model, 2048, 1024)(x_decoder, x)
x = Dense(vocab_size, activation="softmax")(x)

model = Model(inputs=[input_encoder, input_decoder], outputs=x)
model.compile("adam", "sparse_categorical_crossentropy", ["accuracy"])
print(model.summary())