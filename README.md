# tensorflow transformer
I am a fan of OpenAI technologies, it was from their GPT that i learned about the transformer. Well, i decided to try to create it on tensorflow. Here s what happened

# What do you need to work correctly?
The structure of the network should be something like this:
Embedding\n
PositionalEncoding\n
TransformerEncoder\n
TransformerDecoder (2 inputs: input for decoder, encoder output. input for decoder on the example of GPT should be generated text)\n
Flatten (optional)\n
Dense (softmax for classification, mse for regression. or linear)

# What if you dont want to submit data to PosEncoding and Embedding?
You can set the attn_axes value in the encoder and decoder. For example, 1d data ([1, 2, 3]) is attn_axes=(1,)/(0,)
