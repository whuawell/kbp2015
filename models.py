__author__ = 'victor'

from keras.models import Sequential
from keras.layers.core import *
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import *
from keras.constraints import *
from keras.objectives import *
from keras.optimizers import *
from keras.regularizers import *

import os
import numpy as np

RNN = GRU

def load_pretrained(word2emb, word_vocab, W):
    for i, word in enumerate(word_vocab.index2word):
        if word in word2emb:
            W[i] = word2emb[word]
    return W

def sent(vocab, word2emb, num_word, emb_dim, hidden=(300,), dropout=0.5, activation='tanh', truncate_gradient=-1):
    n_mem = hidden[0]
    net = Sequential()
    word_emb = Embedding(num_word, emb_dim)
    W = word_emb.get_weights()[0]
    W = load_pretrained(word2emb, vocab, W)
    word_emb.set_weights([W])
    word_emb.constraints = word_emb.params = word_emb.regularizers = []
    net.add(word_emb)
    n_in = emb_dim
    for n_out in hidden[:-1]:
        net.add(RNN(n_in, n_out, truncate_gradient=truncate_gradient, return_sequences=True))
        net.add(Activation(activation))
        n_in = n_out
    n_out = hidden[-1]
    net.add(RNN(n_in, n_out, truncate_gradient=truncate_gradient, return_sequences=False))

    if dropout:
        net.add(Dropout(dropout))

    return net
