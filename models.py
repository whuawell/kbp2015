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

def load_pretrained(word2emb, word_vocab, W):
    for i, word in enumerate(word_vocab.index2word):
        if word in word2emb:
            W[i] = word2emb[word]
    return W

def sent(vocab, word2emb, num_word, emb_dim, hidden=(300,), dropout=0.5, activation='tanh', truncate_gradient=-1):
    n_mem = hidden[0]
    hidden = hidden[1:]
    net = Sequential()
    word_emb = Embedding(num_word, emb_dim, W_constraint=unitnorm)
    W = word_emb.get_weights()[0]
    W = load_pretrained(word2emb, vocab, W)
    word_emb.set_weights([W])
    word_emb.constraints = word_emb.params = word_emb.regularizers = []
    net.add(word_emb)
    net.add(LSTM(emb_dim, n_mem, truncate_gradient=truncate_gradient))

    n_in = n_mem
    for n_out in hidden:
        net.add(Dense(n_in, n_out))
        net.add(Activation(activation))

    if dropout:
        net.add(Dropout(dropout))

    return net
