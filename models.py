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

RNN = LSTM

def load_pretrained(word2emb, vocab, W):
    for i, word in enumerate(vocab['word'].index2word):
        if word in word2emb:
            W[i] = word2emb[word]
    return W

def ner(vocab, word2emb, emb_dim, hidden=(300,), dropout=0.5, activation='tanh', truncate_gradient=-1, reg=1e-3):
    net = Sequential()
    edge_emb = Embedding(len(vocab['ner']), emb_dim, W_constraint=unitnorm)
    net.add(edge_emb)
    n_in = emb_dim
    for n_out in hidden[:-1]:
        net.add(RNN(n_in, n_out, truncate_gradient=truncate_gradient, return_sequences=True))
        if dropout:
            net.add(Dropout(dropout))
        net.add(Activation(activation))
        n_in = n_out
    n_out = hidden[-1]
    net.add(RNN(n_in, n_out, truncate_gradient=truncate_gradient, return_sequences=False))

    return net, n_out

def parse(vocab, word2emb, emb_dim, hidden=(300,), dropout=0.5, activation='tanh', truncate_gradient=-1, reg=1e-3):
    net = Sequential()
    edge_emb = Embedding(len(vocab['dep']), emb_dim, W_constraint=unitnorm)
    net.add(edge_emb)
    n_in = emb_dim
    for n_out in hidden[:-1]:
        net.add(RNN(n_in, n_out, truncate_gradient=truncate_gradient, return_sequences=True))
        if dropout:
            net.add(Dropout(dropout))
        net.add(Activation(activation))
        n_in = n_out
    n_out = hidden[-1]
    net.add(RNN(n_in, n_out, truncate_gradient=truncate_gradient, return_sequences=False))

    return net, n_out

def sent(vocab, word2emb, emb_dim, hidden=(300,), dropout=0.5, activation='tanh', truncate_gradient=-1, reg=1e-3):
    net = Sequential()
    word_emb = Embedding(len(vocab['word']), emb_dim)
    W = word_emb.get_weights()[0]
    W = load_pretrained(word2emb, vocab, W)
    word_emb.set_weights([W])
    word_emb.constraints = word_emb.params = word_emb.regularizers = []
    net.add(word_emb)
    n_in = emb_dim
    for n_out in hidden[:-1]:
        net.add(RNN(n_in, n_out, truncate_gradient=truncate_gradient, return_sequences=True))
        if dropout:
            net.add(Dropout(dropout))
        net.add(Activation(activation))
        n_in = n_out
    n_out = hidden[-1]
    net.add(RNN(n_in, n_out, truncate_gradient=truncate_gradient, return_sequences=False))

    return net, n_out

def sent_ner(vocab, word2emb, emb_dim, hidden=(300,), dropout=0.5, activation='tanh', truncate_gradient=-1, reg=1e-3):
    word_emb = Embedding(len(vocab['word']), emb_dim)
    W = word_emb.get_weights()[0]
    W = load_pretrained(word2emb, vocab, W)
    word_emb.set_weights([W])
    word_emb.constraints = word_emb.params = word_emb.regularizers = []
    word_net = Sequential()
    word_net.add(word_emb)
    ner_emb = Embedding(len(vocab['ner']), emb_dim)
    ner_net = Sequential()
    ner_net.add(ner_emb)
    net = Sequential()
    net.add(Merge([word_net, ner_net], mode='concat'))
    n_in = emb_dim * 2
    for n_out in hidden[:-1]:
        net.add(RNN(n_in, n_out, truncate_gradient=truncate_gradient, return_sequences=True))
        if dropout:
            net.add(Dropout(dropout))
        net.add(Activation(activation))
        n_in = n_out
    n_out = hidden[-1]
    net.add(RNN(n_in, n_out, truncate_gradient=truncate_gradient, return_sequences=False))

    return net, n_out

def sent_parse(vocab, word2emb, emb_dim, hidden=(300,), dropout=0.5, activation='tanh', truncate_gradient=-1, reg=1e-3):
    word_emb = Embedding(len(vocab['word']), emb_dim)
    W = word_emb.get_weights()[0]
    W = load_pretrained(word2emb, vocab, W)
    word_emb.set_weights([W])
    word_emb.constraints = word_emb.params = word_emb.regularizers = []
    word_net = Sequential()
    word_net.add(word_emb)
    dep_emb = Embedding(len(vocab['dep']), emb_dim)
    dep_net = Sequential()
    dep_net.add(dep_emb)
    net = Sequential()
    net.add(Merge([word_net, dep_net], mode='concat'))
    n_in = emb_dim * 2
    for n_out in hidden[:-1]:
        net.add(RNN(n_in, n_out, truncate_gradient=truncate_gradient, return_sequences=True))
        if dropout:
            net.add(Dropout(dropout))
        net.add(Activation(activation))
        n_in = n_out
    n_out = hidden[-1]
    net.add(RNN(n_in, n_out, truncate_gradient=truncate_gradient, return_sequences=False))

    return net, n_out

def sent_parse_ner(vocab, word2emb, emb_dim, hidden=(300,), dropout=0.5, activation='tanh', truncate_gradient=-1, reg=1e-3):
    word_emb = Embedding(len(vocab['word']), emb_dim)
    W = word_emb.get_weights()[0]
    W = load_pretrained(word2emb, vocab, W)
    word_emb.set_weights([W])
    word_emb.constraints = word_emb.params = word_emb.regularizers = []
    word_net = Sequential()
    word_net.add(word_emb)
    dep_emb = Embedding(len(vocab['dep']), emb_dim)
    dep_net = Sequential()
    dep_net.add(dep_emb)
    ner_emb = Embedding(len(vocab['ner']), emb_dim)
    ner_net = Sequential()
    ner_net.add(ner_emb)
    net = Sequential()
    net.add(Merge([word_net, dep_net, ner_net], mode='concat'))
    if dropout:
        net.add(Dropout(dropout))
    n_in = emb_dim * 3
    for n_out in hidden[:-1]:
        net.add(RNN(n_in, n_out, truncate_gradient=truncate_gradient, return_sequences=True))
        if dropout:
            net.add(Dropout(dropout))
        net.add(Activation(activation))
        n_in = n_out
    n_out = hidden[-1]
    net.add(RNN(n_in, n_out, truncate_gradient=truncate_gradient, return_sequences=False))

    return net, n_out
