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

def load_pretrained(word2emb, vocab, W):
    for i, word in enumerate(vocab['word'].index2word):
        if word in word2emb:
            W[i] = word2emb[word]
    return W

def ner(vocab, word2emb, emb_dim, hidden=(300,), dropout=0.5, activation='tanh', truncate_gradient=-1, reg=1e-3):
    net = Sequential()
    net.add(Embedding(len(vocab['ner']), emb_dim))
    net.add(Flatten())
    net.add(Dense(emb_dim * 2, hidden[-1]))
    net.add(Activation(activation))
    return net, hidden[-1]

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
    sent_net, sent_nout = sent(vocab, word2emb, emb_dim, hidden, dropout, activation, truncate_gradient, reg)
    ner_net, ner_nout = ner(vocab, word2emb, emb_dim, hidden, dropout, activation, truncate_gradient, reg)

    net = Sequential()
    net.add(Merge([sent_net, ner_net], mode='concat'))
    return net, ner_nout + sent_nout

def sent_parse(vocab, word2emb, emb_dim, hidden=(300,), dropout=0.5, activation='tanh', truncate_gradient=-1, reg=1e-3):
    word_emb = Embedding(len(vocab['word']), emb_dim)
    W = word_emb.get_weights()[0]
    W = load_pretrained(word2emb, vocab, W)
    word_emb.set_weights([W])
    word_emb.constraints = word_emb.params = word_emb.regularizers = []
    word_net = Sequential()
    word_net.add(word_emb)
    dep_emb = Embedding(len(vocab['dep']), emb_dim, W_constraint=unitnorm)
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
    sent_net, sent_nout = sent(vocab, word2emb, emb_dim, hidden, dropout, activation, truncate_gradient, reg)
    ner_net, ner_nout = ner(vocab, word2emb, emb_dim, hidden, dropout, activation, truncate_gradient, reg)
    parse_net, parse_nout = parse(vocab, word2emb, emb_dim, hidden, dropout, activation, truncate_gradient, reg)

    net = Sequential()
    net.add(Merge([sent_net, parse_net, ner_net], mode='concat'))
    return net, ner_nout + sent_nout + parse_nout

