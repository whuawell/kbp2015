__author__ = 'victor'

from keras.models import Graph
from keras.layers.core import *
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import *
from keras.layers.convolutional import *
from keras.constraints import *
from keras.objectives import *
from keras.optimizers import *
from keras.regularizers import *

import os
import json
import numpy as np
from data.pretrain import Senna


def get_model(config, vocab, typechecker):
    fetch = {
        'concat': concatenated,
        'single': single,
        'single_conv': single_conv,
        'single_small': single_small,
    }[config.model]
    graph, out = fetch(vocab, config)
    graph.compile(rmsprop(lr=config.lr, clipnorm=25.), {out: typechecker.filtered_crossentropy})
    return graph

def get_rnn(config):
    return {'lstm': LSTM, 'gru': GRU, 'mut1': JZS1, 'mut2': JZS2, 'mut3': JZS3}[config.rnn]

def pretrained_word_emb(vocab, emb_dim):
    word2emb = vocab['word'].load_word2emb()
    word_emb = Embedding(len(vocab['word']), emb_dim)
    W = word_emb.get_weights()[0]
    for i, word in enumerate(word2emb.keys()):
        W[i] = word2emb[word]
    word_emb.set_weights([W])
    return word_emb

def concatenated(vocab, config):
    RNN = get_rnn(config)

    graph = Graph()
    graph.add_input(name='word_input', ndim=2, dtype='int')
    graph.add_input(name='ner_input', ndim=2, dtype='int')
    graph.add_input(name='dep_input', ndim=2, dtype='int')
    graph.add_input(name='pos_input', ndim=2, dtype='int')
    graph.add_node(pretrained_word_emb(vocab, config.emb_dim), name='word_emb', input='word_input')
    graph.add_node(Embedding(len(vocab['ner']), config.emb_dim, W_constraint=UnitNorm()), name='ner_emb', input='ner_input')
    graph.add_node(Embedding(len(vocab['dep']), config.emb_dim, W_constraint=UnitNorm()), name='dep_emb', input='dep_input')
    graph.add_node(Embedding(len(vocab['pos']), config.emb_dim, W_constraint=UnitNorm()), name='pos_emb', input='pos_input')
    graph.add_node(Dropout(config.dropout), 'drop0', inputs=['word_emb', 'ner_emb', 'pos_emb', 'dep_emb'], merge_mode='concat')

    n_in = 4 * config.emb_dim
    n_out = config.hidden[0]
    graph.add_node(
        RNN(n_in, n_out, truncate_gradient=config.truncate_gradient, return_sequences=True),
        name='RNN1', input='drop0')
    graph.add_node(Dropout(config.dropout), 'drop1', input='RNN1')
    n_in = n_out
    n_out = config.hidden[1]
    graph.add_node(
        RNN(n_in, n_out, truncate_gradient=config.truncate_gradient, return_sequences=False),
        name='RNN2', input='drop1')
    graph.add_node(Dropout(config.dropout), 'drop2', input='RNN2')
    graph.add_node(Dense(n_out, len(vocab['rel']), W_regularizer=l2(config.reg)), 'dense2', input='drop2')
    graph.add_output(name='p_relation', input='dense2')

    return graph, 'p_relation'

def single(vocab, config):
    RNN = get_rnn(config)

    graph = Graph()
    graph.add_input(name='word_input', ndim=2, dtype='int')
    graph.add_node(pretrained_word_emb(vocab, config.emb_dim), name='word_emb', input='word_input')
    graph.add_node(Dropout(config.dropout), 'drop0', input='word_emb')

    n_in = config.emb_dim
    n_out = config.hidden[0]
    graph.add_node(
        RNN(n_in, n_out, truncate_gradient=config.truncate_gradient, return_sequences=True),
        name='RNN1', input='drop0')
    graph.add_node(Dropout(config.dropout), 'drop1', input='RNN1')

    n_in = n_out
    n_out = config.hidden[1]
    graph.add_node(
        RNN(n_in, n_out, truncate_gradient=config.truncate_gradient, return_sequences=False),
        name='RNN2', input='drop1')
    graph.add_node(Dropout(config.dropout), 'drop2', input='RNN2')
    graph.add_node(Dense(n_out, len(vocab['rel'])), 'dense2', input='drop2')
    graph.add_output(name='p_relation', input='dense2')

    return graph, 'p_relation'

def single_small(vocab, config):
    RNN = get_rnn(config)

    graph = Graph()
    graph.add_input(name='word_input', ndim=2, dtype='int')
    graph.add_node(pretrained_word_emb(vocab, config.emb_dim), name='word_emb', input='word_input')
    graph.add_node(Dropout(config.dropout), 'drop0', input='word_emb')

    n_in = config.emb_dim
    n_out = config.hidden[0]
    graph.add_node(
        RNN(n_in, n_out, truncate_gradient=config.truncate_gradient, return_sequences=False),
        name='RNN1', input='drop0')
    graph.add_node(Dropout(config.dropout), 'drop1', input='RNN1')
    graph.add_node(Dense(n_out, len(vocab['rel'])), 'dense2', input='drop1')
    graph.add_output(name='p_relation', input='dense2')

    return graph, 'p_relation'

def single_conv(vocab, config):
    RNN = get_rnn(config)

    graph = Graph()
    graph.add_input(name='word_input', ndim=2, dtype='int')
    graph.add_node(pretrained_word_emb(vocab, config.emb_dim), name='word_emb', input='word_input')
    graph.add_node(Dropout(config.dropout), 'emb_out', input='word_emb')

    n_in = config.emb_dim
    n_out = config.hidden[0]
    pool = 2
    graph.add_node(Convolution1D(n_in, n_out, 3, activation=config.activation), name='conv1', input='emb_out')
    graph.add_node(Dropout(config.dropout), 'drop1', input='conv1')
    graph.add_node(MaxPooling1D(pool_length=pool), 'conv_out', input='drop1')

    n_in = n_out / pool
    n_out = config.hidden[1]
    graph.add_node(RNN(n_in, n_out), 'rnn', input='conv_out')
    graph.add_node(Dropout(config.dropout), 'rnn_out', input='rnn')

    n_in = n_out
    n_out = len(vocab['rel'])
    graph.add_node(Dense(n_in, n_out), 'dense_out', input='rnn_out')
    graph.add_output(name='p_relation', input='dense_out')

    return graph, 'p_relation'
