__author__ = 'victor'

from keras.models import Graph
from keras.layers.core import *
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import *
from keras.constraints import *
from keras.objectives import *
from keras.optimizers import *
from keras.regularizers import *

import os
import json
import numpy as np
from data.pretrain import Senna

RNN = LSTM



def get_model(config, vocab, typechecker):
    if config.model == 'concat':
        graph, out = concatenated(vocab, config)
    elif config.model == 'seq':
        graph, out = sequence(vocab, config)
    else:
        raise Exception("unknown model type %s" % config.model)
    graph.compile('rmsprop', {out: typechecker.filtered_crossentropy})
    return graph

def pretrained_word_emb(vocab, emb_dim):
    word2emb = vocab['word'].load_word2emb()
    word_emb = Embedding(len(vocab['word']), emb_dim)
    W = word_emb.get_weights()[0]
    for i, word in enumerate(word2emb.keys()):
        W[i] = word2emb[word]
    word_emb.set_weights([W])
    return word_emb

def concatenated(vocab, config):
    graph = Graph()
    graph.add_input(name='word_input', ndim=2, dtype='int')
    graph.add_input(name='ner_input', ndim=2, dtype='int')
    graph.add_input(name='dep_input', ndim=2, dtype='int')
    graph.add_input(name='pos_input', ndim=2, dtype='int')
    graph.add_node(pretrained_word_emb(vocab, config.emb_dim), name='word_emb', input='word_input')
    graph.add_node(Embedding(len(vocab['ner']), config.emb_dim), name='ner_emb', input='ner_input')
    graph.add_node(Embedding(len(vocab['dep']), config.emb_dim), name='dep_emb', input='dep_input')
    graph.add_node(Embedding(len(vocab['pos']), config.emb_dim), name='pos_emb', input='pos_input')

    n_in = 4 * config.emb_dim
    n_out = config.hidden[0]
    graph.add_node(
        RNN(n_in, n_out, truncate_gradient=config.truncate_gradient, return_sequences=True),
        name='RNN1', inputs=['word_emb', 'ner_emb', 'pos_emb', 'dep_emb'], merge_mode='concat')
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

def sequence(vocab, config):
    graph = Graph()
    graph.add_input(name='word_input', ndim=2, dtype='int')
    graph.add_node(pretrained_word_emb(vocab, config.emb_dim), name='word_emb', input='word_input')

    n_in = config.emb_dim
    n_out = config.hidden[0]
    graph.add_node(
        RNN(n_in, n_out, truncate_gradient=config.truncate_gradient, return_sequences=True),
        name='RNN1', input='word_emb')
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
