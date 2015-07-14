__author__ = 'victor'
import csv
import os
import numpy as np
from theano import tensor as T, shared
from keras.objectives import categorical_crossentropy

class TypeCheckAdaptor(object):

    def __init__(self, fname, vocab):
        from theano import shared
        self.vocab = vocab
        self.valid_types = self.load_valid_types(fname, vocab)
        self.valid = shared(self.valid_types)
        # self.class_weights = np.array([vocab['rel'].counts[r] for r in vocab['rel'].index2word], dtype='float32')
        # self.class_weights = self.class_weights.max() / self.class_weights
        # self.class_weights[vocab['rel']['no_relation']] *= 9.
        # self.class_weights = np.ones_like(self.class_weights)
        # self.class_weights = shared(self.class_weights)

    def get_valid(self, ner1, ner2):
        return self.valid[ner1, ner2]

    def get_valid_cpu(self, ner1, ner2):
        valid = self.valid_types[ner1, ner2]
        return valid

    def filtered_crossentropy(self, targ, pred):
        ner1 = T.cast(targ[:, -2], 'int32')
        ner2 = T.cast(targ[:, -1], 'int32')
        valid = self.get_valid(ner1, ner2)
        y_pred = T.nnet.softmax(valid * pred)
        y_targ = targ[:, :-2]

        y_pred = T.clip(y_pred, 1e-7, 1.0 - 1e-7)
        y_pred /= y_pred.sum(axis=-1, keepdims=True)
        log_prob = y_targ * T.log(y_pred)
        # log_prob *= self.class_weights[np.newaxis, :]
        return -T.sum(log_prob, axis=y_targ.ndim-1)

    def load_valid_types(self, fname, vocab):
        valid_types = np.zeros((len(vocab['ner']), len(vocab['ner']), len(vocab['rel'])), dtype='float32')
        with open(fname) as f:
            reader = csv.reader(f)
            for row in reader:
                relation, subject_ner, object_ner = [e.strip() for e in row]
                if relation not in vocab['rel'] or subject_ner not in vocab['ner'] or object_ner not in vocab['ner']:
                    continue
                valid_types[vocab['ner'][subject_ner], vocab['ner'][object_ner], vocab['rel'][relation]] = 1
        # any ner type can express no_relation
        for ner1 in xrange(len(vocab['ner'])):
            for ner2 in xrange(len(vocab['ner'])):
                valid_types[ner1, ner2, vocab['rel']['no_relation']] = 1
        # allow misc types
        valid_types[vocab['ner']['MISC'], :, :] = 1
        valid_types[:, vocab['ner']['MISC'], :] = 1
        return valid_types
