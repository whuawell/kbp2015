__author__ = 'victor'
import csv
import os
import numpy as np

class TypeCheckAdaptor(object):

    def __init__(self, fname, vocab):
        from theano import shared
        self.vocab = vocab
        self.valid_types = self.load_valid_types(fname, vocab)
        self.valid = shared(self.valid_types)

    def get_valid(self, ner1, ner2):
        return self.valid[ner1, ner2]

    def get_valid_cpu(self, ner1, ner2):
        valid = self.valid_types[ner1, ner2]
        return valid

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

