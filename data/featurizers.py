__author__ = 'victor'
from dataset import Example, Vocab
import numpy as np
from dependency import NoPathException


def one_hot(y, num_classes):
    Y = np.zeros((len(y), num_classes), dtype='float32')
    Y[np.arange(len(y)), y] = 1.
    return Y


class Featurizer(object):

    def __init__(self, vocab=None):
        if vocab is None:
            vocab = {
                'rel': Vocab(unk=False),
                'ner': Vocab(unk=False),
                'dep': Vocab(unk=False),
                'pos': Vocab(unk=False),
                'word': Vocab(unk='UNKNOWN'),
            }
        self.vocab = vocab

    @classmethod
    def get_token(cls, ex, index):
        if index >= ex.subject_begin and index < ex.subject_end:
            return ex.subject_ner
        if index >= ex.object_begin and index < ex.object_end:
            return ex.object_ner
        return ex.words[index]

    def featurize(self, ex, add=False):
        if not ex.dependency:
            raise NoPathException(str(ex))

        return Example(**{
            'relation': self.vocab['rel'].get(ex.relation, add=add) if ex.relation else None,
            'subject_ner': self.vocab['ner'].get(ex.subject_ner, add=add),
            'object_ner': self.vocab['ner'].get(ex.object_ner, add=add),
            'orig': self,
        })


class SinglePathFeaturizer(Featurizer):

    def featurize(self, ex, add=False):
        feat = super(SinglePathFeaturizer, self).featurize(ex, add)
        sequence = []
        for from_, to_, arc in ex.dependency:
            if arc == 'root':
                continue
            sequence += [self.get_token(ex, from_), arc]
        sequence += [self.get_token(ex, to_)]
        ex.sequence = sequence
        feat.sequence = [self.vocab['word'].get(w, add) for w in ex.sequence]
        ex.length = feat.length = len(sequence)
        return feat


class ConcatenatedFeaturizer(Featurizer):

    def featurize(self, ex, add=False):
        feat = super(ConcatenatedFeaturizer, self).featurize(ex, add)
        sequence = []
        for child, parent, arc in ex.dependency:
            if arc.endswith('_from') or arc == 'root':
                token = self.get_token(ex, child)
                ner = ex.ner[child]
                pos = ex.pos[child]
            elif arc.endswith('_to'):
                token = self.get_token(ex, parent)
                ner = ex.ner[parent]
                pos = ex.pos[parent]
            else:
                raise Exception('Unknown arc type ' + arc)
            sequence.append([token, ner, pos, arc])
        ex.sequence = sequence
        feat.sequence = []
        for i, tup in enumerate(ex.sequence):
            word, ner, pos, arc = tup
            feat.sequence.append([self.vocab['word'].get(word, add),
                                  self.vocab['ner'].get(ner, add),
                                  self.vocab['pos'].get(pos, add),
                                  self.vocab['dep'].get(arc, add)])
        ex.length = feat.length = len(sequence)
        return feat

