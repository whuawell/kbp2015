__author__ = 'victor'
from dataset import Example, Vocab
import numpy as np
from dependency import NoPathException, DependencyParse


class Featurizer(object):

    def __init__(self, **vocab_kwargs):
        self.vocab = {
            'rel': vocab_kwargs.get('rel', Vocab()),
            'ner': vocab_kwargs.get('ner', Vocab(unk='O')),
            'dep': vocab_kwargs.get('dep', Vocab()),
            'pos': vocab_kwargs.get('pos', Vocab(unk='.')),
            'word': vocab_kwargs.get('word', Vocab(unk='UNKNOWN')),
        }

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
            'dependency': DependencyParse(ex.dependency, enhanced=True).get_path_from_parse(
                ex.subject_begin, ex.subject_end, ex.object_begin, ex.object_end),
            'orig': self,
        })

    def one_hot(self, y):
        Y = np.zeros((y.size, len(self.vocab['rel'])))
        Y[np.arange(len(y)), y] = 1
        return Y.astype('float32')


class SinglePathFeaturizer(Featurizer):

    def featurize(self, ex, add=False):
        feat = super(SinglePathFeaturizer, self).featurize(ex, add)
        sequence = []
        for from_, to_, arc in feat.dependency:
            if arc == 'root':
                continue
            sequence += [self.get_token(ex, from_), arc]
        sequence += [self.get_token(ex, to_)]
        ex.sequence = sequence
        feat.sequence = [self.vocab['word'].get(w, add) for w in ex.sequence]
        ex.length = feat.length = len(sequence)
        return feat

    def to_matrix(self, examples):
        X = {'word_input': []}
        Y = {'p_relation': []}
        types = []
        for ex in examples:
            X['word_input'].append(ex.sequence)
            Y['p_relation'].append(ex.relation)
            types.append([ex.subject_ner, ex.object_ner])
        X['word_input'] = np.array(X['word_input'])
        Y['p_relation'] = self.one_hot(np.array(Y['p_relation']))
        # double check lengths
        for k, v in X.items():
            assert len(v) == len(Y['p_relation'])
        return X, Y, np.array(types)


class ConcatenatedFeaturizer(Featurizer):

    def featurize(self, ex, add=False):
        feat = super(ConcatenatedFeaturizer, self).featurize(ex, add)
        sequence = []
        for child, parent, arc in feat.dependency:
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
            child = ex.words[child]
            parent = ex.words[parent] if parent else None
            sequence.append([token, ner, pos, arc])
        ex.sequence = sequence
        feat.words, feat.ner, feat.pos, feat.arc = [], [], [], []
        for i, tup in enumerate(ex.sequence):
            word, ner, pos, arc = tup
            if not add and arc not in self.vocab['dep']:
                arc = 'dep_from' if arc.endswith('_from') else 'dep_to'
            feat.words.append(self.vocab['word'].get(word, add))
            feat.ner.append(self.vocab['ner'].get(ner, add))
            feat.pos.append(self.vocab['pos'].get(pos, add))
            feat.arc.append(self.vocab['dep'].get(arc, add))
        ex.length = feat.length = len(sequence)
        return feat

    def to_matrix(self, examples):
        X = {k: [] for k in ['word_input', 'ner_input', 'pos_input', 'dep_input']}
        Y = {'p_relation': []}
        types = []
        for ex in examples:
            X['word_input'].append(ex.words)
            X['ner_input'].append(ex.ner)
            X['pos_input'].append(ex.pos)
            X['dep_input'].append(ex.arc)
            Y['p_relation'].append(ex.relation)
            types.append([ex.subject_ner, ex.object_ner])
        Y['p_relation'] = self.one_hot(np.array(Y['p_relation']))
        # double check lengths
        for k, v in X.items():
            assert len(v) == len(Y['p_relation'])
        return {k: np.array(v) for k, v in X.items()}, Y, np.array(types)
