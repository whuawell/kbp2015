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
        raise NotImplementedError()

    def one_hot(self, y):
        Y = np.zeros((y.size, len(self.vocab['rel'])))
        Y[np.arange(len(y)), y] = 1
        return Y.astype('float32')


class DependencyFeaturizer(Featurizer):

    def featurize(self, ex, add=False):
        if not ex.dependency: # no dependency parse
            raise NoPathException(str(ex))

        feat = Example(**{
            'relation': self.vocab['rel'].get(ex.relation, add=add) if ex.relation else None,
            'subject_ner': self.vocab['ner'].get(ex.subject_ner, add=add),
            'object_ner': self.vocab['ner'].get(ex.object_ner, add=add),
            'dependency': DependencyParse(ex.dependency, enhanced=True).get_path_from_parse(
                ex.subject_begin, ex.subject_end, ex.object_begin, ex.object_end),
            'orig': ex,
        })

        if not feat.dependency: # no shortest path between entities
            raise NoPathException(str(ex))
        return feat


class SinglePathDependencyFeaturizer(DependencyFeaturizer):

    def featurize(self, ex, add=False):
        feat = super(SinglePathDependencyFeaturizer, self).featurize(ex, add)
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

    def decode_sequence(self, ex):
        sequence = [self.vocab['word'].index2word[w] for w in ex.sequence]
        return sequence

    def to_matrix(self, examples):
        X = {'word_input': []}
        Y = {'p_relation': []}
        types = []
        for ex in examples:
            X['word_input'].append(ex.sequence)
            Y['p_relation'].append(ex.relation)
            types.append([ex.subject_ner, ex.object_ner])
        X['word_input'] = np.array(X['word_input'])
        Y['p_relation'] = [None] * len(Y['p_relation']) if None in Y['p_relation'] else self.one_hot(np.array(Y['p_relation']))
        # double check lengths
        for k, v in X.items():
            assert len(v) == len(Y['p_relation'])
        return X, Y, np.array(types)


class ConcatenatedDependencyFeaturizer(DependencyFeaturizer):

    def featurize(self, ex, add=False):
        feat = super(ConcatenatedDependencyFeaturizer, self).featurize(ex, add)
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

    def decode_sequence(self, ex):
        sequence = []
        for word, ner, pos, arc in zip(ex.words, ex.ner, ex.pos, ex.arc):
            sequence.append([
                self.vocab['word'].index2word[word],
                self.vocab['ner'].index2word[ner],
                self.vocab['pos'].index2word[pos],
                self.vocab['dep'].index2word[arc],
            ])
        return sequence

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
        Y['p_relation'] = [None] * len(Y['p_relation']) if None in Y['p_relation'] else self.one_hot(np.array(Y['p_relation']))
        # double check lengths
        for k, v in X.items():
            assert len(v) == len(Y['p_relation'])
        return {k: np.array(v) for k, v in X.items()}, Y, np.array(types)


class SentenceFeaturizer(Featurizer):

    def featurize(self, ex, add=False):
        isbetween = lambda x, start, end: x >= start and x < end
        if isbetween(ex.subject_begin, ex.object_begin, ex.object_end) or isbetween(ex.object_begin, ex.subject_begin, ex.subject_end):
            raise NoPathException(str(ex))

        feat = Example(**{
            'relation': self.vocab['rel'].get(ex.relation, add=add) if ex.relation else None,
            'subject_ner': self.vocab['ner'].get(ex.subject_ner, add=add),
            'object_ner': self.vocab['ner'].get(ex.object_ner, add=add),
            'orig': ex,
        })

        return feat


class SinglePathSentenceFeaturizer(SentenceFeaturizer):

    def featurize(self, ex, add=False):
        feat = super(SinglePathSentenceFeaturizer, self).featurize(ex, add)
        feat['sequence'] = [self.vocab['word'].get(w, add=add) for w in ex.words]
        ex.length = feat.length = len(feat.sequence)
        return feat

    def decode_sequence(self, ex):
        sequence = [self.vocab['word'].index2word[w] for w in ex.sequence]
        return sequence

    def to_matrix(self, examples):
        X = {'word_input': []}
        Y = {'p_relation': []}
        types = []
        for ex in examples:
            X['word_input'].append(ex.sequence)
            Y['p_relation'].append(ex.relation)
            types.append([ex.subject_ner, ex.object_ner])
        X['word_input'] = np.array(X['word_input'])
        Y['p_relation'] = [None] * len(Y['p_relation']) if None in Y['p_relation'] else self.one_hot(np.array(Y['p_relation']))
        # double check lengths
        for k, v in X.items():
            assert len(v) == len(Y['p_relation'])
        return X, Y, np.array(types)
