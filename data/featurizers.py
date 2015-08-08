__author__ = 'victor'
from dataset import Example, Vocab
import numpy as np
from copy import deepcopy
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

    def corrupt(self, feat, add=False):
        raise NotImplementedError()


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

    def corrupt(self, feat, add=False):
        corrupted = Example(**deepcopy(feat.__dict__))
        corrupted.corrupt = True
        # randomly drop a node
        drop = np.random.randint(0, len(corrupted.sequence))
        sequence = corrupted.sequence[:drop]
        if drop < len(corrupted.sequence) - 1:
            sequence += corrupted.sequence[drop+1:]
        corrupted.sequence = sequence
        corrupted.relation = self.vocab['rel'].add('no_relation')
        corrupted.length = len(corrupted.sequence)
        return corrupted if corrupted.length else None

    def decode_sequence(self, ex):
        sequence = [self.vocab['word'].index2word[w] for w in ex.sequence]
        return sequence

    def to_matrix(self, examples):
        X, Y, types = [], [], []
        for ex in examples:
            X.append(ex.sequence)
            Y.append(ex.relation)
            types.append([ex.subject_ner, ex.object_ner])
        X = np.array(X)
        Y = [None] * len(Y) if None in Y else self.one_hot(np.array(Y))
        # double check lengths
        assert len(X) == len(Y)
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

    def __init__(self, scope=-1, **vocab_kwargs):
        super(SentenceFeaturizer, self).__init__(**vocab_kwargs)
        self.scope = scope

    def featurize(self, ex, add=False):
        isbetween = lambda x, start, end: x >= start and x < end
        if isbetween(ex.subject_begin, ex.object_begin, ex.object_end) or isbetween(ex.object_begin, ex.subject_begin, ex.subject_end):
            raise NoPathException(str(ex))

        first = 'subject' if ex.subject_begin < ex.object_begin else 'object'
        second = 'object' if ex.subject_begin < ex.object_begin else 'subject'
        chunk0 = ex.words[:ex[first + '_begin']] 
        chunk1 = chunk0 + [ex[first + '_ner']] 
        chunk2 = chunk1 + ex.words[ex[first + '_end']:ex[second + '_begin']]
        sequence = chunk2 + [ex[second + '_ner']] + ex.words[ex[second + '_end']:]
        first_pos = len(chunk0)
        second_pos = len(chunk2)

        if self.scope > 0:
            start = max(0, first_pos - self.scope)
            end = min(len(sequence), second_pos + self.scope + 1)
            sequence = sequence[start:end]

        feat = Example(**{
            'relation': self.vocab['rel'].get(ex.relation, add=add) if ex.relation else None,
            'subject_ner': self.vocab['ner'].get(ex.subject_ner, add=add),
            'object_ner': self.vocab['ner'].get(ex.object_ner, add=add),
            'orig': ex,
            'sequence': [self.vocab['word'].get(w, add=add) for w in sequence],
            'subject_pos': first_pos if first == 'subject' else second_pos,
            'object_pos': first_pos if first == 'object' else second_pos,
        })
        ex.length = feat.length = len(feat.sequence)

        return feat


class SinglePathSentenceFeaturizer(SentenceFeaturizer):

    def decode_sequence(self, ex):
        sequence = [self.vocab['word'].index2word[w] for w in ex.sequence]
        return sequence

    def to_matrix(self, examples):
        X, Y = [], []
        types = []
        for ex in examples:
            X.append(ex.sequence)
            Y.append(ex.relation)
            types.append([ex.subject_ner, ex.object_ner])
        X = np.array(X)
        Y = [None] * len(Y) if None in Y else self.one_hot(np.array(Y))
        # double check lengths
        assert len(X) == len(Y)
        return X, Y, np.array(types)
