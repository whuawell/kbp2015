__author__ = 'victor'
from collections import Counter
import cPickle as pkl
import numpy as np
import sys
from pprint import pformat
import os
from dependency import NoPathException
from utils import np_softmax


class Example(dict):

    def __init__(self, *args, **kwargs):
        super(Example, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Vocab(object):
    def __init__(self, unk=None):
        self.word2index = {}
        self.index2word = []
        self.counts = Counter()
        self.unk = unk

        if unk:
            self.add(unk)

    def clear_counts(self):
        self.counts = Counter()

    def __repr__(self):
        return str(self.word2index)

    def __len__(self):
        return len(self.index2word)

    def __getitem__(self, word):
        if self.unk:
            return self.word2index.get(word, self.word2index[self.unk])
        else:
            return self.word2index[word]

    def __contains__(self, word):
        return word in self.word2index

    def get(self, word, add=False):
        return self.add(word) if add else self[word]

    def add(self, word, count=1):
        if word not in self.word2index:
            self.word2index[word] = len(self)
            self.index2word.append(word)
        self.counts[word] += count
        return self.word2index[word]

    def sent2index(self, sent, add=False):
        if add:
            return [self.add(w) for w in sent]
        else:
            return [self[w] for w in sent]

    def index2sent(self, indices):
        return [self.index2word[i] for i in indices]

    def prune_rares(self, cutoff=2):
        v = Vocab(unk=self.unk)
        for w in self.index2word:
            if self.counts[w] > cutoff or w == self.unk:
                v.add(w, count=self.counts[w])
        return v


class Split(object):

    ignore_relations = {'org:website', 'org:date_founded'}

    def __init__(self, generator, featurizer, add=False, handle_no_path='short'):
        self.featurizer = featurizer
        self.examples = []
        self.num_examples = 0

        for ex in generator:
            def raise_error(ex):
                print >> sys.stderr, 'Warning: Could not find path between entities in parse'
                print >> sys.stderr, ' '.join(ex.words)
                print >> sys.stderr, "subject %s (%s, %s) object %s (%s, %s)" % (
                    ex.subject, ex.subject_begin, ex.subject_end,
                    ex.object, ex.object_begin, ex.object_end)
                print >> sys.stderr, 'dependency parse'
                print >> sys.stderr, ex.dependency
                print >> sys.stderr, "\n"
                raise

            if ex.relation in self.ignore_relations:
                continue
            try:
                feat = self.featurizer.featurize(ex, add=add)
            except NoPathException as e:
                if handle_no_path == 'ignore':
                    continue
                elif handle_no_path == 'short':
                    print >> sys.stderr, 'Warning: Could not find path between entities in parse'
                    print >> sys.stderr, ' '.join(ex.words)
                    print >> sys.stderr, "subject %s (%s, %s) object %s (%s, %s)" % (
                        ex.subject, ex.subject_begin, ex.subject_end,
                        ex.object, ex.object_begin, ex.object_end)
                    print >> sys.stderr, "\n"
                    continue
                else:
                    raise_error(ex)
            except:
                print >> sys.stderr, "Error: Could not featurize example"
                raise_error(ex)

            feat.id = self.num_examples
            feat.orig = ex
            self.num_examples += 1
            self.examples += [feat]

    def __len__(self):
        return self.num_examples

    def remove_invalid_examples(self, typechecker):
        invalids = [e for e in self.examples if not typechecker.get_valid(e.subject_ner, e.object_ner)]
        self.examples = [e for e in self.examples if typechecker.get_valid(e.subject_ner, e.object_ner)]
        return invalids

    def get_length_map(self, examples):
        length_map = {}
        for ex in examples:
            if ex.length not in length_map:
                length_map[ex.length] = []
            length_map[ex.length] += [ex]
        return length_map

    def get_label_map(self, examples):
        label_map = {}
        for ex in examples:
            if ex.relation not in label_map:
                label_map[ex.relation] = []
            label_map[ex.relation] += [ex]
        return label_map

    def batches(self, batch_size=128):
        length_map = self.get_length_map(self.examples)
        lengths = length_map.keys()
        np.random.shuffle(lengths)
        for length in lengths:
            examples = length_map[length]
            ids = np.array([e.id for e in examples])
            for i in xrange(0, len(examples), batch_size):
                end = min(i+batch_size, len(examples))
                Xbatch, Ybatch, types = self.featurizer.to_matrix(examples[i:end])
                if not len(types):
                    continue
                yield ids[i:end], Xbatch, Ybatch, types


class Dataset(object):

    def __init__(self, train, dev, featurizer):
        self.train, self.dev, self.featurizer = train, dev, featurizer

    @classmethod
    def build(cls, train_generator, dev_generator, featurizer):
        train = Split(train_generator, featurizer, add=True)
        dev = Split(dev_generator, featurizer, add=False)
        return Dataset(train, dev, featurizer)

    def save(self, to_dir):
        if not os.path.isdir(to_dir):
            os.makedirs(to_dir)
        with open(os.path.join(to_dir, 'data.pkl'), 'wb') as f:
            pkl.dump(self, f, protocol=pkl.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, from_dir):
        with open(os.path.join(from_dir, 'data.pkl')) as f:
            return pkl.load(f)


if __name__ == '__main__':
    from adaptors import *
    from featurizers import *
    from pretrain import Senna

    train_generator = SupervisedDataAdaptor().to_examples('raw/supervision.csv')
    dev_generator = KBPEvaluationDataAdaptor().to_examples('raw/evaluation.tsv')
    featurizer = ConcatenatedFeaturizer(word=Senna())

    save = 'saves/supervision_evaluation'
    if not os.path.isdir(save):
        dataset = Dataset.build(train_generator, dev_generator, featurizer)
        dataset.save(save)
    else:
        dataset = Dataset.load(save)
