#!/usr/bin/env python
"""dataset.py
Usage: dataset.py [--context=<SIZE>] [--pretrained=<NAME>] [--unk=<UNK>] [--overwrite]

Options:
    --context=<SIZE>        -1 means entire sentence, n means n words around the entities    [default: -1]
    --pretrained=<NAME>     glove or senna      [default: glove]
    --unk=<UNK>             the token used for unknown, for senna, set this to 'UNKNOWN'    [default: ]
"""
import numpy as np
import cPickle as pkl
import os
import json
from text.dataset import *
from text.vocab import Vocab
from pretrained.load import load_pretrained

mydir = os.path.dirname(__file__)

def one_hot(y, num_classes):
    Y = np.zeros((len(y), num_classes), dtype='float32')
    Y[np.arange(len(y)), y] = 1.
    return Y

def get_first_n_words(sent, n):
    if n == 0:
        return []
    words = sent.split()
    end = min(len(words), n)
    return words[:n]

def get_last_n_words(sent, n):
    if n == 0:
        return []
    words = sent.split()
    start = min(n, len(sent))
    return words[-start:]

class ExampleAdaptor(object):

    def __init__(self, word_vocab, rel_vocab):
        self.word_vocab, self.rel_vocab = word_vocab, rel_vocab

    def convert(self, example, context=0, unk='UNKNOWN'):
        sent = example.sentence.lower()
        if context == -1:
            words = sent.split()
        else:
            start = min(int(example.entityCharOffsetEnd), int(example.slotValueCharOffsetEnd))
            end = max(int(example.entityCharOffsetBegin), int(example.slotValueCharOffsetBegin))
            words = get_last_n_words(sent[:start], context) + sent[start:end].split() + get_first_n_words(sent[end:], context)
        if unk:
            get_idx = lambda w: self.word_vocab.word2index.get(w, self.word_vocab[unk])
        else:
            get_idx = lambda w: self.word_vocab[w]
        nums = [get_idx(w) for w in words]
        rel = self.rel_vocab.add(example.relation)
        return Example(sentence=nums, relation=rel)


class SplitAdaptor(object):

    def __init__(self, example_adaptor):
        self.example_adaptor = example_adaptor

    def convert(self, split, context=0, unk='UNKNOWN'):
        my_split = Split()
        lengths = {}
        idx = 0
        for ex in split.examples:
            ex = self.example_adaptor.convert(ex, context=context, unk=unk)
            l = len(ex.sentence)
            if l == 0:
                continue
            my_split.add(ex)
            if l not in lengths:
                lengths[l] = []
            lengths[l].append(idx)
            idx += 1
        my_split.lengths = lengths
        return my_split


class AnnotatedData(object):

    def __init__(self, splits, word_vocab, rel_vocab, word2emb):
        self.word_vocab, self.rel_vocab, self.word2emb = word_vocab, rel_vocab, word2emb
        self.splits = splits

    @classmethod
    def build(cls, pretrained='senna', context=0, unk='UNKNOWN'):
        word_vocab, word2emb = load_pretrained(pretrained)
        rel_vocab = Vocab(unk=False)
        raw = Dataset.load(os.path.join(mydir, 'annotated'))
        example_adaptor = ExampleAdaptor(word_vocab, rel_vocab)
        split_adaptor = SplitAdaptor(example_adaptor)
        splits = {name:split_adaptor.convert(split, context=context, unk=unk) for name, split in raw.splits.items()}
        return AnnotatedData(splits, word_vocab, rel_vocab, word2emb)

    def save(self, to_dir):
        if not os.path.isdir(to_dir):
            os.makedirs(to_dir)
        with open(os.path.join(to_dir, 'config.json'), 'wb') as f:
            json.dump({'splits': self.splits.keys()}, f)
        with open(os.path.join(to_dir, 'vocabs.pkl'), 'wb') as f:
            pkl.dump({'word': self.word_vocab, 'rel': self.rel_vocab}, f, protocol=pkl.HIGHEST_PROTOCOL)
        with open(os.path.join(to_dir, 'word2emb.pkl'), 'wb') as f:
            pkl.dump(self.word2emb, f, protocol=pkl.HIGHEST_PROTOCOL)
        for name, split in self.splits.items():
            split.save(os.path.join(to_dir, name))

    @classmethod
    def load(cls, from_dir):
        with open(os.path.join(from_dir, 'config.json')) as f:
            config = json.load(f)
        with open(os.path.join(from_dir, 'vocabs.pkl')) as f:
            vocabs = pkl.load(f)
        splits = {name:Split.load(os.path.join(from_dir, name)) for name in config['splits']}
        with open(os.path.join(from_dir, 'word2emb.pkl')) as f:
            word2emb = pkl.load(f)
        return AnnotatedData(splits, vocabs['word'], vocabs['rel'], word2emb)

    def generate_batches(self, name, batch_size=128, label='classification', to_one_hot=True):
        assert label in ['classification', 'filter']
        split = self.splits[name]
        order = split.lengths.keys()
        random.shuffle(order)

        for l in order:
            x, y = [], []
            for idx in split.lengths[l]:
                ex = split.examples[idx]
                x.append(ex.sentence)
                y.append(ex.relation)
            X = np.array(x)
            a, b = X.shape
            Y = np.array(y)
            related = Y != self.rel_vocab['no_relation']
            if label == 'classification':
                X = X[related]
                Y = Y[related]
                if to_one_hot:
                    Y = one_hot(Y, len(self.rel_vocab))
            else:
                Y.fill(0.)
                Y[related] = 1.
                Y.reshape((-1, 1))
            if len(X):
                # this can turn out to be false if non of the examples pass the filter
                yield X, Y


if __name__ == '__main__':
    from docopt import docopt
    from pprint import pprint
    args = docopt(__doc__)
    pprint(args)
    max_printed = 20
    context = int(args['--context'])
    pretrained = args['--pretrained']
    unk = args['--unk']

    save = pretrained + str(context)
    if os.path.isdir(save) and not args['--overwrite']:
        d = AnnotatedData.load(save)
    else:
        d = AnnotatedData.build(context=context, unk=unk, pretrained=pretrained)
        d.save(save)
    n_printed = total = 0
    for X, Y in d.generate_batches('train', to_one_hot=False):
        print X.shape, Y.shape
        total += len(X)
        for x, y in zip(X, Y):
            words = [d.word_vocab.index2word[i] for i in x]
            rel = d.rel_vocab.index2word[y]
            if n_printed < max_printed:
                print words, rel
                n_printed += 1
    print 'done', total, 'in total'
