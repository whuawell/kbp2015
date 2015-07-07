#!/usr/bin/env python
"""dataset.py
Usage: dataset.py [--pretrained=<NAME>] [--unk=<UNK>] [--overwrite] [--max_printed=<MAX>]

Options:
    --pretrained=<NAME>     glove or senna      [default: senna]
    --unk=<UNK>             the token used for unknown, for senna, set this to 'UNKNOWN'    [default: UNKNOWN]
    --max_printed=<MAX>       how many examples to print (use -1 to print all)  [default: 10]
"""
import numpy as np
import cPickle as pkl
import os
import json
from text.dataset import *
from text.vocab import Vocab
from pretrained.load import load_pretrained
from dependency import get_path_from_parse, parse_words, NoPathException
import csv
import string
from pprint import pprint

mydir = os.path.dirname(__file__)

def one_hot(y, num_classes):
    Y = np.zeros((len(y), num_classes), dtype='float32')
    Y[np.arange(len(y)), y] = 1.
    return Y

class TypeCheckAdaptor(object):

    def __init__(self, vocab):
        from theano import shared
        self.vocab = vocab
        self.valid_types = self.load_valid()
        self.valid = shared(self.valid_types)

    def get_valid(self, ner1, ner2):
        return self.valid[ner1, ner2]

    def get_valid_cpu(self, ner1, ner2):
        valid = self.valid_types[ner1, ner2]
        return valid

    def load_valid(self):
        fname = os.path.join(mydir, 'typecheck.csv')
        valid_types = np.zeros((len(self.vocab['ner']), len(self.vocab['ner']), len(self.vocab['rel'])), dtype='float32')

        with open(fname) as f:
            reader = csv.reader(f)
            for row in reader:
                relation, subject_ner, object_ner = [e.strip() for e in row]
                if relation not in self.vocab['rel'] or subject_ner not in self.vocab['ner'] or object_ner not in self.vocab['ner']:
                    if subject_ner != 'MISC' and object_ner != 'MISC':
                        continue
                valid_types[self.vocab['ner'][subject_ner], self.vocab['ner'][object_ner], self.vocab['rel'][relation]] = 1
        # any ner type can express no_relation
        for ner1 in xrange(len(self.vocab['ner'])):
            for ner2 in xrange(len(self.vocab['ner'])):
                valid_types[ner1, ner2, self.vocab['rel']['no_relation']] = 1
        # allow misc types
        valid_types[self.vocab['ner']['MISC'], :, :] = 1
        valid_types[:, self.vocab['ner']['MISC'], :] = 1
        return valid_types


class ExampleAdaptor(object):

    punctuation = set(string.punctuation)
    NUM = '1'
    PUNC = '.'

    def __init__(self, vocab):
        self.vocab = vocab

    def convert(self, example, unk='UNKNOWN', tokenize=parse_words):
        index2word = tokenize(example.words)
        index2word = [w.lower() for w in index2word]
        index2ner = tokenize(example.ner)
        dependency_path = get_path_from_parse(example.dependency,
                                              int(example.subject_begin), int(example.subject_end),
                                              int(example.object_begin), int(example.object_end))

        seen_start = False
        START = '***START***'
        words = []
        parse = []
        ner = []
        start_index = self.vocab['dep'].add(START)

        for from_, to_, edge_ in dependency_path:
            from_word, to_word = [index2word[i] for i in [from_, to_]]
            from_word_index, to_word_index = [self.vocab['word'].word2index.get(w, self.vocab['word'][unk]) for w in [from_word, to_word]]
            from_ner, to_ner = [index2ner[i] for i in [from_, to_]]
            from_ner_index, to_ner_index = [self.vocab['ner'].add(w) for w in [from_ner, to_ner]]
            edge = self.vocab['dep'].add(edge_)

            if not seen_start:
                words.append(from_word_index)
                ner.append(from_ner_index)
                parse.append(start_index)
                seen_start = True

            words.append(to_word_index)
            ner.append(to_ner_index)
            parse.append(edge)
        rel = self.vocab['rel'].add(example.relation) if 'relation' in example.keys() else None
        subject_ner, object_ner = [self.vocab['ner'].add(k) for k in [example.subject_ner, example.object_ner]]

        return Example(words=words, parse=parse, ner=ner, subject_ner=subject_ner, object_ner=object_ner, relation=rel,
                       subject=' '.join(index2word[int(example.subject_begin):int(example.subject_end)]),
                       object=' '.join(index2word[int(example.object_begin):int(example.object_end)]), debug=index2word)


class SplitAdaptor(object):

    def __init__(self, example_adaptor):
        self.example_adaptor = example_adaptor

    def convert(self, split, unk='UNKNOWN'):
        my_split = Split()
        lengths = {}
        idx = no_path = 0
        for i, ex in enumerate(split.examples):
            try:
                ex = self.example_adaptor.convert(ex, unk=unk)
            except NoPathException as e:
                no_path += 1
                continue
            l = len(ex.words)
            if l == 0:
                continue
            my_split.add(ex)
            if l not in lengths:
                lengths[l] = []
            lengths[l].append(idx)
            idx += 1
            if i % 1000 == 0:
                print 'converted', i, 'out of', len(split.examples)
        my_split.lengths = lengths
        print 'found', no_path, 'no path exceptions'
        return my_split


class AnnotatedData(object):

    def __init__(self, splits, vocab, word2emb):
        self.vocab = vocab
        self.splits = splits
        self.word2emb = word2emb

    @classmethod
    def build(cls, pretrained='senna', unk='UNKNOWN'):
        word_vocab, word2emb = load_pretrained(pretrained)
        vocab = {'word':word_vocab,
                 'rel': Vocab(unk=False),
                 'ner': Vocab(unk=False),
                 'dep': Vocab(unk=False)}
        raw = Dataset.load(os.path.join(mydir, 'annotated'))
        example_adaptor = ExampleAdaptor(vocab)
        split_adaptor = SplitAdaptor(example_adaptor)
        splits = {name:split_adaptor.convert(split, unk=unk) for name, split in raw.splits.items()}
        return AnnotatedData(splits, vocab, word2emb)

    def save(self, to_dir):
        if not os.path.isdir(to_dir):
            os.makedirs(to_dir)
        with open(os.path.join(to_dir, 'config.json'), 'wb') as f:
            json.dump({'splits': self.splits.keys()}, f)
        with open(os.path.join(to_dir, 'vocabs.pkl'), 'wb') as f:
            pkl.dump(self.vocab, f, protocol=pkl.HIGHEST_PROTOCOL)
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
        return AnnotatedData(splits, vocabs, word2emb)

    def generate_batches(self, name, batch_size=128, label='classification', to_one_hot=True, debug=False):
        assert label in ['classification', 'filter', 'raw']
        split = self.splits[name]
        order = split.lengths.keys()
        random.shuffle(order)

        for l in order:
            x_words, x_parse, x_ner, x_types, y = [], [], [], [], []
            debug_dump = []
            for idx in split.lengths[l]:
                ex = split.examples[idx]
                x_words.append(ex.words)
                x_parse.append(ex.parse)
                x_ner.append(ex.ner)
                x_types.append([ex.subject_ner, ex.object_ner])
                y.append(ex.relation)
                if debug:
                    debug_dump.append([ex.debug, ex.subject, ex.object])
            X = [np.array(x_words), np.array(x_parse), np.array(x_ner), np.array(x_types)]
            Y = np.array(y)
            if label == 'classification':
                if to_one_hot:
                    Y = one_hot(Y, len(self.vocab['rel']))
            elif label == 'filter':
                related = Y != self.vocab['rel']['no_relation']
                Y.fill(0.)
                Y[related] = 1.
                Y = Y.reshape((-1, 1))
            if len(Y):
                # this can turn out to be false if non of the examples pass the filter
                if debug:
                    yield X, Y, debug_dump
                else:
                    yield X, Y

if __name__ == '__main__':
    from docopt import docopt
    from pprint import pprint
    args = docopt(__doc__)
    pprint(args)
    max_printed = int(args['--max_printed'])
    if max_printed < 0:
        max_printed = np.inf
    pretrained = args['--pretrained']
    unk = args['--unk']

    save = pretrained
    if os.path.isdir(save) and not args['--overwrite']:
        d = AnnotatedData.load(save)
    else:
        d = AnnotatedData.build(unk=unk, pretrained=pretrained)
        d.save(save)

    typechecker = TypeCheckAdaptor(d.vocab)
    valid_types = typechecker.load_valid()
    print 'loaded', valid_types.sum(), 'valid types'

    num_neg = num_pos = 0
    for X, Y in d.generate_batches('train', label='filter'):
        num_pos += Y.sum()
        num_neg += len(Y) - Y.sum()
    print 'num_pos', num_pos, 'num_neg', num_neg

    n_printed = total = 0
    for X, Y, Debug in d.generate_batches('train', to_one_hot=False, debug=True):
        Xwords, Xparse, Xner, Xtypes = X
        total += len(X)
        args = X + [Debug, Y]
        for xwords, xparse, xner, xtypes, debug, y in zip(*args):
            debug_words, subj, obj = debug
            words = [d.vocab['word'].index2word[i] for i in xwords]
            parse = [d.vocab['dep'].index2word[i] for i in xparse]
            ner = [d.vocab['ner'].index2word[i] for i in xner]
            types = [d.vocab['ner'].index2word[i] for i in xtypes]
            rel = d.vocab['rel'].index2word[y]
            subj_ner, obj_ner = types
            try:
                if n_printed < max_printed:
                    print '# sentence'
                    print ' '.join(debug_words)
                    print '# span (word, parse, ner)'
                    for word, parse, ner in zip(words, parse, ner):
                        print word, parse, ner
                    print '# subject type'
                    print subj_ner
                    print '# object type'
                    print obj_ner
                    print '# relation'
                    print rel
                    print '# subject'
                    print subj
                    print '# object'
                    print obj
                    print
                    n_printed += 1
            except UnicodeEncodeError as e:
                continue
    print 'done', total, 'in total'
