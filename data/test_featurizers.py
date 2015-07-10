__author__ = 'victor'

import unittest
from featurizers import *
import csv

class TestFeaturizer(object):

    words = ['yesterday', ',', 'Steph', 'Curry', 'had', 'curry', 'for', 'dinner']
    ner = ['DATE', 'O', 'PERSON', 'PERSON', 'O', 'O', 'O', 'O']
    pos = ['NN', ',', 'NNP', 'NNP', 'VBD', 'NN', 'IN', 'NN']
    dep = [
        [words.index('curry'), words.index('had'), 'dobj_from'],
        [words.index('had'), None, 'root'],
        [words.index('had'), words.index('Curry'), 'nsubj_to'],
    ]


class TestSingleSequence(unittest.TestCase, TestFeaturizer):

    def setUp(self):
        self.featurizer = SinglePathFeaturizer()

    def test_featurize(self):
        ex = Example(words=self.words, dependency=self.dep, ner=self.ner, pos=self.pos,
                     relation='no_relation', subject_ner='PERSON', object_ner='O',
                     subject_begin=2, subject_end=4, object_begin=5, object_end=6)
        got = self.featurizer.featurize(ex, True)
        self.assertItemsEqual(self.featurizer.vocab['ner'].index2word, ['PERSON', 'O'])
        self.assertItemsEqual(self.featurizer.vocab['word'].index2word, ['UNKNOWN', 'O', 'dobj_from', 'had', 'nsubj_to', 'PERSON'])
        self.assertEqual(got.subject_ner, 0)
        self.assertEqual(got.object_ner, 1)
        self.assertEqual([self.featurizer.vocab['word'].index2word[w] for w in got.sequence],
                         ['O', 'dobj_from', 'had', 'nsubj_to', 'PERSON'])


class TestConcatenated(unittest.TestCase, TestFeaturizer):

    def setUp(self):
        self.featurizer = ConcatenatedFeaturizer()

    def test_featurize(self):
        ex = Example(words=self.words, dependency=self.dep, ner=self.ner, pos=self.pos,
                     relation='no_relation', subject_ner='PERSON', object_ner='O',
                     subject_begin=2, subject_end=4, object_begin=5, object_end=6)
        got = self.featurizer.featurize(ex, True)
        self.assertItemsEqual(self.featurizer.vocab['ner'].index2word, ['PERSON', 'O'])
        self.assertItemsEqual(self.featurizer.vocab['word'].index2word, ['UNKNOWN', 'O', 'had', 'PERSON'])
        self.assertItemsEqual(self.featurizer.vocab['pos'].index2word, ['NN', 'VBD', 'NNP'])
        self.assertItemsEqual(self.featurizer.vocab['dep'].index2word, ['dobj_from', 'root', 'nsubj_to'])
        self.assertEqual(got.subject_ner, 0)
        self.assertEqual(got.object_ner, 1)
        word = lambda w: self.featurizer.vocab['word'][w]
        ner = lambda w: self.featurizer.vocab['ner'][w]
        dep = lambda w: self.featurizer.vocab['dep'][w]
        pos = lambda w: self.featurizer.vocab['pos'][w]
        expect = [
            [word('O'), ner('O'), pos('NN'), dep('dobj_from')],
            [word('had'), ner('O'), pos('VBD'), dep('root')],
            [word('PERSON'), ner('PERSON'), pos('NNP'), dep('nsubj_to')],
        ]
        self.assertEqual(got.sequence, expect)

if __name__ == '__main__':
    unittest.main()