__author__ = 'victor'

import unittest
from data.featurizers import *

class TestFeaturizer(object):

    words = ['yesterday', ',', 'Steph', 'Curry', 'had', 'curry', 'for', 'dinner']
    ner = ['DATE', 'O', 'PERSON', 'PERSON', 'O', 'O', 'O', 'O']
    pos = ['NN', ',', 'NNP', 'NNP', 'VBD', 'NN', 'IN', 'NN']
    dep = [
        [words.index('curry'), words.index('had'), 'dobj'],
        [words.index('had'), None, 'root'],
        [words.index('Curry'), words.index('had'), 'nsubj'],
    ]


class TestSingleSequence(unittest.TestCase, TestFeaturizer):

    def setUp(self):
        self.featurizer = SinglePathDependencyFeaturizer()

    def test_featurize(self):
        ex = Example(words=self.words, dependency=self.dep, ner=self.ner, pos=self.pos,
                     relation='no_relation', subject_ner='PERSON', object_ner='O',
                     subject_begin=2, subject_end=4, object_begin=5, object_end=6)
        got = self.featurizer.featurize(ex, True)
        self.assertItemsEqual(self.featurizer.vocab['ner'].index2word, ['PERSON', 'O'])
        self.assertItemsEqual(self.featurizer.vocab['word'].index2word, ['UNKNOWN', 'O', 'dobj_from', 'had', 'nsubj_to', 'PERSON'])
        self.assertEqual(got.subject_ner, self.featurizer.vocab['ner']['PERSON'])
        self.assertEqual(got.object_ner, self.featurizer.vocab['ner']['O'])
        self.assertEqual([self.featurizer.vocab['word'].index2word[w] for w in got.sequence],
                         ['O', 'dobj_from', 'had', 'nsubj_to', 'PERSON'])


class TestConcatenated(unittest.TestCase, TestFeaturizer):

    def setUp(self):
        self.featurizer = ConcatenatedDependencyFeaturizer()

    def test_featurize(self):
        ex = Example(words=self.words, dependency=self.dep, ner=self.ner, pos=self.pos,
                     relation='no_relation', subject_ner='PERSON', object_ner='O',
                     subject_begin=2, subject_end=4, object_begin=5, object_end=6)
        got = self.featurizer.featurize(ex, True)
        self.assertItemsEqual(self.featurizer.vocab['ner'].index2word, ['PERSON', 'O'])
        self.assertItemsEqual(self.featurizer.vocab['word'].index2word, ['UNKNOWN', 'O', 'had', 'PERSON'])
        self.assertItemsEqual(self.featurizer.vocab['pos'].index2word, [self.featurizer.vocab['pos'].unk, 'NN', 'VBD', 'NNP'])
        self.assertItemsEqual(self.featurizer.vocab['dep'].index2word, ['dobj_from', 'root', 'nsubj_to'])
        word = lambda w: self.featurizer.vocab['word'][w]
        ner = lambda w: self.featurizer.vocab['ner'][w]
        dep = lambda w: self.featurizer.vocab['dep'][w]
        pos = lambda w: self.featurizer.vocab['pos'][w]
        self.assertEqual(got.subject_ner, ner('PERSON'))
        self.assertEqual(got.object_ner, ner('O'))
        self.assertEqual(got.words, [word(k) for k in ['O', 'had', 'PERSON']])
        self.assertEqual(got.ner, [ner(k) for k in ['O', 'O', 'PERSON']])
        self.assertEqual(got.pos, [pos(k) for k in ['NN', 'VBD', 'NNP']])
        self.assertEqual(got.arc, [dep(k) for k in ['dobj_from', 'root', 'nsubj_to']])

if __name__ == '__main__':
    unittest.main()