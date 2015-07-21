__author__ = 'victor'

import unittest
from data.adaptors import *
import csv
import os
mydir = os.path.dirname(os.path.abspath(__file__))

class TestAdaptor(object):

    def test_words(self):
        ex = self.adaptor.to_example(self.raw)
        self.assertEqual(ex.words, [w.lower() for w in  self.words])

    def test_lemmas(self):
        ex = self.adaptor.to_example(self.raw)
        self.assertEqual(ex.lemmas, [w.lower() for w in  self.lemmas])

    def test_ner(self):
        ex = self.adaptor.to_example(self.raw)
        self.assertEqual(ex.ner, self.ner)

    def test_pos(self):
        ex = self.adaptor.to_example(self.raw)
        self.assertEqual(ex.pos, self.pos)

    def test_subject(self):
        ex = self.adaptor.to_example(self.raw)
        self.assertEqual(ex.subject, self.subject.lower())
        self.assertEqual(ex.subject_ner, self.subject_ner)
        self.assertEqual(ex.subject_begin, self.subject_begin)
        self.assertEqual(ex.subject_end, self.subject_end)

    def test_object(self):
        ex = self.adaptor.to_example(self.raw)
        self.assertEqual(ex.object, self.object.lower())
        self.assertEqual(ex.object_ner, self.object_ner)
        self.assertEqual(ex.object_begin, self.object_begin)
        self.assertEqual(ex.object_end, self.object_end)
        
    def test_relation(self):
        ex = self.adaptor.to_example(self.raw)
        self.assertEqual(ex.relation, self.relation)

    def test_read_file(self):
        for ex in self.adaptor.to_examples(self.file):
            pass


class TestSupervised(unittest.TestCase, TestAdaptor):

    def setUp(self):
        self.file = os.path.join(mydir, '..', 'data', 'raw', 'supervision.csv')

        with open(self.file) as f:
            reader = csv.reader(f)
            self.raw = reader.next()
        self.adaptor = SupervisedDataAdaptor()

        self.words = [
            "Alexandra", "of", "Denmark", "-LRB-", "0000", "-", "0000", "-RRB-", "was", "Queen",
            "Consort", "to", "Edward", "VII", "of", "the", "United", "Kingdom", "and", "thus",
            "Empress", "of", "India", "during", "her", "husband", "\'s", "reign", "."
        ]
        
        self.lemmas = [
            "Alexandra", "of", "Denmark", "-lrb-", "0000", "-", "0000", "-rrb-", "be", "Queen",
            "Consort", "to", "Edward", "VII", "of", "the", "United", "Kingdom", "and", "thus",
            "empress", "of", "India", "during", "she", "husband", "'s", "reign", "."
        ]

        self.ner = [
            "PERSON", "PERSON", "PERSON", "O", "DATE", "DURATION", "DATE", "O", "O", "LOCATION",
            "LOCATION", "O", "PERSON", "PERSON", "O", "O", "LOCATION", "LOCATION", "O", "O", "O",
            "O", "LOCATION", "O", "O", "O", "O", "O", "O"
        ]

        self.pos = [
            "NNP", "IN", "NNP", "-LRB-", "CD", ":", "CD", "-RRB-", "VBD", "NNP", "NNP", "TO", "NNP",
            "NNP", "IN", "DT", "NNP", "NNP", "CC", "RB", "NN", "IN", "NNP", "IN", "PRP$", "NN",
            "POS", "NN", ".",
        ]

        self.subject_begin = 0
        self.subject_end = 3
        self.subject = 'Alexandra of Denmark'
        self.subject_ner = 'PERSON'

        self.object_begin = 12
        self.object_end = 13
        self.object = 'Edward'
        self.object_ner = 'PERSON'

        self.relation = 'per:spouse'


class TestKBPTest(unittest.TestCase, TestAdaptor):

    def setUp(self):
        self.file = os.path.join(mydir, '..', 'data', 'raw', 'test.sample.tsv')

        with open(self.file) as f:
            reader = csv.reader(f, delimiter="\t")
            self.raw = reader.next()
        self.adaptor = KBPDataAdaptor()

        self.words = [
            'This', 'recipe', 'from', 'Sean', 'Baker', 'of', 'Gather', 'in', 'Berkeley', 'is', 'a',
            'vegan', 'interpretation', 'of', 'a', 'rustic', 'seafood', 'salad', 'that', 'typically',
            'includes', 'mussels', ',', 'squid', 'and', 'other', 'shellfish', '.'
        ]

        self.lemmas = ['this', 'recipe', 'from', 'Sean', 'Baker', 'of', 'Gather', 'in',
                       'Berkeley', 'be', 'a', 'vegan', 'interpretation', 'of', 'a', 'rustic',
                       'seafood', 'salad', 'that', 'typically', 'include', 'mussel', ',',
                       'squid', 'and', 'other', 'shellfish', '.']

        self.ner = [
            'O', 'O', 'O', 'PERSON', 'PERSON', 'O', 'O', 'O', 'CITY', 'O', 'O', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'CAUSE_OF_DEATH', 'O'
        ]

        self.pos = [
            'DT', 'NN', 'IN', 'NNP', 'NNP', 'IN', 'NNP', 'IN', 'NNP', 'VBZ', 'DT', 'JJ', 'NN',
            'IN', 'DT', 'JJ', 'NN', 'NN', 'WDT', 'RB', 'VBZ', 'NNS', ',', 'NN', 'CC', 'JJ',
            'NN', '.'
         ]

        self.subject_begin = 3
        self.subject_end = 5
        self.subject = 'Sean Baker'
        self.subject_ner = 'PERSON'

        self.object_begin = 8
        self.object_end = 9
        self.object = 'Berkeley'
        self.object_ner = 'CITY'

        self.relation = None


class TestKBPEvaluationTest(unittest.TestCase, TestAdaptor):

    def setUp(self):
        self.file = os.path.join(mydir, '..', 'data', 'raw', 'evaluation.tsv')
        with open(self.file) as f:
            reader = csv.reader(f, delimiter="\t")
            self.raw = reader.next()
        self.adaptor = KBPEvaluationDataAdaptor()

        self.words = [
            'She', 'waited', 'for', 'him', 'to', 'phone', 'her', 'that', 'night', 'so', 'they',
            'could', 'continue', 'their', 'discussion', ',', 'but', 'Pekar', 'never', 'called',
            ';', 'he', 'was', 'found', 'dead', 'early', 'the', 'next', 'morning', 'by', 'his',
            'wife', ',', 'Joyce', 'Brabner', '.']

        self.lemmas = [
            'she', 'wait', 'for', 'he', 'to', 'phone', 'she', 'that', 'night', 'so', 'they',
            'could', 'continue', 'they', 'discussion', ',', 'but', 'Pekar', 'never', 'call', ';',
            'he', 'be', 'find', 'dead', 'early', 'the', 'next', 'morning', 'by', 'he', 'wife',
            ',', 'Joyce', 'Brabner', '.']

        self.ner = [
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'TIME', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
            'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'DATE', 'DATE', 'DATE', 'DATE', 'O', 'O',
            'O', 'O', 'PERSON', 'PERSON', 'O'
         ]

        self.pos = [
            'PRP', 'VBD', 'IN', 'PRP', 'TO', 'VB', 'PRP$', 'DT', 'NN', 'IN', 'PRP', 'MD', 'VB',
            'PRP$', 'NN', ',', 'CC', 'NNP', 'RB', 'VBD', ':', 'PRP', 'VBD', 'VBN', 'JJ', 'RB',
            'DT', 'JJ', 'NN', 'IN', 'PRP$', 'NN', ",", 'NNP', 'NNP', '.'
        ]

        self.subject_begin = 17
        self.subject_end = 18
        self.subject = 'Pekar'
        self.subject_ner = 'PERSON'

        self.object_begin = 33
        self.object_end = 35
        self.object = 'Joyce Brabner'
        self.object_ner = 'PERSON'

        self.relation = 'per:spouse'


class TestSelfTrainingAdaptor(unittest.TestCase, TestAdaptor):

    def setUp(self):
        self.file = os.path.join(mydir, '..', 'data', 'raw', 'self_training.tsv')
        with open(self.file) as f:
            reader = csv.reader(f, delimiter="\t")
            self.raw = reader.next()

        self.adaptor = SelfTrainingAdaptor()

        self.words = ['-LSB-', '00', '-RSB-', 'Y.F.', 'Sasaki', ',', 'K.', 'Fujikawa', ',', 'K.',
                      'Ishida', ',', 'N.', 'Kawamura', ',', 'Y.', 'Nishikawa', ',', 'S.', 'Ohta',
                      ',', 'M.', 'Satoh', ',', 'H.', 'Madarame', ',', 'S.', 'Ueno', ',', 'N.',
                      'Susa', ',', 'N.', 'Matsusaka', ',', 'S.', 'Tsuda', ',', 'The', 'alkaline',
                      'single-cell', 'gel', 'electrophoresis', 'assay', 'with', 'mouse',
                      'multiple', 'organs', ':', 'results', 'with', '00', 'aromatic', 'amines',
                      'evaluated', 'by', 'the', 'IARC', 'and', 'US', 'NTP', ',', 'Mutat', '.']

        self.lemmas = ['-lsb-', '00', '-rsb-', 'Y.F.', 'Sasaki', ',', 'K.', 'Fujikawa', ',', 'K.',
                       'Ishida', ',', 'N.', 'Kawamura', ',', 'Y.', 'Nishikawa', ',', 'S.', 'Ohta',
                       ',', 'M.', 'Satoh', ',', 'H.', 'Madarame', ',', 'S.', 'Ueno', ',', 'N.',
                       'Susa', ',', 'N.', 'Matsusaka', ',', 'S.', 'Tsuda', ',', 'the', 'alkaline',
                       'single-cell', 'gel', 'electrophoresis', 'assay', 'with', 'mouse',
                       'multiple', 'organ', ':', 'result', 'with', '00', 'aromatic', 'amine',
                       'evaluate', 'by', 'the', 'iarc', 'and', 'US', 'NTP', ',', 'Mutat', '.']

        self.ner = [
            'O', 'NUMBER', 'O', 'PERSON', 'PERSON', 'O', 'PERSON', 'PERSON', 'O', 'PERSON',
            'PERSON', 'O', 'PERSON', 'PERSON', 'O', 'PERSON', 'PERSON', 'O', 'PERSON', 'PERSON',
            'O', 'PERSON', 'PERSON', 'O', 'PERSON', 'PERSON', 'O', 'PERSON', 'PERSON', 'O',
            'PERSON', 'PERSON', 'O', 'PERSON', 'PERSON', 'O', 'PERSON', 'PERSON', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'NUMBER', 'O', 'O', 'O', 'O',
            'O', 'ORGANIZATION', 'O', 'COUNTRY', 'ORGANIZATION', 'O', 'PERSON', 'O'
         ]

        self.pos = [
            '-LRB-', 'CD', '-RRB-', 'NNP', 'NNP', ',', 'NNP', 'NNP', ',', 'NNP', 'NNP', ',', 'NNP',
            'NNP', ',', 'NNP', 'NNP', ',', 'NNP', 'NNP', ',', 'NNP', 'NNP', ',', 'NNP', 'NNP', ',',
            'NNP', 'NNP', ',', 'NNP', 'NNP', ',', 'NNP', 'NNP', ',', 'NNP', 'NNP', ',', 'DT', 'NN',
            'JJ', 'NN', 'NN', 'NN', 'IN', 'NN', 'JJ', 'NNS', ':', 'NNS', 'IN', 'CD', 'JJ', 'NNS',
            'VBN', 'IN', 'DT', 'NN', 'CC', 'NNP', 'NNP', ',', 'NNP', '.'
        ]

        self.subject_begin = 30
        self.subject_end = 32
        self.subject = 'N. Susa'
        self.subject_ner = 'PERSON'

        self.object_begin = 33
        self.object_end = 35
        self.object = 'N. Matsusaka'
        self.object_ner = 'PERSON'

        self.relation = 'no_relation'


if __name__ == '__main__':
    unittest.main()
