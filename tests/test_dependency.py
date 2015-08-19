import unittest
from data.dependency import DependencyParse
import os
mydir = os.path.dirname(os.path.abspath(__file__))

__author__ = 'victor'


class TestDependencyParse(unittest.TestCase):

    def setUp(self):
        self.words = ['yesterday', ',', 'Steph', 'Curry', 'had', 'curry', 'for', 'dinner']
        self.ner = ['DATE', 'O', 'PERSON', 'PERSON', 'O', 'O', 'O', 'O']
        self.pos = ['NN', ',', 'NNP', 'NNP', 'VBD', 'NN', 'IN', 'NN']
        self.dep = [
            ('yesterday', 'had', 'nmod:tmod'),
            ('Steph', 'Curry', 'compound_from'),
            ('Curry', 'had', 'nsubj'),
            ('had', '.', 'root'),
            ('curry', 'had', 'dobj'),
            ('for', 'dinner', 'case'),
            ('dinner', 'curry', 'nmod'),
        ]
        for i, tup in enumerate(self.dep):
            child, parent, arc = tup
            if arc == 'root':
                self.dep[i] = (self.words.index(child), -1, 'root')
            else:
                self.dep[i] = (self.words.index(child), self.words.index(parent), arc)

    def test_shortest(self):
        parse = DependencyParse(self.dep, enhanced=True)
        shortest = parse.get_path_from_parse(2, 4, 5, 6)
        self.assertEqual(shortest, [
            (self.words.index('curry'), self.words.index('had'), 'dobj_from'),
            (self.words.index('had'), self.words.index('Curry'), 'nsubj_to'),
        ])

    def test_shorest_real(self):
        import csv
        from data.adaptors import KBPDataAdaptor
        with open(os.path.join(mydir, '..', 'data', 'raw', 'test.sample.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            raw = reader.next()
        adaptor = KBPDataAdaptor()
        ex = adaptor.to_example(raw)
        parse = DependencyParse(ex.dependency, enhanced=True)
        shortest = parse.get_path_from_parse(ex.subject_begin, ex.subject_end, ex.object_begin, ex.object_end)

        self.assertEqual(shortest, [
            (ex.words.index('berkeley'), ex.words.index('baker'), 'nmod:in_from'),
        ])

if __name__ == '__main__':
    unittest.main()
