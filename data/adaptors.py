__author__ = 'victor'
from dataset import Example
import csv

class DatasetAdaptor(object):

    keep = ['dependency', 'words', 'lemmas', 'pos', 'ner', 'subject_begin', 'subject_end', 'subject',
            'subject_ner', 'object_begin', 'object_end', 'object', 'object_ner', 'relation']

    def parse_dependency(self, dependency, ex, use_lemma=True):
        deps = []
        words = ex.lemmas if use_lemma else ex.words
        for line in dependency.split("\n"):
            child, parent, arc = line.split("\t")
            deps.append((int(child)-1, int(parent)-1, arc))
        return deps

    def parse_array(self, words, zero_numbers=False):
        parsed = words[2:-2].split('","')
        if zero_numbers:
            isdigit = unicode.isdigit if isinstance(parsed[0], unicode) else str.isdigit
            parsed = ['0'*len(word) if isdigit(word) else word for word in parsed]
        return parsed

    def convert_types(self, ex):
        for e in ['lemmas', 'words']:
            ex[e] = self.parse_array(ex[e], True)
        for e in ['pos', 'ner']:
            ex[e] = self.parse_array(ex[e])
        for e in ['subject_begin', 'subject_end', 'object_begin', 'object_end']:
            ex[e] = int(ex[e])

        ex.subject = ' '.join(ex.words[ex.subject_begin:ex.subject_end])
        ex.object = ' '.join(ex.words[ex.object_begin:ex.object_end])
        ex.dependency = self.parse_dependency(ex.dependency, ex)

        for i in xrange(ex.subject_begin, ex.subject_end):
            ex.ner[i] = ex.subject_ner

        for i in xrange(ex.object_begin, ex.object_end):
            ex.ner[i] = ex.object_ner

        if not hasattr(ex, 'relation'):
            ex.relation = None

        return Example(**{k:v for k, v in ex.__dict__.items() if k in self.keep})

    def to_example(self, row):
        raise NotImplementedError()

    def to_examples(self, fname):
        raise NotImplementedError()


class SupervisedDataAdaptor(DatasetAdaptor):

    headers = [
            'dependency', 'words', 'lemmas', 'pos', 'ner', 'subject_begin', 'subject_end', 'subject_head',
            'subject_ner', 'object_begin', 'object_end', 'object_head', 'object_ner', 'relation'
            ]

    def to_example(self, row):

        d = dict(zip(self.headers, row))
        assert len(d) == len(self.headers), "could not convert row to example %s\n%s" % (row, d)
        ex = Example(**d)
        return self.convert_types(ex)

    def to_examples(self, fname):
        with open(fname) as f:
            reader = csv.reader(f)
            for row in reader:
                yield self.to_example(row)


class KBPDataAdaptor(DatasetAdaptor):
    headers = ['gloss', 'dependency', 'dep_extra', 'dep_malt', 'words', 'lemmas', 'pos', 'ner', 'subject_id',
               'subject_entity', 'subject_link_score', 'subject_ner', 'object_id', 'object_entity', 'object_link_score',
               'object_ner', 'subject_begin', 'subject_end', 'object_begin', 'object_end']

    def parse_array(self, words, zero_numbers=False):
        words = words.replace('"', '').replace(',,,', ',COMMA,')
        parsed = words[1:-1].split(',')
        parsed = [',' if p == 'COMMA' else p for p in parsed]
        if zero_numbers:
            isdigit = unicode.isdigit if isinstance(parsed[0], unicode) else str.isdigit
            parsed = ['0'*len(word) if isdigit(word) else word for word in parsed]
        return parsed

    def to_example(self, row):
        d = dict(zip(self.headers, row))
        assert len(d) == len(self.headers), "could not convert row to example %s\n%s" % (row, d)
        ex = Example(**d)
        for k in ['dependency', 'dep_extra', 'dep_malt']:
            ex[k] = ex[k].replace("\\n", "\n").replace("\\t", "\t")
        return self.convert_types(ex)

    def to_examples(self, fname):
        with open(fname) as f:
            for row in f:
                yield self.to_example(row.split("\t"))


class KBPEvaluationDataAdaptor(KBPDataAdaptor):

    headers = ['gloss', 'dependency', 'dep_extra', 'dep_malt', 'words', 'lemmas',
               'pos', 'ner', 'subject_id', 'subject', 'subject_link_score', 'subject_ner',
               'object_id', 'object', 'object_link_score', 'object_ner',
               'subject_begin', 'subject_end', 'object_begin', 'object_end',
               'known_relations', 'incompatible_relations', 'annotated_relations']

    def to_example(self, row):
        d = dict(zip(self.headers, row))
        assert len(d) == len(self.headers), "could not convert row to example %s\n%s" % (row, d)
        ex = Example(**d)
        rels = self.parse_array(ex.known_relations)
        ex.relation = rels[0] if rels else 'no_relation'
        for k in ['dependency', 'dep_extra', 'dep_malt']:
            ex[k] = ex[k].replace("\\n", "\n").replace("\\t", "\t")
        return self.convert_types(ex)
