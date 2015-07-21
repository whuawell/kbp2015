__author__ = 'victor'
from dataset import Example
import csv
import os
mydir = os.path.dirname(os.path.abspath(__file__))
rawdir = os.path.join(mydir, 'raw')

class DatasetAdaptor(object):

    keep = ['dependency', 'words', 'lemmas', 'pos', 'ner', 'subject_begin', 'subject_end', 'subject',
            'subject_ner', 'object_begin', 'object_end', 'object', 'object_ner', 'relation',
            'subject_id', 'object_id']

    def parse_dependency(self, dependency, ex, use_lemma=False):
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
            ex[e] = [w.lower() for w in ex[e]]
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
        assert len(row) == len(self.headers), "could not convert row to example %s\n%s" % (row, self.headers)
        d = dict(zip(self.headers, row))
        ex = Example(**d)
        return self.convert_types(ex)

    def to_examples(self, fname=None):
        if fname is None:
            fname = os.path.join(rawdir, 'supervision.csv')
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
        assert len(row) == len(self.headers), "could not convert row to example %s\n%s" % (row, self.headers)
        d = dict(zip(self.headers, row))
        ex = Example(**d)
        for k in ['dependency', 'dep_extra', 'dep_malt']:
            ex[k] = ex[k].replace("\\n", "\n").replace("\\t", "\t")
        return self.convert_types(ex)

    def to_examples(self, fname=None):
        if fname is None:
            fname = os.path.join(rawdir, 'test.sample.tsv')
        with open(fname) as f:
            for row in f:
                yield self.to_example(row.split("\t"))

    def online_to_examples(self, disable_interrupts=False):
        import sys
        if disable_interrupts:
            import signal
            s = signal.signal(signal.SIGINT, signal.SIG_IGN)
        for line in sys.stdin:
            row = line.split("\t")
            if len(row) < 2:
               continue 
            yield self.to_example(row)


class KBPEvaluationDataAdaptor(KBPDataAdaptor):

    headers = ['gloss', 'dependency', 'dep_extra', 'dep_malt', 'words', 'lemmas',
               'pos', 'ner', 'subject_id', 'subject', 'subject_link_score', 'subject_ner',
               'object_id', 'object', 'object_link_score', 'object_ner',
               'subject_begin', 'subject_end', 'object_begin', 'object_end',
               'known_relations', 'incompatible_relations', 'annotated_relations']

    relation_map = {
        'per:employee_or_member_of': 'per:employee_of',
        'org:top_members_employees': 'org:top_members/employees',
        'per:statesorprovinces_of_residence': 'per:stateorprovinces_of_residence',
        'org:number_of_employees_members': 'org:number_of_employees/members',
        'org:political_religious_affiliation': 'org:political/religious_affiliation',
        '': 'no_relation',
    }

    def to_example(self, row):
        assert len(row) == len(self.headers), "could not convert row to example %s\n%s" % (row, self.headers)
        d = dict(zip(self.headers, row))
        ex = Example(**d)
        rels = self.parse_array(ex.known_relations)
        ex.relation = rels[0]
        for k in ['pos']:
            ex[k] = ex[k].replace('`', "'")
        if ex.relation in self.relation_map:
            ex.relation = self.relation_map[ex.relation]
        for k in ['dependency', 'dep_extra', 'dep_malt']:
            ex[k] = ex[k].replace("\\n", "\n").replace("\\t", "\t")
        return self.convert_types(ex)

    def to_examples(self, fname=None):
        if fname is None:
            fname = os.path.join(rawdir, 'evaluation.tsv')
        for example in super(KBPEvaluationDataAdaptor, self).to_examples(fname):
            yield example


class SelfTrainingAdaptor(KBPEvaluationDataAdaptor):

    relation_map = {
        'per:member_of': 'per:employee_of',
        '': 'no_relation',
        'false': 'no_relation',
        '???': 'no_relation',
    }

    headers = ['gloss', 'dependency', 'dep_extra', 'dep_malt', 'words', 'lemmas',
               'pos', 'ner', 'subject_id', 'subject', 'subject_link_score', 'subject_ner',
               'object_id', 'object', 'object_link_score', 'object_ner',
               'subject_begin', 'subject_end', 'object_begin', 'object_end', 'corpus_id',
               'known_relations', 'incompatible_relations', 'annotated_relations']

    def to_example(self, row):
        assert len(row) == len(self.headers), "could not convert row to example %s\n%s" % (row, self.headers)
        d = dict(zip(self.headers, row))
        ex = Example(**d)
        ex.relation = ex.annotated_relations.strip()
        for k in ['pos']:
            ex[k] = ex[k].replace('`', "'")

        if ex.relation in self.relation_map:
            ex.relation = self.relation_map[ex.relation]
        for k in ['dependency', 'dep_extra', 'dep_malt']:
            ex[k] = ex[k].replace("\\n", "\n").replace("\\t", "\t")
        return self.convert_types(ex)

    def to_examples(self, fname=None):
        if fname is None:
            fname = os.path.join(rawdir, 'self_training.tsv')
        for example in super(SelfTrainingAdaptor, self).to_examples(fname):
            yield example


class AllAnnotatedAdaptor(DatasetAdaptor):

    def __init__(self):
        super(AllAnnotatedAdaptor, self).__init__()
        self.supervised = SupervisedDataAdaptor()
        self.annotated = SelfTrainingAdaptor()

    def to_examples(self, fname=None):
        for ex in self.supervised.to_examples():
            yield ex
        for ex in self.annotated.to_examples():
            yield ex
