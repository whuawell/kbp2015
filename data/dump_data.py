#!/usr/bin/env python
"""dump_data.py

"""
import os
from text.vocab import Vocab
from text.dataset import Dataset, Split, Example
from dataset import parse_words
import csv
import numpy as np

if __name__ == '__main__':

    mydir = os.path.dirname(__file__)
    fname = os.path.join(mydir, 'supervision.csv')
    typecheck = os.path.join(mydir, 'typecheck.csv')
    valid = set()
    with open(typecheck) as f:
        reader = csv.reader(f)
        for rel, ner1, ner2 in reader:
            valid.add((ner1, ner2, rel))

    dataset = Dataset(splits={'train': Split(), 'dev': Split(), 'test': Split()})
    probs = {'train':0.7, 'dev':0.2, 'test':0.1}
    seen = set()
    headers = ['dependency', 'words', 'lemma', 'pos', 'ner', 'subject_begin',
               'subject_end', 'subject_head', 'subject_ner', 'object_begin', 'object_end', 'object_head', 'object_ner', 'relation']

    with open(fname, 'rb') as f:
        reader = csv.reader(f)
        print headers
        num_bad = 0
        for i, row in enumerate(reader):
            row = {h:e for h,e in zip(headers,row)}
            ex = Example(**row)
            key = ex.words, ex.relation
            if key in seen:
                continue

            words = parse_words(ex.words)
            if ex.relation != 'no_relation' and (ex.subject_ner, ex.object_ner, ex.relation) not in valid and (ex.object_ner, ex.subject_ner, ex.relation) not in valid:
                if ex.subject_ner != 'MISC' and ex.object_ner != 'MISC':
                    # print words
                    # print ' '.join(words[int(ex.subject_begin):int(ex.subject_end)]), ex.subject_ner
                    # print ' '.join(words[int(ex.object_begin):int(ex.object_end)]), ex.object_ner
                    # print ex.relation
                    num_bad += 1

            seen.add(key)
            dataset.add_random(ex, probs)

            if i%5000==0:
                print 'processed', i
                for name, split in dataset.splits.items():
                    print name, len(split.examples),
                print
        print 'processed', i, 'num_bad', num_bad

    dataset.save('annotated')
