#!/usr/bin/env python
"""dump_data.py

"""
import os
from text.vocab import Vocab
from text.dataset import Dataset, Split, Example
import csv
import numpy as np

if __name__ == '__main__':

    mydir = os.path.dirname(__file__)
    fname = os.path.join(mydir, 'supervision.csv')

    dataset = Dataset(splits={'train': Split(), 'dev': Split(), 'test': Split()})
    probs = {'train':0.7, 'dev':0.2, 'test':0.1}
    seen = set()
    headers = ['dependency', 'words', 'lemma', 'pos', 'ner', 'subject_begin',
               'subject_end', 'subject_head', 'subject_ner', 'object_begin', 'object_end', 'object_head', 'object_ner', 'relation']

    with open(fname, 'rb') as f:
        reader = csv.reader(f)
        print headers
        for i, row in enumerate(reader):
            row = {h:e for h,e in zip(headers,row)}
            ex = Example(**row)
            key = ex.words, ex.relation
            if key in seen:
                continue
            seen.add(key)
            dataset.add_random(ex, probs)

            if i%5000==0:
                print 'processed', i
                for name, split in dataset.splits.items():
                    print name, len(split.examples),
                print

    dataset.save('annotated')
