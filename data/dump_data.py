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
    fname = os.path.join(mydir, 'annotated_sentences.csv')

    dataset = Dataset(splits={'train': Split(), 'dev': Split(), 'test': Split()})
    probs = {'train':0.7, 'dev':0.2, 'test':0.1}
    seen = set()

    with open(fname, 'rb') as f:
        reader = csv.reader(f)
        headers = reader.next()
        print headers
        for i, row in enumerate(reader):
            row = {h:e for h,e in zip(headers,row)}
            if row['entityCharOffsetBegin'] == 'entityCharOffsetBegin':
                continue # this dataset has bugs, some rows are garbage
            ex = Example(**row)
            key = ex.sentence, ex.relation
            if key in seen:
                continue
            seen.add(key)
            dataset.add_random(ex, probs)

            if i%5000==0:
                print 'processed', i

    dataset.save('annotated')
