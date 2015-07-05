#!/usr/bin/env python
#!/u/nlp/packages/anaconda/bin/safepython
import cPickle as pkl
import numpy as np
import json
import os
import signal
import fileinput
from pprint import pprint
from train import get_model_from_arg

context = 0
filter_threshold = 0.7


from data.dataset import AnnotatedData, ExampleAdaptor
from text.dataset import Example
from train import *
d = AnnotatedData.load('data/senna')
typechecker = TypeCheckAdaptor(d.vocab)

def load_model(folder):
    with open(os.path.join(folder, 'args.json')) as f:
        args = json.load(f)
    with open(os.path.join(folder, 'model.weights.pkl')) as f:
        weights = pkl.load(f)
    model = get_model_from_arg(args, d, typechecker)
    model.set_weights(weights)
    return model

mydir = os.path.abspath(os.path.dirname(__file__))
filter_dir = os.path.join(mydir, 'deploy', 'filter')
classifier_dir = os.path.join(mydir, 'deploy', 'classifier')

if __name__ == '__main__':
    # hack: disable keyboard interrupt (something funky with greenplum?)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # filter_model = load_model(filter_dir)
    # classifier = load_model(classifier_dir)
    adaptor = ExampleAdaptor(d.vocab)

    headers = ['gloss', 'dependency', 'dep_extra', 'dep_malt', 'words', 'lemmas', 'pos', 'ner', 'subject_id',
               'subject_entity', 'subject_link_score', 'subject_ner', 'object_id', 'object_entity', 'object_link_score',
               'object_ner', 'subject_begin', 'subject_end', 'object_begin', 'object_end']

    def hack_word_list(words):
        words = words[1:-1].replace('","', 'COMMA').split(',')
        for i, word in enumerate(words):
            if word == 'COMMA':
                words[i] = ','
        return words

    for line in fileinput.input():
        line = line.strip()
        fields = line.split("\t")
        row = dict(zip(headers, fields))

        # hack the data formatting issues away
        row['dependency'] = row['dependency'].replace("\\n", "\n").replace("\\t", "\t")

        ex = Example(**row)
        s_begin, s_end, o_begin, o_end = [int(row[k]) for k in ['subject_begin', 'subject_end', 'object_begin', 'object_end']]

        convert = adaptor.convert(ex, unk='UNKNOWN', tokenize=hack_word_list)
        Xin = [convert.words, convert.parse, convert.ner]
        Xin = [np.array(x).reshape((1, -1)) for x in Xin]
        types = np.array([convert.subject_ner, convert.object_ner]).reshape((1, -1))

        # pred = classifier.predict(Xin, verbose=0)
        # pred *= typechecker.get_valid_cpu(types[:, 0], types[:, 1])
        # rel = pred.argmax(axis=1)
        #
        # relation = d.vocab['rel'].index2word[rel[0]]
        # if relation == 'no_relation':
        #     continue
        # confidence = pred[rel]
        # print "\t".join([str(s) for s in [row['subject_id'], relation, row['object_id'], confidence]])
