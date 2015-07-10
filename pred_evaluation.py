#!/usr/bin/env python
__author__ = 'victor'

import cPickle as pkl
import numpy as np
import json
import os
import sys
import signal
import fileinput
from pprint import pprint
from train import get_model_from_arg
import csv

mydir = os.path.dirname(os.path.abspath(__file__))

evaluation_file = os.path.join(mydir, 'evaluation.tsv')

from data.dataset import AnnotatedData, ExampleAdaptor, UnseenExampleError
from text.dataset import Example
from train import *

def load_from_dir(folder):
    with open(os.path.join(folder, 'args.json')) as f:
        args = json.load(f)
    with open(os.path.join(folder, 'best_weights.pkl')) as f:
        weights = pkl.load(f)
    data_dir = os.path.join(mydir, args['<data>'])
    data = AnnotatedData.load(data_dir)
    typechecker = TypeCheckAdaptor(data.vocab)

    model = get_model_from_arg(args, data, typechecker)
    model.set_weights(weights)
    return model, data, typechecker, args

def database_array(words):
    words = words[1:-1].replace('","', 'COMMA').split(',')
    for i, word in enumerate(words):
        if word == 'COMMA':
            words[i] = ','
    return words

relation_hack = {
    'per:employee_or_member_of': 'per:employee_of',
    'org:top_members_employees': 'org:top_members/employees',
    'per:statesorprovinces_of_residence': 'per:stateorprovinces_of_residence',
    'org:number_of_employees_members': 'org:number_of_employees/members',
    'org:political_religious_affiliation': 'org:political/religious_affiliation',
}

def softmax(a):
    a -= a.max()
    e = np.exp(a)
    return e / e.sum(keepdims=True)

if __name__ == '__main__':
    model_dir = sys.argv[1]
    model, dataset, typechecker, args = load_from_dir(model_dir)
    adaptor = ExampleAdaptor(dataset.vocab)
    labels = [i for i in range(len(dataset.vocab['rel'])) if i != dataset.vocab['rel']['no_relation']]

    def hack_word_list(words):
        words = words[1:-1].replace('","', 'COMMA').split(',')
        for i, word in enumerate(words):
            if word == 'COMMA':
                words[i] = ','
        return words

    def get_XY(X):
        Xwords, Xparse, Xner = X
        Xin = {
            'sent': Xwords,
            'ner': Xner,
            'parse': Xparse,
            'sent_ner': [Xwords, Xner],
            'sent_parse': [Xwords, Xparse],
            'sent_parse_ner': [Xwords, Xparse, Xner]
        }[args['--model']]
        return Xin

    total = total_loss = 0
    func = model.test
    preds, targs = [], []

    name = 'eval_pred.out'

    import json
    fout = open(name, 'wb')

    preds, targs = [], []
    headers = ['gloss', 'dependency', 'dependency_extra', 'dependency_malt', 'words', 'lemma',
               'pos', 'ner', 'subject_id', 'subject', 'subject_link_score', 'subject_ner',
               'object_id', 'object', 'object_link_score', 'object_ner',
               'subject_begin', 'subject_end', 'object_begin', 'object_end',
               'known_relations', 'incompatible_relations', 'annotated_relations']

    tsvin = open(evaluation_file)
    for i, line in enumerate(tsvin):
        row = dict(zip(headers, line.split("\t")))
        row['dependency'] = row['dependency'].replace("\\n", "\n").replace("\\t", "\t")
        rels = [e for e in row['known_relations'][1:-1].split(',') if e.strip()]
        row['relation'] = rels[0] if rels else 'no_relation'
        if row['relation'] in relation_hack:
            row['relation'] = relation_hack[row['relation']]
        ex = Example(**row)

        # skip badly tagged examples
        if ex.subject == ex.object:
            continue

        try:
            ex = adaptor.convert(ex, unk='UNKNOWN', tokenize=database_array, kbp=True)
        except UnseenExampleError as e:
            if hasattr(e, 'skippable'):
                sys.stderr.write('ignoring relation ' + str(ex.relation) + "\n")
                continue
            else:
                raise e

        Xin = get_XY([ex.words, ex.parse, ex.ner])
        Xin = [np.array(x).reshape((1, -1)) for x in Xin]
        types = np.array([ex.subject_ner, ex.object_ner]).reshape((1, -1))

        try:
            pred = model.predict(Xin, verbose=0)
        except Exception as e:
            sys.stderr.write("input %s\n" % ex)
            raise e
        pred *= typechecker.get_valid_cpu(types[:, 0], types[:, 1])
        pred = softmax(pred)
        rel = pred.argmax(axis=1)

        ex.predicted_prob = pred.flatten().tolist()
        ex.predicted_relation = rel[0]
        preds.append(ex.predicted_relation)
        targs.append(ex.relation)

        ex.confidence = ex.predicted_prob[ex.predicted_relation]
        ex.ner = [dataset.vocab['ner'].index2word[n] for n in ex.ner]
        ex.words = [dataset.vocab['word'].index2word[w] for w in ex.words]
        ex.parse = [dataset.vocab['dep'].index2word[w] for w in ex.parse]
        ex.p_relation = ex.predicted_prob[ex.relation]
        ex.p_predicted_relation = ex.predicted_prob[ex.predicted_relation]
        ex.relation = dataset.vocab['rel'].index2word[ex.relation]
        ex.predicted_relation = dataset.vocab['rel'].index2word[ex.predicted_relation]
        ex.subject_ner, ex.object_ner = [dataset.vocab['ner'].index2word[w] for w in [ex.subject_ner, ex.object_ner]]
        fout.write(json.dumps(ex, sort_keys=True) + "\n")

        if i > 0 and i % 1000 == 0:
            print 'predicted', i, 'examples'
            print 'runing f1', f1_score(targs, preds, average='micro', labels=labels)

    with open(name + '.preds', 'wb') as f:
        print >> f, "targ\tpred"
        for targ, pred in zip(targs, preds):
            print >> f, dataset.vocab['rel'].index2word[targ] + "\t" + dataset.vocab['rel'].index2word[pred]

    from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score
    report = classification_report(targs, preds, target_names=dataset.vocab['rel'].index2word)

    with open(name + '.report', 'wb') as f:
        f.write(report)

    scores = {
        'f1': f1_score(targs, preds, average='micro', labels=labels),
        'precision': precision_score(targs, preds, average='micro', labels=labels),
        'recall': recall_score(targs, preds, average='micro', labels=labels),
        'accuracy': accuracy_score(targs, preds),
    }

    from pprint import pprint
    pprint(scores)

    with open(name + '.scores', 'wb') as f:
        json.dump(scores, f)
