#!/usr/bin/env python
#!/u/nlp/packages/anaconda/bin/safepython
import cPickle as pkl
import numpy as np
import json
import os
import sys
import signal
import fileinput
from pprint import pprint
from train import get_model_from_arg

mydir = os.path.dirname(os.path.abspath(__file__))

context = 0
filter_threshold = 0.7

from data.dataset import AnnotatedData, ExampleAdaptor
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

if __name__ == '__main__':
    model_dir, split = sys.argv[1:]
    model, dataset, typechecker, args = load_from_dir(model_dir)
    adaptor = ExampleAdaptor(dataset.vocab)

    headers = ['gloss', 'dependency', 'dep_extra', 'dep_malt', 'words', 'lemmas', 'pos', 'ner', 'subject_id',
               'subject_entity', 'subject_link_score', 'subject_ner', 'object_id', 'object_entity', 'object_link_score',
               'object_ner', 'subject_begin', 'subject_end', 'object_begin', 'object_end']

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

    name = split + '_pred.out'

    import json
    fout = open(name, 'wb')

    preds, targs = [], []
    for i, ex in enumerate(dataset.splits[split].examples):
        Xin = get_XY([ex.words, ex.parse, ex.ner])
        Xin = [np.array(x).reshape((1, -1)) for x in Xin]
        types = np.array([ex.subject_ner, ex.object_ner]).reshape((1, -1))

        pred = model.predict(Xin, verbose=0)
        pred *= typechecker.get_valid_cpu(types[:, 0], types[:, 1])
        rel = pred.argmax(axis=1)

        ex.predicted_prob = pred.flatten().tolist()
        ex.predicted_relation = rel[0]
        fout.write(json.dumps(ex, sort_keys=True) + "\n")

        preds.append(ex.predicted_relation)
        targs.append(ex.relation)

        if i % 1000 == 0:
            print 'predicted', i, 'examples'

    with open(name + '.preds', 'wb') as f:
        print >> f, "targ\tpred"
        for targ, pred in zip(targs, preds):
            print >> f, dataset.vocab['rel'].index2word[targ] + "\t" + dataset.vocab['rel'].index2word[pred]

    from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score
    labels = [i for i in range(len(dataset.vocab['rel'])) if i != dataset.vocab['rel']['no_relation']]
    print classification_report(targs, preds)

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
