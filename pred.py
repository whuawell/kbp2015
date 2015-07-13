#!/usr/bin/env python
#!/u/nlp/packages/anaconda/bin/safepython
"""pred.py
Usage: pred.py <model_dir> [--evaluation=<DATA>] [--debug]

Options:
    --evaluation=<DATA>     [default: eval]

"""
import os
mydir = os.path.dirname(os.path.abspath(__file__))
from docopt import docopt
import numpy as np
from utils import np_softmax
from pprint import pprint
from configs.config import Config
from data.dataset import Dataset, Split
from data.adaptors import *
from data.typecheck import TypeCheckAdaptor
from models import get_model


if __name__ == '__main__':
    args = docopt(__doc__)

    root = os.path.abspath(args['<model_dir>'])
    config = Config.load(os.path.join(root, 'config.json'))
    dataset = Dataset.load(config.data)
    typechecker = TypeCheckAdaptor(os.path.join(mydir, 'data', 'raw', 'typecheck.csv'), dataset.featurizer.vocab)

    model = get_model(config, dataset.featurizer.vocab, typechecker)
    model.load_weights(os.path.join(root, 'best_weights'))

    dev_generator = {
        'kbp_eval': KBPEvaluationDataAdaptor().to_examples(os.path.join(mydir, 'data', 'raw', 'evaluation.tsv')),
        'supervised': SupervisedDataAdaptor().to_examples(os.path.join(mydir, 'data', 'raw', 'supervision.csv')),
        'online': KBPDataAdaptor().online_to_examples(disable_interrupts=False), #args['--evaluation'] == 'online'),
        'kbp_sample': KBPDataAdaptor().to_examples(os.path.join(mydir, 'data', 'raw', 'test.sample.tsv')),
    }[args['--evaluation']]

    if args['--evaluation'] == 'online':
        for ex in dev_generator:
            try:
                feat = dataset.featurizer.featurize(ex, add=False)
            except Exception as e:
                continue
            X, Y, types = dataset.featurizer.to_matrix([feat])
            prob = model.predict(X, verbose=0)['p_relation']
            prob *= typechecker.get_valid_cpu(types[:, 0], types[:, 1])
            pred = prob.argmax(axis=1)
            confidence = np_softmax(prob.flatten())[pred[0]]
            if args['--debug']:
                print "\t".join([str(s) for s in [ex.subject, dataset.featurizer.vocab['rel'].index2word[pred[0]], ex.object, confidence, dataset.featurizer.decode_sequence(feat)]])
            else:
                print "\t".join([str(s) for s in [ex.subject_id, pred[0], ex.object_id, confidence]])
    else:
        from train import Trainer
        dev_split = Split(dev_generator, dataset.featurizer, add=False)
        scoring_labels = [i for i in xrange(len(dataset.featurizer.vocab['rel'])) if i != dataset.featurizer.vocab['rel']['no_relation']]
        trainer = Trainer('.', model, typechecker, scoring_labels)
        best_scores = trainer.run_epoch(dev_split, train=False, return_pred=True)

        import json
        with open('preds.json', 'wb') as f:
            json.dump({k: v for k, v in best_scores.items() if k in ['preds', 'targs', 'ids']}, f)
        with open('pred_scores.json', 'wb') as f:
            del best_scores['preds']
            del best_scores['targs']
            del best_scores['ids']
            json.dump(best_scores, f, sort_keys=True)
        print 'best scores'
        pprint(best_scores)
