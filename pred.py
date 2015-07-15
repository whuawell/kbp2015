#!/usr/bin/env python
#!/u/nlp/packages/anaconda/bin/safepython
"""pred.py
Usage: pred.py <model_dir> [--evaluation=<DATA>] [--debug]

Options:
    --evaluation=<DATA>     [default: kbp_eval]

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
        'kbp_sample': KBPDataAdaptor().to_examples(os.path.join(mydir, 'data', 'raw', 'test.sample.tsv')),
    }[args['--evaluation']]

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
