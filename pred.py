#!/usr/bin/env python
#!/u/nlp/packages/anaconda/bin/safepython
"""pred.py
Usage: pred.py <model_dir> [--evaluation=<DATA>] [--debug]

Options:
    --evaluation=<DATA>     [default: kbp_eval]

"""
import os
from docopt import docopt
import numpy as np
from utils import np_softmax
from pprint import pprint
from configs.config import Config
from data.dataset import Dataset, Split
from data.adaptors import *
from data.typecheck import TypeCheckAdaptor
from models import get_model
import cPickle as pkl


if __name__ == '__main__':
    mydir = os.path.dirname(os.path.abspath(__file__))
    args = docopt(__doc__)

    root = os.path.abspath(args['<model_dir>'])
    config = Config.load(os.path.join(root, 'config.json'))
    with open(os.path.join(root, 'featurizer.pkl')) as f:
        featurizer = pkl.load(f)
    
    typechecker = TypeCheckAdaptor(os.path.join(mydir, 'data', 'raw', 'typecheck.csv'), featurizer.vocab)

    model = get_model(config, featurizer.vocab, typechecker)
    model.load_weights(os.path.join(root, 'best_weights'))

    dev_generator = {
        'kbp_eval': KBPEvaluationDataAdaptor().to_examples(os.path.join(mydir, 'data', 'raw', 'evaluation.tsv')),
        'supervised': SupervisedDataAdaptor().to_examples(os.path.join(mydir, 'data', 'raw', 'supervision.csv')),
        'kbp_sample': KBPDataAdaptor().to_examples(os.path.join(mydir, 'data', 'raw', 'test.sample.tsv')),
    }[args['--evaluation']]

    from train import Trainer
    dev_split = Split(dev_generator, featurizer, add=False)
    scoring_labels = [i for i in xrange(len(featurizer.vocab['rel'])) if i != featurizer.vocab['rel']['no_relation']]
    trainer = Trainer('.', model, typechecker, scoring_labels)
    best_scores = trainer.run_epoch(dev_split, train=False, return_pred=True)

    todir = os.path.join(root, 'preds')
    if not os.path.isdir(todir):
        os.makedirs(todir)
    print 'predictions output at', todir

    from plot_utils import plot_confusion_matrix, plot_histogram, get_sorted_labels, parse_gabor_report, parse_sklearn_report, combine_report, retrieve_wrong_examples
    import json
    from sklearn.metrics import classification_report

    wrongs = retrieve_wrong_examples(dev_split.examples,
                                     best_scores['ids'],
                                     best_scores['preds'],
                                     best_scores['targs'],
                                     featurizer.vocab
    )
    with open(os.path.join(todir, 'wrongs.json'), 'wb') as f:
        json.dump(wrongs, f, indent=2, sort_keys=True)

    sklearn_report = classification_report(
        best_scores['targs'], best_scores['preds'],
        target_names=featurizer.vocab['rel'].index2word)
    with open(os.path.join(mydir, 'data', 'raw', 'gabor_report.txt')) as f:
        gabor = f.read()
    gabor_report = parse_gabor_report(gabor)
    sklearn_report = parse_sklearn_report(str(sklearn_report))
    combined_report = combine_report(sklearn_report, gabor_report, featurizer.vocab['rel'].counts)

    with open(os.path.join(todir, 'classification_report.txt'), 'wb') as f:
        f.write(combined_report)

    order, labels, counts = get_sorted_labels(best_scores['targs'], featurizer.vocab)
    fig = plot_confusion_matrix(best_scores['targs'], best_scores['preds'], order, labels)
    fig.savefig(os.path.join(todir, 'confusion_matrix.png'))

    fig = plot_histogram(labels, counts)
    fig.savefig(os.path.join(todir, 'relation_histogram.png'))

    with open(os.path.join(todir, 'best_scores.json'), 'wb') as f:
        del best_scores['preds']
        del best_scores['targs']
        del best_scores['ids']
        json.dump(best_scores, f, sort_keys=True)
    print 'best scores'
    pprint(best_scores)

    print 'best scores'
    pprint(best_scores)
