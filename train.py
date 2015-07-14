""" train.py
Usage: train.py <name> [--config=<CONFIG>]

Options:
    --config=<CONFIG>     [default: default]
"""
import numpy as np
np.random.seed(42)
import os
import sys
import json
sys.setrecursionlimit(50000)
mydir = os.path.dirname(__file__)
sys.path.append(os.path.join(mydir, 'data'))
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from models import get_model
from configs.config import Config
from data.dataset import *
from data.adaptors import *
from data.featurizers import *
from data.typecheck import *
from data.pretrain import Senna
from time import time
from keras.utils.generic_utils import Progbar


class Trainer(object):

    def __init__(self, log_dir, model, typechecker, labels):
        self.model, self.typechecker, self.labels = model, typechecker, labels
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir
        self.logs = {
            'train': open(os.path.join(self.log_dir, 'train.log'), 'wb'),
        }

    def log(self, name, payload):
        self.logs[name].write(json.dumps(payload, sort_keys=True) + "\n")

    def run_epoch(self, split, train=False, batch_size=128, return_pred=False):
        total = total_loss = 0
        func = self.model.train_on_batch if train else self.model.test_on_batch
        ids, preds, targs = [], [], []
        prog = Progbar(split.num_examples)
        for idx, X, Y, types in split.batches(batch_size):
            X.update({k: np.concatenate([v, types], axis=1) for k, v in Y.items()})
            batch_end = time()
            loss = func(X)
            prob = self.model.predict(X, verbose=0)['p_relation']
            prob *= self.typechecker.get_valid_cpu(types[:, 0], types[:, 1])
            pred = prob.argmax(axis=1)

            targ = Y['p_relation'].argmax(axis=1)
            ids.append(idx)
            targs.append(targ)
            preds.append(pred)
            total_loss += loss
            total += 1
            prog.add(idx.size, values=[('loss', loss), ('acc', np.mean(pred==targ))])
        preds = np.concatenate(preds).astype('int32')
        targs = np.concatenate(targs).astype('int32')
        ids = np.concatenate(ids).astype('int32')

        ret = {
            'f1': f1_score(targs, preds, average='micro', labels=self.labels),
            'precision': precision_score(targs, preds, average='micro', labels=self.labels),
            'recall': recall_score(targs, preds, average='micro', labels=self.labels),
            'accuracy': accuracy_score(targs, preds),
            'loss': total_loss / float(total),
            }
        if return_pred:
            ret.update({'ids': ids.tolist(), 'preds': preds.tolist(), 'targs': targs.tolist()})
        return ret

    def run_epoch_stochastic(self, split, train=False, batch_size=128, return_pred=False, losses=None):
        if not train:
            return self.run_epoch(split, train, batch_size, return_pred)

        total = total_loss = 0
        ids, preds, targs = [], [], []
        prog = Progbar(split.num_examples)
        for i in xrange(split.num_examples):
            if losses is None:
                idx, X, Y, types = split.stochastic_curriculum_class_based()
            else:
                idx, X, Y, types = split.stochastic_curriculum(losses)
            X.update({k: np.concatenate([v, types], axis=1) for k, v in Y.items()})
            batch_end = time()
            loss = self.model.train_on_batch(X)
            if losses is not None:
                losses[i] = loss
            prob = self.model.predict(X, verbose=0)['p_relation']
            prob *= self.typechecker.get_valid_cpu(types[:, 0], types[:, 1])
            pred = prob.argmax(axis=1)

            targ = Y['p_relation'].argmax(axis=1)
            ids.append(idx)
            targs.append(targ)
            preds.append(pred)
            total_loss += loss
            total += 1
            prog.add(len(idx), values=[('loss', loss), ('acc', np.mean(pred==targ))])
        preds = np.concatenate(preds).astype('int32')
        targs = np.concatenate(targs).astype('int32')
        ids = np.concatenate(ids).astype('int32')

        ret = {
            'f1': f1_score(targs, preds, average='micro', labels=self.labels),
            'precision': precision_score(targs, preds, average='micro', labels=self.labels),
            'recall': recall_score(targs, preds, average='micro', labels=self.labels),
            'accuracy': accuracy_score(targs, preds),
            'loss': total_loss / float(total),
            }
        if return_pred:
            ret.update({'ids': ids.tolist(), 'preds': preds.tolist(), 'targs': targs.tolist()})
        return ret

    def train(self, train_split, dev_split=None, max_epoch=150):
        best_scores, best_weights, dev_scores = {}, None, None
        compare = 'precision'
        best_scores[compare] = 0.

        losses = np.ones(train_split.num_examples) * 100.
        for epoch in xrange(max_epoch+1):
            start = time()
            print 'starting epoch', epoch
            print 'training...'
            # train_result = self.run_epoch_stochastic(train_split, True, losses=losses)
            train_result = self.run_epoch(train_split, True)
            if dev_split:
                print 'evaluating...'
                dev_scores = self.run_epoch(dev_split, False)
            scores = {'train': train_result, 'dev': dev_scores, 'epoch': epoch, 'time': time()-start}
            pprint(scores)
            self.log('train', scores)
            if dev_scores is not None:
                if dev_scores[compare] > best_scores[compare]:
                    best_scores = dev_scores.copy()
                    best_weights = self.model.get_weights()

        if best_weights is not None:
            self.model.set_weights(best_weights)
            scores = self.run_epoch(dev_split, False, return_pred=True)
            assert scores[compare] == best_scores[compare]
            best_scores = scores

        return best_scores


if __name__ == '__main__':
    from docopt import docopt
    from pprint import pprint
    args = docopt(__doc__)
    pprint(args)

    config = Config.default() if args['--config'] == 'default' else Config.load(args['--config'])

    if not os.path.isdir(config.data):
        train_generator = SupervisedDataAdaptor().to_examples('data/raw/supervision.csv')
        dev_generator = KBPEvaluationDataAdaptor().to_examples('data/raw/evaluation.tsv')
        featurizer = ConcatenatedFeaturizer(word=Senna()) if 'concat' in config.model else SinglePathFeaturizer(word=Senna())
        dataset = Dataset.build(train_generator, dev_generator, featurizer)
        dataset.save(config.data)
    else:
        dataset = Dataset.load(config.data)
    print 'using train split', dataset.train
    print 'using dev split', dataset.dev
    print 'using featurizer', dataset.featurizer
    print 'using config'
    pprint(config)

    name = os.path.join('experiments', args['<name>'])
    todir = os.path.join(mydir, name)
    if not os.path.isdir(todir):
        os.makedirs(todir)
    config.save(os.path.join(todir, 'config.json'))

    typechecker = TypeCheckAdaptor(os.path.join(mydir, 'data', 'raw', 'typecheck.csv'), dataset.featurizer.vocab)
    scoring_labels = [i for i in xrange(len(dataset.featurizer.vocab['rel'])) if i != dataset.featurizer.vocab['rel']['no_relation']]

    invalids = dataset.train.remove_invalid_examples(typechecker)
    print 'removed', len(invalids), 'invalid training examples'
    invalids = dataset.dev.remove_invalid_examples(typechecker)
    print 'removed', len(invalids), 'invalid dev examples'

    model = get_model(config, dataset.featurizer.vocab, typechecker)
    trainer = Trainer(todir, model, typechecker, scoring_labels)
    best_scores = trainer.train(dataset.train, dataset.dev, max_epoch=config.max_epoch)

    model.save_weights(os.path.join(todir, 'best_weights'), overwrite=True)

    with open(os.path.join(todir, 'classification_report.txt'), 'wb') as f:
        report = classification_report(best_scores['targs'], best_scores['preds'], target_names=dataset.featurizer.vocab['rel'].index2word)
        f.write(report)
    print report

    from plot_utils import plot_confusion_matrix, plot_histogram, get_sorted_labels
    order, labels, counts = get_sorted_labels(best_scores['targs'], dataset.featurizer.vocab)
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

