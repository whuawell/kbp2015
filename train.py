""" train.py
Usage: train.py <name> [--config=<CONFIG>] [--options=<KWARGS>]

Options:
    --config=<CONFIG>     [default: default]
    --options=<KWARGS>    key value pair options like --options=train:supervised,dev:kbp_eval   [default: ]
"""
import os
import sys
import json
sys.setrecursionlimit(50000)
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from configs.config import Config
from data.dataset import *
from data.adaptors import *
from data.featurizers import *
from data.typecheck import *
from data.pretrain import Senna
from time import time
from keras.utils.generic_utils import Progbar
import cPickle as pkl

import autograd.numpy as np
from autograd import value_and_grad
from autograd.util import quick_grad_check
np.random.seed(42)


mydir = os.path.dirname(__file__)
sys.path.append(os.path.join(mydir, 'data'))

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

    def train(self, train_split, dev_split=None, max_epoch=150):
        best_scores, best_weights, dev_scores = {}, None, None
        compare = 'precision'
        best_scores[compare] = 0.

        for epoch in xrange(max_epoch+1):
            start = time()
            print 'starting epoch', epoch
            print 'training...'
            train_result = self.run_epoch(train_split, True)
            if dev_split:
                print 'evaluating...'
                dev_scores = self.run_epoch(dev_split, False)
            scores = {'train': train_result, 'dev': dev_scores, 'epoch': epoch, 'time': time()-start}
            pprint(scores)
            self.log('train', scores)
            if dev_scores is not None:
                if dev_scores[compare] > best_scores[compare] and dev_scores['f1'] > 0.3:
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
    if args['--options']:
        for spec in args['--options'].split(','):
            spec = spec.split(':')
            assert len(spec) == 2, 'invalid option specified: %' % spec
            k, v = spec
            if isinstance(config[k], int): v = int(v)
            if isinstance(config[k], float): v = float(v)
            config[k] = v

    if 'data' not in config:
        config.data = '_'.join([config.train, config.dev, config.featurizer, 'corrupt' + str(config.num_corrupt)])
    if os.path.isdir(os.path.join(mydir, 'data', 'saves', config.data)):
        dataset = Dataset.load(os.path.join(mydir, 'data', 'saves', config.data))
    else:
        datasets = {
            'supervised': SupervisedDataAdaptor(),
            'kbp_eval': KBPEvaluationDataAdaptor(),
            'all_annotated': AllAnnotatedAdaptor(),
            'self_training': SelfTrainingAdaptor(),
        }
        train_generator = datasets[config.train].to_examples()
        dev_generator = datasets[config.dev].to_examples()
        featurizer = SinglePathSentenceFeaturizer(word=Senna())
        dataset = Dataset.build(train_generator, dev_generator, featurizer, num_corrupt=config.num_corrupt)
        dataset.save(os.path.join(mydir, 'data', 'saves', config.data))
    print 'using train split', dataset.train, 'of size', len(dataset.train)
    print 'using dev split', dataset.dev, 'of size', len(dataset.dev)
    print 'using featurizer', dataset.featurizer
    print 'using config'
    pprint(config)

    name = os.path.join('experiments', args['<name>'])
    todir = os.path.join(mydir, name)
    if not os.path.isdir(todir):
        os.makedirs(todir)
    print 'saving'
    config.save(os.path.join(todir, 'config.json'))
    with open(os.path.join(todir, 'featurizer.pkl'), 'wb') as f:
        pkl.dump(dataset.featurizer, f, protocol=pkl.HIGHEST_PROTOCOL)

    typechecker = TypeCheckAdaptor(os.path.join(mydir, 'data', 'raw', 'typecheck.csv'), dataset.featurizer.vocab)
    scoring_labels = [i for i in xrange(len(dataset.featurizer.vocab['rel'])) if i != dataset.featurizer.vocab['rel']['no_relation']]

    invalids = dataset.train.remove_invalid_examples(typechecker)
    print 'removed', len(invalids), 'invalid training examples'
    invalids = dataset.dev.remove_invalid_examples(typechecker)
    print 'removed', len(invalids), 'invalid dev examples'

    word_vocab = dataset.featurizer.vocab['word']
    rel_vocab = dataset.featurizer.vocab['rel']
    # when computing precison/recall/f1, do not take into account no_relation
    valid_classes = [i for i in xrange(len(rel_vocab)) if i != rel_vocab['no_relation']]

    from pystacks.param import ParamServer
    from pystacks.layers.recurrent import LSTMMemoryLayer
    from pystacks.layers.embedding import LookupTable
    from pystacks.regularizers import Dropout
    from pystacks.layers.core import Dense
    from pystacks.layers.activations import LogSoftmax
    from pystacks.initialization import Hardcode
    from pystacks.optimizers import *
    from pystacks.utils.logging import Progbar
    from pystacks.utils.math import make_batches_by_len
    from pystacks.grad_transformer import *

    server = ParamServer()

    emb_size = 50
    state_size = 128
    num_epoch = 20
    batch_size = 128
    learning_rate = 1e-2

    lookup_layer = LookupTable(len(word_vocab), emb_size)
    lookup_layer.E.init = Hardcode(word_vocab.load_embeddings())
    lookup_layer.register_params(server)
    lstm_layer = LSTMMemoryLayer(emb_size, state_size, dropout=Dropout(0.5)).register_params(server)

    rel_output_layer = Dense(state_size, len(rel_vocab)).register_params(server)
    rel_softmax_layer = LogSoftmax().register_params(server)

    word_output_layer = Dense(state_size, len(word_vocab)).register_params(server)
    word_softmax_layer = LogSoftmax().register_params(server)

    server.finalize()

    def pred_fun(weights, x, types, train=False):
        server.param_vector = weights
        ht, ct = None, None
        output_rel, output_word = [], []
        for t in xrange(x.shape[1]):
            emb = lookup_layer.forward(x[:, t], train=train)
            ht, ct = lstm_layer.forward(emb, ht, ct, train=train)
            pt_rel = rel_output_layer.forward(ht)
            pt_rel = pt_rel * typechecker.get_valid_cpu(types[:, 0], types[:, 1])

            output_rel.append(rel_softmax_layer.forward(pt_rel))
            pt_word = word_output_layer.forward(ht)
            output_word.append(word_softmax_layer.forward(pt_word))
        return output_rel, output_word # Output normalized log-probabilities.

    def loss_fun(weights, x, targets, types, train=False):
        logprobs_rel, logprobs_word = pred_fun(weights, x, types, train=train)
        loss_sum = - np.sum(logprobs_rel[-1] * targets)
        for t in xrange(len(targets)-1):
            loss_sum -= np.sum(logprobs_word[t][np.arange(len(x[:, t+1])), x[:, t+1]])
        return loss_sum / float(x.shape[0])

    def score(weights, inputs, targets, types, train=False):
        targs = np.argmax(targets, axis=-1)
        logprobs_rel, logprobs_word = pred_fun(weights, inputs, types, train)
        preds = np.argmax(logprobs_word[-1], axis=-1)
        acc = accuracy_score(targs, preds)
        return targs, preds, acc
        # return {'acc': acc, 'precision':precision, 'recall':recall, 'f1':f1}

    # Build gradient of loss function using autograd.
    loss_and_grad = value_and_grad(loss_fun)

    # Check the gradients numerically, just to be safe
    idx, x, y, types = dataset.train.batches(1).next()
    quick_grad_check(loss_fun, server.param_vector, (x, y, types))

    print("Training LSTM...")
    optimizer = Adam()
    start = time()
    for epoch in xrange(1, num_epoch+1):
        epoch_loss = 0.
        targs, preds, num_seen = [], [], 0
        print 'epoch', epoch
        bar = Progbar('train', track=['loss', 'acc'])
        for idx, x, y, types in dataset.train.batches(batch_size):
            loss, dparams = loss_and_grad(server.param_vector, x, y, types, train=True)
            epoch_loss += loss
            num_seen += len(y)
            server.update_params(optimizer, dparams, learning_rate)
            targs_, preds_, acc = score(server.param_vector, x, y, types, train=False)
            targs.append(targs_)
            preds.append(preds_)
            bar.update(num_seen/float(len(dataset.train)), new_values={'loss':loss, 'acc':acc})
        bar.finish()

        targs = np.concatenate(targs)
        preds = np.concatenate(preds)
        pprint({
            'precison': precision_score(targs, preds, labels=valid_classes, average='micro'),
            'recall': recall_score(targs, preds, labels=valid_classes, average='micro'),
            'f1': f1_score(targs, preds, labels=valid_classes, average='micro')})

        targs, preds, num_seen = [], [], 0
        bar = Progbar('eval', track=['acc'])
        for idx, x, y, types in dataset.dev.batches(batch_size):
            targs_, preds_, acc = score(server.param_vector, x, y, train=False)
            targs.append(targs_)
            preds.append(preds_)
            num_seen += len(y)
            bar.update(num_seen/float(len(dataset.dev)), new_values={'acc': acc})
        bar.finish()

        targs = np.concatenate(targs)
        preds = np.concatenate(preds)
        pprint({
            'precison': precision_score(targs, preds, labels=valid_classes, average='micro'),
            'recall': recall_score(targs, preds, labels=valid_classes, average='micro'),
            'f1': f1_score(targs, preds, labels=valid_classes, average='micro')})

        print("epoch %s train loss %s in %s" % (epoch, epoch_loss, time() - start))
        start = time()