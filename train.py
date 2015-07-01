""" train.py
Usage: train.py <data> [--model=<MODEL>] [--optim=<OPTIM>] [--epoch=<EPOCH>] [--activation=<ACT>] [--hidden=<HID>]
                [--dropout=<RATE>] [--truncate_grad=<STEPs>] [--mode=<MODE>] [--lr=<LR>] [--reg=<REG>]


Options:
    --model=<MODEL>     [default: sent]
    --optim=<OPTIM>     [default: rmsprop]
    --epoch=<EPOCH>     [default: 50]
    --activation=<ACT>  [default: tanh]
    --hidden=<HID>      [default: 100,100]
    --truncate_grad=<STEPS>     [default: 25]
    --dropout=<RATE>        [default: 0.50]
    --mode=<MODE>         [default: classification]
    --lr=<LR>               [default: 0.001]
    --reg=<REG>             [default: 1e-3]
"""
import cPickle as pkl
import numpy as np
np.random.seed(42)
import os
import sys
sys.setrecursionlimit(50000)
mydir = os.path.dirname(__file__)
sys.path.append(os.path.join(mydir, 'data'))
from data.dataset import AnnotatedData
from models import *
from keras.layers.core import *
from keras.optimizers import *
from keras.regularizers import *

def one_hot(y, num_classes):
    Y = np.zeros((len(y), num_classes), dtype='float32')
    Y[np.arange(len(y)), y] = 1.
    return Y

mydir = os.path.dirname(__file__)

if __name__ == '__main__':
    from docopt import docopt
    from pprint import pprint
    args = docopt(__doc__)
    pprint(args)

    name = '_'.join([str(args[k]).strip('/').replace('/', '_') for k in sorted(args.keys())])
    todir = os.path.join(mydir, name)
    if not os.path.isdir(todir):
        os.makedirs(todir)

    dataset = AnnotatedData.load(args['<data>'])
    os.chdir(todir)

    import json
    with open('args.json', 'wb') as f:
        json.dump(args, f)

    word_emb_dim = len(dataset.word2emb.values()[0])
    hidden = [int(d) for d in args['--hidden'].split(',')]
    dropout, lr, reg = [float(args[d]) for d in ['--dropout', '--lr', '--reg']]
    max_epoch, truncate_grad = [int(args[d]) for d in ['--epoch', '--truncate_grad']]
    activation = args['--activation']
    mode = args['--mode']

    optim = {
        'adagrad': adagrad(lr=lr, clipnorm=5.),
        'rmsprop': rmsprop(lr=lr, clipnorm=5.),
        'sgd': sgd(lr=lr, momentum=0.9, decay=1e-7, nesterov=True, clipnorm=5.),
        'adadelta': adadelta(lr=lr, clipnorm=5.)
    }[args['--optim']]

    get_model = {
        'ner': ner,
        'sent': sent,
        'parse': parse,
        'sent_ner': sent_ner,
        'sent_parse_ner': sent_parse_ner,
    }[args['--model']]

    model, model_nout = get_model(dataset.vocab, dataset.word2emb, word_emb_dim, hidden, dropout, activation, truncate_grad, reg)
    n_out = len(dataset.vocab['rel']) if mode == 'classification' else 1
    out_layer = Activation('softmax') if mode == 'classification' else Activation('sigmoid')
    loss = 'categorical_crossentropy' if mode == 'classification' else 'binary_crossentropy'

    model.add(Dense(model_nout, n_out, W_regularizer=l2(reg)))
    model.add(out_layer)
    model.compile(optim, loss=loss)

    best_acc, best_weights = 0, None
    best_loss = np.inf

    def get_input(X):
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

    def run_epoch(split, train=False):
        total = total_acc = total_loss = 0
        func = model.train if train else model.test
        for X, Y in dataset.generate_batches(split, label=mode):
            Xin = get_input(X)
            if mode == 'classification':
                loss, acc = func(Xin, Y, accuracy=True)
            else:
                loss = func(Xin, Y)
                pred = model.predict(Xin, verbose=0).flatten() > 0.5
                acc = np.mean(pred == Y.flatten())
            total_loss += loss
            total_acc += acc
            total += 1
        total_acc /= float(total)
        total_loss /= float(total)
        return total_acc, total_loss

    log = open('train.log', 'wb')
    for epoch in xrange(max_epoch+1):

        train_acc, train_loss = run_epoch('train', train=True)
        dev_acc, dev_loss = run_epoch('dev', train=False)

        if epoch > 5 and dev_acc > best_acc:
            best_acc = dev_acc
            best_weights = model.get_weights()
            with open('best_weights.pkl', 'wb') as f:
                pkl.dump(best_weights, f, protocol=pkl.HIGHEST_PROTOCOL)

        d = json.dumps({'epoch': epoch, 'dev_loss': dev_loss, 'dev_acc': dev_acc,
                        'train_loss': train_loss, 'train_acc': train_acc}, sort_keys=True)
        print d
        log.write(d + "\n")

        if epoch % 10 == 0:
            np.save('weights.' + str(epoch), model.get_weights())
    log.close()

    model.set_weights(best_weights)
    test_total = test_loss = test_acc = 0
    preds = []
    targs = []
    for X, Y in dataset.generate_batches('test', label=mode):
        Xin = get_input(X)
        if mode == 'classification':
            loss, acc = model.test(Xin, Y, accuracy=True)
            pred = model.predict_classes(Xin, verbose=0)
            targs.append(Y.argmax(axis=1))
        else:
            loss = model.test(X, Y)
            pred = model.predict(X, verbose=0).flatten() > 0.5
            acc = np.mean(pred == Y.flatten())
            targs.append(Y.flatten())
        preds.append(pred)
        test_loss += loss
        test_acc += acc
        test_total += 1
    test_loss /= float(test_total)
    test_acc /= float(test_total)
    preds = np.concatenate(preds)
    targs = np.concatenate(targs)

    from sklearn.metrics import f1_score

    d = {'test_loss': test_loss, 'test_acc': test_acc, 'dev_acc': best_acc}
    if mode == 'classification':
        d['test_f1_macro'] = f1_score(targs, preds, average='macro')
        d['test_f1_micro'] = f1_score(targs, preds, average='micro')
    else:
        d['test_f1'] = f1_score(targs, preds)
    with open('test.json', 'wb') as f:
        json.dump(d, f)
    with open('test.pkl', 'wb') as f:
        pkl.dump({'pred': preds, 'targ': targs}, f, pkl.HIGHEST_PROTOCOL)
    pprint(d)

    def process_for_pickle(model):
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                process_for_pickle(layer)
            else:
                layer.constraints = layer.regularizers = []
    process_for_pickle(model)

    with open('model.pkl', 'wb') as f:
        pkl.dump(model, f, protocol=pkl.HIGHEST_PROTOCOL)

