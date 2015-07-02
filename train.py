""" train.py
Usage: train.py <data> [--model=<MODEL>] [--optim=<OPTIM>] [--epoch=<EPOCH>] [--activation=<ACT>] [--hidden=<HID>]
                [--dropout=<RATE>] [--truncate_grad=<STEPs>] [--mode=<MODE>] [--lr=<LR>] [--reg=<REG>]


Options:
    --model=<MODEL>     [default: sent]
    --optim=<OPTIM>     [default: rmsprop]
    --epoch=<EPOCH>     [default: 50]
    --activation=<ACT>  [default: relu]
    --hidden=<HID>      [default: 100,100]
    --truncate_grad=<STEPS>     [default: 25]
    --dropout=<RATE>        [default: 0.5]
    --mode=<MODE>         [default: classification]
    --lr=<LR>               [default: 0.001]
    --reg=<REG>             [default: 0]
"""
import cPickle as pkl
import numpy as np
np.random.seed(42)
import os
import sys
import json
sys.setrecursionlimit(50000)
mydir = os.path.dirname(__file__)
sys.path.append(os.path.join(mydir, 'data'))
from data.dataset import AnnotatedData, one_hot, TypeCheckAdaptor
from models import *
from keras.layers.core import *
from keras.optimizers import *
from keras.regularizers import *
from keras.objectives import *
from theano import shared
from sklearn.metrics import f1_score, accuracy_score

mydir = os.path.dirname(__file__)

def get_model_from_arg(args, dataset):
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
        'sent_parse': sent_parse,
        'sent_parse_ner': sent_parse_ner,
    }[args['--model']]

    model, model_nout = get_model(dataset.vocab, dataset.word2emb, word_emb_dim, hidden, dropout, activation, truncate_grad, reg)
    n_out = len(dataset.vocab['rel']) if mode == 'classification' else 1

    def filtered_categorical_crossentropy(targ, pred):
        # return categorical_crossentropy(targ, T.nnet.softmax(pred))
        ner1 = T.cast(targ[:, -2], 'int32')
        ner2 = T.cast(targ[:, -1], 'int32')
        valid = typechecker.get_valid(ner1, ner2)
        y_pred = T.nnet.softmax(valid * pred)
        y_targ = targ[:, :-2]
        return categorical_crossentropy(y_targ, y_pred)

    def filtered_binary_crossentropy(targ, pred):
        y_targ = targ
        y_pred = pred
        # ner1 = T.cast(targ[:, -2], 'int32')
        # ner2 = T.cast(targ[:, -1], 'int32')
        # valid = T.gt(typechecker.get_valid(ner1, ner2).sum(axis=1), 0).reshape((-1, 1))
        # y_pred = pred * valid
        return binary_crossentropy(y_targ, y_pred)

    loss = filtered_categorical_crossentropy if mode == 'classification' else filtered_binary_crossentropy

    model.add(Dense(model_nout, n_out))
    if mode == 'filter':
        model.add(Activation('sigmoid'))
    model.compile(optim, loss=loss)
    return model


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

    sizes = {name:len(split) for name, split in dataset.splits.items()}
    print json.dumps(sizes)

    typechecker = TypeCheckAdaptor(dataset.vocab)
    os.chdir(todir)

    with open('args.json', 'wb') as f:
        json.dump(args, f)

    model = get_model_from_arg(args, dataset)

    best_f1, best_weights = 0, None
    best_loss = np.inf

    def get_XY(X, Y):
        Xwords, Xparse, Xner = X
        Xin = {
            'sent': Xwords,
            'ner': Xner,
            'parse': Xparse,
            'sent_ner': [Xwords, Xner],
            'sent_parse': [Xwords, Xparse],
            'sent_parse_ner': [Xwords, Xparse, Xner]
        }[args['--model']]
        Yin = np.concatenate([Y, Xner], axis=1) if args['--mode'] == 'classification' else Y
        return Xin, Yin, Xner

    def run_epoch(split, train=False):
        total = total_loss = 0
        func = model.train if train else model.test
        preds, targs = [], []
        for X, Y in dataset.generate_batches(split, label=args['--mode']):
            Xin, Yin, Xner = get_XY(X, Y)
            loss = func(Xin, Yin)
            if args['--mode'] == 'classification':
                pred = model.predict(Xin, verbose=0)
                pred *= typechecker.get_valid_cpu(Xner[:, 0], Xner[:, 1])
                pred = pred.argmax(axis=1)
                targs.append(Y.argmax(axis=1))
            else:
                pred = model.predict(Xin, verbose=0).flatten() > 0.5
                valid = typechecker.get_valid_cpu(Xner[:, 0], Xner[:, 1]).sum(axis=1) > 0
                pred = pred * valid
                targs.append(Y.flatten())
            preds.append(pred)
            total_loss += loss
            total += 1
        preds = np.concatenate(preds).astype('int32')
        targs = np.concatenate(targs).astype('int32')
        total_f1 = f1_score(targs, preds, average='micro') if args['--mode'] == 'classification' else f1_score(targs, preds)
        total_loss /= float(total)
        return total_f1, total_loss, preds, targs

    log = open('train.log', 'wb')
    for epoch in xrange(int(args['--epoch'])+1):

        train_f1, train_loss, preds, targs = run_epoch('train', train=True)
        dev_f1, dev_loss, preds, targs = run_epoch('dev', train=False)

        if epoch > 5 and dev_f1 > best_f1:
            best_f1 = dev_f1
            best_weights = model.get_weights()
            with open('best_weights.pkl', 'wb') as f:
                pkl.dump(best_weights, f, protocol=pkl.HIGHEST_PROTOCOL)

        d = json.dumps({'epoch': epoch, 'dev_loss': dev_loss, 'dev_f1': dev_f1,
                        'train_loss': train_loss, 'train_f1': train_f1}, sort_keys=True)
        print d
        log.write(d + "\n")

        if epoch % 10 == 0:
            np.save('weights.' + str(epoch), model.get_weights())
    log.close()

    model.set_weights(best_weights)
    test_f1, test_loss, preds, targs = run_epoch('test', train=False)

    from sklearn.metrics import f1_score

    d = {'test_loss': test_loss, 'test_f1': test_f1, 'dev_f1': best_f1}
    if args['--mode'] == 'classification':
        d['test_f1_macro'] = f1_score(targs, preds, average='macro')
        d['test_f1_micro'] = f1_score(targs, preds, average='micro')
    else:
        d['test_f1'] = f1_score(targs, preds)
    with open('test.json', 'wb') as f:
        json.dump(d, f)
    with open('test.pkl', 'wb') as f:
        pkl.dump({'pred': preds, 'targ': targs}, f, pkl.HIGHEST_PROTOCOL)
    pprint(d)

    with open('model.weights.pkl', 'wb') as f:
        pkl.dump(model.get_weights(), f, protocol=pkl.HIGHEST_PROTOCOL)
