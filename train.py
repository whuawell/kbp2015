""" train.py
Usage: train.py <data> [--optim=<OPTIM>] [--epoch=<EPOCH>] [--activation=<ACT>] [--hidden=<HID>] [--reg=<REG>]
                [--dropout=<RATE>] [--truncate_grad=<STEPs>] [--model=<MODEL>] [--lr=<LR>]


Options:
    --optim=<OPTIM>     [default: adagrad]
    --epoch=<EPOCH>     [default: 50]
    --activation=<ACT>  [default: relu]
    --hidden=<HID>      [default: 300,300]
    --truncate_grad=<STEPS>     [default: 50]
    --dropout=<RATE>        [default: 0.70]
    --model=<MODEL>         [default: classification]
    --lr=<LR>               [default: 0.1]
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
from models import sent
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

    num_rel, num_word = len(dataset.rel_vocab), len(dataset.word_vocab)
    word_emb_dim = len(dataset.word2emb.values()[0])
    hidden = [int(d) for d in args['--hidden'].split(',')]
    dropout, lr = [float(args[d]) for d in ['--dropout', '--lr']]
    max_epoch, truncate_grad = [int(args[d]) for d in ['--epoch', '--truncate_grad']]
    activation = args['--activation']
    mode = args['--model']

    optim = {
        'adagrad': adagrad(lr=lr, clipnorm=5.),
        'rmsprop': rmsprop(lr=lr, clipnorm=5.),
        'sgd': sgd(lr=lr, momentum=0.9, decay=1e-7, nesterov=True, clipnorm=5.),
    }[args['--optim']]

    model = sent(dataset.word_vocab, dataset.word2emb, num_word, word_emb_dim, hidden, dropout, activation, truncate_grad)
    n_out = len(dataset.rel_vocab) if mode == 'classification' else 1
    out_layer = Activation('softmax') if mode == 'classification' else Activation('sigmoid')
    loss = 'categorical_crossentropy' if mode == 'classification' else 'binary_crossentropy'

    model.add(Dense(hidden[-1], n_out))
    model.add(out_layer)
    model.compile(optim, loss=loss)

    best_acc, best_weights = 0, None
    best_loss = np.inf

    log = open('train.log', 'wb')
    for epoch in xrange(max_epoch+1):
        train_total = dev_total = train_loss = train_acc = dev_loss = dev_acc = 0
        for X, Y in dataset.generate_batches('train', label=mode):
            if mode == 'classification':
                loss, acc = model.train(X, Y, accuracy=True)
            else:
                loss = model.train(X, Y)
                pred = model.predict(X, verbose=0).flatten() > 0.5
                acc = np.mean(pred == Y.flatten())
            train_loss += loss
            train_acc += acc
            train_total += 1
        train_acc /= float(train_total)
        train_loss /= float(train_total)

        for X, Y in dataset.generate_batches('dev', label=mode):
            if mode == 'classification':
                loss, acc = model.test(X, Y, accuracy=True)
            else:
                loss = model.test(X, Y)
                pred = model.predict(X, verbose=0).flatten() > 0.5
                acc = np.mean(pred == Y.flatten())
            dev_loss += loss
            dev_acc += acc
            dev_total += 1
        dev_loss /= float(dev_total)
        dev_acc /= float(dev_total)

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
        if mode == 'classification':
            loss, acc = model.test(X, Y, accuracy=True)
            pred = model.predict_classes(X, verbose=0)
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

    with open('model.pkl', 'wb') as f:
        pkl.dump(model, f, protocol=pkl.HIGHEST_PROTOCOL)

