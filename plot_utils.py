__author__ = 'victor'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as P
from sklearn.metrics import confusion_matrix
from collections import Counter
import numpy as np

def get_sorted_labels(targs, vocab):
    counter = Counter(targs)
    order = [i for i, count in counter.most_common()]
    labels = [vocab['rel'].index2word[i] for i in order]
    counts = [count for i, count in counter.most_common()]
    return order, labels, counts

def plot_confusion_matrix(targs, preds, order, labels, title='Confusion Matrix'):
    fig, ax = P.subplots(figsize=(10, 10))
    cm = confusion_matrix(targs, preds, labels=order)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    handle = ax.imshow(cm, interpolation='nearest', cmap=matplotlib.cm.Blues)
    ax.set_title(title)
    P.colorbar(handle, ax=ax)
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    return fig


def plot_histogram(labels, counts, title='Relation Histogram'):
    fig, ax = P.subplots(figsize=(10, 5))
    tick_marks = np.arange(len(labels))
    ax.bar(tick_marks, counts)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_xlabel('relation')
    ax.set_ylabel('counts')
    return fig

from collections import OrderedDict
from tabulate import tabulate

def parse_sklearn_report(str_report):
    report = OrderedDict()
    lines = str_report.splitlines()[2:-2] # first two lines are headers, last two lines are averages
    for line in lines:
        #                         no_relation       0.86      0.34      0.49      6191
        relation, precision, recall, f1, support = line.split()
        precision, recall, f1 = ["{:.2%}".format(float(e)) for e in [precision, recall, f1]]
        report[relation] = (precision, recall, f1, support)
    return report

def parse_gabor_report(str_report):
    report = OrderedDict()
    for line in str_report.splitlines():
        # [org:number_of_employees/members]  #: 9  P: 100.00%  R: 0.00%  F1: 0.00%
        relation, _, support, _, precision, _, recall, _, f1 = line.split()
        report[relation.strip('[]')] = (precision, recall, f1, support)
    return report

def combine_report(report, gabor, train_count):
    header = [
        'relation', 'nn_precision', 'nn_recall', 'nn_f1', 'nn_support',
        'sup_precision', 'sup_recall', 'sup_f1', 'sup_support',
        'train_support'
    ]
    table = []
    for relation in report.keys():
        precision, recall, f1, support = report[relation]
        g_precision, g_recall, g_f1, g_support = gabor[relation] if relation in gabor else 4 * ['N/A']
        row = [relation, precision, recall, f1, support, g_precision, g_recall, g_f1, g_support, train_count[relation]]
        table += [row]

    return tabulate(table, headers=header)

def retrieve_wrong_examples(examples, ids, preds, targs, vocab):
    examples_by_ids = {e.id:e for e in examples}
    wrongs = []
    for idx, pred, targ in zip(ids, preds, targs):
        if pred != targ:
            ex = examples_by_ids[idx]
            debug = {
                    'pred': vocab['rel'].index2word[pred],
                    'targ': vocab['rel'].index2word[targ],
                    'sequence': ' '.join([vocab['word'].index2word[w] for w in ex.sequence]),
                    'sentence': ' '.join(ex.orig.words),
                    'subj': ex.orig.subject,
                    'obj': ex.orig.object,
                    'length': len(ex.sequence),
                    }
            for k in ['sentence', 'sequence']:
              debug[k] = debug[k].replace(ex.orig.subject, '***' + ex.orig.subject + '***')
              debug[k] = debug[k].replace(ex.orig.object, '***' + ex.orig.object + '***')

            wrongs += [debug]
    return wrongs
