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

