#!/u/nlp/packages/anaconda/bin/safepython
import cPickle as pkl
import numpy as np
import json
import os
import signal
import fileinput
from pprint import pprint

from data.dataset import get_first_n_words, get_last_n_words

context = 0
filter_dir = os.path.join('deploy/filter')
classifier_dir = os.path.join('deploy/classifier')
filter_threshold = 0.7

def load_model(folder):
    with open(os.path.join(folder, 'model.pkl')) as f:
        return pkl.load(f)

mydir = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':
    # hack: disable keyboard interrupt (something funky with greenplum?)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    vocab_file = os.path.join('deploy/vocabs.pkl')
    with open(vocab_file) as f:
        vocab = pkl.load(f)
    filter_model = load_model(filter_dir)
    classifier = load_model(classifier_dir)

    for line in fileinput.input():
        line = line.strip()
        fields = line.split("\t")
        sent, dep, words, lemmas, pos_tags, ner_tags, s_id, s_ent, s_link_score, s_ner, o_id, o_ent, o_link_score, o_ner, s_begin, s_end, o_begin, o_end = fields
        s_begin, s_end, o_begin, o_end = [int(e) for e in [s_begin, s_end, o_begin, o_end]]
        sent = sent.lower()

        if context == -1:
            words = sent.split()
        else:
            start = min(int(s_end), int(o_end))
            end = max(int(s_begin), int(o_begin))
            words = get_last_n_words(sent[:start], context) + sent[start:end].split() + get_first_n_words(sent[end:], context)

        get_idx = lambda w: vocab['word'].word2index.get(w, vocab['word']['UNKNOWN'])
        nums = [get_idx(w) for w in words]

        X = np.array(nums).reshape((1, -1))

        filter_pred = filter_model.predict(X, verbose=False).flatten() > filter_threshold
        if filter_pred:
            # guessed that there is a relation
            pred = classifier.predict(X, verbose=False).flatten()
            rel = pred.argmax()
            relation = vocab['rel'].index2word[pred[0]]
            confidence = pred[rel]
            print "\t".join([str(s) for s in [s_id, relation, o_id, confidence]])
