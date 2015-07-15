#!/u/nlp/packages/anaconda/bin/safepython
#!/usr/bin/env python
import os
mydir = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from utils import np_softmax
from pprint import pprint
from configs.config import Config
from data.dataset import Split
from data.adaptors import *
from data.typecheck import TypeCheckAdaptor
from models import get_model
import cPickle as pkl

class Cache(object):

    def __init__(self):
        self.examples = []

    def __len__(self):
        return len(self.examples)
    
    def group_by_len(self):
        lens = {}
        for e in self.examples:
            if e.length not in lens:
                lens[e.length] = []
            lens[e.length] += [e]
        return lens

    def batches(self):
        lens = self.group_by_len()
        for l, ex in lens.items():
            yield l, ex

mydir = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    root = os.path.join(mydir, 'experiments', 'deploy')
    config = Config.load(os.path.join(root, 'config.json'))
    with open(os.path.join(root, 'featurizer.pkl')) as f:
        featurizer = pkl.load(f)
    typechecker = TypeCheckAdaptor(os.path.join(mydir, 'data', 'raw', 'typecheck.csv'), featurizer.vocab)

    model = get_model(config, featurizer.vocab, typechecker)
    model.load_weights(os.path.join(root, 'best_weights'))

    dev_generator = KBPDataAdaptor().online_to_examples(disable_interrupts='victor'!=os.environ['USER'])
    cache = Cache()
    max_cache_size = 2**15
    log = open(os.path.join(mydir, 'kbp.log'), 'wb')

    def process_cache(cache):
        for length, examples in cache.batches():
            X, Y, types = featurizer.to_matrix(examples)
            prob = model.predict(X, verbose=0)['p_relation']
            prob *= typechecker.get_valid_cpu(types[:, 0], types[:, 1])
            pred = prob.argmax(axis=1)
            confidence = np_softmax(prob)[np.arange(len(pred)), pred]
            for ex, rel, conf in zip(cache.examples, pred, confidence):
                rel = featurizer.vocab['rel'].index2word[rel]
                if rel == 'no_relation':
                    continue
                print "\t".join([str(s) for s in [ex.orig.subject_id, rel, ex.orig.object_id, conf]])

    for i, ex in enumerate(dev_generator):
        log.write(str(i) + "\n")
        try:
            feat = featurizer.featurize(ex, add=False)
        except Exception as e:
            continue
        if len(cache) < max_cache_size:
            cache.examples += [feat]
            continue
        process_cache(cache)
        cache.examples = []
        log.write(str(i) + "\n")
    process_cache(cache)
    cache.examples = []
    
