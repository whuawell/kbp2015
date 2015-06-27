import numpy as np
import os
import cPickle as pkl
mydir = os.path.dirname(__file__)
from text.vocab import Vocab

def load_pretrained(pretrained, cache=True):
    cache_file = os.path.join(mydir, pretrained, 'cache.pkl')

    if cache and os.path.isfile(cache_file):
        with open(cache_file) as f:
            return pkl.load(f)

    emb = np.loadtxt(os.path.join(mydir, pretrained, 'embeddings.txt'))
    words = open(os.path.join(mydir, pretrained, 'words.lst')).read().split("\n")
    vocab = Vocab(unk=True)
    for word in words:
        vocab.add(word)

    # senna has 'UNKNOWN' and 'PADDING'
    ret = vocab, dict(zip(words, emb))

    if cache:
        with open(cache_file, 'wb') as f:
            pkl.dump(ret, f, protocol=pkl.HIGHEST_PROTOCOL)

    return ret