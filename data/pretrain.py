__author__ = 'victor'
from dataset import Vocab
import os
import autograd.numpy as np


class Senna(Vocab):

    def __init__(self, *kargs, **kwargs):
        super(Senna, self).__init__(unk='UNKNOWN')
        for word in self.load_wordlist():
            self.add(word)

    def load_wordlist(self):
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw', 'senna')
        with open(os.path.join(root, 'words.lst')) as f:
            lines = [l.strip("\n") for l in f]
        return lines

    def load_word2emb(self):
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw', 'senna')
        embs = np.loadtxt(os.path.join(root, 'embeddings.txt'))
        words = self.load_wordlist()
        return dict(zip(words, embs))

    def load_embeddings(self):
        E = np.random.uniform(low=-0.1, high=0.1, size=(len(self), 50))
        word2emb = self.load_word2emb()
        for i, word in enumerate(self.index2word):
            if word in word2emb:
                E[i] = word2emb[word]
        return E
