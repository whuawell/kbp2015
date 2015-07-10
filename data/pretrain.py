__author__ = 'victor'
from dataset import Vocab
import os
import numpy as np


class Senna(Vocab):

    def __init__(self, *kargs, **kwargs):
        super(Senna, self).__init__(unk='UNKNOWN')
        self.root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw', 'senna')
        for word in self.load_wordlist():
            self.add(word)

    def load_wordlist(self):
        with open(os.path.join(self.root, 'words.lst')) as f:
            lines = [l.strip("\n") for l in f]
        return lines

    def load_word2emb(self):
        embs = np.loadtxt(os.path.join(self.root, 'embeddings.txt'))
        words = self.load_wordlist()
        return dict(zip(words, embs))
