__author__ = 'victor'
import numpy as np

def np_softmax(y):
    y -= y.max(axis=-1, keepdims=True)
    e = np.exp(y)
    return e / e.sum(axis=-1, keepdims=True)
