#!/u/nlp/packages/anaconda/bin/safepython
import os
mydir = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from utils import np_softmax
from pprint import pprint
from configs.config import Config
from data.dataset import Dataset, Split
from data.adaptors import *
from data.typecheck import TypeCheckAdaptor
from models import get_model


if __name__ == '__main__':
    root = os.path.join(mydir, 'experiment', 'deploy')
    config = Config.load(os.path.join(root, 'config.json'))
    dataset = Dataset.load(config.data)
    typechecker = TypeCheckAdaptor(os.path.join(mydir, 'data', 'raw', 'typecheck.csv'), dataset.featurizer.vocab)

    model = get_model(config, dataset.featurizer.vocab, typechecker)
    model.load_weights(os.path.join(root, 'best_weights'))

    dev_generator = KBPDataAdaptor().online_to_examples(disable_interrupts=True)
    for ex in dev_generator:
        try:
            feat = dataset.featurizer.featurize(ex, add=False)
        except Exception as e:
            continue
        X, Y, types = dataset.featurizer.to_matrix([feat])
        prob = model.predict(X, verbose=0)['p_relation']
        prob *= typechecker.get_valid_cpu(types[:, 0], types[:, 1])
        pred = prob.argmax(axis=1)
        confidence = np_softmax(prob.flatten())[pred[0]]
        print "\t".join([str(s) for s in [ex.subject_id, pred[0], ex.object_id, confidence]])
