__author__ = 'victor'
import json


class Config(dict):

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def load(cls, from_file):
        with open(from_file) as f:
            return Config(**json.load(f))

    def save(self, to_file):
        with open(to_file, 'wb') as f:
            json.dump(self.__dict__, f, sort_keys=True, indent=2)

    @classmethod
    def default(cls):
        return Config(**{
            'rnn': 'lstm',
            'train': 'supervised',
            'dev': 'kbp_eval',
            'featurizer': 'sent',
            'model': 'single_small',
            'emb_dim': 50,
            'hidden': (512, 512),
            'dropout': 0.5,
            'activation': 'relu',
            'truncate_gradient': 50,
            'reg': 1e-4,
            'optim': 'rmsprop',
            'lr': 1e-2,
            'max_epoch': 20,
        })
