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
            json.dump(self.__dict__, f, sort_keys=True)

    @classmethod
    def default(cls):
        return Config(**{
            'data': 'data/saves/supervision_evaluation',
            'model': 'concat',
            'emb_dim': 50,
            'hidden': (300,300),
            'dropout': 0.5,
            'activation': 'relu',
            'truncate_gradient': 50,
            'reg': 1e-3,
            'optim': 'rmsprop',
            'lr': 1e-2,
        })
