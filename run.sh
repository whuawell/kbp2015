#!/usr/bin/env zsh

set -x
set -e

# entity only
python train.py ent

# sentence only, train word embeddings from scratch
python train.py sent

# both, train word embeddings from scratch
python train.py both

# sentence only, use pretrained words and don't keep training
python train.py sent --pretrained pretrained/glove --epoch=100
python train.py sent --pretrained pretrained/senna --epoch=100

# both, use pretrained words and don't keep training
python train.py both --pretrained pretrained/glove --epoch=100
python train.py both --pretrained pretrained/senna --epoch=100

# sentence only, use pretrained words and keep training
python train.py sent --pretrained pretrained/glove --train_words
python train.py sent --pretrained pretrained/senna --train_words

# both, use pretrained words and keep training
python train.py both --pretrained pretrained/glove --train_words
python train.py both --pretrained pretrained/senna --train_words


