# KBP 2015

This repository contains code that were used to develop the LSTM classifier for Stanford's 2015 KBP submission. 

### Dataset

The dataset that were used to train/evaluate the models can be downloaded by running `download_raw.sh` inside `data/raw`.

### Train/evaluation

The model is trained using `train.py` and evaluated using `pred.py`. You need to specify the training/dev/evaluation dataset for both cases. `kbp.py` is to be used with the Stanford internal KBP pipeline as a classifier. `pred.py` and `kbp.py` can only be run after a trained model has been dumped via `train.py`, as they load up the previously trained model weights.

The model used for submission is the `sent` model, which is a recurrent network over the sentence. The optional `scope` parameter for the sentence models truncates the sentence such that only the phrase `scope` words before the first entity token and `scope` words after the last entity token are considered as opposed to the entire sentence. 
