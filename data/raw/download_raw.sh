#!/usr/bin/env bash
wget https://dl.dropboxusercontent.com/u/9015381/datasets/kbp/evaluation.tsv
wget https://dl.dropboxusercontent.com/u/9015381/datasets/kbp/test.sample.tsv
wget https://dl.dropboxusercontent.com/u/9015381/datasets/kbp/supervision.csv
wget https://dl.dropboxusercontent.com/u/9015381/datasets/kbp/typecheck.csv
wget https://dl.dropboxusercontent.com/u/9015381/datasets/kbp/senna.zip
unzip senna.zip
rm senna.zip
wget https://dl.dropboxusercontent.com/u/9015381/datasets/kbp/gabor_report.txt