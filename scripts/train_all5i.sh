#!/bin/bash

args=--rnn_layers=2\ --rnn_layers_parser=1\ --rnn_layers_tagger=1
for code in en_{ewt,gum,lines}; do
#  for embedding in binarizee/bin_${code%%_*}_{256,320,384}_{1,2,4,8}_{50,50-10,50-20,100-20}; do
#  for embedding in binarizee/bin_${code%%_*}_{320_{1,2,4,8}_{50-20,100-20},320_4_{50,50-10},{256,384}_4_{50-20,100-20}}; do
  for embedding in binarizee/bin_${code%%_*}_320_{2,4,5}_200-20; do
    qsub -q gpu*@dll* -cwd -p -101 -b y -l gpu=1,gpu_cc_min3.5=1 -j y withcuda venv/bin/python3 ud_parser5i.py ud-2.2-conll18/$code/$code --embeddings=$embedding $args
  done
done
