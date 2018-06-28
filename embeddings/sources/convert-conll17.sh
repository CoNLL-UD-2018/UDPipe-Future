#!/bin/bash

lang=$1; shift
source=$1; shift

python3 convert-conll17.py $lang <(xzcat ~/troja/conll2017-data/raw-z-emb/$source/$source.vectors.xz)

for tbank in $@; do
  for f in $lang-conll17.*; do
    ln -s $f $tbank.conll17.${f#*.}
  done
done
