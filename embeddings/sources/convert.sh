#!/bin/sh

lang=$1; shift

wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.$lang.vec || exit
wget https://raw.githubusercontent.com/Babylonpartners/fastText_multilingual/master/alignment_matrices/$lang.txt

python3 convert.py $lang

rm -f wiki.$lang.vec $lang.txt

for tbank in $@; do
  for f in $lang.*; do
    ln -s $f $tbank.${f#*.}
  done
done
