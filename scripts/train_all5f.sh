#!/bin/sh

while read code; do
  args=$(grep -o "$code .*" models-chosen/$code-*-last1.cmd | sed 's/^[^ ]* //; s#=../embeddings#=embeddings#')
  [ -z "$args" ] && { echo Skipping $code; continue; }
  echo For $code have args $args

  args=$(echo "$args" | sed 's/--rnn_layers\(\|_parser\|_tagger\)=[^ ]*//g')
  for reg in {--rnn_layers=0\ --rnn_layers_parser=2\ --rnn_layers_tagger=2,--rnn_layers=2\ --rnn_layers_parser=1\ --rnn_layers_tagger=1}; do
    qsub -q gpu*@dll* -cwd -p -200 -b y -l gpu=1,gpu_cc_min3.5=1 -j y withcuda venv/bin/python3 ud_parser5f.py ud-2.2-conll18/$code/$code $args $reg
  done
done <langs.big
