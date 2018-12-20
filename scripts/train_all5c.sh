#!/bin/sh

while read code; do
  args=$(grep -o "$code .*" models-chosen/$code-*-last1.cmd | sed 's/^[^ ]* //; s#=../embeddings#=embeddings#')
  [ -z "$args" ] && { echo Skipping $code; continue; }
  echo For $code have args $args

#  for reg in --label_smoothing={0.005,0.01,0.03,0.05,0.1,0.2}; do
  for reg in --label_smoothing={0.005,0.01,0.03,0.05,0.1,0.2}\ --parse=0\ --tags=UPOS; do
    qsub -q gpu*@dll* -cwd -p -200 -b y -l gpu=1,gpu_cc_min3.5=1 -j y withcuda venv/bin/python3 ud_parser5c.py ud-2.2-conll18/$code/$code $args $reg
  done
done <langs.big.sub
