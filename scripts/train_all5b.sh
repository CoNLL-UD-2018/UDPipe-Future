#!/bin/sh

while read code; do
#for code in cs_fictree; do
  args=$(grep -o "$code .*" models-chosen/$code-*-last1.cmd | sed 's/^[^ ]* //; s#=../embeddings#=embeddings#')
  [ -z "$args" ] && { echo Skipping $code; continue; }
  echo For $code have args $args

  for reg in --confidence_penalty={0.1,0.3,0.5}\ --label_smoothing=0; do
    qsub -q gpu*@dll* -cwd -b y -l gpu=1,gpu_cc_min3.5=1 -j y withcuda venv/bin/python3 ud_parser5b.py ud-2.2-conll18/$code/$code $args $reg
  done
done <langs.big.sub
