#!/bin/sh

while read code; do
  args=$(grep -o "$code .*" models-chosen/$code-*-last1.cmd | sed 's/^[^ ]* //; s#=../embeddings#=embeddings#')
  [ -z "$args" ] && { echo Skipping $code; continue; }
  echo For $code have args $args

  for reg in --parse=0\ --tags={UPOS,XPOS,FEATS,LEMMAS}; do
    qsub -q gpu*@dll* -cwd -p -210 -b y -l gpu=1,gpu_cc_min3.5=1 -j y withcuda venv/bin/python3 ud_parser5d.py ud-2.2-conll18/$code/$code $args $reg
  done
done <langs.big
