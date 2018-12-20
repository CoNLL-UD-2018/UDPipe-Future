#!/bin/sh

while read code; do
  args=$(grep -o "$code .*" models-chosen/$code-*-last1.cmd | sed 's/^[^ ]* //; s#=../embeddings#=embeddings#')
  [ -z "$args" ] && { echo Skipping $code; continue; }
  echo For $code have args $args

  for reg in "--dropout=0 --label_smoothing=0" "--dropout=0"; do
    qsub -q gpu*@dll* -p -101 -cwd -b y -l gpu=1,gpu_cc_min3.5=1 -j y withcuda venv/bin/python3 ud_parser5a.py ud-2.2-conll18/$code/$code $args $reg
  done
done <langs.big
