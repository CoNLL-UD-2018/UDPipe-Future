#!/bin/sh

lang="$1"; shift

for d in logs/ud_parser3.py-*b=$lang,*; do
  target=evals/$lang/${d#logs/}
  for 
  mkdir -p $target

  qsub -q gpu*@dll* -cwd -b y -l gpu=1,gpu_cc_min3.5=1 -j y withcuda venv/bin/python3 ud_parser$variant.py $d$code  $args
done
