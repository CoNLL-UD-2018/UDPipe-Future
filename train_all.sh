#!/bin/sh

for d in ud-2.2-conll18/*/; do
  code=${d%/}
  code=${code##*/}
  qsub -q gpu*@dll* -cwd -b y -l gpu=1,gpu_cc_min3.5=1 withcuda ~/venvs/tf-1.5-gpu/bin/python ud_parser.py $d$code "$@"
done
