#!/bin/bash

for d in 320; do
  for reg in 2 4 5; do
    for epochs in 200:1e-3,20:1e-4; do
      out=${epochs//:1e-[0-9]/}
      out=bin_en_${d}_${reg}_${out//,/-}
      qsub -q gpu*@dll* -l gpu=1,gpu_ram=7G -j y -o $out.log withcuda ~/venvs/tf-1.5-gpu/bin/python binarize.py en $out --dimension $d --regularization $reg --epochs $epochs --threads 4
    done
  done
done
