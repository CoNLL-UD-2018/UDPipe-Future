#!/bin/bash

for d in 256 320 384; do
  for reg in 1 2 4 8; do
    for epochs in 50:1e-3 50:1e-3,10:1e-4 50:1e-3,20:1e-4 100:1e-3,20:1e-4; do
      out=${epochs//:1e-[0-9]/}
      out=bin_en_${d}_${reg}_${out//,/-}
      qsub -q cpu-troja.q@* -pe smp 10 -j y -o $out.log python3 binarize.py en $out --dimension $d --regularization $reg --epochs $epochs --threads 10
    done
  done
done
