#!/bin/sh

dir="$1"; shift

for cmd in "$dir"/*.cmd; do
  lang=$(basename ${cmd%%-*})
  echo $cmd $lang
  qsub -q cpu-ms.q@* -pe smp 8 -j y -o /dev/null ./eval.sh $lang $cmd
done
