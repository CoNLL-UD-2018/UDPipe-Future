#!/bin/sh

while read line; do
  qsub -q cpu-ms.q@* -j y sh convert.sh $line
done <langs
